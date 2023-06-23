import math

import torch
import torch.nn as nn

import numpy as np
from scipy.optimize import linear_sum_assignment

from decoders import CTCDecoder, CTCTransformerDecoder, Decoder
from encoders import Encoder
from aligner import PermutationModule, LengthClassDuplicationModule, GroupingModule
from data.utils import generate_square_subsequent_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model, dtype=torch.float16)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LSTM2CTC(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim, hidden_dim=256):
        super().__init__()
        self.encoder = Encoder(encoder_index_dim, hidden_dim=hidden_dim)
        self.ctc = CTCDecoder(decoder_index_dim)
        self.encoder_index_dim = encoder_index_dim

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        enc_out = nn.functional.pad(enc_out, (0, 0, 0, enc_out.shape[1]), "constant", 0)

        return self.ctc(enc_out)


class Transformer2CTC(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim, embedding_dim=300,
                 heads=4, hidden_dim=256, layers=6, dropout=0.1, length_mul=3):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layers)
        self.embedding = nn.Embedding(encoder_index_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim * length_mul)
        self.length_mul = length_mul
        self.decoder = CTCDecoder(decoder_index_dim, embedding_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask):
        x = self.embedding(x).permute(1, 0, 2) * math.sqrt(self.embedding_dim)
        output = self.linear(self.transformer_encoder(x, mask).permute(1, 0, 2)).reshape((x.shape[1], x.shape[0] * self.length_mul, -1))
        return self.decoder(output)


class Transformer2TransformerCTC(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim, embedding_dim=256,
                 heads=16, hidden_dim=512, layers=6, dropout = 0.1, length_mul=3):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, heads, 2048, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layers)
        self.embedding = nn.Embedding(encoder_index_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = CTCTransformerDecoder(decoder_index_dim, hidden_dim, layers=layers)
        self.linear = nn.Linear(hidden_dim, hidden_dim * length_mul)
        self.length_mul = length_mul

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask):
        x = self.embedding(x).permute(1, 0, 2) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, mask)
        output = self.linear(output.permute(1, 0, 2)).reshape((x.shape[1], x.shape[0] * self.length_mul, -1)).permute(1, 0, 2)
        return self.decoder(output).permute(1, 0, 2)


class AligNART(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim, embedding_dim=256,
                 heads=8, hidden_dim=512, layers=6, dropout = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, heads, 2048, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layers)

        self.embedding = nn.Embedding(encoder_index_dim, embedding_dim)
        self.embedding_dim = embedding_dim

        self.decoder = CTCTransformerDecoder(decoder_index_dim, embedding_dim, layers=6, heads=8)

        self.permute = PermutationModule(embedding_dim, hidden_dim, hidden_dim)
        self.group = GroupingModule(embedding_dim, hidden_dim)
        self.duplicate = LengthClassDuplicationModule(embedding_dim, hidden_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask, gt=None, teacher_forcing=False):
        x = self.embedding(x).permute(1, 0, 2) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)

        enc_output = self.transformer_encoder(x, mask).permute(1, 0, 2)

        duplication_probs, duplication_matrix = self.duplicate(enc_output)

        output = self.pos_encoder(gt[0].to(dtype=torch.float32)@enc_output) if gt is not None and teacher_forcing else self.pos_encoder(duplication_matrix@enc_output)
        permutation_matrix = self.permute(output)
        if teacher_forcing and gt is not None:
            output = gt[1].to(dtype=torch.float32)@output
        else:
            permutation_matrix_c = permutation_matrix.cpu().detach().numpy()
            for i in range(permutation_matrix_c.shape[0]):
                permutation_matrix_c[i] += 1e-16
                xs, ys = linear_sum_assignment(-1 * np.log(permutation_matrix_c[i]))
                permutation_matrix_c[i] *= 0
                permutation_matrix_c[i, xs, ys] = 1
            permutation_matrix_c = torch.Tensor(permutation_matrix_c).to(device=x.device, dtype=torch.float32)
            output = permutation_matrix_c@output
        grouping_probs, grouping_matrix = self.group(output)
        if gt is not None and teacher_forcing:
            output = gt[2].to(dtype=torch.float32)@output
        else:
            output = grouping_matrix@output
        output = self.decoder(output.permute(1, 0, 2)).permute(1, 0, 2)
        return output, (duplication_probs, duplication_matrix, permutation_matrix, grouping_probs, grouping_matrix)


class CTCAligner(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim, embedding_dim=256,
                 heads=8, hidden_dim=512, layers=6, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, heads, 2048, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layers)
        self.embedding = nn.Embedding(encoder_index_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = CTCTransformerDecoder(decoder_index_dim, embedding_dim, layers=6, heads=8)
        self.permute = PermutationModule(embedding_dim, hidden_dim, hidden_dim)
        self.duplicate = LengthClassDuplicationModule(embedding_dim, hidden_dim)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask, gt=None, teacher_forcing=False):
        x = self.embedding(x).permute(1, 0, 2) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)

        enc_output = self.transformer_encoder(x, mask).permute(1, 0, 2)
        duplication_probs, duplication_matrix = self.duplicate(enc_output)

        output = gt[0].to(dtype=torch.float32)@enc_output if gt is not None and teacher_forcing else self.pos_encoder(duplication_matrix@enc_output)
        permutation_matrix = self.permute(output)
        if teacher_forcing and gt is not None:
            output = gt[1].to(dtype=torch.float32)@output
        else:
            permutation_matrix_c = permutation_matrix.cpu().detach().numpy()
            for i in range(permutation_matrix_c.shape[0]):
                permutation_matrix_c[i] += 1e-16
                xs, ys = linear_sum_assignment(-1 * np.log(permutation_matrix_c[i]))
                permutation_matrix_c[i] *= 0
                permutation_matrix_c[i, xs, ys] = 1
            permutation_matrix_c = torch.Tensor(permutation_matrix_c).to(device=x.device, dtype=torch.float32)
            output = permutation_matrix_c@output
        output = self.decoder(output.permute(1, 0, 2)).permute(1, 0, 2)
        return output, (duplication_probs, duplication_matrix, permutation_matrix)
