import math
import warnings

import torch
import torch.nn as nn

import numpy as np
from scipy.optimize import linear_sum_assignment

from decoders import CTCDecoder, CTCTransformerDecoder
from encoders import Encoder
from aligner import PermutationModule, LengthClassDuplicationModule, GroupingModule, \
    LengthRegDuplicationModule


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Seq2CTC(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim, hidden_dim=256):
        super(Seq2CTC, self).__init__()
        self.encoder = Encoder(encoder_index_dim, hidden_dim=hidden_dim)
        self.ctc = CTCDecoder(decoder_index_dim)
        # self.output_mask = OutputMask(2 * hidden_dim)
        self.encoder_index_dim = encoder_index_dim

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        device = x.device

        input_length = x.shape[1]

        enc_out, _ = self.encoder(x)

        enc_out = nn.functional.pad(enc_out, (0, 0, 0, enc_out.shape[1]), "constant", 0)

        # mask = self.output_mask(enc_out.permute(1, 0, 2))
        # ctc = torch.mul(mask.unsqueeze(-1), self.ctc(enc_out.permute(1, 0, 2)))

        return self.ctc(enc_out.permute(1, 0, 2))


class Transformer2CTC(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim, embedding_dim=256,
                 heads=4, hidden_dim=256, layers=4, dropout=0.3):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layers)
        self.encoder = nn.Embedding(encoder_index_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = CTCDecoder(decoder_index_dim, hidden_dim)
        # self.output_mask = OutputMask(hidden_dim=hidden_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask):
        # x = nn.functional.pad(x, (0, 0, 0, x.shape[0]), "constant", 0)
        x = self.encoder(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, mask)
        # out_mask = self.output_mask(output)
        # output = torch.mul(out_mask.unsqueeze(-1), self.decoder(output))
        return self.decoder(output)


class Transformer2TransformerCTC(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim, embedding_dim=256,
                 heads=4, hidden_dim=256, layers=4, dropout=0.3):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layers)
        self.encoder = nn.Embedding(encoder_index_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = CTCTransformerDecoder(decoder_index_dim, hidden_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask):
        # x = nn.functional.pad(x, (0, 0, 0, x.shape[0]), "constant", 0)
        x = self.encoder(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, mask)
        # out_mask = self.output_mask(output)
        # output = torch.mul(out_mask.unsqueeze(-1), self.decoder(output))
        return self.decoder(output)


class AligNART(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim, embedding_dim=256,
                 heads=4, hidden_dim=256, layers=4, dropout=0.3):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layers)
        self.encoder = nn.Embedding(encoder_index_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = CTCTransformerDecoder(decoder_index_dim, hidden_dim)
        self.permute = PermutationModule(hidden_dim, hidden_dim, hidden_dim)
        self.group = GroupingModule(hidden_dim, hidden_dim)
        self.duplicate = LengthClassDuplicationModule(hidden_dim, hidden_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask, gt=None):
        x = self.encoder(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, mask).permute(1, 0, 2)
        duplication_probs, duplication_matrix = self.duplicate(output)
        output = self.pos_encoder(gt[0] @ output) if gt is not None else self.pos_encoder(duplication_matrix @ output)
        permutation_matrix = self.permute(output)
        if gt is not None:
            output = gt[1] @ output
        else:
            permutation_matrix = permutation_matrix.cpu().detach().numpy()
            for i in range(permutation_matrix.shape[0]):
                try:
                    permutation_matrix[i] += 1e-10
                    xs, ys = linear_sum_assignment(-1 * np.log(permutation_matrix[i]))
                    permutation_matrix[i] *= 0
                    permutation_matrix[i, xs, ys] = 1
                except Exception as e:
                    permutation_matrix[i] = np.floor(permutation_matrix[i])
                    warnings.warn("Warning: permutation_matrix {}".format(e))
            permutation_matrix = torch.Tensor(permutation_matrix).to(device=x.device)
            output = permutation_matrix @ output
        grouping_probs, grouping_matrix = self.group(output)
        output = gt[2] @ output if gt is not None else grouping_matrix @ output
        output = self.decoder(output.permute(1, 0, 2))
        return output, (duplication_probs, duplication_matrix, permutation_matrix, grouping_probs, grouping_matrix)


class CTCAligner(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim, embedding_dim=256,
                 heads=4, hidden_dim=256, layers=4, dropout=0.3):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layers)
        self.encoder = nn.Embedding(encoder_index_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = CTCDecoder(decoder_index_dim, hidden_dim)
        self.permute = PermutationModule(hidden_dim, hidden_dim, hidden_dim)
        self.duplicate = LengthRegDuplicationModule(hidden_dim, hidden_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask, gt=None):
        x = self.encoder(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, mask).permute(1, 0, 2)
        duplication_matrix = self.duplicate(output)
        # if gt is not None:
        #     print(gt[0].shape, output.shape)
        output = gt[0] @ output if gt is not None else self.pos_encoder(duplication_matrix @ output)
        permutation_matrix = self.permute(output)
        if gt is not None:
            output = gt[1] @ output
        else:
            permutation_matrix = permutation_matrix.cpu().detach().numpy()
            for i in range(permutation_matrix.shape[0]):
                try:
                    permutation_matrix[i] += 1e-10
                    xs, ys = linear_sum_assignment(-1 * np.log(permutation_matrix[i]))
                    permutation_matrix[i] *= 0
                    permutation_matrix[i, xs, ys] = 1
                except Exception as e:
                    permutation_matrix[i] = np.floor(permutation_matrix[i])
                    warnings.warn("Warning: permutation_matrix {}".format(e))
            permutation_matrix = torch.Tensor(permutation_matrix).to(device=x.device)
            output = permutation_matrix @ output
        output = self.decoder(output.permute(1, 0, 2))
        return output, (duplication_matrix, permutation_matrix)
