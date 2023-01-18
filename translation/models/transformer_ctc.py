import math
import torch
import torch.nn as nn

from decoders import CTCDecoder, CTCDecoderN


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
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


class Transformer2CTC(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim, embedding_dim=256,
                 heads=4, hidden_dim=256, layers=4, dropout = 0.3):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layers)
        self.encoder = nn.Embedding(encoder_index_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = CTCDecoder(decoder_index_dim, hidden_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask):
        x = nn.functional.pad(x, (0, 0, 0, x.shape[0]), "constant", 0)
        x = self.encoder(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, mask)
        return self.decoder(output)
