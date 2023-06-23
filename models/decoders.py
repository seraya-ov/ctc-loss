import torch
import torch.nn as nn

from attention import LSTMAttention
from models.models import PositionalEncoding


class CTCDecoder(nn.Module):
    def __init__(self, index_dim, hidden_dim=512):
        super(CTCDecoder, self).__init__()
        self.ctc_linear = nn.Linear(hidden_dim, index_dim)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.ctc_linear(x))


class CTCTransformerDecoder(nn.Module):
    def __init__(self, index_dim, hidden_dim=256, heads=16, dim_feedforward=2048, layers=6, length_mul=3, dropout=0.1):
        super(CTCTransformerDecoder, self).__init__()
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim,
                                                    nhead=heads,
                                                    dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layers, layers)
        self.ctc_linear = nn.Linear(hidden_dim, index_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

    def forward(self, x):
        return self.softmax(self.ctc_linear(self.decoder(self.pos_encoder(x.detach()), x)))


class Decoder(nn.Module):
    def __init__(self, index_dim, embedding_dim=256, hidden_dim=256, attention_dim=128, enc_directions=2):
        super(Decoder, self).__init__()
        self.attention = LSTMAttention(hidden_dim, attention_dim)
        self.embedding = nn.Embedding(index_dim, embedding_dim)
        self.linear = nn.Linear(hidden_dim * enc_directions, index_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim * enc_directions, hidden_dim * enc_directions, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, x, hidden, enc_out):
        context_vector = self.attention(hidden[0], enc_out)
        emb = self.relu(self.embedding(x))
        emb = torch.cat([context_vector.unsqueeze(1).repeat(1, emb.shape[1], 1), emb], dim=-1)
        out, hidden = self.lstm(emb, hidden)
        out = self.linear(out)
        return out, hidden
