import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, index_dim, embedding_dim=256, hidden_dim=256, dropout=0.2, layers=2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(index_dim, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layers, dropout=dropout, bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.directions = 2
        self.layers = layers

    def forward(self, x):
        mask = (x != 0).to(torch.long)
        lengths = mask.sum(dim=1).to('cpu')

        emb = self.embedding(x)
        emb = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        out, hidden = self.lstm(emb)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.dropout(out), hidden
