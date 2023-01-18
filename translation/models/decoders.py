import torch
import torch.nn as nn


class CTCDecoder(nn.Module):
    def __init__(self, index_dim, hidden_dim=512):
        super(CTCDecoder, self).__init__()
        self.ctc_linear = nn.Linear(hidden_dim, index_dim)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = torch.repeat_interleave(x.permute(1, 0, 2), 1, dim=1)
        return self.softmax(self.ctc_linear(x))


class CTCTransformerDecoder(nn.Module):
    def __init__(self, index_dim, hidden_dim=512, heads=4):
        super(CTCTransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(index_dim, hidden_dim, padding_idx=0)
        self.decoder = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim)
        self.ctc_linear = nn.Linear(hidden_dim, index_dim)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        # x = torch.repeat_interleave(x.permute(1, 0, 2), 1, dim=1)
        y = self.embedding(self.softmax(self.ctc_linear(x)).argmax(-1))
        return self.softmax(self.ctc_linear(self.decoder(y.detach(), x))).permute(1, 0, 2)
