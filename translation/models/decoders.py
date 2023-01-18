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


class CTCDecoderN(nn.Module):
    def __init__(self, index_dim, hidden_dim=512):
        super(CTCDecoderN, self).__init__()
        self.ctc_linear = nn.Linear(hidden_dim, index_dim)
        self.ctc_weight = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = torch.bmm(torch.bmm(x, self.ctc_weight(x).permute(0, 2, 1)), x)
        return self.softmax(self.ctc_linear(x)).permute(1, 0, 2)
