import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_dim=256, heads=8, dropout=0.1):
        super(Attention, self).__init__()
        self.kqv_proj = nn.Linear(hidden_dim, 3 * attention_dim)
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // heads
        self.heads = heads
        self.attention = torch.nn.MultiheadAttention(attention_dim, heads, dropout=dropout, batch_first=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        Q, K, V = self.kqv_proj(x).reshape((x.shape[0], x.shape[1], 3 * self.attention_dim)).chunk(3, dim=-1)
        context_vector, _ = self.attention(Q, K, V)
        return context_vector

class GatedAttention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super(GatedAttention, self).__init__()
        self.Q = nn.Linear(hidden_dim, attention_dim)
        self.K = nn.Linear(hidden_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        Qw = self.Q(x)
        Kw = self.K(x)
        g = self.sigmoid(self.V(Qw)).squeeze(-1)
        D = torch.diag_embed(g).to(device=x.device)
        M = torch.diag(torch.ones((x.shape[1]), dtype=torch.float32) + -torch.inf).repeat(x.shape[0], 1, 1).to(device=x.device)
        I = torch.eye(x.shape[1], dtype=torch.float16).repeat(x.shape[0], 1, 1).to(device=x.device)
        P = self.softmax(M + Qw@(Kw.permute(0, 2, 1)))
        P = D + (I - D) @ P
        return P


class LSTMAttention(nn.Module):
    def __init__(self, hidden_dim, attention_dim, directions=2):
        super(LSTMAttention, self).__init__()
        self.W1 = nn.Linear(hidden_dim * directions, attention_dim)
        self.W2 = nn.Linear(hidden_dim * directions, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, enc_out):
        score = self.V(self.tanh(self.W1(enc_out) + self.W2(hidden.view(hidden.shape[1], -1).unsqueeze(1)))).squeeze(-1)
        attention_weights = self.softmax(score)
        assert len(attention_weights.shape) == 2

        context_vector = attention_weights.unsqueeze(-1) * enc_out
        context_vector = context_vector.sum(axis=1)
        return context_vector
