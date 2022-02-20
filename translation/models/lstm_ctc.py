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


class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_dim, directions=2):
        super(Attention, self).__init__()
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


class Decoder(nn.Module):
    def __init__(self, index_dim, embedding_dim=256, hidden_dim=256, attention_dim=128, enc_directions=2):
        super(Decoder, self).__init__()
        self.attention = Attention(hidden_dim, attention_dim)
        self.embedding = nn.Embedding(index_dim, embedding_dim)
        self.linear = nn.Linear(hidden_dim * enc_directions, index_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim * enc_directions, hidden_dim * enc_directions, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, x, hidden, enc_out):
        context_vector = self.attention(hidden[0], enc_out)
        emb = self.relu(self.embedding(x))
        emb = torch.cat([context_vector.unsqueeze(1), emb], dim=-1)
        out, hidden = self.lstm(emb, hidden)
        out = self.linear(out)
        return out, hidden


class CTCDecoder(nn.Module):
    def __init__(self, index_dim, hidden_dim=512):
        super(CTCDecoder, self).__init__()
        self.ctc_linear = nn.Linear(hidden_dim, index_dim)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        return self.softmax(self.ctc_linear(x))


class Seq2Seq(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(encoder_index_dim)
        self.ctc = CTCDecoder(decoder_index_dim)
        self.decoder = Decoder(decoder_index_dim)

        self.encoder_index_dim = encoder_index_dim
        self.decoder_index_dim = decoder_index_dim

    def forward(self, x, y, use_teacher_forcing):
        assert x.shape[0] == y.shape[0]
        if y is not None:
            assert x.device == y.device

        batch_size = x.shape[0]
        device = x.device

        if y is not None:
            target_length = y.shape[1]
        else:
            target_length = x.shape[1]

        enc_out, hidden = self.encoder(x)

        hidden = [hidden[0].view(self.encoder.layers, batch_size, -1).mean(dim=0).unsqueeze(0),
                  hidden[1].view(self.encoder.layers, batch_size, -1).mean(dim=0).unsqueeze(0)]
        ctc = self.ctc(enc_out)
        out = y[:, 0].unsqueeze(1)

        outputs = torch.zeros(batch_size, target_length, self.decoder_index_dim).to(device)

        if use_teacher_forcing and y is not None:
            for di in range(1, target_length):
                out, hidden = self.decoder(out.to(dtype=torch.long), hidden, enc_out)
                outputs[:, di:di + 1] = out

                out = y[:, di].unsqueeze(1)
        else:
            for di in range(1, target_length):
                out, hidden = self.decoder(out.to(dtype=torch.long), hidden, enc_out)
                outputs[:, di:di + 1] = out

                out = out.argmax(dim=-1)

        return ctc, outputs
