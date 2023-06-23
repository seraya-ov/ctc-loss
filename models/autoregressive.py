import torch
import torch.nn as nn

from decoders import Decoder
from data.utils import generate_square_subsequent_mask
from models.encoders import Encoder
from models.models import PositionalEncoding


class Seq2Seq(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim, hidden_dim=256):
        super().__init__()
        self.encoder = Encoder(encoder_index_dim, hidden_dim=hidden_dim)
        self.decoder = Decoder(decoder_index_dim, hidden_dim=hidden_dim)

        self.encoder_index_dim = encoder_index_dim
        self.decoder_index_dim = decoder_index_dim

    def forward(self, x, y, use_teacher_forcing):
        if y is not None:
            assert x.device == y.device
            assert x.shape[0] == y.shape[0]

        batch_size = x.shape[0]
        device = x.device

        if y is not None:
            target_length = y.shape[1]
        else:
            target_length = x.shape[1]

        enc_out, hidden = self.encoder(x)
        out = x[:, 0].unsqueeze(1)

        hidden = [hidden[0].view(self.encoder.layers, batch_size, -1).mean(dim=0).unsqueeze(0),
                  hidden[1].view(self.encoder.layers, batch_size, -1).mean(dim=0).unsqueeze(0)]

        outputs = torch.zeros(batch_size, target_length, self.decoder_index_dim).to(device)
        outputs[:, 0, 3] = 1  # SOS

        if use_teacher_forcing and y is not None:
            outputs[:, 1:], _ = self.decoder(y[:, :-1].to(dtype=torch.long), hidden, enc_out)
        else:
            for di in range(1, target_length):
                out, hidden = self.decoder(out.to(dtype=torch.long), hidden, enc_out)
                outputs[:, di:di + 1] = out

                out = out.argmax(dim=-1)
        return outputs


class Transformer(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim,
                 heads=8, hidden_dim=256, encoder_layers=6, decoder_layers=6,
                 dropout=0.1, dim_feedforward=512):
        super().__init__()
        self.model_type = "Transformer"
        self.dim_model = hidden_dim
        self.output_dim = decoder_index_dim

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.encoder_embedding = nn.Embedding(encoder_index_dim, hidden_dim, padding_idx=0)
        self.decoder_embedding = nn.Embedding(decoder_index_dim, hidden_dim, padding_idx=0)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward
        )
        self.out = nn.Linear(hidden_dim, decoder_index_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder_embedding.weight.data.uniform_(-initrange, initrange)

    def encode(self, x, src_mask=None):
        return self.transformer.encoder(x, src_mask)

    def decode(self, y, memory, trg_mask=None):
        return self.transformer.decoder(y, memory, trg_mask)

    def forward(self, x, y, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask,
                use_teacher_forcing=False):

        batch_size = x.shape[0]
        device = x.device

        if y is not None:
            target_length = y.shape[1]
        else:
            target_length = x.shape[1]

        if use_teacher_forcing and y is not None:
            x = self.encoder_embedding(x).permute(1, 0, 2) * math.sqrt(self.dim_model)
            x = self.pos_encoder(x)
            y = self.decoder_embedding(y).permute(1, 0, 2) * math.sqrt(self.dim_model)
            y = self.pos_encoder(y)
            transformer_out = self.transformer(x, y, src_mask=src_mask, tgt_mask=tgt_mask,
                                               src_key_padding_mask=src_padding_mask,
                                               tgt_key_padding_mask=tgt_padding_mask).permute(1, 0, 2)
            output = self.out(transformer_out)
        else:
            y = x[:, 0].unsqueeze(1)
            output = torch.zeros(batch_size, 1, self.output_dim).to(device)

            x = self.encoder_embedding(x).permute(1, 0, 2) * math.sqrt(self.dim_model)
            x = self.pos_encoder(x)
            memory = self.encode(x, src_mask)
            for _ in range(target_length * 2):
                y_input = self.decoder_embedding(y).permute(1, 0, 2) * math.sqrt(self.dim_model)
                y_input = self.pos_encoder(y_input)
                pred = self.out(self.decode(y_input, memory, (
                    generate_square_subsequent_mask(y_input.shape[0]).to(dtype=torch.bool))).permute(1, 0, 2))
                next_item = torch.argmax(pred[:, -1:], dim=-1)
                y = torch.cat((y, next_item), dim=1)
                output = torch.cat((output, pred[:, -1:]), dim=1)
            output = output[:, 1:]
        return output
