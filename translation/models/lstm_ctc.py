import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from decoders import CTCDecoder


class Seq2CTC(nn.Module):
    def __init__(self, encoder_index_dim, decoder_index_dim, hidden_dim=256):
        super(Seq2CTC, self).__init__()
        self.encoder = Encoder(encoder_index_dim, hidden_dim=hidden_dim)
        self.ctc = CTCDecoder(decoder_index_dim)
        # self.output_mask = OutputMask(2 * hidden_dim)
        self.encoder_index_dim = encoder_index_dim

    def forward(self, x, y=None):

        enc_out, _ = self.encoder(x)

        enc_out = nn.functional.pad(enc_out, (0, 0, 0, enc_out.shape[1]), "constant", 0)

        # mask = self.output_mask(enc_out.permute(1, 0, 2))
        # ctc = torch.mul(mask.unsqueeze(-1), self.ctc(enc_out.permute(1, 0, 2)))

        return self.ctc(enc_out.permute(1, 0, 2))

