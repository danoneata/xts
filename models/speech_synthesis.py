import pdb
import sys

import torch

from torch import nn
from torch.nn import functional as F

sys.path.insert(0, "tacotron2")
from tacotron2.layers import ConvNorm, LinearNorm
from tacotron2.model import Prenet, Postnet


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim

        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim],
        )

        self.decoder_rnn = nn.LSTM(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim,
        )

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step,
        )

        self.postnet = Postnet(hparams)

    def shift(self, y):
        B, _, D = y.shape
        return torch.cat((torch.zeros(B, 1, D).to(y.device), y[:, :-1]), dim=1)

    def forward(self, x, y):
        # Glossary:
        # B →  batch size
        # S →  sequence size
        # D →  feature size

        # x.shape →  B, S / 3, Dx
        # z.shape →  B, S    , Dz

        # shifts labels such that they are not seen at training
        z = self.shift(y)

        # B, S, D1
        z = self.prenet(z)

        # B, S, Dx
        # upsamples the `x` sequence to the size of `y` by repeating elements
        x = x.repeat_interleave(3, dim=1)
        # B, S, Dx + D1
        z = torch.cat((z, x), dim=2)

        # S, B, Dx + D1
        z = z.permute(1, 0, 2)
        # S, B, D2
        z, _ = self.decoder_rnn(z)
        # S, B, D2
        z = F.dropout(z, self.p_decoder_dropout, self.training)
        # S, D2, B
        z = z.permute(1, 0, 2)

        # B, S, Dx + D2
        z = torch.cat((z, x), dim=2)
        # B, S, D3
        z = self.linear_projection(z)

        # B, D3, S
        z = z.permute(0, 2, 1)
        # B, D4, S
        z_post = self.postnet(z) + z
        # B, S, D4
        z_post = z_post.permute(0, 2, 1)
        # B, S, D4
        z = z.permute(0, 2, 1)

        return z, z_post
