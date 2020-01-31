import pdb
import sys

from toolz import concat, partition, take

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

sys.path.insert(0, "tacotron2")
from tacotron2.layers import ConvNorm, LinearNorm
from tacotron2.model import Prenet, Postnet


class Tacotron2(nn.Module):
    def __init__(self, hparams, dataset_params):
        super(Tacotron2, self).__init__()
        self.len_inp = dataset_params["len-inp"]
        self.len_out = dataset_params["len-out"]
        self.out_inp_factor = int(np.ceil(self.len_out / self.len_inp))

        self.n_mel_channels = hparams.n_mel_channels
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim

        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels,
            [hparams.prenet_dim, hparams.prenet_dim],
        )

        self.decoder_rnn = nn.LSTM(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim,
        )

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channels,
        )

        self.postnet = Postnet(hparams)

    def shift(self, y):
        B, _, D = y.shape
        return torch.cat((torch.zeros(B, 1, D).to(y.device), y[:, :-1]), dim=1)

    def post_process(self, z):
        # B, D, S
        z = z.permute(0, 2, 1)
        # B, D, S
        z_post = self.postnet(z) + z
        # B, S, D'
        z_post = z_post.permute(0, 2, 1)
        # B, S, D'
        z = z.permute(0, 2, 1)
        return z, z_post

    def get_selected_indices(self):
        indices = range(self.len_inp * self.out_inp_factor)
        num_extra_elems = self.out_inp_factor * self.len_inp - self.len_out
        selected_groups = set(np.random.choice(self.len_inp, num_extra_elems, replace=False))
        selected_indices = list(concat(
            take(self.out_inp_factor - 1, group) if i in selected_groups else group
            for i, group in enumerate(partition(self.out_inp_factor, indices))
        ))
        return selected_indices

    def upsample_video(self, x):
        # Upsamples the video sequence to the size of audio sequence by
        # repeating the feature frames
        # return x.repeat_interleave(3, dim=1)
        x = x.repeat_interleave(self.out_inp_factor, dim=1)
        i = self.get_selected_indices()
        return x[:, i]

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
        x = self.upsample_video(x)
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

        return self.post_process(z)

    def predict(self, x):
        x = self.upsample_video(x)
        B, S, _ = x.shape

        z = torch.zeros(B, 1, self.n_mel_channels).to(x.device)
        h = torch.zeros(1, B, self.decoder_rnn_dim).to(x.device)
        c = torch.zeros(1, B, self.decoder_rnn_dim).to(x.device)

        y = []

        for i in range(S):
            # B, 1, D1
            z = self.prenet(z)
            # B, 1, Dx + D1
            z = torch.cat((z, x[:, i].unsqueeze(1)), dim=2)
            # 1, B, Dx + D1
            z = z.permute(1, 0, 2)
            # 1, B, D2
            z, (h, c) = self.decoder_rnn(z, (h, c))
            # 1, B, D2
            z = F.dropout(z, self.p_decoder_dropout, self.training)
            # S, D2, 1
            z = z.permute(1, 0, 2)

            # B, 1, Dx + D2
            z = torch.cat((z, x[:, i].unsqueeze(1)), dim=2)
            # B, 1, D3
            z = self.linear_projection(z)

            y.append(z)

        y = torch.cat(y, dim=1)
        return self.post_process(y)


    def predict2(self, x, t):
        x = self.upsample_video(x)
        B, S, _ = x.shape

        z = torch.zeros(B, 1, self.n_mel_channels).to(x.device)
        h = torch.zeros(1, B, self.decoder_rnn_dim).to(x.device)
        c = torch.zeros(1, B, self.decoder_rnn_dim).to(x.device)

        y = []

        for i in range(S):
            # B, 1, D1
            z = self.prenet(z)
            # B, 1, Dx + D1
            z = torch.cat((z, x[:, i].unsqueeze(1)), dim=2)
            # 1, B, Dx + D1
            z = z.permute(1, 0, 2)
            # 1, B, D2
            z, (h, c) = self.decoder_rnn(z, (h, c))
            # 1, B, D2
            z = F.dropout(z, self.p_decoder_dropout, self.training)
            # S, D2, 1
            z = z.permute(1, 0, 2)

            # B, 1, Dx + D2
            z = torch.cat((z, x[:, i].unsqueeze(1)), dim=2)
            # B, 1, D3
            z = self.linear_projection(z)

            y.append(z)
            z = t[:, i].unsqueeze(1)

        y = torch.cat(y, dim=1)
        return self.post_process(y)
