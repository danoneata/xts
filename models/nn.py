from typing import List, Union, Dict

import collections
import enum
import pdb

from types import SimpleNamespace

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from torchvision.models import resnet18

import src.dataset

from hparams import hparams
from models.speech_synthesis import Tacotron2


get_same_padding = lambda s: (s - 1) // 2


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        kwargs = dict(kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = nn.Conv3d(1, 4, **kwargs)
        self.conv2 = nn.Conv3d(4, 4, **kwargs)
        self.conv3 = nn.Conv3d(4, 1, **kwargs)
        self.gru = nn.GRU(64 * 64, 16)
        self.linear = nn.Linear(16, 3 * 1025)

    def forward(self, x):
        B, S, H, W = x.shape
        x = x.float() / 255  # scale data in [0, 1]
        x = x.unsqueeze(1)  # B x 1 x S x H x W
        x = self.conv1(x)  # B x C x S x H x W
        # x = self.max_pool(x.reshape(B, -1 , H, W))
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(2, 0, 1, 3, 4).reshape(S, B, -1)
        x, _ = self.gru(x)
        x = x.permute(1, 0, 2)
        x = self.linear(x)
        x = x.reshape(B, S, 3, 1025).reshape(B, S * 3, 1025)
        x = F.pad(x, pad=(0, 0, 0, 14))
        x = x.permute(0, 2, 1)
        return x


class Sven(nn.Module):
    def __init__(self, params):
        super(Sven, self).__init__()
        self.params = SimpleNamespace(**params)
        # use 3d convolution to extract features in time
        self.conv0_3d = nn.Sequential(
            nn.Conv3d(
                1,  # single channel as the images are converted to gray-scale
                self.params.conv3d_num_filters,
                kernel_size=self.params.conv3d_kernel_size,
                stride=(1, 1, 1),
                padding=tuple(get_same_padding(s) for s in self.params.conv3d_kernel_size),
                bias=False,
            ),
            nn.BatchNorm3d(self.params.conv3d_num_filters),
            nn.ReLU(inplace=True),
        )
        self.encoder = resnet18()
        # update first layer of the resnet to match the `conv0_3d` layer
        self.encoder.conv1 = nn.Conv2d(
            self.params.conv3d_num_filters,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        # drop the last layer of the resnet corresponding to the softmax classification
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.encoder_rnn = nn.LSTM(
            input_size=hparams.encoder_embedding_dim,
            hidden_size=hparams.encoder_embedding_dim,
            num_layers=self.params.encoder_rnn_num_layers,
            dropout=self.params.encoder_rnn_dropout,
        )
        self.decoder = Tacotron2(hparams)

    def encode(self, x):
        B, S, H, W = x.shape
        # B, 1, S, H, W
        x = x.unsqueeze(1)
        # B, conv3d_num_filters, S, H, W
        x = self.conv0_3d(x)
        # B, S, conv3d_num_filters, H, W
        x = x.permute(0, 2, 1, 3, 4)
        # BS, conv3d_num_filters, H, W
        x = x.reshape(B * S, self.params.conv3d_num_filters, H, W)
        # BS, Dx, 1, 1
        x = self.encoder(x)
        # B, S, Dx
        x = x.squeeze().view(B, S, hparams.encoder_embedding_dim)
        # S, B, Dx
        x = x.permute(1, 0, 2)
        # S, B, Dx
        x, _ = self.encoder_rnn(x)
        # B, S, Dx
        x = x.permute(1, 0, 2)
        return x

    def forward(self, x_y):
        # Batch decoding with teacher forcing
        x, y = x_y
        x = self.encode(x)
        return self.decoder(x, y)

    def predict(self, x):
        # Step-by-step decoding
        x = self.encode(x)
        _, y = self.decoder.predict(x)
        return y

    def predict2(self, x_y):
        # Step-by-step decoding with auxilary information; equivalent to the `forward` method.
        x, y = x_y
        x = self.encode(x)
        _, y = self.decoder.predict2(x, y)
        return y
