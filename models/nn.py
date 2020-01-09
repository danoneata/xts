from typing import List, Union, Dict

import collections
import enum
import pdb

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from torchvision.models import resnet18

import src.dataset

from hparams import hparams
from models.speech_synthesis import Tacotron2


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
    def __init__(self):
        super(Sven, self).__init__()
        self.encoder = resnet18()
        # use grayscale images
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # drop the last layer corresponind to the softmax classification
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.decoder = Tacotron2(hparams)

    def forward(self, x_y):
        x, y = x_y
        B, S, H, W = x.shape

        # scales data in [0, 1]
        x = x.float() / 255
        # BS, 1, H, W
        x = x.view(B * S, H, W).unsqueeze(1)
        # BS, Dx, 1, 1
        x = self.encoder(x)
        # B, S, Dx
        x = x.squeeze().view(B, S, hparams.encoder_embedding_dim)

        return self.decoder(x, y)
