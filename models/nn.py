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


class SpeakerInfo(enum.Enum):
    NOTHING = enum.auto()
    ID = enum.auto()
    EMBEDDING = enum.auto()


class VideoEncoder(nn.Module):
    def __init__(self, params):
        super(VideoEncoder, self).__init__()
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

    def forward(self, x):
        B, S, H, W = x.shape
        # B, 1, S, H, W
        x = x.unsqueeze(1)
        # B, conv3d_num_filters, S, H, W
        x = self.conv0_3d(x)
        # B, S, conv3d_num_filters, H, W
        x = x.permute(0, 2, 1, 3, 4)
        # BS, conv3d_num_filters, H, W
        x = x.reshape(B * S, self.params.conv3d_num_filters, H, W)
        # BS, D, 1, 1
        x = self.encoder(x)
        # B, S, D
        x = x.squeeze().view(B, S, hparams.encoder_embedding_dim)
        # S, B, D
        x = x.permute(1, 0, 2)
        # S, B, D
        x, _ = self.encoder_rnn(x)
        # B, S, D
        x = x.permute(1, 0, 2)
        return x


class Bjorn(nn.Module):
    """xTS model that uses pre-computed speaker embeddings"""
    def __init__(self, dataset_params, params):
        super(Bjorn, self).__init__()
        E_DIM_IN = 512
        E_DIM_OUT = params["speaker_embedding_dim"]
        hparams_copy = SimpleNamespace(**vars(hparams))
        hparams_copy.encoder_embedding_dim += E_DIM_OUT
        self.video_encoder = VideoEncoder(params)
        self.linear = nn.Linear(E_DIM_IN, E_DIM_OUT)
        self.decoder = Tacotron2(hparams_copy, dataset_params)
        self.speaker_info = SpeakerInfo.EMBEDDING

    def _concat_embedding(self, x, e):
        _, S, _ = x.shape
        e = self.linear(e)
        e = e.unsqueeze(1)
        e = e.repeat(1, S, 1)
        x = torch.cat((x, e), dim=2)
        return x

    def forward(self, inp):
        x, y, emb = inp
        x = self.video_encoder(x)
        x = self._concat_embedding(x, emb)
        return self.decoder(x, y)

    def predict(self, inp):
        x, emb = inp
        x = self.video_encoder(x)
        x = self._concat_embedding(x, emb)
        _, y = self.decoder.predict(x)
        return y


class Sven(nn.Module):
    """xTS model that can use speaker id's to learn speaker embeddings"""
    def __init__(self, dataset_params, params):
        super(Sven, self).__init__()
        self.params = SimpleNamespace(**params)
        self.speaker_info = SpeakerInfo.ID
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
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        if "speaker_embedding_dim" in params and params["speaker_embedding_dim"] > 0:
            num_embeddings = 14
            speaker_embedding_dim = params["speaker_embedding_dim"]
            self.speaker_embedding = nn.Embedding(num_embeddings, speaker_embedding_dim)
            hparams.encoder_embedding_dim = hparams.encoder_embedding_dim + speaker_embedding_dim
        else:
            self.speaker_embedding= None
        self.decoder = Tacotron2(hparams, dataset_params)

    def _encode_video(self, x):
        B, S, H, W = x.shape
        # B, 1, S, H, W
        x = x.unsqueeze(1)
        # B, conv3d_num_filters, S, H, W
        x = self.conv0_3d(x)
        # B, S, conv3d_num_filters, H, W
        x = x.permute(0, 2, 1, 3, 4)
        # BS, conv3d_num_filters, H, W
        x = x.reshape(B * S, self.params.conv3d_num_filters, H, W)
        # BS, D, 1, 1
        x = self.encoder(x)
        # B, S, D
        x = x.squeeze().view(B, S, self.encoder_embedding_dim)
        # S, B, D
        x = x.permute(1, 0, 2)
        # S, B, D
        x, _ = self.encoder_rnn(x)
        # B, S, D
        x = x.permute(1, 0, 2)
        return x

    def _concat_embedding(self, x, e):
        _, S, _ = x.shape
        # B, S, E
        e = e.unsqueeze(1).repeat(1, S, 1)
        # B, S, E + D
        x = torch.cat((x, e), dim=2)
        return x

    def encode(self, x, ids):
        x = self._encode_video(x)
        if self.speaker_embedding:
            # B, E
            e = self.speaker_embedding(ids)
            x = self._concat_embedding(x, e)
        return x

    def forward(self, inp):
        # Batch decoding with teacher forcing
        x, y, ids = inp
        x = self.encode(x, ids)
        return self.decoder(x, y)

    def predict(self, inp):
        # Step-by-step decoding
        x, ids = inp
        x = self.encode(x, ids)
        _, y = self.decoder.predict(x)
        return y

    def predict2(self, inp):
        # Step-by-step decoding with auxilary information; equivalent to the `forward` method.
        x, y, ids = inp
        x = self.encode(x)
        _, y = self.decoder.predict2(x, y)
        return y
