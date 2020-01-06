from typing import List, Union, Dict
import collections
import enum

import numpy as np
import torch
import torch.nn as torch_nn
import pdb
import src.dataset

from torch.nn import functional as F


class baseline(torch_nn.Module):
    def __init__(self):
        super(baseline, self).__init__()
        kwargs = dict(kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = torch_nn.Conv3d(1, 4, **kwargs)
        self.conv2 = torch_nn.Conv3d(4, 4, **kwargs)
        self.conv3 = torch_nn.Conv3d(4, 1, **kwargs)
        self.gru = torch_nn.GRU(64 * 64, 16)
        self.linear = torch_nn.Linear(16, 3 * 1025)

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
