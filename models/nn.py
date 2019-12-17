from typing import List, Union, Dict
import collections
import enum

import numpy as np
import torch
import torch.nn as torch_nn
import pdb
import src.dataset


class baseline(torch_nn.Module):

    def __init__(self, history: int):
        super(baseline, self).__init__()
        self.history = history
        kwargs = dict(kernel_size=3, stride=1, padding=0, bias=True)
        self.conv1 = torch_nn.Conv3d(history, 4, **kwargs)
        self.conv2 = torch_nn.Conv3d(4, 4, **kwargs)
        self.conv3 = torch_nn.Conv3d(4, 6, **kwargs)
        self.gru = torch_nn.GRU(6, 10)
        self.linear = torch_nn.Linear(,)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gru(x)
        x = self.linear(x)
        return x


