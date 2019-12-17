from typing import List, Callable, Union, Tuple
import datetime
import inspect
import math
import os
import random


from torch.nn import functional as F
from moviepy.editor import *
import json
import cv2
import numpy as np
import h5py
import torch
import torch.utils.data
import pdb
import scipy
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy import signal
from scipy.io import wavfile


class xTSSample(object):
    def __init__(self,
                 root: str,
                 person: str,
                 file: str):
        self.root = root
        self.person = person
        self.data = None
        self.file = file
        self.crop = None
        self.spec = None

    def load(self):
        """ Crop lips"""

        f = open(os.path.join(self.root, "grid", "face-landmarks", self.person, self.file + ".json"))
        fl = json.load(f)
        top = fl[0][51][1] - 10
        bot = fl[0][58][1] + 10
        left = fl[0][49][0] - 10
        right = fl[0][55][0] + 10
        cap = cv2.VideoCapture(os.path.join(self.root, "grid", "Video",  self.person, self.file + ".mpg"))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        buf = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))
        self.crop = np.empty((frameCount, bot - top + 20, right - left + 20), np.dtype('uint8'))
        fc = 0
        ret = True

        while fc < frameCount and ret:
            ret, frame = cap.read()
            buf[fc] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.crop[fc] = buf[fc][top-10:bot + 10, left-10:right + 10]
            fc += 1

        cap.release()
        self.data = torch.from_numpy(self.crop)

        """ Create spectrogram"""
        sample_rate, samples = scipy.io.wavfile.read(
            os.path.join(self.root, "grid", "Audio", self.person, self.file + ".wav"))
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        fmin = 10
        fmax = 4000
        freq_slice = np.where((frequencies >= fmin) & (frequencies <= fmax))
        "frequencies = frequencies [freq_slice]"
        "spectrogram = spectrogram[freq_slice, :][0]"

        self.spec = spectrogram





class xTSDataset(torch.utils.data.Dataset):
    """ Implementation of the pytorch Dataset. """

    def __init__(self,
                 root: str,
                 type: str,
                 transform: List[Callable] = None):
        """ Initializes the xTSDataset
        Args:
            root (string): Path to the root data directory.
            type (string): name of the txt file containing the data split
        """
        self.root = root
        f = open(type + ".txt", 'r')
        self.folder = []
        self.file = []
        content = f.read()
        res = content.split()
        i = 0
        for idx in res:
            if i % 2 == 0:
                self.file.append(idx)
            if i % 2 == 1:
                self.folder.append(idx)
            i = i + 1
        self.size = len(self.file)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        if idx >= self.size:
            raise IndexError
        stream = xTSSample(self.root, self.folder[idx], self.file[idx])
        stream.load()

        return stream.data, stream.spec
