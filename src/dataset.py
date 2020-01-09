from typing import List, Callable, Union, Tuple

import datetime
import inspect
import json
import math
import os
import pdb
import random

import cv2
import numpy as np

import librosa
from PIL import Image

from torch.nn import functional as F
from moviepy.editor import *

import torch
import torch.utils.data

from hparams import hparams

sys.path.insert(0, "tacotron2")
from tacotron2.audio_processing import griffin_lim
from tacotron2.layers import TacotronSTFT
from tacotron2.utils import load_wav_to_torch


MAX_WAV_VALUE = 32_768


TACO_STFT = TacotronSTFT(
    filter_length=hparams.filter_length,
    hop_length=hparams.hop_length,
    win_length=hparams.win_length,
    sampling_rate=hparams.sampling_rate,
    mel_fmin=hparams.mel_fmin,
    mel_fmax=hparams.mel_fmax,
)


def audio_to_mel(audio):
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = TACO_STFT.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec


def get_mel_from_path(path: str):
    SAMPLING_RATE = 16_000
    audio, sampling_rate = load_wav_to_torch(path)
    assert sampling_rate == SAMPLING_RATE
    if audio.ndim == 2:
        audio = audio.mean(dim=1)
    return audio_to_mel(audio).T


def mel_to_audio(mel):
    # TODO make it work in batch mode
    mel = mel.unsqueeze(0)
    mel_decompress = TACO_STFT.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()

    spec_from_mel_scaling = 1000

    spec_from_mel = torch.mm(mel_decompress[0], TACO_STFT.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    GRIFFIN_ITERS = 60
    audio = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), TACO_STFT.stft_fn, GRIFFIN_ITERS)
    audio = audio.squeeze()
    audio = audio.cpu().numpy()

    return audio


class xTSSample(object):
    def __init__(self, root: str, person: str, file: str):
        self.root = root
        self.person = person
        self.data = None
        self.file = file
        self.crop = None
        self.spec = None

        self.paths = {
            "face": os.path.join(root, "face-landmarks"),
            "audio": os.path.join(root, "audio-from-video"),
            "video": os.path.join(root, "video"),
        }

    def get_video_lips(self):
        """Crop lips"""
        with open(os.path.join(self.paths["face"], self.person, self.file + ".json")):
            fl = json.load(f)

        top = fl[0][51][1] - 10
        bot = fl[0][58][1] + 10
        left = fl[0][49][0] - 10
        right = fl[0][55][0] + 10

        cap = cv2.VideoCapture(
            os.path.join(self.paths["video"], self.person, self.file + ".mpg")
        )
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((frameCount, frameHeight, frameWidth), np.dtype("uint8"))
        self.crop = np.empty(
            (frameCount, bot - top + 20, right - left + 20), np.dtype("uint8")
        )
        fc = 0
        ret = True

        while fc < frameCount and ret:
            ret, frame = cap.read()
            buf[fc] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.crop[fc] = buf[fc][top - 10 : bot + 10, left - 10 : right + 10]
            fc += 1

        cap.release()
        self.crop = [np.array(Image.fromarray(im).resize((64, 64))) for im in self.crop]
        self.crop = np.stack(self.crop)
        self.data = torch.from_numpy(self.crop)

    def load(self):
        _get_path = lambda m, e: os.path.join(self.paths[m], self.person, self.file + e)
        self.video = get_video_lips(_get_path("video", ".mpg"))
        self.spect = get_mel_spect(_get_path("audio", ".wav"))


class xTSDataset(torch.utils.data.Dataset):
    """Implementation of the pytorch Dataset."""

    def __init__(self, root: str, type: str, transform: List[Callable] = None):
        """ Initializes the xTSDataset
        Args:
            root (string): Path to the root data directory.
            type (string): name of the txt file containing the data split
        """
        self.root = root

        path = os.path.join(self.root, "filelists", type + ".txt")
        with open(path, "r") as f:
            content = f.read()

        self.folder = []
        self.file = []
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

        return stream.video, stream.spect
