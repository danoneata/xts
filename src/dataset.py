from typing import Callable, Dict, List, Tuple, Union

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


def get_mel_from_path(path: str, transform=None):
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

    def get_video_lips(self, path_face, path_video, transform):
        """Crop lips"""
        with open(path_face) as f:
            face_landmarks = json.load(f)

        delta = 15

        top = face_landmarks[0][51][1] - delta
        bottom = face_landmarks[0][58][1] + delta
        left = face_landmarks[0][49][0] - delta
        right = face_landmarks[0][55][0] + delta

        capture = cv2.VideoCapture(path_video)
        frames = []

        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame = frame[top: bottom, left: right]
            frames.append(transform(frame))


        frames = torch.cat(frames)
        capture.release()

        return frames

    def load(self, transforms):
        _get_path = lambda m, e: os.path.join(self.paths[m], self.person, self.file + e)
        self.video = self.get_video_lips(_get_path("face", ".json"), _get_path("video", ".mpg"), transforms["video"])
        self.spect = get_mel_from_path(_get_path("audio", ".wav"), transforms["spect"])


class xTSDataset(torch.utils.data.Dataset):
    """Implementation of the pytorch Dataset."""

    def __init__(self, root: str, type: str, transforms: Dict[str, Callable] = None):
        """ Initializes the xTSDataset
        Args:
            root (string): Path to the root data directory.
            type (string): name of the txt file containing the data split
        """
        self.root = root
        self.transforms = transforms

        with open(os.path.join(self.root, "filelists", type + ".txt"), "r") as f:
            file_folder = [line.split() for line in f.readlines()]

        self.size = len(file_folder)
        self.file, self.folder = zip(*file_folder)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        if idx >= self.size:
            raise IndexError

        stream = xTSSample(self.root, self.folder[idx], self.file[idx])
        stream.load(self.transforms)

        return stream.video, stream.spect
