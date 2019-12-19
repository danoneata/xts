from typing import List, Callable, Union, Tuple
import datetime
import inspect
import math
import os
import random

import librosa
from PIL import Image

from torch.nn import functional as F
from moviepy.editor import *
import json
import cv2
import numpy as np
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

        self.paths = {
            "face": os.path.join(root, "face-landmarks"),
            "audio": os.path.join(root, "audio-from-video"),
            "video": os.path.join(root, "video"),
        }

    def load(self):
        """ Crop lips"""

        f = open(os.path.join(self.paths["face"], self.person, self.file + ".json"))
        fl = json.load(f)
        top = fl[0][51][1] - 10
        bot = fl[0][58][1] + 10
        left = fl[0][49][0] - 10
        right = fl[0][55][0] + 10
        cap = cv2.VideoCapture(os.path.join(self.paths["video"],  self.person, self.file + ".mpg"))
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
        self.crop = [np.array(Image.fromarray(im).resize((64, 64))) for im in self.crop]
        self.crop = np.stack(self.crop)
        self.data = torch.from_numpy(self.crop)

        """ Create spectrogram"""
        path = os.path.join(self.paths["audio"], self.person, self.file + ".wav")
        wav = load_wav(path)
        self.spec = torch.from_numpy(spectrogram(wav))
        # sample_rate, samples = scipy.io.wavfile.read(path)
        # frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        # fmin = 10
        # fmax = 4000
        # freq_slice = np.where((frequencies >= fmin) & (frequencies <= fmax))
        "frequencies = frequencies [freq_slice]"
        "spectrogram = spectrogram[freq_slice, :][0]"

        # self.spec = spectrogram


class HParams:
    num_mels=80
    num_freq=1025
    sample_rate=16000
    frame_length_ms=50
    frame_shift_ms=12.5
    preemphasis=0.97
    min_level_db=-100
    ref_level_db=20

hparams = HParams()


def load_wav(path):
  return librosa.core.load(path, sr=hparams.sample_rate)[0]


def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))


def preemphasis(x):
  return signal.lfilter([1, -hparams.preemphasis], [1], x)


def inv_preemphasis(x):
  return signal.lfilter([1], [1, -hparams.preemphasis], x)


def spectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
  return _normalize(S)


def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
  return inv_preemphasis(_griffin_lim(S ** hparams.power))          # Reconstruct phase


def melspectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D)))
  return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(hparams.sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)


def _griffin_lim(S):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles)
  for i in range(hparams.griffin_lim_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
  return y


def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
  _, hop_length, win_length = _stft_parameters()
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
  n_fft = (hparams.num_freq - 1) * 2
  hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
  win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
  return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
  n_fft = (hparams.num_freq - 1) * 2
  return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _normalize(S):
  return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

def _denormalize(S):
  return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db



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

        path = os.path.join(self.root, "filelists", type + ".txt")
        with open(path, 'r') as f:
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

        return stream.data, stream.spec
