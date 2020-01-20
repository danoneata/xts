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
from audio import AUDIO_PROCESSING


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

    def load(self, transforms, audio_proc):
        _get_path = lambda m, e: os.path.join(self.paths[m], self.person, self.file + e)
        self.video = self.get_video_lips(_get_path("face", ".json"), _get_path("video", ".mpg"), transforms["video"])
        self.spect = audio_proc.audio_to_mel(audio_proc.load_audio(_get_path("audio", ".wav")))


class xTSDataset(torch.utils.data.Dataset):
    """Implementation of the pytorch Dataset."""

    def __init__(self, root: str, type: str, transforms: Dict[str, Callable] = None):
        """ Initializes the xTSDataset
        Args:
            root (string): Path to the root data directory.
            type (string): name of the txt file containing the data split
        """
        self.root = root
        self.SAMPLING_RATE = 16_000

        self.transforms = transforms

        with open(os.path.join(self.root, "filelists", type + ".txt"), "r") as f:
            file_folder = [line.split() for line in f.readlines()]

        self.size = len(file_folder)
        self.file, self.folder = zip(*file_folder)
        self.audio_processing = AUDIO_PROCESSING[hparams.audio_processing](self.SAMPLING_RATE)
        self.speaker_to_id = {s: i for i, s in enumerate(sorted(set(self.folder)))}

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        if idx >= self.size:
            raise IndexError

        try:
            stream = xTSSample(self.root, self.folder[idx], self.file[idx])
            id_ = self.speaker_to_id[stream.person]
            stream.load(self.transforms, self.audio_processing)

            return stream.video, stream.spect, id_
        except Exception as e:
            print(e)
            print(id_)
            return None
