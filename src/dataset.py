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

from torch.utils.data._utils.collate import default_collate

from hparams import hparams
from audio import AUDIO_PROCESSING


def prepare_batch_2(batch, device, non_blocking):
    batch_x, batch_y = batch
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    return (batch_x, batch_y), batch_y


def prepare_batch_3(batch, device, non_blocking):
    batch_x, batch_y, extra = batch
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    extra = extra.to(device)
    return (batch_x, batch_y, extra), batch_y


def collate_fn(batches):
    videos = [batch[0] for batch in batches if batch[0] is not None]
    spects = [batch[1] for batch in batches if batch[1] is not None]

    max_v = max(video.shape[0] for video in videos)
    pad_v = lambda video: (0, 0, 0, 0, 0, max_v - video.shape[0])

    max_s = max(spect.shape[0] for spect in spects)
    pad_s = lambda spect: (0, 0, 0, max_s - spect.shape[0])

    videos = [F.pad(video, pad=pad_v(video)) for video in videos]
    spects = [F.pad(spect, pad=pad_s(spect)) for spect in spects]

    video = torch.stack(videos)
    spect = torch.stack(spects)

    if len(batches[0]) == 2:
        return video, spect
    else:
        extra = [batch[2:] for batch in batches if batch[0] is not None]
        extra = default_collate(extra)
        extra = torch.cat(extra)
        return video, spect, extra


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

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        if idx >= self.size:
            raise IndexError

        try:
            self.stream = xTSSample(self.root, self.folder[idx], self.file[idx])
            self.stream.load(self.transforms, self.audio_processing)
            return self.stream.video, self.stream.spect
        except Exception as e:
            print(e)
            print(self.folder[idx], self.file[idx])
            return None, None


class xTSDatasetSpeakerId(xTSDataset):
    def __init__(self, *args, **kwargs):
        super(xTSDatasetSpeakerId, self).__init__(*args, **kwargs)
        self.speaker_to_id = {s: i for i, s in enumerate(sorted(set(self.folder)))}

    def __getitem__(self, idx: int):
        video, spect = super().__getitem__(idx)
        id_ = self.speaker_to_id[self.stream.person]
        id_ = torch.tensor(id_).long()
        return video, spect, id_


class xTSDatasetSpeakerEmbedding(xTSDataset):
    def __init__(self, *args, **kwargs):
        super(xTSDatasetSpeakerEmbedding, self).__init__(*args, **kwargs)
        data_embedding = np.load(os.path.join(self.root, "speaker-embeddings/full.npz"))
        self.speaker_embeddings = data_embedding["feats"]
        self.file_to_index = {f: i for i, (f, _) in enumerate(data_embedding["files_and_folders"])}

    def __getitem__(self, idx: int):
        video, spect = super().__getitem__(idx)
        i = self.file_to_index[self.file[idx]]
        embedding = self.speaker_embeddings[i]
        embedding = torch.tensor(embedding).float()
        return video, spect, embedding
