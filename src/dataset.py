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

import torch
import torch.utils.data

from torch.utils.data._utils.collate import default_collate

from toolz import compose

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
    pad_v = lambda video: [0, 0, 0, 0, 0, max_v - video.shape[0]]

    max_s = max(spect.shape[0] for spect in spects)
    pad_s = lambda spect: [0, 0, 0, max_s - spect.shape[0]]

    videos = [F.pad(video, pad=pad_v(video)) for video in videos]
    spects = [F.pad(spect, pad=pad_s(spect)) for spect in spects]

    video = torch.stack(videos)
    spect = torch.stack(spects)

    if len(batches[0]) == 2:
        return video, spect
    else:
        extra = [batch[2:] for batch in batches if batch[0] is not None]
        extra = default_collate(extra)
        # extra = torch.cat(extra)
        return [video, spect] + extra


DATASET_PARAMETERS = {
    "lrw": {"len-inp": 29, "len-out": 98,},
    "grid": {"len-inp": 75, "len-out": 239,},
}


class PathLoader:
    def __init__(self, root: str, dataset: str, filelist: str):
        self.folders = {
            "base": os.path.join(root, dataset),
            "face": os.path.join(root, dataset, "face-landmarks"),
            "audio": os.path.join(root, dataset, "audio-from-video"),
            "video": os.path.join(root, dataset, "video"),
        }
        filelist = os.path.join(root, dataset, "filelists", filelist + ".txt")
        with open(filelist, "r") as f:
            self.ids = [line.strip() for line in f.readlines()]


class GridPathLoader(PathLoader):
    def __init__(self, root: str, filelist: str):
        super(GridPathLoader, self).__init__(root, "grid", filelist)
        self.extensions = {
            "face": ".json",
            "audio": ".wav",
            "video": ".mpg",
        }
        self.paths = {
            k: [
                os.path.join(self.folders[k], self.id_to_filename(i, k))
                for i in self.ids
            ]
            for k in ("face", "audio", "video")
        }
        self.paths["speaker-embeddings"] = [
            os.path.join(self.folders["base"], "speaker-embeddings", filelist + ".npz")
        ]
        # Speaker information
        self.speakers = [i.split()[1] for i in self.ids]
        self.speaker_to_id = {s: i for i, s in enumerate(sorted(set(self.speakers)))}

    def id_to_filename(self, id1, type1):
        file1, subject = id1.split()
        return os.path.join(subject, file1 + self.extensions[type1])


class LRWPathLoader(PathLoader):
    def __init__(self, root: str, filelist: str):
        super(LRWPathLoader, self).__init__(root, "lrw", filelist)
        self.extensions = {
            "face": ".json",
            "audio": ".wav",
            "video": ".mp4",
        }
        self.paths = {
            k: [
                os.path.join(self.folders[k], self.id_to_filename(p, k))
                for p in self.ids
            ]
            for k in ("face", "audio", "video")
        }
        self.paths["speaker-embeddings"] = [
            os.path.join(self.folders["base"], "speaker-embeddings", filelist + ".npz")
        ]

    def id_to_filename(self, id1, type1):
        return os.path.join(id1 + self.extensions[type1])


PATH_LOADERS = {
    "grid": GridPathLoader,
    "lrw": LRWPathLoader,
}


def get_video_lips(path_face, path_video, transform):
    """Loads video and crops frames around lips."""
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
        frame = frame[top:bottom, left:right]
        frames.append(transform(frame))

    frames = torch.cat(frames)
    capture.release()

    return frames


class xTSDataset(torch.utils.data.Dataset):
    """Implementation of the pytorch Dataset."""

    def __init__(
        self, hparams, path_loader: PathLoader, transforms: Dict[str, Callable]
    ):
        """ Initializes the xTSDataset
        Args:
            root (string): Path to the root data directory.
            type (string): name of the txt file containing the data split
        """
        self.path_loader = path_loader
        self.paths = path_loader.paths
        self.transforms = transforms
        self.size = len(self.path_loader.ids)

        # Data loaders
        audio_proc = AUDIO_PROCESSING[hparams.audio_processing](
            hparams.sampling_rate, hparams.n_mel_channels
        )
        self.get_spect = compose(audio_proc.audio_to_mel, audio_proc.load_audio)
        self.get_video = get_video_lips

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        if idx >= self.size:
            raise IndexError
        try:
            video = self.get_video(
                self.paths["face"][idx],
                self.paths["video"][idx],
                self.transforms["video"],
            )
            spect = self.get_spect(self.paths["audio"][idx])
            return video, spect
        except Exception as e:
            print(e)
            print(self.path_loader.ids[idx])
            return None, None


class xTSDatasetSpeakerId(xTSDataset):
    def __init__(self, *args, **kwargs):
        super(xTSDatasetSpeakerId, self).__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        video, spect = super(xTSDatasetSpeakerId, self).__getitem__(idx)
        id_ = self.path_loader.speaker_to_id[self.path_loader.speakers[idx]]
        id_ = torch.tensor(id_).long()
        return video, spect, id_


class xTSDatasetSpeakerIdFilename(xTSDataset):
    def __init__(self, *args, **kwargs):
        super(xTSDatasetSpeakerIdFilename, self).__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        video, spect = super(xTSDatasetSpeakerIdFilename, self).__getitem__(idx)
        id_ = self.path_loader.speaker_to_id[self.path_loader.speakers[idx]]
        id_ = torch.tensor(id_).long()
        return video, spect, id_, self.path_loader.ids[idx]


def get_embedding_stats(emb):
    emb = torch.tensor(emb).float()
    ε = 1e-7 * torch.ones(1, emb.shape[1])
    return {
        "μ": emb.mean(dim=0, keepdim=True),
        "σ": torch.max(emb.var(dim=0, keepdim=True).sqrt(), ε) / 3,
    }


class xTSDatasetSpeakerEmbedding(xTSDataset):
    def __init__(self, *args, **kwargs):
        super(xTSDatasetSpeakerEmbedding, self).__init__(*args, **kwargs)
        data_embedding = np.load(self.paths["speaker-embeddings"][0])
        self.speaker_embeddings = data_embedding["feats"]
        self.embedding_stats = get_embedding_stats(data_embedding["feats"])
        self.id_to_index = {
            id1: index for index, id1 in enumerate(data_embedding["ids"])
        }

    def __getitem__(self, idx: int):
        video, spect = super(xTSDatasetSpeakerEmbedding, self).__getitem__(idx)
        i = self.id_to_index[self.path_loader.ids[idx]]
        embedding = self.speaker_embeddings[i]
        embedding = torch.tensor(embedding).float()
        return video, spect, embedding


class xTSDatasetSpeakerFixedEmbedding(xTSDataset):
    def __init__(self, *args, **kwargs):
        super(xTSDatasetSpeakerFixedEmbedding, self).__init__(*args, **kwargs)
        data_embedding = np.load(self.paths["speaker-embeddings"][0])
        self.embedding_stats = get_embedding_stats(data_embedding["feats"])
        self.speaker_embeddings = self._get_speaker_fixed_embeddings(
            data_embedding["feats"], data_embedding["ids"]
        )

    def _get_speaker_fixed_embeddings(self, features, ids):
        speakers = [utt_id.split()[1] for utt_id in ids.tolist()]
        num_speakers = len(set(speakers))
        embeddings_speaker = np.zeros((num_speakers, features.shape[1]))
        for speaker in set(speakers):
            i = self.path_loader.speaker_to_id[speaker]
            idxs = [speaker == t for t in speakers]
            embeddings_speaker[i] = np.mean(features[idxs], axis=0)
        return embeddings_speaker

    def __getitem__(self, idx: int):
        video, spect = super(xTSDatasetSpeakerFixedEmbedding, self).__getitem__(idx)
        id_ = self.path_loader.speaker_to_id[self.path_loader.speakers[idx]]
        embedding = self.speaker_embeddings[id_]
        embedding = torch.tensor(embedding).float()
        return video, spect, embedding


class xTSDatasetSpeakerIdEmbedding(xTSDataset):
    def __init__(self, *args, **kwargs):
        super(xTSDatasetSpeakerIdEmbedding, self).__init__(*args, **kwargs)
        data_embedding = np.load(self.paths["speaker-embeddings"][0])
        self.speaker_embeddings = data_embedding["feats"]
        self.embedding_stats = get_embedding_stats(data_embedding["feats"])
        self.id_to_index = {
            id1: index for index, id1 in enumerate(data_embedding["ids"])
        }

    def __getitem__(self, idx: int):
        video, spect = super(xTSDatasetSpeakerIdEmbedding, self).__getitem__(idx)
        # id
        id_ = self.path_loader.speaker_to_id[self.path_loader.speakers[idx]]
        id_ = torch.tensor(id_).long()
        # embedding
        i = self.id_to_index[self.path_loader.ids[idx]]
        embedding = self.speaker_embeddings[i]
        embedding = torch.tensor(embedding).float()
        return video, spect, id_, embedding
