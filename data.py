import json
import os
import pdb

from collections import namedtuple
from itertools import groupby

from constants import FPS


Sample = namedtuple("Sample", "key video_path audio_path frame_path speaker sentence landmarks audio_alignment video_alignment")


DATA_DIR = "data"
FILELISTS = {
    "tiny": os.path.join(DATA_DIR, "filelists", "tiny.txt"),
    "small": os.path.join(DATA_DIR, "filelists", "small.txt"),
    "full": os.path.join(DATA_DIR, "filelists", "full.txt"),
}


class Time:
    def __init__(self, t):
        self.t = t

    def __str__(self):
        return str(self.t)

    def __repr__(self):
        return "Time({})".format(self.t)

    @classmethod
    def from_frame(cls, n, fps):
        return cls(n / fps)

    @classmethod
    def from_seconds(cls, t):
        return cls(t)

    def to_seconds(self):
        return self.t


def load_audio_alignments():
    def process(line):
        key, _, start, duration, phone = line.split()
        _, key = key.split('_')
        phone, *_ = phone.split('_')
        start = float(start)
        end = start + float(duration)
        return key, phone, start, end
    def drop_first(xs):
        _, *rest = xs
        return rest
    PATH = "/home/doneata/work/experiments-tedlium-r2/exp_grid/chain_cleaned/tdnn1f_sp_bi_ali/alignments.txt"
    with open(PATH, 'r') as f:
        alignments_iter = (process(line) for line in f.readlines())
        alignments = {
            k: list(map(drop_first, g))
            for k, g in groupby(alignments_iter, key=lambda t: t[0])
        }
    return alignments


def load_video_alignments(key, speaker):
    def process(line):
        start, end, word = line.split()
        start = Time.from_frame(int(start) / 1000, fps=FPS)
        end = Time.from_frame(int(end) / 1000, fps=FPS)
        return word, start, end
    path = os.path.join(DATA_DIR, "align", speaker, key + ".align")
    with open(path, "r") as f:
        return [process(line) for line in f.readlines()]


def load_sentence(key, speaker):
    return " ".join(t[0] for t in load_video_alignments(key, speaker))


def load_face_landmarks(key, speaker):
    path = os.path.join(DATA_DIR, "face-landmarks", speaker, key + ".json")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def load_data(split):
    with open(FILELISTS[split], 'r') as f:
        key_and_speaker = [line.split() for line in f.readlines()]
    audio_alignments = load_audio_alignments()
    get_video_path = lambda key, speaker: os.path.join(DATA_DIR, "video", speaker, key + ".mpg")
    get_audio_path = lambda key, speaker: os.path.join(DATA_DIR, "audio", speaker, key + ".wav")
    get_frame_path = lambda key, speaker: os.path.join(DATA_DIR, "frame", speaker, key + ".jpg")
    return [
        Sample(
            key,
            get_video_path(key, speaker),
            get_audio_path(key, speaker),
            get_frame_path(key, speaker),
            speaker,
            load_sentence(key, speaker),
            load_face_landmarks(key, speaker),
            audio_alignments[key],
            load_video_alignments(key, speaker),
        )
        for key, speaker in key_and_speaker
	if speaker != "s21"
    ]


if __name__ == "__main__":
    data = load_data("tiny")
    pdb.set_trace()
