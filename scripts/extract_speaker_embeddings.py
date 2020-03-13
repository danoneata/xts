# source ~/src/vgg-speaker-recognition/venv/bin/activate
#
import argparse
import os
import pdb
import random
import sys

from abc import ABCMeta, abstractmethod
from itertools import product
from typing import Any, Callable

import numpy as np
import tensorflow as tf

from tqdm import tqdm

sys.path.insert(0, "/home/doneata/src/vgg-speaker-recognition/tool")
sys.path.insert(0, "/home/doneata/src/vgg-speaker-recognition/src")
import model
import toolkits
import utils as ut


AUDIO_EXT = ".wav"
ROOT = os.environ.get("ROOT", "data")
PARAMS = {
    "dim": (257, None, 1),
    "nfft": 512,
    "spec_len": 250,
    "win_length": 400,
    "hop_length": 160,
    "n_classes": 5994,
    "sampling_rate": 16000,
    "feat_dim": 512,
    "normalize": True,
    "model_path": "/home/doneata/src/vgg-speaker-recognition/model/weights.h5",
}

SEED = 1337
random.seed(SEED)

class Dataset(metaclass=ABCMeta):
    @abstractmethod
    def load_ids_and_paths(self, filelist: str):
        pass

    @abstractmethod
    def get_path_emb(self, filelist: str):
        pass


class GridDataset(Dataset):
    def load_ids_and_paths(self, filelist):
        with open(os.path.join(ROOT, "grid", "filelists", filelist + ".txt"), "r") as f:
            ids = [line.strip() for line in f.readlines()]

        def get_path(id1):
            file_, folder = id1.split()
            return os.path.join(ROOT, "grid", "audio-16khz", folder, file_ + AUDIO_EXT)

        paths = list(map(get_path, ids))
        return ids, paths

    def get_path_emb(self, filelist):
        return os.path.join(ROOT, "grid", "speaker-embeddings", filelist + ".npz")


class LRWDataset(Dataset):
    def load_ids_and_paths(self, filelist):
        with open(os.path.join(ROOT, "lrw", "filelists", filelist + ".txt"), "r") as f:
            ids = [line.strip() for line in f.readlines()]
        get_path = lambda i: os.path.join(ROOT, "lrw", "audio-from-video", i + ".wav")
        paths = list(map(get_path, ids))
        return ids, paths

    def get_path_emb(self, filelist):
        return os.path.join(ROOT, "lrw", "speaker-embeddings", filelist + ".npz")


class GridSyntheticDataset(Dataset):
    def __init__(self, model_name):
        self.model_name = model_name

    def load_ids_and_paths(self, filelist):
        with open(os.path.join(ROOT, "grid", "filelists", filelist + ".txt"), "r") as f:
            ids = [line.strip() for line in f.readlines()]

        def get_path(id1):
            file_, folder = id1.split()
            return os.path.join(
                "output",
                "synth-samples",
                f"grid-multi-speaker-{filelist}-{self.model_name}",
                folder,
                file_ + AUDIO_EXT,
            )

        paths = list(map(get_path, ids))
        return ids, paths

    def get_path_emb(self, filelist):
        return os.path.join(
            "output",
            "speaker-embeddings",
            f"grid-multi-speaker-{filelist}-{self.model_name}.npz",
        )


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--dataset", choices=DATASETS, required=True, help="dataset to extract embeddings on")
    parser.add_argument("--to-evaluate", action="store_true", help="EER evaluation")

    # set up training configuration.
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--resume", default=PARAMS["model_path"], type=str)

    # set up data to load
    parser.add_argument("--filelist", type=str, default="tiny2", help="name of the filelist to use")

    # set up network configuration.
    parser.add_argument("--net", default="resnet34s", choices=["resnet34s", "resnet34l"], type=str)
    parser.add_argument("--ghost-cluster", default=2, type=int)
    parser.add_argument("--vlad-cluster", default=8, type=int)
    parser.add_argument("--bottleneck-dim", default=512, type=int)
    parser.add_argument("--aggregation-mode", default="gvlad", choices=["avg", "vlad", "gvlad"], type=str)
    parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
    # fmt: on

    return parser


def extract_features(paths, args):
    # GPU configuration
    toolkits.initialize_GPU(args)

    network_eval = model.vggvox_resnet2d_icassp(
        input_dim=PARAMS["dim"], num_class=PARAMS["n_classes"], mode="eval", args=args
    )
    network_eval.load_weights(os.path.join(args.resume), by_name=True)

    num_paths = len(paths)
    feats = np.zeros((num_paths, PARAMS["feat_dim"]))

    for i, path in enumerate(tqdm(paths)):
        specs = ut.load_data(
            path,
            win_length=PARAMS["win_length"],
            sr=PARAMS["sampling_rate"],
            hop_length=PARAMS["hop_length"],
            n_fft=PARAMS["nfft"],
            spec_len=PARAMS["spec_len"],
            mode="eval",
        )
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)
        feats[i] = network_eval.predict(specs)

    return feats


def evaluate(feats, labels, num_pairs=10_000):
    indices = range(len(labels))
    pairs = [(i, j) for i, j in product(indices, indices) if i != j]
    pairs = random.sample(pairs, num_pairs)

    sim = lambda u, v: u @ v

    x = np.array([sim(feats[i], feats[j]) for i, j in pairs])
    y = np.array([int(labels[i] == labels[j]) for i, j in pairs])

    eer, thresh = toolkits.calculate_eer(y, x)
    eer = eer * 100
    print("τ:   {:.3f} ".format(thresh))
    print("EER: {:.3f}%".format(eer))


TR_SPEAKERS = "s1 s10 s12 s14 s15 s17 s22 s26 s28 s3 s32 s5 s6 s7".split()
DATASETS = {
    "grid": lambda: GridDataset(),
    "lrw": lambda: LRWDataset(),
}

for s in TR_SPEAKERS:
    key = f"magnus-multi-speaker-best-emb-spk-{s}"
    DATASETS[f"synth:{key}"] = lambda key=key: GridSyntheticDataset(key)


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]()
    ids, paths = dataset.load_ids_and_paths(args.filelist)
    feats = extract_features(paths, args)

    np.savez(dataset.get_path_emb(args.filelist), ids=ids, feats=feats)

    # we have speaker information only for Grid
    if args.to_evaluate and args.dataset == "grid":
        labels = [folder for _, folder in ids]
        evaluate(feats, labels)


if __name__ == "__main__":
    main()
