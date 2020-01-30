# source ~/src/vgg-speaker-recognition/venv/bin/activate
# 
import argparse
import os
import pdb
import random
import sys

from itertools import product

import numpy as np
import tensorflow as tf

sys.path.insert(0, "/home/doneata/src/vgg-speaker-recognition/tool")
sys.path.insert(0, "/home/doneata/src/vgg-speaker-recognition/src")
import model
import toolkits
import utils as ut


AUDIO_EXT = ".wav"
ROOT = os.environ.get("ROOT", "")
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


class GridDataset:
    def load_ids_and_paths(filelist):
        with open(os.path.join(ROOT, "grid", "filelists", filelist + ".txt"), "r") as f:
            files_and_folders = [line.split() for line in f.readlines()]
        ids = files_and_folders
        paths = [
            os.path.join(ROOT, "audio-16khz", folder, file_ + AUDIO_EXT)
            for file_, folder in files_and_folders
        ]
        return ids, paths


class LRWDataset:
    def load_ids_and_paths():
        with open(os.path.join(ROOT, "lrw", "filelists", filelist + ".txt"), "r") as f:
            paths = [line.split() for line in f.readlines()]
        ids = paths
        paths_audio = [os.path.join(ROOT, "lrw", "audio", p + ".wav") for p in paths]
        return paths, paths_audio


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--dataset", action="store_true", required=True, help="dataset to extract embeddings on")
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

    for i, path in enumerate(paths):
        if i % 50 == 0:
            print("extracting features {:7d} / {:7d}".format(i, num_paths))

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


DATASETS = {
    'grid': GridDataset,
    'lrw': LRWDataset,
}


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]()
    ids, paths = dataset.load_ids_and_paths(args.filelist)
    feats = extract_features(paths, args)

    path_emb = os.path.join(args.ROOT, args.dataset, "speaker-embeddings", args.filelist + ".npz")
    np.savez(path_emb, ids=ids, feats=feats)

    if args.to_evaluate and args.dataset == "grid":  # we don't have speaker information for LRW
        labels = [folder for _, folder in ids]
        evaluate(feats, labels)


if __name__ == "__main__":
    main()
