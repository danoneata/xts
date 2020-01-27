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


SEED = 1337
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

random.seed(SEED)


def load_filelist(filelist):
    with open(os.path.join(ROOT, "filelists", filelist + ".txt"), "r") as f:
        return [line.split() for line in f.readlines()]


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # fmt: off
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


def main():
    AUDIO_EXT = ".wav"

    parser = get_arg_parser()
    args = parser.parse_args()

    files_and_folders = load_filelist(args.filelist)
    paths = [
        os.path.join(ROOT, "audio-16khz", folder, file_ + AUDIO_EXT)
        for file_, folder in files_and_folders
    ]

    feats = extract_features(paths, args)

    path_emb = "{}/speaker-embeddings/{}.npz".format(args.ROOT, args.filelist)
    np.savez(path_emb, files_and_folders=files_and_folders, feats=feats)

    if args.to_evaluate:
        labels = [folder for _, folder in files_and_folders]
        evaluate(feats, labels)


if __name__ == "__main__":
    main()
