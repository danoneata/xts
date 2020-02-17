import argparse
import pdb
import os
import sys

import librosa
import numpy as np

from pystoi.stoi import stoi

from pypesq import pesq

sys.path.insert(0, "/home/doneata/work/xts")
from train import (
    PATH_LOADERS,
    ROOT,
)


def compute_stoi(gt, pr):
    gt = np.round(gt * (2 ** 16)).astype(np.int16)
    pr = np.round(pr * (2 ** 16)).astype(np.int16)
    return stoi(gt, pr, SAMPLING_RATE)


def compute_pesq(gt, pr):
    sampling_rate = 8_000
    gt1 = librosa.resample(gt, SAMPLING_RATE, sampling_rate)
    pr1 = librosa.resample(pr, SAMPLING_RATE, sampling_rate)
    return pesq(gt1, pr1, sampling_rate)


SAMPLING_RATE = 16_000
METRICS = {
    "stoi": compute_stoi,
    "pesq": compute_pesq,
    # "mcd": ???,
}


def get_argument_parser():
    parser = argparse.ArgumentParser(description="Evaluate a given model")
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        required=True,
        choices=METRICS,
        help="what metric to use for evaluation",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=PATH_LOADERS,
        help="which dataset to train on",
    )
    parser.add_argument(
        "--filelist", type=str, required=True, help="name of the filelist to use",
    )
    parser.add_argument(
        "-p", "--predictions", type=str, required=True, help="path to wav predictions",
    )
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="test",
        choices={"train", "valid", "test"},
        required=False,
        help="which data to use",
    )
    parser.add_argument("-v", "--verbose", action="count", help="verbosity level")
    return parser


def evaluate(args):
    load_audio = lambda p: librosa.core.load(p, SAMPLING_RATE)[0]
    compute_metric = METRICS[args.metric]
    path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-" + args.split)

    for id1 in path_loader.ids:
        filename = path_loader.id_to_filename(id1, "audio")

        path_gt = os.path.join(path_loader.folders["audio"], filename)
        path_pr = os.path.join(args.predictions, filename)

        audio_gt = load_audio(path_gt)
        audio_pr = load_audio(path_pr)

        #
        # print(len(audio_pr))
        metric = compute_metric(audio_gt, audio_pr[:len(audio_gt)])
        print(metric)
        pdb.set_trace()


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
