import argparse
import pdb
import os
import sys

import librosa
import numpy as np

from scipy import stats

from mcd import dtw
from mcd.metrics_fast import logSpecDbDist as log_spec_db_dist

import python_speech_features

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


def compute_mcd(gt, pr, type1="librosa"):

    def get_mfcc_psf(audio):
        return python_speech_features.mfcc(audio, sr=16_000)

    def get_mfcc_librosa(audio):
        # n_mfcc=60, n_fft=2048, hop_length=275, win_length=1100
        mfcc = librosa.feature.mfcc(audio, sr=16_000, htk=False)
        return mfcc.astype("float64").T

    GET_MFCC = {
        "librosa": get_mfcc_librosa,
        "python-speech-features": get_mfcc_psf,
    }

    get_mfcc = GET_MFCC[type1]

    mfcc_gt = get_mfcc(gt)
    mfcc_pr = get_mfcc(pr)

    # cost, _ = dtw.dtw(mfcc_gt, mfcc_pr, log_spec_db_dist)
    cost = sum(log_spec_db_dist(g, p) for g, p in zip(mfcc_gt, mfcc_pr))
    num_frames = len(mfcc_gt)
    # NumPy implementation of MCD:
    # > K = 10 / np.log(10) * np.sqrt(2)
    # > K * np.mean(np.sqrt(np.sum((mfcc_gt - mfcc_pr) ** 2, axis=1)))

    return cost / num_frames


def postprocess(gt, pr):
    def trim(a):
        return librosa.effects.trim(a)[0]

    def scale(a):
        return a * 1 / max(abs(a))

    def align(x, y):
        n = min(len(x), len(y))
        return x[:n], y[:n]

    # return align(gt, pr)
    return align(scale(gt), scale(pr))
    # return align(scale(trim(gt)), scale(trim(pr)))


SAMPLING_RATE = 16_000
METRICS = {
    "stoi": compute_stoi,
    "pesq": compute_pesq,
    "mcd": compute_mcd,
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


def evaluate(metric_name, path_loader, path_predictions):
    def evaluate1(id1):
        filename = path_loader.id_to_filename(id1, "audio")

        path_gt = os.path.join(path_loader.folders["audio"], filename)
        path_pr = os.path.join(path_predictions, filename)

        audio_gt = load_audio(path_gt)
        audio_pr = load_audio(path_pr)

        return compute_metric(*postprocess(audio_gt, audio_pr))

    load_audio = lambda p: librosa.core.load(p, SAMPLING_RATE)[0]
    compute_metric = METRICS[metric_name]
    return [evaluate1(id1) for id1 in path_loader.ids]


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-" + args.split)
    results = evaluate(args.metric, path_loader, args.predictions)
    # Print results
    print(np.mean(results))
    print(stats.describe(results))


if __name__ == "__main__":
    main()
