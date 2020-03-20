import argparse
import pdb
import random
import sys

from itertools import product

import numpy as np


random.seed(1337)


def load_data(type_, key=None):
    if type_ == "real":
        path = "data/grid/speaker-embeddings/multi-speaker-test.npz"
    elif type_ == "synth":
        path = f"output/speaker-embeddings/grid-multi-speaker-{key}.npz"
    else:
        assert False
    data = np.load(path)
    return data["ids"].tolist(), data["feats"]


def evaluate(feats, labels, pairs):
    sys.path.insert(0, "/home/doneata/src/vgg-speaker-recognition/tool")
    import toolkits

    feats_r, feats_s = feats
    labels_r, labels_s = labels

    sim = lambda u, v: u @ v

    x = np.array([sim(feats_r[i], feats_s[j]) for i, j in pairs])
    y = np.array([int(labels_r[i] == labels_s[j]) for i, j in pairs])

    eer, thresh = toolkits.calculate_eer(y, x)
    eer = eer * 100

    print(f"+: {sum(y)} / {len(y)}")
    print("τ:   {:.3f} ".format(thresh))
    print("EER: {:.3f}%".format(eer))


def generate_pairs_random(labels1, labels2, num_pairs=20_000):
    indices1 = range(len(labels1))
    indices2 = range(len(labels2))
    pairs = [(i, j) for i, j in product(indices1, indices2) if i != j]
    pairs = random.sample(pairs, num_pairs)
    print(pairs[:10])
    return pairs


def generate_pairs(ids1, ids2):
    # TODO
    pdb.set_trace()


def main():
    FILELIST = "unseen-k-tiny-test"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="what to evaluate")
    args = parser.parse_args()

    ids_r, feats_r = load_data(type_="real")
    ids_s, feats_s = load_data(type_="synth", key=f"{FILELIST}-{args.model}")

    labels_r = [i.split()[1] for i in ids_r]
    labels_s = [i.split()[0].split("-")[1] for i in ids_s]

    pairs = generate_pairs_random(labels_r, labels_s)
    # pairs = generate_pairs(ids_r, ids_s)

    print(args.model)
    evaluate((feats_r, feats_s), (labels_r, labels_s), pairs)


if __name__ == "__main__":
    main()
