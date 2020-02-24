import argparse
import os
import pdb
import sys

sys.path.insert(0, ".")
from train import link_best_model


def main():
    parser = argparse.ArgumentParser(description="Create sym link to best model")
    parser.add_argument("model")
    args = parser.parse_args()
    link_best_model(args.model)


if __name__ == "__main__":
    main()
