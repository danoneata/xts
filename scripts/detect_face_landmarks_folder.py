import argparse
import glob
import json
import pdb
import os

import numpy as np

import cv2
import dlib
import tqdm


SHAPE_PREDICTOR_PATH = (
    "/home/doneata/src/dlib-models/shape_predictor_68_face_landmarks.dat"
)


def shape_to_list(shape):
    return [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]


def detect_face_landmarks(detector, predictor, path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    landmarks = [shape_to_list(predictor(gray, rect)) for rect in rects]

    return landmarks


GLOB_PATTERN = {
    "grid": "**/*.jpg",
    "lrw": "**/**/*.jpg",
}


def main():
    parser = argparse.ArgumentParser(
        description="""Extracts face landmarks for the frames in a dataset.
        This script assumes that the frames are located at `data/$DATASET/frames`
        and it outputs the face landmarks at `data/$DATA/face-landmarks`."""
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=GLOB_PATTERN,
        help="name of the dataset",
    )
    args = parser.parse_args()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    glob_path = os.path.join("data", args.dataset, "frames", GLOB_PATTERN[args.dataset])

    def get_path_out(path):
        _, _, _, *folders, filename = path.split(os.path.sep)
        filename, _ = os.path.splitext(filename)
        filename = filename + ".json"

        output_dir = os.path.join("data", args.dataset, "face-landmarks", *folders)
        os.makedirs(output_dir, exist_ok=True)

        return os.path.join(output_dir, filename)

    for path in tqdm.tqdm(glob.glob(glob_path)):

        path_out = get_path_out(path)

        if os.path.exists(path_out):
            continue

        try:
            landmarks = detect_face_landmarks(detector, predictor, path)
        except Exception as e:
            print(path)
            print(e)
            print()
            continue

        with open(path_out, "w") as f:
            json.dump(landmarks, f, indent=True)


if __name__ == "__main__":
    main()
