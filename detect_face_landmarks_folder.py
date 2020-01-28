import argparse
import glob
import json
import pdb
import os

import numpy as np

import cv2
import dlib
import tqdm


SHAPE_PREDICTOR_PATH = "/home/doneata/src/dlib-models/shape_predictor_68_face_landmarks.dat"


def shape_to_list(shape):
    return [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]


def detect_face_landmarks(detector, predictor, path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    landmarks = [shape_to_list(predictor(gray, rect)) for rect in rects]

    return landmarks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", required=True, help="path to images")
    args = parser.parse_args()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    glob_path = os.path.join(args.input_path, "**/*.jpg")

    for path in tqdm.tqdm(glob.glob(glob_path)):

        try:
            landmarks = detect_face_landmarks(detector, predictor, path)
        except Exception:
            print(path)
            continue

        *_, subject, filename = path.split(os.path.sep)
        filename, _ = os.path.splitext(filename)
        filename = filename + ".json"

        output_dir = os.path.join("data", "face-landmarks", subject)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(landmarks, f, indent=True)


if __name__ == "__main__":
    main()
