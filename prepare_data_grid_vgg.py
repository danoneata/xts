"""Prepares data for the deep-lip-reading code."""
import itertools
import os
import pdb
import sys

import cv2

from constants import FPS
from data import load_data
from generate_crops import crop_frame


PATH = "/home/doneata/work/deep-lip-reading/media"
HEIGHT, WIDTH = 160, 160


def load_video(path):
    reader = cv2.VideoCapture(path)
    images = []
    while True:
        success, image = reader.read()
        if not success:
            break
        images.append(image)
    return images


def write_video(frames, path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(path, fourcc, FPS, (HEIGHT, WIDTH))
    for frame in frames:
        video.write(cv2.resize(frame, (HEIGHT, WIDTH)))
    video.release()


def get_lip_point(landmarks):
    lip_marks = landmarks[0][49:]
    n = len(lip_marks)
    x = sum(t[0] for t in lip_marks) / n
    y = sum(t[1] for t in lip_marks) / n
    return int(x), int(y)


def get_box_around(point):
    x, y = point

    # use `min` in case the lips are too close to the bottom border
    y2 = min(y + 160 // 2, 288)
    size = y2 - y
    y1 = y - size

    x1 = x - size
    x2 = x + size

    return (x1, y1), (x2, y2)


def main():

    split = sys.argv[1]
    data = load_data(split)
    annotations = []

    for i, sample in enumerate(load_data(split)):

        print(i, sample.key, sample.speaker, sample.sentence, end=" ")

        if not sample.landmarks:
            print("x")
            continue
        else:
            print("OK")

        lip_point = get_lip_point(sample.landmarks)
        box = get_box_around(lip_point)

        delta_x = box[1][0] - box[0][0]
        delta_y = box[1][1] - box[0][1]
        print(f"\tΔ = {delta_x}, {delta_y}")

        frames = load_video(sample.video_path)
        frames = [crop_frame(frame, box) for frame in frames]

        get_segment1 = lambda *args: get_segment(sample.video_alignment, *args)
        words = sample.sentence.split()
        frames_group = itertools.groupby(enumerate(frames), key=get_segment1)

        for j, (word, start, end) in enumerate(sample.video_alignment):

            if word == "sil":
                continue

            path = os.path.join(PATH, "grid", sample.key + f"_{j:03d}" + ".mp4")
            annotations.append((path, word))

            s = int(start.to_frame())
            e = int(end.to_frame()) + 1

            if e - s == 1:
                e += 1

            print(f"\t{word:10s} → {s:3d}:{e:3d}", end=" ")
            write_video(frames[s: e], path)
            print()

        if i >= 30:
            break

    with open(os.path.join(PATH, f"grid-{split}.txt"), "w") as f:
        for path, word in annotations:
            f.write(f"{path}, {word.upper()}\n")


if __name__ == "__main__":
    main()
