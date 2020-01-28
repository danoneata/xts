import pdb
import os
import sys

import cv2
import imageio

from pydub import AudioSegment

from toolz import (
    first as fst,
    second as snd,
)

from constants import FPS, SELECTED_PHONES
from data import DATA_DIR, load_data


def get_lips_bounding_box(landmarks):
    assert len(landmarks) == 1
    lip_landmarks = landmarks[0][49:]
    ε = 25  # Extra margin
    return [
        (
            min(map(fst, lip_landmarks)) - ε,
            min(map(snd, lip_landmarks)) - ε,
        ),
        (
            max(map(fst, lip_landmarks)) + ε,
            max(map(snd, lip_landmarks)) + ε,
        ),
    ]


def crop_frame(frame, box):
    return frame[box[0][1]: box[1][1], box[0][0]: box[1][0]]


class GifWriter:
    def __init__(self, path):
        self.images = []
        self.path_still = path + "_still.gif"
        self.path_anima = path + "_anima.gif"

    def add_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.images.append(image_rgb)

    def write(self):
        imageio.mimsave(self.path_still, self.images[:1])
        imageio.mimsave(self.path_anima, self.images)


def crop_video(key, video_path, alignment, box):
    def init_writer(n):
        path = os.path.join(DATA_DIR, "crops", key + "_{:04}".format(n))
        return GifWriter(path)
    def get_phone(n):
        for i, (phone, start, end) in enumerate(alignment):
            if start * FPS <= n < end * FPS:
                return i, phone
        return None, None
    reader = cv2.VideoCapture(video_path) 
    writer = None
    frame_n = 0
    phone_n_prev = None
    while True:
        success, image = reader.read()
        phone_n, phone = get_phone(frame_n)
        # print(frame_n, phone_n, phone)
        if not success:
            break
        if phone_n != phone_n_prev:
            phone_n_prev = phone_n
            if writer:
                writer.write()
            if phone in SELECTED_PHONES:
                writer = init_writer(phone_n)
        if phone in SELECTED_PHONES:
            writer.add_image(crop_frame(image, box))
        frame_n += 1
    reader.release()


def crop_audio(key, audio_path, alignment):
    audio = AudioSegment.from_wav(audio_path)
    def get_path(i):
        return os.path.join(DATA_DIR, "crops", key + "_{:04}.wav".format(i))
    for i, (phone, start, end) in enumerate(alignment):
        if phone in SELECTED_PHONES:
            chunk = audio[start * 1000: end * 1000]  # TODO `to_ms`
            chunk.export(get_path(i), format="wav")

def main():
    split = sys.argv[1]
    for i, sample in enumerate(load_data(split)):
        try:
            box = get_lips_bounding_box(sample.landmarks)
        except:
            print(sample.key, sample.speaker)
            continue
        print(i, sample.key, sample.speaker, sample.sentence)
        sil, start, end = sample.video_alignment[0]
        assert sil == "sil"
        delta = end.to_seconds()
        phone_alignment = [(phone, start + delta, end + delta) for phone, start, end in sample.audio_alignment]
        crop_video(sample.key, sample.video_path, phone_alignment, box)
        crop_audio(sample.key, sample.audio_path, sample.audio_alignment)


if __name__ == "__main__":
    main()
