import json
import os
import pdb
import random
import shutil
import subprocess

import dominate

from dominate import tags

from toolz import partition


SEED = 1337
random.seed(SEED)


VIDEO_PATH = "data/grid/video"
AUDIO_PATH = "data/grid/audio-from-video"

FACE_PATH = "data/grid/face-landmarks"
OUT_PATH = "output/www/is20"


def load_text():
    def parse(line):
        _, key, *words = line.split()
        return key, " ".join(w for w in words if w != "sil")

    with open("data/grid/text/full.txt", "r") as f:
        return dict(parse(line) for line in f.readlines())


def load_filelist(path):
    with open(path, "r") as f:
        return [line.split() for line in f.readlines()]


def get_lip_box(id_, speaker):
    Δ = 15
    with open(os.path.join(FACE_PATH, speaker, id_ + ".json")) as f:
        face_landmarks = json.load(f)
    t = face_landmarks[0][51][1] - Δ
    b = face_landmarks[0][58][1] + Δ
    l = face_landmarks[0][49][0] - Δ
    r = face_landmarks[0][55][0] + Δ
    w = r - l
    h = b - t
    return w, w, l, t


def get_video_path(id_, speaker):
    folder_dst = os.path.join("data", "video", speaker)
    file_dst = id_ + ".webm"

    video_src = os.path.join(VIDEO_PATH, speaker, id_ + ".mpg")
    video_dst = os.path.join(OUT_PATH, folder_dst, file_dst)

    if os.path.exists(video_dst):
        return os.path.join(folder_dst, file_dst)

    os.makedirs(os.path.join(OUT_PATH, folder_dst), exist_ok=True)
    w, h, x, y = get_lip_box(id_, speaker)
    subprocess.call(
        [
            "/usr/bin/ffmpeg",
            "-i",
            video_src,
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "2M",
            "-c:a",
            "libopus",
            "-f",
            "webm",
            "-filter:v",
            f"crop={w}:{h}:{x}:{y}",
            video_dst,
        ]
    )
    return os.path.join(folder_dst, file_dst)


def get_audio_path(id_, speaker, method):
    METHOD_TO_FOLDER_SRC = {
        "ours-baseline": "output/synth-samples/grid-k-seen-test-magnus-best",
        "theirs": "data/grid/samples-konstantinos/seen",
    }

    folder_dst = os.path.join("data", "audio", method, speaker)
    file_dst = id_ + ".wav"

    audio_path = METHOD_TO_FOLDER_SRC[method]
    audio_src = os.path.join(audio_path, speaker, id_ + ".wav")
    audio_dst = os.path.join(OUT_PATH, folder_dst, file_dst)

    if os.path.exists(audio_dst):
        return os.path.join(folder_dst, file_dst)

    os.makedirs(os.path.join(OUT_PATH, folder_dst), exist_ok=True)
    shutil.copy(audio_src, audio_dst)

    return os.path.join(folder_dst, file_dst)


def load_results_1(num_results=12):
    text = load_text()

    def load_result(id_, speaker):
        audio_orig_src = os.path.join(AUDIO_PATH, speaker, id_ + ".jpg")
        return {
            "text": text[id_],
            "speaker": speaker,
            "sample-id": id_,
            "video-path": get_video_path(id_, speaker),
            "audio-path-orig": None,
            "audio-path-theirs": get_audio_path(id_, speaker, "theirs"),
            "audio-path-ours": get_audio_path(id_, speaker, "ours-baseline"),
        }

    ids = load_filelist("data/grid/filelists/k-seen-test.txt")
    random.shuffle(ids)
    selected_ids = sorted(ids[:num_results], key=lambda t: int(t[1][1:]))
    return [load_result(*i) for i in selected_ids]


doc = dominate.document(title="Speaker disentanglement in video-to-speech conversion")

with doc.head:
    # tags.link(rel="stylesheet", href="static/style.css")
    # tags.script(type="text/javascript", src="script.js")
    # jQuery
    tags.script(
        type="text/javascript", src="https://code.jquery.com/jquery-3.5.1.min.js",
    )
    # Bootstrap
    tags.link(
        rel="stylesheet",
        href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css",
    )
    tags.script(
        type="text/javascript",
        src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js",
    )


# fmt: off
with doc:
    with tags.body():
        with tags.div(cls="container"):

            tags.h1("Speaker disentanglement in video-to-speech conversion", cls="mt-5")
            tags.p(
                "This web-page presents results for our Interspeech 2020 submission. "
                "We show qualitative results for two sets of experiments:"
            )

            with tags.ul():

                with tags.li():
                    tags.a("video-to-speech", href="#video-to-speech.html")
                    tags.span("in which we evaluate our baseline system with respect to the previous work;")

                with tags.li():
                    tags.a("speaker control", href="#speaker-control.html")
                    tags.span("in which we evaluate a speaker-dependent model to generate audio in a new voice.")

            tags.p("Our code is available here.")
            tags.span("Note: If you are having trouble playing the videos below, please consider using the Chrome browser.", cls="text-muted")

            tags.h2("Video-to-speech", name="video-to-speech", cls="mt-3")
            tags.p(
                "We show results for the seen scenario (four speakers which were also used at traning). "
                "We randomly selected 12 samples and showing the synthesized audio for our method (baseline or B) and for the work of Vougioukas et al. (Interspeech, 2019), denoted by V2S GAN. "
                "We show the lip crops, similar to the input of our network (although the network gets grayscale videos)."
            )

            data = load_results_1()
            for row in partition(6, data):
                with tags.div(cls="row mt-2 align-items-end"):
                    for col in row:
                        with tags.div(cls="col-2 text-center"):
                            with tags.div():
                                tags.span(col["speaker"], cls="text-muted")
                                tags.code(col["sample-id"], cls="ml-1")
                            tags.span(col["text"], cls="font-italic")
                            with tags.video(controls=True, cls="embed-responsive"):
                                tags.source(src=col["video-path"], type="video/webm")
                            tags.span("Ours")
                            with tags.audio(controls=True, cls="embed-responsive"):
                                tags.source(src=col["audio-path-ours"], type="audio/wav")
                            tags.span("V2S GAN")
                            with tags.audio(controls=True, cls="embed-responsive"):
                                tags.source(src=col["audio-path-theirs"], type="audio/wav")

            tags.h2("Speaker control", name="speaker-control", cls="mt-3")
            tags.p("We show results for the unseen scenario.")


with open("output/www/is20/index.html", "w") as f:
    f.write(str(doc))
# fmt: on
