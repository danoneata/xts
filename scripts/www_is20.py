# Generates static web-page: http://speed.pub.ro/xts/

import json
import os
import pdb
import random
import shutil
import subprocess

from itertools import groupby

import dominate

from dominate import tags
from dominate.util import raw

from toolz import compose, first, partition_all, second


SEED = 1337
random.seed(SEED)


VIDEO_PATH = "data/grid/video"
AUDIO_PATH = "data/grid/audio-from-video"

FACE_PATH = "data/grid/face-landmarks"
OUT_PATH = "output/www/is20"


TARGET_SPEAKERS = "s1 s3 s5 s6 s7 s10 s12 s14 s15 s17 s22 s26 s28 s32".split()


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


def get_audio_path(id_, speaker, method, filelist=None):
    METHOD_TO_FOLDER_SRC = {
        "ours-baseline": "output/synth-samples/grid-k-seen-test-magnus-best",
        "ours-speaker-identity-dispel": f"output/synth-samples/grid-multi-speaker-{filelist}-magnus-multi-speaker-drop-frames-linear-speaker-dispel-best-emb-all-speakers",
        "ours-speaker-embedding-dispel": f"output/synth-samples/grid-multi-speaker-{filelist}-bjorn-drop-frames-linear-speaker-dispel-best-emb-all-speakers",
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


get_speaker_id = lambda s: int(s[1:])


def load_results_1(num_results=12):
    text = load_text()

    def load_result(id_, speaker):
        audio_orig_src = os.path.join(AUDIO_PATH, speaker, id_ + ".jpg")
        return {
            "text": text[id_],
            "speaker": speaker,
            "sample-id": id_,
            "video-path": get_video_path(id_, speaker),
            "audio-path-theirs": get_audio_path(id_, speaker, "theirs"),
            "audio-path-ours": get_audio_path(id_, speaker, "ours-baseline"),
        }

    ids = load_filelist("data/grid/filelists/k-seen-test.txt")
    random.shuffle(ids)
    selected_ids = sorted(ids[:num_results], key=lambda t: int(t[1][1:]))
    return [load_result(*i) for i in selected_ids]


def load_results_2(filelist, method):
    text = load_text()

    ids = load_filelist(os.path.join("data/grid/filelists", filelist + ".txt"))
    # Pick first sample from each audio
    selected_ids = [first(g) for _, g in groupby(ids, key=lambda t: t[1])]
    selected_ids = sorted(selected_ids, key=compose(get_speaker_id, second))
    # random.shuffle(ids)
    # selected_ids = sorted(ids[:num_results], key=lambda t: int(t[1][1:]))

    def load_result(id_, speaker):
        audio_orig_src = os.path.join(AUDIO_PATH, speaker, id_ + ".jpg")
        return {
            "text": text[id_],
            "speaker": speaker,
            "sample-id": id_,
            "video-path": get_video_path(id_, speaker),
            "audio-paths-ours": [
                (target, get_audio_path(f"{id_}-{target}", speaker, method, filelist))
                for target in TARGET_SPEAKERS
            ],
        }

    return [load_result(*i) for i in selected_ids]


doc = dominate.document(title="Speaker disentanglement in video-to-speech conversion")

with doc.head:
    tags.meta(**{'content': 'text/html;charset=utf-8', 'http-equiv': 'Content-Type'})
    tags.meta(**{'content': 'utf-8', 'http-equiv': 'encoding'})

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
        src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js",
        integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo",
        crossorigin="anonymous",
    )
    tags.script(
        type="text/javascript",
        src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js",
    )
    # My scripts
    # tags.link(rel="stylesheet", href="static/style.css")


# fmt: off
with doc:
    with tags.body():
        with tags.div(cls="container"):

            tags.h1("Speaker disentanglement in video-to-speech conversion", cls="mt-5")
            with tags.p():
                tags.span("This web-page presents results for our Interspeech 2020 submission:")
                tags.blockquote(raw(r"Dan Oneață, Adriana Stan, Horia Cucu. Speaker disentanglement in video-to-speech conversion."), cls="blockqoute ml-4 font-italic")
                tags.span("We show qualitative results for two sets of experiments:")

            with tags.ul():

                with tags.li():
                    tags.a("video-to-speech", href="#video-to-speech")
                    tags.span("in which we evaluate our baseline system with respect to the previous work;")

                with tags.li():
                    tags.a("speaker control", href="#speaker-control")
                    tags.span("in which we evaluate a speaker-dependent model to generate audio in a target voice.")

            p = tags.p("Our code is available ")
            p += tags.a("here", href="https://github.com/danoneata/xts")
            p += "."

            tags.span("Note: If you are having trouble playing the videos below, please consider using the Chrome or Firefox browsers.", cls="text-muted")

            tags.h2("Video-to-speech", name="video-to-speech", cls="mt-3")
            raw(
                "<p>We show results for the <em>seen</em> scenario, in which we consider videos from four speakers encountered at training. "
                "We have randomly selected 12 video samples and show the synthesized audio for our baseline method (denoted by B in the paper) and for the work of <a href='https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1445.pdf'>Vougioukas et al. (Interspeech, 2019)</a> (denoted by V2S GAN). "
                "The videos are cropped around the lips, corresponding to the input to our network. "
                "These results correspond to section 4.1 in our paper.</p>"
            )

            data1 = load_results_1()
            for row in partition_all(6, data1):
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

            p = tags.p("In this experiment, we synthesize audio based on two inputs:")
            p += tags.em("(i) ")
            p += "the video stream showing the lip movement and "
            p += tags.em("(ii) ")
            p += "a target identity. "
            p += "For each test video sample we synthesize audio in all target voices encountered at train time. "
            p += "Based on whether we have seen (or not) the identity shown in the video at train time, we have two scenarios: "
            p += tags.em("seen")
            p += " and "
            p += tags.em("unseen")
            p += ". "
            p += "You can select the desired target identity using the drop-down menu beneath each video."
            # p += "We show a randomly selected sample for each subject."

            # raw("""<p>In this experiment, we synthesize audio based on two inputs:
            # <em>(i)</em> the video stream showing the lip movement and
            # <em>(ii)</em> a target identity.
            # Based on whether we have seen (or not) the identity shown in the video at train time, we consider two scenarios:
            # <em>seen</em> and  <em>unseen</em>.
            # For each test video sample we synthesize audio in all target voices encountered at train time&mdash;you can select the desired target identity using the drop-down menu beneath each video.
            # We show a randomly selected sample for each subject.</p>""")

            tags.h3("Seen scenario")

            p = tags.p("In this scenario the input videos at test time have identities also encountered at train time (14 identities), ")
            p += "but neither the video samples nor the word sequence were seen during training. "
            p += "Due to space limitations, we were not able to present quantitative results for this setting in our paper."

            with tags.div(cls="form-group form-inline"):
                tags.label("Method:")
                with tags.select(cls="form-control ml-2 method", **{"data-scenario": "seen"}):
                    tags.option("speaker identity (SI)", selected=True, data_name="ours-speaker-identity-dispel")
                    tags.option("speaker embedding (SE)", data_name="ours-speaker-embedding-dispel")

            data2 = load_results_2(
                filelist="multi-speaker-tiny-test",
                method="ours-speaker-embedding-dispel",
            )

            data2 = load_results_2(
                filelist="multi-speaker-tiny-test",
                method="ours-speaker-identity-dispel",
            )

            for row in partition_all(6, data2):
                with tags.div(cls="row mt-2 align-items-end"):
                    for col in row:
                        key = f"seen-{col['sample-id']}"
                        with tags.div(cls="col-2 text-center"):
                            with tags.div():
                                tags.span(col["speaker"], cls="text-muted")
                                tags.code(col["sample-id"], cls="ml-1")
                            tags.span(col["text"], cls="font-italic")
                            with tags.video(controls=True, cls="embed-responsive"):
                                tags.source(src=col["video-path"], type="video/webm")
                            with tags.div(cls="form-group"):
                                tags.label("Target identity:", fr=key)
                                with tags.select(cls="form-control target-identity", id=key, data_scenario="seen"):
                                    for t, _ in col["audio-paths-ours"]:
                                        is_target_identity = t == col["speaker"]
                                        tags.option(t, selected=is_target_identity, data_target=t, data_speaker=col["speaker"], data_sample=col["sample-id"])
                            p, = [p for t, p in col["audio-paths-ours"] if t == col["speaker"]]
                            # tags.source(src=p, type="audio/wav")
                            tags.audio(controls=True, cls="embed-responsive", id=key + "-audio", data_scenario="seen", src=p)

            tags.h3("Unseen scenario", cls="mt-3")

            p = tags.p("In this scenario the identities of the people in the input videos at test time (9 identities) are different from the identities of the people at train time (14 identities). ")
            p += "We still synthesize speech in all target voices encountered at train time (14 voices). "
            p += "These results correspond to section 4.2 in our paper."


            with tags.div(cls="form-group form-inline"):
                tags.label("Method:")
                with tags.select(cls="form-control ml-2 method", data_scenario="unseen"):
                    tags.option("speaker identity (SI)", selected=True, data_name="ours-speaker-identity-dispel")
                    tags.option("speaker embedding (SE)", data_name="ours-speaker-embedding-dispel")

            data3 = load_results_2(
                filelist="unseen-k-tiny-test",
                method="ours-speaker-embedding-dispel",
            )

            data3 = load_results_2(
                filelist="unseen-k-tiny-test",
                method="ours-speaker-identity-dispel",
            )

            for row in partition_all(6, data3):
                with tags.div(cls="row mt-2 align-items-end"):
                    for col in row:
                        key = f"seen-{col['sample-id']}"
                        with tags.div(cls="col-2 text-center"):
                            with tags.div():
                                tags.span(col["speaker"], cls="text-muted")
                                tags.code(col["sample-id"], cls="ml-1")
                            tags.span(col["text"], cls="font-italic")
                            with tags.video(controls=True, cls="embed-responsive"):
                                tags.source(src=col["video-path"], type="video/webm")
                            with tags.div(cls="form-group"):
                                tags.label("Target identity:", fr=key)
                                with tags.select(cls="form-control target-identity", id=key, data_scenario="unseen"):
                                    for t, _ in col["audio-paths-ours"]:
                                        tags.option(t, data_target=t, data_speaker=col["speaker"], data_sample=col["sample-id"])
                            with tags.audio(controls=True, cls="embed-responsive", id=key + "-audio", data_scenario="unseen"):
                                _, p = first(col["audio-paths-ours"])
                                tags.source(src=p, type="audio/wav")

    tags.script(type="text/javascript", src="script.js")

with open("output/www/is20/index.html", "w") as f:
    f.write(str(doc))
# fmt: on
