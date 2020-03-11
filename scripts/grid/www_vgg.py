import itertools
import os
import pdb
import random
import shutil
import subprocess

import dominate
import dominate.tags as T

from data import load_data as load_samples


def load_data():
    def process(line):
        return line.split()
    with open(os.path.join("www", "preds.txt"), "r") as f:
        return [process(line) for line in f.readlines()]


def create_page(data, samples):
    doc = dominate.document(title="Deep Lip Reading on the Grid data")

    with doc.head:
        T.meta(**{'content': 'text/html;charset=utf-8', 'http-equiv': 'Content-Type'})
        T.meta(**{'content': 'utf-8', 'http-equiv': 'encoding'})
        T.link(rel='stylesheet', href='style.css')
        T.script(
            src="https://code.jquery.com/jquery-3.4.1.min.js",
            integrity="sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo=",
            crossorigin="anonymous",
        )
        T.script(type='text/javascript', src='script.js')

    with doc:
        T.h3("Deep Lip Reading on the Grid data")

        T.p("""
            We run the pre-trained transformer model from VGG on word-level video snippets (shown as GIFs);
            these are taken from the Grid corpus (see also videos below).
            For each word-level snippet we provide the following information:
        """)

        with T.ul():
            T.li("the sample's name")
            T.li("the input to the visual frontend")
            T.li("the prediction")
            T.li("the groundtruth")

        T.p("""
            The GIFs are at 4 frames per second and can be animated by hovering over.
        """)

        for k, group in itertools.groupby(data, key=lambda t: t[0].split("_")[0]):

            # Show video
            sample, *_ = [s for s in samples if s.key == k]
            src = sample.video_path
            dst = os.path.join("www", "data-vgg", k + ".mp4")
            if not os.path.exists(dst):
                subprocess.run(["ffmpeg", "-i", src, dst])

            with T.video(controls="controls"):
                path = os.path.join("data-vgg", k + ".mp4")
                T.source(src=path, **{"type": "video/mp4"})
                T.br()

            # Show crops
            with T.div():
                for key, pred, gt in group:
                    with T.div(cls="sample"):
                        src = os.path.join("data-vgg", key + "_still.gif")
                        with T.div(cls="info"):
                            T.span(key)
                        T.img(src=src, width=100, height=100)
                        with T.div():
                            T.span(pred)
                        with T.div():
                            T.span(gt)

    path = os.path.join("www", "vgg.html")

    with open(path, "w") as f:
        f.write(doc.render())

    print(path)


def main():
    samples = load_samples("tiny")
    data = load_data()
    create_page(data, samples)


if __name__ == "__main__":
    main()
