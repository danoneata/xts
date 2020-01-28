import itertools
import os
import pdb
import random
import shutil

import dominate
import dominate.tags as T

from constants import SEED, SELECTED_PHONES
from data import DATA_DIR, load_data

random.seed(SEED)
SPLIT = "small"
N_SAMPLES = 128


def load_lexicon():
    def process(line):
        word, *rest = line.split()
        return word, rest
    PATH = "/home/doneata/work/experiments-tedlium-r2/data/local/dict/lexicon.txt"
    with open(PATH, 'r') as f:
        lexicon = dict(process(line) for line in f.readlines())
    return lexicon

def find_word_id(lexicon, text, phones, phone_id):
    lengths = [len(lexicon[word]) for word in text.split()]
    i = phone_id - sum(p in ("SIL", "HH") for p in phones[:phone_id])
    word_id = 0
    while True:
        if i - lengths[word_id] < 0:
            break
        i -= lengths[word_id] 
        word_id += 1
    return word_id


def create_page_index():
    doc = dominate.document(title="Phone alignments")
    lexicon = load_lexicon()

    with doc.head:
        T.meta(**{'content': 'text/html;charset=utf-8', 'http-equiv': 'Content-Type'})
        T.meta(**{'content': 'utf-8', 'http-equiv': 'encoding'})
        T.link(rel='stylesheet', href='../style.css')
        T.script(
            src="https://code.jquery.com/jquery-3.4.1.min.js",
            integrity="sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo=",
            crossorigin="anonymous",
        )
        T.script(type='text/javascript', src='../script.js')

    with doc:
        T.p("""
            The web-pages linked below show alignments of phones with lip movement.
            The data is a subset of 2000 utterances from the Grid corpus.
            We picked a selection of phones and for each of them we randomly selected at most 128 pairs.
        """)

        with T.ul():
            for k, g in itertools.groupby(SELECTED_PHONES, key=lambda w: w[0]):
                with T.li():
                    for i in g:
                        T.a(i, href=i + "/index.html")

        T.p("""Mean duration (s) of each phone:""")
        T.pre("""
AA 0.154
AE 0.066
AH 0.060
AO 0.143
AW 0.277
AY 0.142
EH 0.091
EY 0.124
IH 0.068
IY 0.113
OW 0.123
UW 0.152
            """, style="font-family: Courier")

    path = "www/index.html"
    with open(path, "w") as f:
        f.write(doc.render())


def create_page_1(phone, data):
    doc = dominate.document(title="Phone alignments – " + phone)
    lexicon = load_lexicon()

    with doc.head:
        T.meta(**{'content': 'text/html;charset=utf-8', 'http-equiv': 'Content-Type'})
        T.meta(**{'content': 'utf-8', 'http-equiv': 'encoding'})
        T.link(rel='stylesheet', href='../style.css')
        T.script(
            src="https://code.jquery.com/jquery-3.4.1.min.js",
            integrity="sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo=",
            crossorigin="anonymous",
        )
        T.script(type='text/javascript', src='../script.js')

    with doc:
        T.h3("Phone: ", phone)
        with T.div(cls="hint"):
            T.p("Notes:")
            with T.ul():
                T.li("hover over the images to play the gif and the audio")
                T.li("below each clip we indicate the utterance id, speaker id and text")
                T.li("the word in bold and brown indicates the \"location\" of the uttered phone")

        for k, sample in data.items():
            # Copy gif
            for t in {"still", "anima"}:
                src = os.path.join(DATA_DIR, "crops", k + f"_{t}.gif")
                dst = os.path.join("www", DATA_DIR, k + f"_{t}.gif")
                shutil.copyfile(src, dst)

            # Copy audio
            src = os.path.join(DATA_DIR, "crops", k + f".wav")
            dst = os.path.join("www", DATA_DIR, k + f".wav")
            shutil.copyfile(src, dst)

            # Create item
            with T.div(cls="sample"):
                src = os.path.join("..", DATA_DIR, k + "_still.gif")
                T.img(src=src, width=100, height=80)
                with T.audio(controls=True):
                    src = os.path.join("..", DATA_DIR, k + ".wav")
                    T.source(src=src, type="audio/wav")
                with T.div(cls="info"):
                    T.span(sample.key + " | " + sample.speaker)
                with T.div(cls="text"):
                    text = [word for word in sample.sentence.split() if word != "sil"]
                    text = " ".join(text)
                    phone_id = k.split("_")[1]
                    phone_id = int(phone_id)
                    phones = [phone for phone, _, _ in sample.audio_alignment]
                    i = find_word_id(lexicon, text, phones, phone_id)
                    words = text.split()
                    T.span(" ".join(words[:i]))
                    T.span(T.b(words[i]), style="color:brown")
                    T.span(" ".join(words[i + 1:]))

    directory = os.path.join("www", phone)
    path = os.path.join(directory, "index.html")

    os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        f.write(doc.render())

    print(path)


def main():
    create_page_index()

    samples = load_data(SPLIT)
    get_key = lambda sample, i: f"{sample.key}_{i:04d}"

    for phone in SELECTED_PHONES:
        print(phone)
        data = {
            get_key(sample, i): sample
            for sample in samples
            for i, (p, _, _) in enumerate(sample.audio_alignment)
            if p == phone
        }
        data = {
            k: v
            for k, v in data.items()
            if os.path.exists(os.path.join(DATA_DIR, "crops", k + "_anima.gif"))
        }
        if len(data) > N_SAMPLES:
            keys = random.sample(data.keys(), N_SAMPLES)
        else:
            keys = data.keys()

        data = {k: data[k] for k in sorted(keys)}
        create_page_1(phone, data)


if __name__ == "__main__":
    main()
