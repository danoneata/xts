import os
import pdb

from glob import glob

from itertools import groupby


I = "data"
O = "data/kaldi"


def get_filename(path):
    *_, speaker, filename = path.split('/')
    filename, _ = os.path.splitext(filename)
    return filename


def get_speaker(path):
    *_, speaker, filename = path.split('/')
    return speaker


def get_utt_id(path):
    id_ = get_spk_id(path)
    filename = get_filename(path)
    return f"s{id_:02d}_{filename}"


def get_spk_id(path):
    speaker = get_speaker(path)
    id_ = speaker[1:]
    return int(id_)


def get_wav_path(spk_id, filename):
    return f"/srv/share/student-share/data/grid/audio-16khz/s{spk_id}/{filename}.wav"


def load_sentence(spk_id, filename):
    def process(line):
        *_, word = line.split()
        return word
    def drop_sil(words):
        return (word for word in words if word != "sil")
    path = f"data/align/s{spk_id}/{filename}.align"
    with open(path, "r") as f:
        return " ".join(drop_sil(process(line) for line in f.readlines()))


def main():
    files = glob(I + "/align/**/*.align", recursive=True)
    files = sorted(files, key=lambda f: get_utt_id(f))

    wav_scp = os.path.join(O, "wav.scp")
    utt2spk = os.path.join(O, "utt2spk")
    spk2utt = os.path.join(O, "spk2utt")
    text = os.path.join(O, "text")

    spk_utt = []

    with open(wav_scp, 'w') as f, open(utt2spk, 'w') as g, open(text, 'w') as h:
        for p in files:
            utt_id = get_utt_id(p)
            spk_id = get_spk_id(p)
            spk_2d = f"s{spk_id:02d}"
            sentence = load_sentence(spk_id, get_filename(p))

            f.write("{} {}\n".format(utt_id, get_wav_path(spk_id, get_filename(p))))
            g.write("{} {}\n".format(utt_id, spk_2d))
            h.write("{} {}\n".format(utt_id, sentence))

            spk_utt.append((spk_2d, utt_id))

    with open(spk2utt, 'w') as f:
        by_spk = lambda t: t[0]
        spk_utt = sorted(spk_utt, key=by_spk)
        for spk, group in groupby(spk_utt, key=by_spk):
            utts = (utt for _, utt in group)
            f.write("{} {}\n".format(spk, " ".join(utts)))


if __name__ == "__main__":
    main()
