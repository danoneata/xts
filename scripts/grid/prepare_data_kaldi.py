import argparse
import os
import pdb

from glob import glob

from itertools import groupby

from data import drop_sil, load_sentence


WAV_DIR = "/srv/share/student-share/data/grid/audio-16khz"


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


def get_wav_path(wav_dir, spk_id, filename):
    path = os.path.join(wav_dir, f"s{spk_id}/{filename}.wav")
    if wav_dir == WAV_DIR:
        return path
    else:
        return f"sox {path} -t wav -r 16000 -b 16 - |"


def get_files_default():
    I = "data/grid"
    files = glob(I + "/align/**/*.align", recursive=True)
    files = sorted(files, key=lambda f: get_utt_id(f))
    return files


def get_files(filelist):
    if not filelist:
        return get_files_default()
    else:
        def parse_line(line):
            id_, speaker = line.split()
            return f"{speaker}/{id_}.align"
        filelist_path = os.path.join("data", "grid", "filelists", filelist + ".txt")
        with open(filelist_path, 'r') as f:
            return [parse_line(line) for line in f.readlines()]


def main():
    parser = argparse.ArgumentParser(description="Generates filelists for Kaldi")
    parser.add_argument("-f", "--filelist", help="which files to use")
    parser.add_argument("-o", "--output-dir", default="data/kaldi/full", help="where to store the generated filed")
    parser.add_argument("-w", "--wav-dir", default=WAV_DIR, help="directory where the audio files (WAV) are stored")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = get_files(args.filelist)

    wav_scp = os.path.join(args.output_dir, "wav.scp")
    utt2spk = os.path.join(args.output_dir, "utt2spk")
    spk2utt = os.path.join(args.output_dir, "spk2utt")
    text = os.path.join(args.output_dir, "text")

    spk_utt = []

    with open(wav_scp, 'w') as f, open(utt2spk, 'w') as g, open(text, 'w') as h:
        for p in sorted(files, key=get_utt_id):
            utt_id = get_utt_id(p)
            spk_id = get_spk_id(p)
            spk_2d = f"s{spk_id:02d}"
            sentence = drop_sil(load_sentence(get_filename(p), 's' + str(spk_id)))

            f.write("{} {}\n".format(utt_id, get_wav_path(args.wav_dir, spk_id, get_filename(p))))
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
