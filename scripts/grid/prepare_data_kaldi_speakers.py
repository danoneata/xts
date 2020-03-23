import argparse
import os
import pdb

from itertools import groupby

from prepare_data_kaldi import (
    WAV_DIR,
    drop_sil,
    get_filename,
    get_files,
    get_files_default,
    get_speaker,
    get_spk_id,
    get_utt_id,
    get_wav_path,
    load_sentence,
)


def get_utt_id_t(path, target_id):
    utt_id = get_utt_id(path)
    return f"{utt_id}-s{target_id:02d}"


def main():
    parser = argparse.ArgumentParser(description="Generates filelists for Kaldi")
    parser.add_argument("-f", "--filelist", help="which files to use")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="data/kaldi/full",
        help="where to store the generated filed",
    )
    parser.add_argument(
        "-w",
        "--wav-dir",
        default=WAV_DIR,
        help="directory where the audio files (WAV) are stored",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = get_files(args.filelist)

    wav_scp = os.path.join(args.output_dir, "wav.scp")
    utt2spk = os.path.join(args.output_dir, "utt2spk")
    spk2utt = os.path.join(args.output_dir, "spk2utt")
    text = os.path.join(args.output_dir, "text")

    spk_utt = []

    TRAINING_SPEAKERS = list(map(int, "1 3 5 6 7 10 12 14 15 17 22 26 28 32".split()))

    with open(wav_scp, "w") as f, open(utt2spk, "w") as g, open(text, "w") as h:
        for p in sorted(files, key=get_utt_id):
            for t in TRAINING_SPEAKERS:
                utt_id = get_utt_id_t(p, t)
                spk_id = get_spk_id(p)
                spk_2d = f"s{spk_id:02d}"

                sentence = drop_sil(load_sentence(get_filename(p), "s" + str(spk_id)))
                wav_path = get_wav_path(args.wav_dir, spk_id, get_filename(p) + f"-s{t}")

                f.write("{} {}\n".format(utt_id, wav_path))
                g.write("{} {}\n".format(utt_id, spk_2d))
                h.write("{} {}\n".format(utt_id, sentence))

                spk_utt.append((spk_2d, utt_id))

    with open(spk2utt, "w") as f:
        by_spk = lambda t: t[0]
        spk_utt = sorted(spk_utt, key=by_spk)
        for spk, group in groupby(spk_utt, key=by_spk):
            utts = (utt for _, utt in group)
            f.write("{} {}\n".format(spk, " ".join(utts)))


if __name__ == "__main__":
    main()
