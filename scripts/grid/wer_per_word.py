import os
import subprocess
import sys


N_WORDS = 6

dataset = os.environ.get("DATA", "unseen-k-small-test")
model_name = sys.argv[1]


def load_data(path):
    def parse_line(line):
        key, *words = line.split()
        return key, words

    with open(path, "r") as f:
        return dict(parse_line(line) for line in f.readlines())


def get_nth(d, n):
    return {k: ws[n] for k, ws in d.items()}


def wer(d1, d2):
    keys = d2.keys()
    n_total = len(keys)
    n_correct = sum(d1[k] == d2[k] for k in keys)
    return 1 - n_correct / n_total


key = f"{dataset}_synth-{model_name}"
subprocess.run(["bash", "scripts/grid/asr_predict.sh", key])
pred = load_data(f"/tmp/pred-{key}")
true = load_data(f"/home/doneata/work/experiments-tedlium-r2/exp_grid/chain_cleaned/tdnn1f_sp_bi/decode-grammar_{key}/scoring/test_filt.txt")

fmt = lambda wer: f"{100 * wer:4.1f}"

print("Filter out shorter predictions:")
print(len(pred), end=" → ")
pred = {k: v for k, v in pred.items() if len(v) == 6}
print(len(pred))

word_wer = (wer(get_nth(true, n), get_nth(pred, n)) for n in range(N_WORDS))
print(" | ".join(fmt(wer) for wer in word_wer))
