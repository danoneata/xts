set -e

P=/home/doneata/work/xts/output/predictions

cd ~/work/dc-tts-xts

# B
venv/bin/python synthesize_spectro.py $P/grid-multi-speaker-unseen-k-small-test-magnus-best.npz

# SI
venv/bin/python synthesize_spectro.py $P/grid-multi-speaker-unseen-k-small-test-magnus-multi-speaker-best-emb-mean.npz

# SI + D
venv/bin/python synthesize_spectro.py $P/grid-multi-speaker-unseen-k-small-test-magnus-multi-speaker-dispel-best-emb-mean.npz

# SE
venv/bin/python synthesize_spectro.py $P/grid-multi-speaker-unseen-k-small-test-bjorn-best-emb-mean.npz

# SE + D
venv/bin/python synthesize_spectro.py $P/grid-multi-speaker-unseen-k-small-test-bjorn-dispel-best-emb-mean.npz
