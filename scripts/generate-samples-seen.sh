set -e

P=/home/doneata/work/xts/output/predictions

cd ~/work/dc-tts-xts

# B / spk
for i in 01 02 04 29; do
    venv/bin/python synthesize_spectro.py $P/grid-k-s${i}-test-magnus-best.npz
done

# B
venv/bin/python synthesize_spectro.py $P/grid-k-seen-test-magnus-best.npz

# SI
venv/bin/python synthesize_spectro.py $P/grid-k-seen-test-magnus-multi-speaker-best.npz

# SI + D
venv/bin/python synthesize_spectro.py $P/grid-k-seen-test-magnus-multi-speaker-dispel-best.npz
