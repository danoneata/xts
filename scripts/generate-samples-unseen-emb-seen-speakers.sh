# Synthesizes speech in the the voices of all seen speakers.

set -e

function get_speakers {
    cat data/grid/filelists/multi-speaker-train.txt | \
        cut -f2 -d" " | \
        sort -u
}

for s in $(get_speakers); do
    p=output/predictions/grid-multi-speaker-unseen-k-small-test-magnus-multi-speaker-best-emb-spk-$s
    python predict.py \
        -d grid \
        -m magnus-multi-speaker \
        --model-path output/models/grid_multi-speaker_magnus-multi-speaker_best.pth \
        --filelist unseen-k-small \
        --filelist-train multi-speaker \
        --emb spk-$s \
        -o $p
    cd ~/work/dc-tts-xts && \
        venv/bin/python synthesize_spectro.py /home/doneata/work/xts/${p}.npz && \
        cd -
done
