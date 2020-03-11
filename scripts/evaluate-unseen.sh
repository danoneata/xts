set -e

for p in magnus-best magnus-multi-speaker-best-emb-mean magnus-multi-speaker-dispel-best-emb-mean bjorn-best-emb-mean bjorn-dispel-best-emb-mean; do
    for m in stoi pesq mcd; do
        python evaluate/quality.py \
            -m $m \
            -d grid \
            --filelist unseen-k-small \
            -p output/synth-samples/grid-multi-speaker-unseen-k-small-test-$p
    done
done
