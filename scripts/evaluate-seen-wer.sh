for k in indep multi-speaker multi-speaker-dispel; do
    python scripts/grid/prepare_data_kaldi.py \
        -f k-seen-test \
        -o data/grid/kaldi/k-seen-test_synth-magnus-${k}-best \
        -w $(realpath output/synth-samples/grid-k-seen-test-magnus-${k}-best)
done

cd ~/work/experiments-tedlium-r2

for k in indep multi-speaker multi-speaker-dispel; do
    bash local/xts/run-grammar.sh --dset k-seen-test_synth-magnus-${k}-best
done
