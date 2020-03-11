for k in magnus-best magnus-multi-speaker-best-emb-mean magnus-multi-speaker-dispel-best-emb-mean bjorn-best-emb-mean bjorn-dispel-best-emb-mean; do
    python scripts/grid/prepare_data_kaldi.py \
        -f unseen-k-small-test \
        -o data/grid/kaldi/unseen-k-small-test_synth-${k} \
        -w $(realpath output/synth-samples/grid-multi-speaker-unseen-k-small-test-${k})
done

cd ~/work/experiments-tedlium-r2

for k in magnus-best magnus-multi-speaker-best-emb-mean magnus-multi-speaker-dispel-best-emb-mean bjorn-best-emb-mean bjorn-dispel-best-emb-mean; do
    bash local/xts/run-grammar.sh --dset unseen-k-small-test_synth-${k}
done
