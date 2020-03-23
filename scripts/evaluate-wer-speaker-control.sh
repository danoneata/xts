set -e

model=$1
model=${model/_/-}-best-emb-all-speakers

python scripts/grid/prepare_data_kaldi_speakers.py \
    -f unseen-k-tiny-test \
    -o data/grid/kaldi/unseen-k-tiny-test_synth-${model} \
    -w $(realpath output/synth-samples/grid-multi-speaker-unseen-k-tiny-test-${model})

cd ~/work/experiments-tedlium-r2

bash local/xts/run-grammar.sh --dset unseen-k-tiny-test_synth-${model}
