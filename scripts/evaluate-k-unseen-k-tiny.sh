# Evaluate Konstantinos's method on the tiny split that I have defined
set -e

python scripts/grid/prepare_data_kaldi.py \
    -f unseen-k-tiny-test \
    -o data/grid/kaldi/unseen-k-tiny-test_synth-k \
    -w $(realpath data/grid/samples-konstantinos/unseen)

cd ~/work/experiments-tedlium-r2

bash local/xts/run-grammar.sh --dset unseen-k-small-test_synth-k
