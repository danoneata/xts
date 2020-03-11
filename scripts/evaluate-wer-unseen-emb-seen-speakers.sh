set -e

function get_speakers {
    cat ~/work/xts/data/grid/filelists/multi-speaker-train.txt | \
        cut -f2 -d" " | \
        sort -u
}

for s in $(get_speakers); do
    method=magnus-multi-speaker-best-emb-spk-$s 
    python scripts/grid/prepare_data_kaldi.py \
        -f unseen-k-small-test \
        -o data/grid/kaldi/unseen-k-small-test_synth-${method} \
        -w $(realpath output/synth-samples/grid-multi-speaker-unseen-k-small-test-${method})
done

cd ~/work/experiments-tedlium-r2

for s in $(get_speakers); do
    method=magnus-multi-speaker-best-emb-spk-$s 
    bash local/xts/run-grammar.sh --dset unseen-k-small-test_synth-${method}
done
