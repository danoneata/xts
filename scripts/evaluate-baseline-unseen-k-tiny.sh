set -e

model=magnus
model_short=magnus
model_long=${model}-best-emb-none
preds_name=grid-multi-speaker-unseen-k-tiny-test-${model_long}

# Mel-spectrograms
python predict.py \
    --hparams ${model_short} \
    -d grid \
    --model-path output/models/grid_multi-speaker_${model}_best.pth \
    --filelist unseen-k-tiny \
    --filelist-train multi-speaker \
    -o output/predictions/$preds_name \
    -v

# WAVs
P=/home/doneata/work/xts/output/predictions
cd ~/work/dc-tts-xts && venv/bin/python synthesize_spectro.py ${P}/${preds_name}.npz && cd -

python scripts/grid/prepare_data_kaldi.py \
    -f unseen-k-tiny-test \
    -o data/grid/kaldi/unseen-k-tiny-test_synth-${model_long} \
    -w $(realpath output/synth-samples/grid-multi-speaker-unseen-k-tiny-test-${model_long})

cd ~/work/experiments-tedlium-r2

bash local/xts/run-grammar.sh --dset unseen-k-tiny-test_synth-${model_long}
