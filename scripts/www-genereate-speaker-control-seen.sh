set -e

model=$1  # e.g. magnus-multi-speaker-drop-frames-linear-speaker_dispel
model_short=${model%%_dispel}
model_short=${model_short%%_revgrad}
model_long=${model/_/-}-best-emb-all-speakers
preds_name=grid-multi-speaker-multi-speaker-tiny-test-${model_long}

# Mel-spectrograms
python predict.py \
    --hparams ${model_short} \
    -d grid \
    --model-path output/models/grid_multi-speaker_${model}_best.pth \
    --filelist multi-speaker-tiny \
    --filelist-train multi-speaker \
    --embedding all-speakers \
    -o output/predictions/$preds_name \
    -v

# WAVs
P=/home/doneata/work/xts/output/predictions
cd ~/work/dc-tts-xts && venv/bin/python synthesize_spectro.py ${P}/${preds_name}.npz && cd -
