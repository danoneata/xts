model=magnus-multi-speaker-drop-frames-linear-speaker-dispel
model_short=${m%%_dispel}
model_long=${model}-best-emb-all-speakers
preds_name=grid-multi-speaker-unseen-k-tiny-test-${model_long}

# Mel-spectrograms
python predict.py \
    --hparams ${model_short} \
    -d grid \
    --model-path output/models/grid_multi-speaker_${model}_best.pth \
    --filelist unseen-k-tiny \
    --filelist-train multi-speaker \
    --embedding all-speakers \
    -o output/predictions/$preds_name \
    -v

# WAVs
P=/home/doneata/work/xts/output/predictions
cd ~/work/dc-tts-xts && venv/bin/python synthesize_spectro.py ${P}/${preds_name}.npz && cd -

# Speaker embeddings
~/src/vgg-speaker-recognition/venv/bin/python scripts/extract_speaker_embeddings.py \
    --dataset synth:${model_long} \
    --filelist unseen-k-tiny-test

# Evaluate
python evaluate/speaker_recognition.py --model ${model_long}
