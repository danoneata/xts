Installation steps:
```bash
conda env create -f environment.yml
pip install -r requirements.txt
```

Clone tacotron2:

```bash
git clone https://github.com/NVIDIA/tacotron2.git
```

The code is organized as follows:

- `data` contains dataset-related information (_e.g._, videos, audio, face landmarks, speaker embeddings)

## Preparing a dataset

- Set paths to video, for example in `data/$DATASET/video`
- Extract middle frame of each video using `scripts/extract_middle_frame.py`
- Extract face landmarks from the middle frame using `scripts/detect_face_landmarks_folder.py`
- Extract speaker embeddings

## Synthesize speech

1. video to mel-spectrogram
```bash
python predict.py -m magnus --model-path output/models/grid_multi-speaker_magnus.pth -d grid --filelist multi-speaker -v -o output/predictions/grid-multi-test-magnus.npz
```

2. mel-spectrogram to WAV:
```bash
# ~/work/dc-tts-xts
# source venv/bin/activate
python synthesize_spectro.py ~/work/xts/output/predictions/grid-multi-test-magnus.npz
```
