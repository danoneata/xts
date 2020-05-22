This repository contains code for video-to-speech conversion.
For more information, please see our Interspeech 2020 submission.

## Installation

Installation steps:

```bash
conda env create -f environment.yml
pip install -r requirements.txt
```

Clone the Tacotron2 repository:

```bash
git clone https://github.com/NVIDIA/tacotron2.git
```

## Code structure

The code is organized as follows:

- `train.py` is the main script, which trains video-to-speech models.
- `train_dispel.py` and `train_revgrad.py` are used to train models that _dispel_ the speaker identity from the visual features.
- `train_asr_clf.py` and `train_speaker_clf.py` train linear probes in the _visual_ feature space.
- `hparams.py` contain hyper-parameter configurations.
- `audio.py` contains audio-processing functionality, _e.g._ extracting Mel spectrograms.
- `models/` contain video-to-speech architectures (video decoders and audio decoders).
- `src/` contains data structures that wrap datasets.
- `evaluate/` implement the evaluation metrics (PESQ, MCD, STOI, WER).
- `scripts/` contain mostly scripts to run experiments or process data.
- `data/` is where the datasets are stored (_i.e._, videos, audio, face landmarks, speaker embeddings).

## Getting started

We provide a data bundle (video, audio, face landmarks, speaker embeddings) for a speaker in GRID (the speaker `s1`).
You can download the data from [here](https://drive.google.com/open?id=1CKBSUKU4kN3xj0keC7zEORexioMJr6W0) and extract it locally:

```bash
unzip grid-s1.zip
```

To train our baseline model just run the following command:

```bash
python train.py --hparams magnus -d grid --filelist k-s01 -v
```

## Preparing a new dataset

- Set paths to video, for example in `data/$DATASET/video`
- Extract middle frame of each video using `scripts/extract_middle_frame.py`
- Extract face landmarks from the middle frame using `scripts/detect_face_landmarks_folder.py`
- Extract speaker embeddings using `scripts/extract_speaker_embeddings`

## Synthesizing speech

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
