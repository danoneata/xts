This repository contains code for video-to-speech conversion.
For more information, please see our EUSIPCO 2021 paper (available on [arXiv](https://arxiv.org/abs/2105.09652)):

> Dan Oneață, Adriana Stan, Horia Cucu.
> Speaker disentanglement in video-to-speech conversion.
> EUSIPCO, 2021.

Qualitative samples are available [here](http://speed.pub.ro/xts/).

## Installation

Installation steps:

```bash
conda env create -f environment.yml
conda activate xts
pip install -r requirements.txt
```

Note: Depending on your GPU, you may need to specify different versions for `cudatoolkit` and `Pytorch` in the `environment.yml` configuration file.

Clone the Tacotron2 repository:

```bash
git clone https://github.com/NVIDIA/tacotron2.git
```

## Structure

We describe how the code and data are organized in the repository.

**Code.**
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

**Data.**
The `data` folder contains a folder for each audio-visual dataset,
which in turn contains sub-folders for the different modalities,
the most important being audio, face landmarks, file-lists, speaker embeddings, video.
An example directory structure for the GRID dataset is the following:
```
data/
└── grid
    ├── audio-from-video
    ├── face-landmarks
    ├── filelists
    ├── speaker-embeddings
    └── video
```

The path names are set by the [`PathLoader`](https://github.com/danoneata/xts/blob/master/src/dataset.py#L75) from [`src/dataset.py`](https://github.com/danoneata/xts/blob/master/src/dataset.py)
and they can vary from dataset to dataset.

## Getting started

We provide a data bundle (video, audio, face landmarks, speaker embeddings) for a speaker in GRID (the speaker `s1`).
You can download the data from [here](https://sharing.speed.pub.ro/owncloud/index.php/s/U1xmWRLc985A12m) and extract it locally in the folder containing the code:

```bash
wget "https://sharing.speed.pub.ro/owncloud/index.php/s/U1xmWRLc985A12m/download" -O grid-s1.zip
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

## Evaluating intelligibility with an ASR

To evaluate the intelligibility of the synthesized speech, we used an automatic speech recognition (ASR) system.
The ASR is based on Kaldi and trained on the [TED-LIUM dataset](https://openslr.magicdatatech.com/19/).
For evaluation, we constrained the language model to GRID's vocabulary by using a finite state grammar constructed from the sentences in GRID.

<details>
<summary>The finite state grammar used to constrain the language model</summary>

```
<command> = bin | lay | place | set;
<color> = blue | green | red | white;
<preposition> = at | by | in | with;
<letter> = a | b | c | d | e | f | g | h | i | j | k | l | m | n | o | p | q | r | s | t | u | v | x | y | z;
<digit> = zero | one | two | three | four | five | six | seven | eight | nine;
<adverb> = again | now | please | soon;
public <utterance> = <command> <color> <preoposition> <letter> <digit> <adverb>;
```

</details>

To replicate our results, you need to follow these steps:
1. Install [Kaldi](https://kaldi-asr.org/)
2. Download [our models and scripts](https://sharing.speed.pub.ro/owncloud/index.php/s/rUkbhaGq5QfI9rW) and extract them locally:
```bash
unzip xts-asr.zip
```
3. Set up the path to Kaldi in `xts-asr/path.sh`; for example:
```bash
export KALDI_ROOT=/home/doneata/src/kaldi
```
4. Link to the `steps` and `utils` folders from Kaldi in `xts-asr`; for example:
```bash
ln -s /home/doneata/src/kaldi/egs/wsj/s5/steps steps
ln -s /home/doneata/src/kaldi/egs/wsj/s5/utils utils
```
5. Run an evaluation by using the `xts-asr/run.sh` script:
```bash
bash run.sh --dset tiny
```
6. To define a new dataset, you will need to prepare the files `wav.scp`, `text`, `utt2spk` and `spk2utt`.
For an example see the files in `xts-asr/data/grid/tiny`.
For more information, please consult the [Kaldi documentation](https://kaldi-asr.org/doc/data_prep.html).
