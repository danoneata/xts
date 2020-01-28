Installation steps:
```
conda env create -f environment.yml
pip install -r requirements.txt
```

The code is organized as follows:

- `data` contains dataset-related information (_e.g._, videos, audio, face landmarks, speaker embeddings)

## Preparing a dataset

- Set paths to video, for example in `data/$DATASET/video`
- Extract middle frame of each video using `scripts/extract_middle_frame.py`
- Extract face landmarks from the middle frame using `scripts/detect_face_landmarks_folder.py`
- Extract speaker embeddings
