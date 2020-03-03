import pdb
import os
import sys

import numpy as np
import torch

from tqdm import tqdm  # type: ignore

from typing import List

sys.path.insert(0, ".")
from train import (
    DATASET_PARAMETERS,
    MODELS,
    ROOT,
    PATH_LOADERS,
    VALID_TRANSFORMS,
    SpeakerInfo,
    collate_fn,
    get_argument_parser,
    prepare_batch_3,
)

from src.dataset import xTSDatasetSpeakerIdFilename  # type: ignore

from hparams import hparams


BATCH_SIZE = 16
DEVICE = "cuda"
NON_BLOCKING = False


def extract_visual_features(args):
    visual_embedding_dim = hparams.encoder_embedding_dim
    input_dim = DATASET_PARAMETERS[args.dataset]["len-inp"]

    model_name, _ = os.path.splitext(os.path.basename(args.model_path))
    output_path = args.output_path or os.path.join(
        "output/visual-features",
        f"{args.dataset}-{args.filelist}-{args.split}_{model_name}.npz",
    )

    path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-" + args.split)

    num_speakers = len(path_loader.speaker_to_id)
    dataset_parameters = DATASET_PARAMETERS[args.dataset]
    dataset_parameters["num_speakers"] = num_speakers

    dataset = xTSDatasetSpeakerIdFilename(path_loader, transforms=VALID_TRANSFORMS)
    prepare_batch = prepare_batch_3
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )

    model = MODELS[args.model_type](dataset_parameters)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.to(DEVICE)

    features = np.zeros((len(dataset), input_dim, visual_embedding_dim))

    num_features = 0
    ids = []  # type: List[str]

    with torch.no_grad():
        for b, batch in enumerate(tqdm(loader)):
            x, _, _, i = batch
            x = x.to(DEVICE)
            z = model._encode_video(x)
            size = len(x)

            features[num_features : num_features + size] = z.cpu().numpy()
            ids.extend(i)

            num_features += size

    features = features[:num_features]
    assert len(features) == len(ids)
    np.savez(output_path, ids=ids, features=features)


def main():
    parser = get_argument_parser()
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="test",
        choices={"train", "valid", "test"},
        required=False,
        help="which data to use",
    )
    parser.add_argument(
        "-o", "--output-path", type=str, help="where to store the visual features",
    )
    args = parser.parse_args()
    print(args)
    extract_visual_features(args)


if __name__ == "__main__":
    main()
