import pdb
import os
import sys

import numpy as np
import torch

from train import (
    DATASET_PARAMETERS,
    MODELS,
    ROOT,
    PATH_LOADERS,
    VALID_TRANSFORMS,
    SpeakerInfo,
    collate_fn,
    get_argument_parser,
    prepare_batch_2,
    prepare_batch_3,
)
import src.dataset


BATCH_SIZE = 16
DEVICE = "cuda"
NON_BLOCKING = False


def predict(args):
    dataset_parameters = DATASET_PARAMETERS[args.dataset]
    model = MODELS[args.model_type](dataset_parameters)

    model_name = f"{args.dataset}_{args.filelist}_{args.model_type}"
    model_path = f"output/models/{model_name}.pth"

    # Initialize model from existing one.
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    if DEVICE == "cuda":
        model.cuda()

    # Select the dataset accoring to the type of speaker information encoded in the model.
    if model.speaker_info is SpeakerInfo.NOTHING:
        Dataset = src.dataset.xTSDataset
        prepare_batch = prepare_batch_2
    elif model.speaker_info is SpeakerInfo.ID:
        Dataset = src.dataset.xTSDatasetSpeakerId
        prepare_batch = prepare_batch_3
    elif model.speaker_info is SpeakerInfo.EMBEDDING:
        Dataset = src.dataset.xTSDatasetSpeakerEmbedding
        prepare_batch = prepare_batch_3
    else:
        assert False, "Unknown speaker info"

    # Prepare data loader
    path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-test")
    dataset = Dataset(path_loader, transforms=VALID_TRANSFORMS)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )

    with torch.no_grad():
        for batch in loader:
            (x, _, extra), _ = prepare_batch(batch, DEVICE, NON_BLOCKING)
            batch_x = x, extra
            p = model.predict(batch_x)


def main():
    parser = get_argument_parser()
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="where to store the predictions",
    )
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
