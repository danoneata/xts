import pdb
import os
import sys

import numpy as np
import torch

from tqdm import tqdm

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


BATCH_SIZE = 8
DEVICE = "cuda"
NON_BLOCKING = False


def predict_emb(model, inp, emb):
    """Predicts with a fixed embedding."""
    x, _ = inp
    x = model._encode_video(x)
    x = model._concat_embedding(x, emb)
    _, y = model.decoder.predict(x)
    return y


def predict(args):
    tr_path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist_train + "-train")
    te_path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-" + args.split)

    num_speakers = len(tr_path_loader.speaker_to_id)
    dataset_parameters = DATASET_PARAMETERS[args.dataset]
    dataset_parameters["num_speakers"] = num_speakers

    model = MODELS[args.model_type](dataset_parameters)

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
        if args.filelist == args.filelist_train:
            Dataset = src.dataset.xTSDatasetSpeakerEmbedding
            prepare_batch = prepare_batch_3
        else:
            Dataset = src.dataset.xTSDatasetSpeakerId
            prepare_batch = prepare_batch_3
    else:
        assert False, "Unknown speaker info"

    # Prepare data loader
    dataset = Dataset(te_path_loader, transforms=VALID_TRANSFORMS)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )

    n_samples = len(dataset)
    t = DATASET_PARAMETERS[args.dataset]["len-out"]
    n_mel_channels = 80
    preds = np.zeros((n_samples, t, n_mel_channels))

    if model.speaker_info is SpeakerInfo.ID and args.embedding == "mean":
        emb = model.speaker_embedding.weight.mean(dim=0, keepdim=True)
        predict1 = lambda model, inp: predict_emb(model, inp, emb.repeat(x.shape[0], 1))
    elif model.speaker_info is SpeakerInfo.EMBEDDING and args.embedding == "mean":
        data_embedding = np.load(tr_path_loader.paths["speaker-embeddings"][0])
        speaker_embeddings = torch.tensor(data_embedding["feats"]).float()
        emb = speaker_embeddings.mean(dim=0, keepdim=True).to(DEVICE)
        predict1 = lambda model, inp: predict_emb(model, inp, emb.repeat(x.shape[0], 1))
    elif (
        model.speaker_info is SpeakerInfo.ID
        and args.embedding
        and args.embedding.startswith("spk")
    ):
        _, speaker = args.embedding.split("-")
        id1 = tr_path_loader.speaker_to_id[speaker]
        emb = model.speaker_embedding.weight[id1]
        predict1 = lambda model, inp: predict_emb(model, inp, emb.repeat(x.shape[0], 1))
    else:
        if model.speaker_info is SpeakerInfo.ID:
            # check that speakers agree
            get_speakers = lambda p: sorted(p.speaker_to_id.keys())
            same_speakers = get_speakers(tr_path_loader) == get_speakers(te_path_loader)
            assert same_speakers, "speakers in the two filelists do not agree"
        elif model.speaker_info is SpeakerInfo.Embedding:
            assert False
        predict1 = lambda model, inp: model.predict(inp)

    with torch.no_grad():
        for b, batch in enumerate(tqdm(loader)):
            (x, _, extra), _ = prepare_batch(batch, DEVICE, NON_BLOCKING)
            p = predict1(model, (x, extra))
            preds[b * BATCH_SIZE : (b + 1) * BATCH_SIZE] = p.cpu().numpy()

    ids = te_path_loader.ids
    filenames = [te_path_loader.id_to_filename(i, "audio") for i in ids]
    np.savez(args.output_path, ids=ids, filenames=filenames, preds=preds)


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
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="where to store the predictions",
    )
    parser.add_argument(
        "--filelist-train",
        help="name of filelist used for training (default, the same as `filelist`)",
    )
    parser.add_argument(
        "-e",
        "--embedding",
        # choices={"mean", None},
        help="what embedding to use",
    )

    args = parser.parse_args()
    args.filelist_train = args.filelist_train or args.filelist
    predict(args)


if __name__ == "__main__":
    main()
