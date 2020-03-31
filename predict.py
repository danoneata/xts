import pdb
import os
import sys

import numpy as np
import torch

from tqdm import tqdm  # type: ignore

from train import (
    DATASET_PARAMETERS,
    HPARAMS,
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

    hparams = HPARAMS[args.hparams]
    model = MODELS[hparams.model_type](dataset_parameters, hparams)

    # Initialize model from existing one.
    # from collections import OrderedDict
    # def updatek(k):
    #     if k.startswith('conv0') or k.startswith('encoder'):
    #         return 'video_encoder.' + k
    #     else:
    #         return k
    # def rename_state_dict(s):
    #     return OrderedDict((updatek(k), v) for k, v in s.items())
    # model.load_state_dict(rename_state_dict(torch.load(args.model_path)))
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    if DEVICE == "cuda":
        model.cuda()

    # Select the dataset according to the type of speaker information encoded in the model.
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
    dataset = Dataset(hparams, te_path_loader, transforms=VALID_TRANSFORMS)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )

    if model.speaker_info is SpeakerInfo.EMBEDDING and hparams.embedding_normalize:
        tr_dataset = src.dataset.xTSDatasetSpeakerEmbedding(hparams, tr_path_loader, transforms=VALID_TRANSFORMS)
        model.embedding_stats = tr_dataset.embedding_stats

    if args.embedding == "all-speakers":
        n_target = num_speakers
        id_to_speaker = {i: s for s, i in tr_path_loader.speaker_to_id.items()}

        def update_id(id_, spk_id_tgt):
            utt_id, spk_src = id_.split()
            spk_tgt = id_to_speaker[spk_id_tgt]
            return f"{utt_id}-{spk_tgt} {spk_src}"

        def get_ids():
            return [
                update_id(id_, tgt)
                for id_ in te_path_loader.ids
                for tgt in range(num_speakers)
            ]

    else:
        n_target = 1
        get_ids = lambda: te_path_loader.ids

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
    elif model.speaker_info is SpeakerInfo.ID and args.embedding == "all-speakers":
        get_i = lambda s, n: torch.zeros(n).long().to(DEVICE) + s

        def predict1(model, inp):
            x, _ = inp
            preds = [model.predict((x, get_i(s, len(x)))) for s in range(num_speakers)]
            preds = torch.stack(preds).transpose(0, 1)
            _, _, T, D = preds.shape
            preds = preds.reshape(-1, T, D)
            return preds

    elif (
        model.speaker_info is SpeakerInfo.EMBEDDING and args.embedding == "all-speakers"
    ):
        data_embedding = np.load(tr_path_loader.paths["speaker-embeddings"][0])
        embeddings = data_embedding["feats"]
        speakers = [utt_id.split()[1] for utt_id in data_embedding["ids"].tolist()]
        embeddings_speaker = np.zeros((num_speakers, embeddings.shape[1]))
        for speaker in set(speakers):
            i = tr_path_loader.speaker_to_id[speaker]
            idxs = [speaker == t for t in speakers]
            embeddings_speaker[i] = np.mean(embeddings[idxs], axis=0)
        embeddings_speaker = torch.tensor(embeddings_speaker).float().to(DEVICE)

        def get_embedding(s, n):
            return embeddings_speaker[s].unsqueeze(0).repeat(n, 1)

        def predict1(model, inp):
            x, _ = inp
            preds = [
                predict_emb(model, inp, get_embedding(s, len(x)))
                for s in range(num_speakers)
            ]
            preds = torch.stack(preds).transpose(0, 1)
            _, _, T, D = preds.shape
            preds = preds.reshape(-1, T, D)
            return preds
    else:
        n_target = 1
        if model.speaker_info is SpeakerInfo.ID:
            # check that speakers agree
            get_speakers = lambda p: sorted(p.speaker_to_id.keys())
            same_speakers = get_speakers(tr_path_loader) == get_speakers(te_path_loader)
            # FIXME?
            # assert same_speakers, "speakers in the two filelists do not agree"
        elif model.speaker_info is SpeakerInfo.Embedding:
            assert False
        predict1 = lambda model, inp: model.predict(inp)

    n_samples = len(dataset)
    t = DATASET_PARAMETERS[args.dataset]["len-out"]
    n_mel_channels = 80
    preds = np.zeros((n_samples * n_target, t, n_mel_channels))

    with torch.no_grad():
        for b, batch in enumerate(tqdm(loader)):
            (x, _, extra), _ = prepare_batch(batch, DEVICE, NON_BLOCKING)
            p = predict1(model, (x, extra))
            α = BATCH_SIZE * n_target * b
            ω = BATCH_SIZE * n_target * (b + 1)
            preds[α:ω] = p.cpu().numpy()

    ids = get_ids()
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
