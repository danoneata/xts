import argparse
import os
import os.path
import pdb
import re
import sys
import time

import numpy as np

from types import SimpleNamespace
from toolz import first

from tqdm import tqdm  # type: ignore

from typing import Dict

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data

from torch.nn import functional as F

from torchvision import transforms

import ignite.engine as engine
import ignite.handlers

from hparams import HPARAMS

import src.dataset

from src.dataset import (
    collate_fn,
    prepare_batch_2,
    prepare_batch_3,
    DATASET_PARAMETERS,
    PATH_LOADERS,
)

from models import MODELS
from models.nn import SpeakerInfo

from my_utils import cache


ROOT = os.environ.get("ROOT", "data")
CHECKPOINTS_DIR = "output/models/checkpoints"
DEVICE = os.environ.get("CUDA_DEVICE", "cuda")


SEED = 1337
EVERY_K_ITERS = 512
MAX_EPOCHS = 128
PATIENCE = 8
BATCH_SIZE = 8
LR_REDUCE_PARAMS = {
    "factor": 0.2,
    "patience": 4,
}


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # rough estimation of the mean and standard deviation
        transforms.Normalize([0.40], [0.15]),
    ]
)

TRAIN_TRANSFORMS = {
    "video": transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomHorizontalFlip(), IMAGE_TRANSFORM,]
    ),
}

VALID_TRANSFORMS = {
    "video": transforms.Compose([transforms.ToPILImage(), IMAGE_TRANSFORM,]),
}


def get_argument_parser():
    parser = argparse.ArgumentParser(description="Evaluate a given model")
    parser.add_argument(
        "--hparams",
        type=str,
        required=True,
        choices=HPARAMS,
        help="which hyper-parameter configuration to use",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=PATH_LOADERS,
        help="which dataset to train on",
    )
    parser.add_argument(
        "--filelist", type=str, default="tiny2", help="name of the filelist to use",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        required=False,
        help="path to model to load",
    )
    parser.add_argument("-v", "--verbose", action="count", help="verbosity level")
    return parser


def link_best_model(model_name):
    pattern = re.compile(r"(.*)_model_.*=([\-0-9.]+).pth")

    def is_match(filename, model_name):
        m = pattern.search(filename)
        return m and m.group(1) == model_name

    def best_score(filename):
        m = pattern.search(filename)
        if m:
            return -float(m.group(2))
        else:
            assert False, "Pattern did not match"

    files = [f for f in os.listdir(CHECKPOINTS_DIR) if is_match(f, model_name)]
    name = first(sorted(files, key=best_score))

    src = os.path.join("checkpoints", name)
    dst = f"output/models/{model_name}_best.pth"

    try:
        os.symlink(src, dst)
    except FileExistsError:
        os.unlink(dst)
        os.symlink(src, dst)

    return dst


def update_namespace(namespace: SimpleNamespace, dict1: Dict) -> SimpleNamespace:
    dict0 = namespace.__dict__
    dict0.update(dict1)
    return SimpleNamespace(**dict0)


def compute_mel_mean(dataset):
    num_samples = len(dataset)
    _, num_mel_channels = dataset[0][1].shape

    mel_means = np.zeros((num_samples, num_mel_channels))
    num_frames = np.zeros((num_samples, 1))

    for i in tqdm(range(num_samples)):
        _, mels, *_ = dataset[i]
        if mels is not None:
            mel_means[i] = mels.mean(dim=0).numpy()
            num_frames[i] = len(mels)
        else:
            pass

    mel_mean = (num_frames * mel_means).sum(axis=0) / num_frames.sum()
    return {"mel_mean": mel_mean}


def train(args, trial, is_train=True, study=None):

    hparams = HPARAMS[args.hparams]

    train_path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-train")
    valid_path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-valid")

    num_speakers = len(train_path_loader.speaker_to_id)
    dataset_parameters = DATASET_PARAMETERS[args.dataset]
    dataset_parameters["num_speakers"] = num_speakers

    hparams = update_namespace(hparams, trial.parameters)
    model = MODELS[hparams.model_type](dataset_parameters, hparams)

    model_name = f"{args.dataset}_{args.filelist}_{args.hparams}"
    model_path = f"output/models/{model_name}.pth"

    # Initialize model from existing one.
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    # Select the dataset accoring to the type of speaker information encoded in the model.
    if model.speaker_info is SpeakerInfo.NOTHING:
        Dataset = src.dataset.xTSDataset
        prepare_batch = prepare_batch_2
    elif model.speaker_info is SpeakerInfo.ID:
        Dataset = src.dataset.xTSDatasetSpeakerId
        prepare_batch = prepare_batch_3
    elif model.speaker_info is SpeakerInfo.EMBEDDING:
        if hparams.use_fixed_embeddings:
            Dataset = src.dataset.xTSDatasetSpeakerFixedEmbedding
        else:
            Dataset = src.dataset.xTSDatasetSpeakerEmbedding
        prepare_batch = prepare_batch_3
    else:
        assert False, "Unknown speaker info"

    train_dataset = Dataset(hparams, train_path_loader, transforms=TRAIN_TRANSFORMS)
    valid_dataset = Dataset(hparams, valid_path_loader, transforms=VALID_TRANSFORMS)

    if model.speaker_info is SpeakerInfo.EMBEDDING and hparams.embedding_normalize:
        model.embedding_stats = train_dataset.embedding_stats

    if hparams.drop_frame_rate:
        path_mel_mean = os.path.join(
            "output", "mel-mean", f"{args.dataset}-{args.filelist}.npz"
        )
        mel_mean = cache(compute_mel_mean, path_mel_mean)(train_dataset)["mel_mean"]
        mel_mean = torch.tensor(mel_mean).float().to(DEVICE)
        mel_mean = mel_mean.unsqueeze(0).unsqueeze(0)
        model.decoder.mel_mean = mel_mean

    kwargs = dict(batch_size=args.batch_size, collate_fn=collate_fn)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, **kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=trial.parameters["lr"])  # 0.001
    mse_loss = nn.MSELoss()

    def loss(pred, true):
        pred1, pred2 = pred
        return mse_loss(pred1, true) + mse_loss(pred2, true)

    trainer = engine.create_supervised_trainer(
        model, optimizer, loss, device=DEVICE, prepare_batch=prepare_batch
    )

    evaluator = engine.create_supervised_evaluator(
        model,
        metrics={"loss": ignite.metrics.Loss(loss)},
        device=DEVICE,
        prepare_batch=prepare_batch,
    )

    @trainer.on(engine.Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print(
            "Epoch {:3d} Train loss: {:8.6f}".format(
                trainer.state.epoch, trainer.state.output
            )
        )

    @trainer.on(engine.Events.ITERATION_COMPLETED(every=EVERY_K_ITERS))
    def log_validation_loss(trainer):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        print(
            "Epoch {:3d} Valid loss: {:8.6f} ←".format(
                trainer.state.epoch, metrics["loss"]
            )
        )

    lr_reduce = lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=args.verbose, **LR_REDUCE_PARAMS
    )

    @evaluator.on(engine.Events.COMPLETED)
    def update_lr_reduce(engine):
        loss = engine.state.metrics["loss"]
        lr_reduce.step(loss)

    @evaluator.on(engine.Events.COMPLETED)
    def terminate_study(engine):
        """Stops underperforming trials."""
        if study and study.should_trial_stop(trial=trial):
            trainer.terminate()

    def score_function(engine):
        return -engine.state.metrics["loss"]

    early_stopping_handler = ignite.handlers.EarlyStopping(
        patience=PATIENCE, score_function=score_function, trainer=trainer
    )
    evaluator.add_event_handler(engine.Events.COMPLETED, early_stopping_handler)

    if is_train:

        def global_step_transform(*args):
            return trainer.state.iteration // EVERY_K_ITERS

        checkpoint_handler = ignite.handlers.ModelCheckpoint(
            CHECKPOINTS_DIR,
            model_name,
            score_name="objective",
            score_function=score_function,
            n_saved=5,
            require_empty=False,
            create_dir=True,
            global_step_transform=global_step_transform,
        )
        evaluator.add_event_handler(
            engine.Events.COMPLETED, checkpoint_handler, {"model": model}
        )

    trainer.run(train_loader, max_epochs=args.max_epochs)

    if is_train:
        torch.save(model.state_dict(), model_path)
        print("Last model @", model_path)

        model_best_path = link_best_model(model_name)
        print("Best model @", model_best_path)

    return evaluator.state.metrics["loss"]


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    args.batch_size = BATCH_SIZE
    args.max_epochs = MAX_EPOCHS
    trial = SimpleNamespace(**{"parameters": {"lr": 5e-4,},})
    print(args)
    print(trial)
    train(args, trial)


if __name__ == "__main__":
    main()
