import argparse
import os
import os.path
import pdb
import sys
import time

import numpy as np

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data

from torch.nn import functional as F

from torchvision import transforms

import ignite.engine as engine
import ignite.handlers

import src.dataset

from models import MODELS

ROOT = os.environ.get("ROOT", "")

SEED = 1337
MAX_EPOCHS = 1
PATIENCE = 4
BATCH_SIZE = 12
LR_REDUCE_PARAMS = {
    "factor": 0.2,
    "patience": 2,
}
DATASET = "grid"


def collate_fn(batches):
    videos = [batch[0] for batch in batches]
    spects = [batch[1] for batch in batches]

    max_v = max(video.shape[0] for video in videos)
    pad_v = lambda video: (0, 0, 0, 0, 0, max_v - video.shape[0])

    max_s = max(spect.shape[0] for spect in spects)
    pad_s = lambda spect: (0, 0, 0, max_s - spect.shape[0])

    videos = [F.pad(video, pad=pad_v(video)) for video in videos]
    spects = [F.pad(spect, pad=pad_s(spect)) for spect in spects]

    video = torch.stack(videos)
    spect = torch.stack(spects)

    return video, spect


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
        "--model-type",
        type=str,
        required=True,
        choices=MODELS,
        help="which model type to train",
    )
    parser.add_argument(
        "--filelist", type=str, default="tiny2", help="name of the filelist to use",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        required=False,
        help="path to model to load",
    )
    parser.add_argument("-v", "--verbose", action="count", help="verbosity level")
    return parser


def train(args, trial, is_train=True, study=None):

    model = MODELS[args.model_type](trial.parameters)
    if args.model is not None:
        model_path = args.model
        model_name = os.path.basename(args.model)
        model.load(model_path)
    else:
        model_name = f"{DATASET}_{args.filelist}_{args.model_type}"
        model_path = f"output/models/{model_name}.pth"

    # fmt: off
    train_dataset = src.dataset.xTSDataset(ROOT, args.filelist + "-train", transforms=TRAIN_TRANSFORMS)
    valid_dataset = src.dataset.xTSDataset(ROOT, args.filelist + "-valid", transforms=VALID_TRANSFORMS)
    # fmt: on

    kwargs = dict(batch_size=args.batch_size, collate_fn=collate_fn)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, **kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=trial.parameters["lr"])  # 0.001
    mse_loss = nn.MSELoss()

    def loss(pred, true):
        pred1, pred2 = pred
        return mse_loss(pred1, true) + mse_loss(pred2, true)

    device = "cuda"

    def prepare_batch(batch, device, non_blocking):
        batch_x, batch_y = batch
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        return (batch_x, batch_y), batch_y

    trainer = engine.create_supervised_trainer(
        model, optimizer, loss, device=device, prepare_batch=prepare_batch
    )

    evaluator = engine.create_supervised_evaluator(
        model,
        metrics={"loss": ignite.metrics.Loss(loss)},
        device=device,
        prepare_batch=prepare_batch,
    )

    @trainer.on(engine.Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print(
            "Epoch {:3d} Train loss: {:8.6f}".format(
                trainer.state.epoch, trainer.state.output
            )
        )

    @trainer.on(engine.Events.EPOCH_COMPLETED)
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
    evaluator.add_event_handler(engine.Events.EPOCH_COMPLETED, early_stopping_handler)

    if is_train:
        checkpoint_handler = ignite.handlers.ModelCheckpoint(
            "output/models/checkpoints",
            model_name,
            score_function=score_function,
            n_saved=5,
            require_empty=False,
            create_dir=True,
        )
        evaluator.add_event_handler(
            engine.Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model}
        )

    trainer.run(train_loader, max_epochs=args.max_epochs)

    if is_train:
        torch.save(model.state_dict(), model_path)
        print("Model saved at:", model_path)

    return evaluator.state.metrics["loss"]


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    args.batch_size = BATCH_SIZE
    args.max_epochs = MAX_EPOCHS
    print(args)
    trial = SimpleNamespace(**{
        "parameters": {
            "lr": 0.001,
        },
    })
    train(args, trial)


if __name__ == "__main__":
    main()
