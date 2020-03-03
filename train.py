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

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data

from torch.nn import functional as F

from torchvision import transforms

import ignite.engine as engine
import ignite.handlers

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

ROOT = os.environ.get("ROOT", "data")
CHECKPOINTS_DIR = "output/models/checkpoints"

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
        "-m",
        "--model-type",
        type=str,
        required=True,
        choices=MODELS,
        help="which model type to train",
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
    pattern = re.compile(r'(.*)_model_[0-9]+_objective=([\-0-9.]+).pth')

    def is_match(filename, model_name):
        m = pattern.search(filename)
        return m and m.group(1) == model_name

    def best_score(filename):
        m = pattern.search(filename)
        return - float(m.group(2))

    files = [f for f in os.listdir(CHECKPOINTS_DIR) if is_match(f, model_name)]
    name = first(sorted(files, key=best_score))

    src = os.path.join('checkpoints', name)
    dst = f"output/models/{model_name}_best.pth"

    try:
        os.symlink(src, dst)
    except FileExistsError:
        os.unlink(dst)
        os.symlink(src, dst)

    return dst


def train(args, trial, is_train=True, study=None):

    train_path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-train")
    valid_path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-valid")

    num_speakers = len(train_path_loader.speaker_to_id)
    dataset_parameters = DATASET_PARAMETERS[args.dataset]
    dataset_parameters["num_speakers"] = num_speakers

    model = MODELS[args.model_type](dataset_parameters, trial.parameters)

    model_name = f"{args.dataset}_{args.filelist}_{args.model_type}"
    model_path = f"output/models/{model_name}.pth"

    # Initialize model from existing one.
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))

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

    train_dataset = Dataset(train_path_loader, transforms=TRAIN_TRANSFORMS)
    valid_dataset = Dataset(valid_path_loader, transforms=VALID_TRANSFORMS)

    kwargs = dict(batch_size=args.batch_size, collate_fn=collate_fn)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, **kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=trial.parameters["lr"])  # 0.001
    mse_loss = nn.MSELoss()

    def loss(pred, true):
        pred1, pred2 = pred
        return mse_loss(pred1, true) + mse_loss(pred2, true)

    device = "cuda"

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
