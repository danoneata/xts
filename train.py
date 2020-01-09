import argparse
import os
import os.path
import pdb
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data

from torch.nn import functional as F

import ignite.engine as engine
import ignite.handlers

import src.dataset

from models import MODELS

ROOT = os.environ.get("ROOT", "")

SEED = 1337
MAX_EPOCHS = 128
PATIENCE = 4
LR_REDUCE_PARAMS = {
    "factor": 0.2,
    "patience": 2,
}


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a given model")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=MODELS,
        help="which model type to train",
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
    args = parser.parse_args()

    print(args)

    model = MODELS[args.model_type]()
    if args.model is not None:
        model_path = args.model
        model_name = os.path.basename(args.model)
        model.load(model_path)
    else:
        model_name = f"{args.model_type}"
        model_path = f"output/models/{model_name}.pth"

    train_dataset = src.dataset.xTSDataset(ROOT, "tiny2")
    valid_dataset = src.dataset.xTSDataset(ROOT, "tiny2")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=3, collate_fn=collate_fn, shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=3, collate_fn=collate_fn, shuffle=False
    )

    # ignite_train = DataLoader(train_loader, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
        # trainer.state.dataloader = model.ignite_random(train_loader, args.num_minibatches, args.minibatch_size, args.epoch_fraction)

    lr_reduce = lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=args.verbose, **LR_REDUCE_PARAMS
    )

    @evaluator.on(engine.Events.COMPLETED)
    def update_lr_reduce(engine):
        loss = engine.state.metrics["loss"]
        lr_reduce.step(loss)

    def score_function(engine):
        return -engine.state.metrics["loss"]

    early_stopping_handler = ignite.handlers.EarlyStopping(
        patience=PATIENCE, score_function=score_function, trainer=trainer
    )
    evaluator.add_event_handler(engine.Events.EPOCH_COMPLETED, early_stopping_handler)

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

    trainer.run(train_loader, max_epochs=MAX_EPOCHS)
    torch.save(model.state_dict(), model_path)
    print("Model saved at:", model_path)


if __name__ == "__main__":
    main()
