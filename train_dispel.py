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

import ignite.engine as engine
import ignite.handlers

from hparams import HPARAMS

from train import (
    DATASET_PARAMETERS,
    DEVICE,
    EVERY_K_ITERS,
    IMAGE_TRANSFORM,
    LR_REDUCE_PARAMS,
    MODELS,
    PATH_LOADERS,
    PATIENCE,
    ROOT,
    SEED,
    SpeakerInfo,
    TRAIN_TRANSFORMS,
    VALID_TRANSFORMS,
    cache,
    collate_fn,
    compute_mel_mean,
    link_best_model,
    get_argument_parser,
    prepare_batch_2,
    prepare_batch_3,
    update_namespace,
)

import src.dataset


MAX_EPOCHS = 256
BATCH_SIZE = 6


class TemporalClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=64):
        super(TemporalClassifier, self).__init__()
        self.nn_pre = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.nn_post = nn.Sequential(
            nn.Linear(hidden_dim, n_classes), nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        B, T, D = x.shape
        p = self.nn_pre(x.reshape(-1, D))
        p = p.view(B, T, -1)
        p = p.mean(dim=1)
        p = self.nn_post(p)
        return p


class LinearTemporalClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(LinearTemporalClassifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, n_classes), nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.linear(x)
        return x


MODELS_SPEAKER = {
    "generic": TemporalClassifier,
    "linear": LinearTemporalClassifier,
}


def train(args, trial, is_train=True, study=None):

    hparams = HPARAMS[args.hparams]

    if hparams.model_type in {"bjorn"}:
        Dataset = src.dataset.xTSDatasetSpeakerIdEmbedding

        def prepare_batch(batch, device, non_blocking):
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)
            batch_x, batch_y, _, emb = batch
            return (batch_x, batch_y, emb), batch_y

    else:
        Dataset = src.dataset.xTSDatasetSpeakerId
        prepare_batch = prepare_batch_3

    train_path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-train")
    valid_path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-valid")

    train_dataset = Dataset(hparams, train_path_loader, transforms=TRAIN_TRANSFORMS)
    valid_dataset = Dataset(hparams, valid_path_loader, transforms=VALID_TRANSFORMS)

    kwargs = dict(batch_size=args.batch_size, collate_fn=collate_fn)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, **kwargs)

    num_speakers = len(train_path_loader.speaker_to_id)
    dataset_parameters = DATASET_PARAMETERS[args.dataset]
    dataset_parameters["num_speakers"] = num_speakers

    hparams = update_namespace(hparams, trial.parameters)
    model = MODELS[hparams.model_type](dataset_parameters, hparams)
    model_speaker = MODELS_SPEAKER[hparams.model_speaker_type](
        hparams.encoder_embedding_dim, num_speakers
    )

    if hparams.drop_frame_rate:
        path_mel_mean = os.path.join(
            "output", "mel-mean", f"{args.dataset}-{args.filelist}.npz"
        )
        mel_mean = cache(compute_mel_mean, path_mel_mean)(train_dataset)["mel_mean"]
        mel_mean = torch.tensor(mel_mean).float().to(DEVICE)
        mel_mean = mel_mean.unsqueeze(0).unsqueeze(0)
        model.decoder.mel_mean = mel_mean

    model_name = f"{args.dataset}_{args.filelist}_{args.hparams}_dispel"
    model_path = f"output/models/{model_name}.pth"

    # Initialize model from existing one.
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))

    if hasattr(hparams, "model_speaker_path"):
        model_speaker.load_state_dict(torch.load(hparams.model_speaker_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=trial.parameters["lr"])  # 0.001
    optimizer_speaker = torch.optim.Adam(model_speaker.parameters(), lr=0.001)

    mse_loss = nn.MSELoss()

    def loss_reconstruction(pred, true):
        pred1, pred2 = pred
        return mse_loss(pred1, true) + mse_loss(pred2, true)

    λ = 0.0002

    model.to(DEVICE)
    model_speaker.to(DEVICE)

    def step(engine, batch):
        model.train()
        model_speaker.train()

        x, y = prepare_batch(batch, device=DEVICE, non_blocking=True)
        i = batch[2].to(DEVICE)

        # Generator: generates audio and dispels speaker identity
        y_pred, z = model.forward_emb(x)
        i_pred = model_speaker.forward(z)

        entropy_s = (-i_pred.exp() * i_pred).sum(dim=1).mean()  # entropy on speakers
        loss_r = loss_reconstruction(y_pred, y)  # reconstruction
        loss_g = loss_r - λ * entropy_s  # generator

        optimizer.zero_grad()
        loss_g.backward(retain_graph=True)
        optimizer.step()

        # Discriminator: predicts speaker identity
        optimizer_speaker.zero_grad()
        loss_s = F.nll_loss(i_pred, i)
        loss_s.backward()
        optimizer_speaker.step()

        return {
            "loss-generator": loss_g.item(),
            "loss-reconstruction": loss_r.item(),
            "loss-speaker": loss_s.item(),
            "entropy-speaker": entropy_s,
        }

    trainer = engine.Engine(step)

    # trainer = engine.create_supervised_trainer(
    #     model, optimizer, loss, device=device, prepare_batch=prepare_batch
    # )

    evaluator = engine.create_supervised_evaluator(
        model,
        metrics={"loss": ignite.metrics.Loss(loss_reconstruction)},
        device=DEVICE,
        prepare_batch=prepare_batch,
    )

    @trainer.on(engine.Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print(
            "Epoch {:3d} | Loss gen.: {:+8.6f} = {:8.6f} - λ * {:8.6f} | Loss disc.: {:8.6f}".format(
                trainer.state.epoch,
                trainer.state.output["loss-generator"],
                trainer.state.output["loss-reconstruction"],
                trainer.state.output["entropy-speaker"],
                trainer.state.output["loss-speaker"],
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
            "output/models/checkpoints",
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
