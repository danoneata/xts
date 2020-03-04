import argparse
import os
import os.path
import pdb

from types import SimpleNamespace

from typing import Dict

import numpy as np

from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data

import ignite.engine as engine
import ignite.handlers

from hparams import hparams

from train import (
    LR_REDUCE_PARAMS,
    PATH_LOADERS,
    ROOT,
)


MAX_EPOCHS = 2 ** 16
BATCH_SIZE = 256
PATIENCE = 16


class LinearTemporalClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(LinearTemporalClassifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, n_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.linear(x)
        return x


class SpeakerClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self, speaker_to_id: Dict[str, int], dataset_name: str, model_name: str
    ):
        super(SpeakerClassificationDataset, self).__init__()
        # Load visual features
        name = f"{dataset_name}_{model_name}.npz"
        path = os.path.join("output/visual-features", name)
        data = np.load(path)

        self.speakers = [id1.split()[1] for id1 in data["ids"]]
        self.features = data["features"].astype(np.float32)

        self.speaker_to_id = speaker_to_id

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError
        x = self.features[idx]
        y = self.speaker_to_id[self.speakers[idx]]
        return x, y


def train(args, trial, is_train=True, study=None):
    visual_embedding_dim = 512
    device = "cuda"

    path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-train")
    get_dataset_name = lambda split: f"{args.dataset}-{args.filelist}-{split}"

    num_speakers = len(path_loader.speaker_to_id)
    speaker_to_id = path_loader.speaker_to_id

    train_dataset = SpeakerClassificationDataset(
        speaker_to_id, get_dataset_name("valid"), args.xts_model_name
    )
    valid_dataset = SpeakerClassificationDataset(
        speaker_to_id, get_dataset_name("test"), args.xts_model_name
    )

    model_speaker = LinearTemporalClassifier(visual_embedding_dim, num_speakers)

    kwargs = dict(batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, **kwargs)

    optimizer = torch.optim.Adam(model_speaker.parameters(), lr=trial.parameters["lr"])
    loss = nn.NLLLoss()

    model_speaker.to(device)

    model_name = f"{args.dataset}_{args.filelist}_speaker-classifier"
    model_path = f"output/models/{model_name}.pth"

    trainer = engine.create_supervised_trainer(
        model_speaker, optimizer, loss, device=device
    )

    metrics_validation = {
        "loss": ignite.metrics.Loss(loss),
        "accuracy": ignite.metrics.Accuracy(),
    }
    evaluator = engine.create_supervised_evaluator(
        model_speaker, metrics=metrics_validation, device=device,
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
            "Epoch {:3d} Valid loss: {:8.6f} Accuracy: {:9.6f} ←".format(
                trainer.state.epoch,
                metrics["loss"],
                metrics["accuracy"],
            )
        )

    lr_reduce = lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=args.verbose, **LR_REDUCE_PARAMS
    )

    @evaluator.on(engine.Events.COMPLETED)
    def update_lr_reduce(engine):
        loss = - engine.state.metrics["accuracy"]
        lr_reduce.step(loss)

    def score_function(engine):
        return engine.state.metrics["accuracy"]

    early_stopping_handler = ignite.handlers.EarlyStopping(
        patience=PATIENCE, score_function=score_function, trainer=trainer
    )
    evaluator.add_event_handler(engine.Events.COMPLETED, early_stopping_handler)

    trainer.run(train_loader, max_epochs=args.max_epochs)

    return evaluator.state.metrics["accuracy"]


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Train speaker classifier on visual features."
    )
    parser.add_argument(
        "-x",
        "--xts-model-name",
        type=str,
        required=True,
        help="name of xTS model used for visual feature extraction",
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
    parser.add_argument("-v", "--verbose", action="count", help="verbosity level")
    return parser


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    args.batch_size = BATCH_SIZE
    args.max_epochs = MAX_EPOCHS
    trial = SimpleNamespace(**{"parameters": {"lr": 1e-2}})
    print(args)
    print(trial)
    accuracy = train(args, trial)
    print(f"{100 * accuracy:.2f}")


if __name__ == "__main__":
    main()
