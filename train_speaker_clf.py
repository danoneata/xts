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
    EVERY_K_ITERS,
    LR_REDUCE_PARAMS,
    PATH_LOADERS,
    PATIENCE,
    ROOT,
    link_best_model,
)

from train_dispel import TemporalClassifier


PATIENCE = 32
EVERY_K_ITERS = 64
MAX_EPOCHS = 256 * 64
BATCH_SIZE = 256


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
        x = self.pca.transform(self.features[idx])
        y = self.speaker_to_id[self.speakers[idx]]
        return x, y


def train(args, trial, is_train=True, study=None):
    visual_embedding_dim = 16
    device = "cuda"

    path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-train")
    get_dataset_name = lambda split: f"{args.dataset}-{args.filelist}-{split}"

    num_speakers = len(path_loader.speaker_to_id)
    speaker_to_id = path_loader.speaker_to_id

    train_dataset = SpeakerClassificationDataset(
        speaker_to_id, get_dataset_name("test"), args.xts_model_name
    )
    valid_dataset = SpeakerClassificationDataset(
        speaker_to_id, get_dataset_name("valid"), args.xts_model_name
    )

    # train PCA
    X = train_dataset.features
    N, T, D = X.shape
    X = X.reshape(N * T, D)
    pca = PCA(n_components=visual_embedding_dim)
    pca.fit(X)

    train_dataset.pca = pca
    valid_dataset.pca = pca

    model_speaker = TemporalClassifier(visual_embedding_dim, num_speakers)

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

    evaluator = engine.create_supervised_evaluator(
        model_speaker, metrics={"loss": ignite.metrics.Loss(loss)}, device=device,
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
            engine.Events.COMPLETED, checkpoint_handler, {"model": model_speaker}
        )

    trainer.run(train_loader, max_epochs=args.max_epochs)

    if is_train:
        torch.save(model_speaker.state_dict(), model_path)
        print("Last model @", model_path)

        model_best_path = link_best_model(model_name)
        print("Best model @", model_best_path)

    return evaluator.state.metrics["loss"]


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
    trial = SimpleNamespace(**{"parameters": {"lr": 1e-3}})
    print(args)
    print(trial)
    train(args, trial)


if __name__ == "__main__":
    main()
