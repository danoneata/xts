import argparse
import os
import os.path
import pdb

from types import SimpleNamespace

from typing import Dict, List

import numpy as np

from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data

from torch.nn import functional as F

import ignite.engine as engine
import ignite.handlers

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

from hparams import hparams

from train import (
    LR_REDUCE_PARAMS,
    PATH_LOADERS,
    ROOT,
)


MAX_EPOCHS = 2 ** 16
BATCH_SIZE = 256
PATIENCE = 16


ASR_DICTIONARY = [
    "bin lay place set".split(),
    "blue green red white".split(),
    "at by in with".split(),
    "a b c d e f g h i j k l m n o p q r s t u v x y z".split(),
    "zero one two three four five six seven eight nine".split(),
    "again now please soon".split(),
]


class MultiTemporalClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: List[int]):
        super(MultiTemporalClassifier, self).__init__()
        self.weights_pos = nn.Parameter(torch.zeros(len(n_classes) + 1, 75))
        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(input_dim, c), nn.LogSoftmax(dim=-1))
                for c in n_classes
            ]
        )

    def _pool(self, x, α):
        _, T, _ = x.shape
        α = α.view(1, T, 1)
        x = α * x / α.sum()
        x = x.sum(dim=1)
        return x

    def forward(self, x):
        # for each classifier pool features based on their position
        α = F.softmax(self.weights_pos.to(x.device), dim=0)
        return [clf(self._pool(x, α[i])) for i, clf in enumerate(self.classifiers)]


class ASRClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self, speaker_to_id: Dict[str, int], dataset_name: str, model_name: str
    ):
        super(ASRClassificationDataset, self).__init__()
        # Load visual features
        name = f"{dataset_name}_{model_name}.npz"
        path = os.path.join("output/visual-features", name)
        data = np.load(path)

        selected_ids = [i.split()[0] for i in data["ids"]]
        self.features = data["features"].astype(np.float32)
        self.id_to_text = self._load_text(selected_ids)

    def _load_text(self, selected_ids):

        invert_dict = lambda d: {v: k for k, v in d.items()}
        word_to_indices = [
            invert_dict(dict(enumerate(words))) for words in ASR_DICTIONARY
        ]

        def drop_sil(words):
            return [word for word in words if word not in {"sil", "sp"}]

        def process_line(line):
            _, key, *words = line.split()
            indices = [
                word_to_indices[i][word] for i, word in enumerate(drop_sil(words))
            ]
            return key, indices

        with open("data/grid/text/full.txt", "r") as f:
            id_to_text = dict(map(process_line, f.readlines()))

        return [id_to_text[i] for i in selected_ids]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError
        x = self.features[idx]
        y = self.id_to_text[idx]
        return x, y


class WER(Metric):
    def __init__(self, n_classes: int):
        self.accuracies = [ignite.metrics.Accuracy() for _ in range(n_classes)]
        super(WER, self).__init__()

    def reset(self):
        for a in self.accuracies:
            a.reset()

    def update(self, output):
        pred, true = output
        for a, p, t in zip(self.accuracies, pred, true):
            a.update((p, t))

    def compute(self):
        fmt = lambda wer: f"{100 * wer:4.1f}"
        print(" | ".join(fmt(1 - a.compute()) for a in self.accuracies))
        return 1 - sum(a.compute() for a in self.accuracies) / len(self.accuracies)


def train(args, trial, is_train=True, study=None):
    visual_embedding_dim = 512
    device = "cuda"

    path_loader = PATH_LOADERS[args.dataset](ROOT, args.filelist + "-train")
    get_dataset_name = lambda split: f"{args.dataset}-{args.filelist}-{split}"

    num_speakers = len(path_loader.speaker_to_id)
    speaker_to_id = path_loader.speaker_to_id

    train_dataset = ASRClassificationDataset(
        speaker_to_id, get_dataset_name("valid"), args.xts_model_name
    )
    valid_dataset = ASRClassificationDataset(
        speaker_to_id, get_dataset_name("test"), args.xts_model_name
    )

    n_classes = [len(words) for words in ASR_DICTIONARY]
    model_asr = MultiTemporalClassifier(visual_embedding_dim, n_classes)

    kwargs = dict(batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, **kwargs)

    optimizer = torch.optim.Adam(model_asr.parameters(), lr=trial.parameters["lr"])

    def loss_multi(true, pred):
        return sum(F.nll_loss(t, p) for t, p in zip(true, pred))

    model_asr.to(device)

    model_name = f"{args.dataset}_{args.filelist}_speaker-classifier"
    model_path = f"output/models/{model_name}.pth"

    trainer = engine.create_supervised_trainer(
        model_asr, optimizer, loss_multi, device=device
    )

    metrics_validation = {
        "loss": ignite.metrics.Loss(loss_multi),
        "wer": WER(len(n_classes)),
    }
    evaluator = engine.create_supervised_evaluator(
        model_asr, metrics=metrics_validation, device=device,
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
            "Epoch {:3d} Valid loss: {:8.6f} WER: {:9.6f} ←".format(
                trainer.state.epoch, metrics["loss"], metrics["wer"],
            )
        )

    lr_reduce = lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=args.verbose, **LR_REDUCE_PARAMS
    )

    @evaluator.on(engine.Events.COMPLETED)
    def update_lr_reduce(engine):
        loss = engine.state.metrics["wer"]
        lr_reduce.step(loss)

    def score_function(engine):
        return -engine.state.metrics["wer"]

    early_stopping_handler = ignite.handlers.EarlyStopping(
        patience=PATIENCE, score_function=score_function, trainer=trainer
    )
    evaluator.add_event_handler(engine.Events.COMPLETED, early_stopping_handler)

    trainer.run(train_loader, max_epochs=args.max_epochs)

    return evaluator.state.metrics["wer"]


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
    wer = train(args, trial)
    print(f"{100 * wer:.2f}")


if __name__ == "__main__":
    main()
