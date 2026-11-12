"""Train the shared cached-feature STA neural baselines."""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import build_action_pairs  # noqa: E402
from src.sta_features import STAFeatureDataset  # noqa: E402
from src.sta_model_0 import FrequencyPriorSTA  # noqa: E402
from src.sta_trainer import train_model  # noqa: E402
from src.sta_training import (  # noqa: E402
    TrainingConfig,
    build_model,
    build_optimizer,
    seed_everything,
)


DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data/processed/sta_features"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data/raw/EPIC_100_train.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results"


class ShardAwareBatchSampler:
    """Shuffle shards and their events while keeping every batch shard-local."""

    def __init__(
        self,
        dataset: STAFeatureDataset,
        batch_size: int,
        *,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self._epoch)
        self._epoch += 1

        rows = list(self.dataset.manifest_rows)
        generator.shuffle(rows)
        for row in rows:
            indices = list(
                range(
                    row["start_index"],
                    row["end_index_exclusive"],
                )
            )
            generator.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum(
            math.ceil(row["event_count"] / self.batch_size)
            for row in self.dataset.manifest_rows
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an STA neural baseline from validated cached feature shards."
        )
    )
    parser.add_argument(
        "--model",
        choices=("mean_pool", "gru"),
        required=True,
        help="Neural STA architecture to train.",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_FEATURE_ROOT,
        help="Root containing train/ and validation/ shard directories.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help=(
            "Complete provided annotation CSV used to define valid verb-noun "
            "action pairs."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for checkpoints, logs, validation tables and configs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as cpu, cuda or cuda:0 (default: auto).",
    )
    parser.add_argument(
        "--smoke-test-epochs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Correctness-test override for the 30-epoch baseline maximum; "
            "do not use for reported training runs."
        ),
    )
    args = parser.parse_args(argv)
    if args.smoke_test_epochs is not None and args.smoke_test_epochs <= 0:
        parser.error("--smoke-test-epochs must be positive")
    return args


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device or choose CUDA when it is available."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} was requested but CUDA is unavailable"
        )
    return device


def training_labels(
    dataset: STAFeatureDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Read Model 0 fitting labels from the training dataset only."""
    verb_labels = np.empty(len(dataset), dtype=np.int64)
    noun_labels = np.empty(len(dataset), dtype=np.int64)
    for index in range(len(dataset)):
        item = dataset[index]
        verb_labels[index] = item["verb_label"]
        noun_labels[index] = item["noun_label"]
    return verb_labels, noun_labels


def load_action_pairs(annotation_path: Path) -> np.ndarray:
    """Build valid actions from every row in the provided annotation CSV."""
    annotations = pd.read_csv(annotation_path)
    required_columns = {"narration_id", "verb_class", "noun_class"}
    missing_columns = required_columns.difference(annotations.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns in {annotation_path}: {sorted(missing_columns)}"
        )
    return build_action_pairs(
        annotations["verb_class"].to_numpy(),
        annotations["noun_class"].to_numpy(),
    )


def run_training(args: argparse.Namespace) -> dict:
    """Build shared training components and run the experiment."""
    config = TrainingConfig(model=args.model)
    if args.smoke_test_epochs is not None:
        config = replace(config, max_epochs=args.smoke_test_epochs)

    seed_everything(config.seed)
    device = resolve_device(args.device)

    train_dataset = STAFeatureDataset(args.feature_root, "train")
    validation_dataset = STAFeatureDataset(
        args.feature_root,
        "validation",
    )

    train_sampler = ShardAwareBatchSampler(
        train_dataset,
        config.train_batch_size,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        num_workers=0,
    )

    verb_labels, noun_labels = training_labels(train_dataset)
    frequency_prior = FrequencyPriorSTA.fit(
        verb_labels,
        noun_labels,
        num_verbs=config.num_verbs,
        num_nouns=config.num_nouns,
    )
    action_pairs = load_action_pairs(args.annotations)

    model = build_model(config)
    optimizer = build_optimizer(model, config)
    return train_model(
        model,
        optimizer,
        train_loader,
        validation_loader,
        frequency_prior,
        action_pairs,
        device,
        config,
        output_root=args.output_root,
    )


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    result = run_training(args)
    fallback_events = result["best_validation_metrics"][
        "validation_fallback_events"
    ]

    print(f"Best epoch:                 {result['best_epoch']}")
    print(
        "Validation Action MT5R:    "
        f"{result['best_action_mt5r']:.4f}"
    )
    print(f"Validation fallback events: {fallback_events:,}")
    print("Output paths:")
    print(f"  Checkpoint:  {result['checkpoint_path']}")
    print(f"  History:     {result['history_path']}")
    print(f"  Validation:  {result['validation_path']}")
    print(f"  Config:      {result['configuration_path']}")
    return result


if __name__ == "__main__":
    main()
