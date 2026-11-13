"""Independently evaluate a trained STA checkpoint on cached validation features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_sta import (  # noqa: E402
    DEFAULT_ANNOTATIONS,
    DEFAULT_FEATURE_ROOT,
    load_action_pairs,
    resolve_device,
    training_labels,
)
from src.sta_features import STAFeatureDataset  # noqa: E402
from src.sta_model_0 import FrequencyPriorSTA  # noqa: E402
from src.sta_trainer import write_json_atomic  # noqa: E402
from src.sta_training import (  # noqa: E402
    TrainingConfig,
    build_model,
    load_checkpoint,
    seed_everything,
    validate_one_epoch,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/tables"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint produced by scripts/train_sta.py.",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_FEATURE_ROOT,
        help="Root containing train/ and validation/ feature shards.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="Complete annotation CSV used to define valid action pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Evaluation JSON path. Defaults to "
            "results/tables/<model>_evaluation.json."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as cpu, cuda or cuda:0 (default: auto).",
    )
    return parser.parse_args(argv)


def config_from_checkpoint(checkpoint: dict) -> TrainingConfig:
    """Rebuild and validate the training configuration stored in a checkpoint."""
    configuration = checkpoint["configuration"]
    if not isinstance(configuration, dict):
        raise ValueError("Checkpoint configuration must be a dictionary")

    try:
        return TrainingConfig(**configuration)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Checkpoint contains an invalid training configuration"
        ) from error


def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    """Load one checkpoint, recompute validation metrics, and save them."""
    device = resolve_device(args.device)

    # Read metadata on CPU before constructing the checkpoint's architecture.
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = config_from_checkpoint(checkpoint)
    seed_everything(config.seed)

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    train_dataset = STAFeatureDataset(args.feature_root, "train")
    validation_dataset = STAFeatureDataset(
        args.feature_root,
        "validation",
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

    metrics = validate_one_epoch(
        model,
        validation_loader,
        device,
        frequency_prior,
        action_pairs,
    )

    output_path = args.output or (
        DEFAULT_OUTPUT_ROOT / f"{config.model}_evaluation.json"
    )
    result = {
        "model": config.model,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_git_commit": checkpoint["git_commit"],
        "checkpoint_git_dirty": checkpoint["git_dirty"],
        "train_events_for_frequency_prior": len(train_dataset),
        "valid_action_classes": len(action_pairs),
        **metrics,
    }
    write_json_atomic(output_path, result)
    result["output_path"] = output_path
    return result


def print_results(result: dict) -> None:
    """Print a compact human-readable evaluation summary."""
    print(f"Model:             {result['model']}")
    print(f"Checkpoint epoch:  {result['checkpoint_epoch']}")
    print(f"Validation events: {result['validation_events']:,}")
    print(f"Usable events:     {result['validation_usable_events']:,}")
    print(f"Fallback events:   {result['validation_fallback_events']:,}")
    print(f"Valid actions:     {result['valid_action_classes']:,}")
    print()
    print("Metric                 Verb       Noun     Action")
    print(
        "Top-1 accuracy     "
        f"{result['verb_top1_accuracy']:10.4f}"
        f"{result['noun_top1_accuracy']:11.4f}"
        f"{result['action_top1_accuracy']:11.4f}"
    )
    print(
        "Top-5 accuracy     "
        f"{result['verb_top5_accuracy']:10.4f}"
        f"{result['noun_top5_accuracy']:11.4f}"
        f"{result['action_top5_accuracy']:11.4f}"
    )
    print(
        "Class-mean Top-5  "
        f"{result['verb_mt5r']:10.4f}"
        f"{result['noun_mt5r']:11.4f}"
        f"{result['action_mt5r']:11.4f}"
    )
    print(f"\nSaved metrics to {result['output_path']}")


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    result = evaluate_checkpoint(args)
    print_results(result)
    return result


if __name__ == "__main__":
    main()
