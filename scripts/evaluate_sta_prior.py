"""Fit and evaluate the Model 0 frequency-prior STA baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (  # noqa: E402
    action_labels_from_pairs,
    build_action_pairs,
    class_mean_top_k_recall,
    joint_action_scores,
    top_k_accuracy,
)
from src.sta_models import FrequencyPriorSTA  # noqa: E402


NUM_VERBS = 97
NUM_NOUNS = 300
DEFAULT_TRAIN = (
    PROJECT_ROOT / "data/processed/EPIC_100_train_participant_split.csv"
)
DEFAULT_VALIDATION = (
    PROJECT_ROOT / "data/processed/EPIC_100_validation_participant_split.csv"
)
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data/raw/EPIC_100_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/tables/sta_model_0_frequency.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the STA training-frequency prior on validation data."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="Full provided training annotations, used only to define valid actions.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require_columns(data: pd.DataFrame, path: Path) -> None:
    required = {"narration_id", "verb_class", "noun_class"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    validation = pd.read_csv(args.validation)
    annotations = pd.read_csv(args.annotations)
    for data, path in (
        (train, args.train),
        (validation, args.validation),
        (annotations, args.annotations),
    ):
        require_columns(data, path)

    if set(train["narration_id"]).intersection(validation["narration_id"]):
        raise ValueError("Train and validation narration IDs overlap")

    model = FrequencyPriorSTA.fit(
        train["verb_class"].to_numpy(),
        train["noun_class"].to_numpy(),
        num_verbs=NUM_VERBS,
        num_nouns=NUM_NOUNS,
    )
    verb_scores, noun_scores = model.predict(len(validation))

    verb_labels = validation["verb_class"].to_numpy()
    noun_labels = validation["noun_class"].to_numpy()
    action_pairs = build_action_pairs(
        annotations["verb_class"].to_numpy(),
        annotations["noun_class"].to_numpy(),
    )
    action_labels = action_labels_from_pairs(
        verb_labels, noun_labels, action_pairs
    )
    action_scores = joint_action_scores(verb_scores, noun_scores, action_pairs)

    metrics = {
        "model": "training_frequency_prior",
        "train_events": len(train),
        "validation_events": len(validation),
        "valid_action_classes": len(action_pairs),
        "verb_top1_accuracy": top_k_accuracy(verb_labels, verb_scores, k=1),
        "noun_top1_accuracy": top_k_accuracy(noun_labels, noun_scores, k=1),
        "action_top1_accuracy": top_k_accuracy(action_labels, action_scores, k=1),
        "verb_mt5r": class_mean_top_k_recall(verb_labels, verb_scores, k=5),
        "noun_mt5r": class_mean_top_k_recall(noun_labels, noun_scores, k=5),
        "action_mt5r": class_mean_top_k_recall(
            action_labels, action_scores, k=5
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")

    print("Model 0: training-frequency prior")
    print(f"Train events:      {metrics['train_events']:,}")
    print(f"Validation events: {metrics['validation_events']:,}")
    print(f"Valid actions:     {metrics['valid_action_classes']:,}")
    print()
    print("Metric                 Verb       Noun     Action")
    print(
        "Top-1 accuracy     "
        f"{metrics['verb_top1_accuracy']:10.4f}"
        f"{metrics['noun_top1_accuracy']:11.4f}"
        f"{metrics['action_top1_accuracy']:11.4f}"
    )
    print(
        "Class-mean Top-5  "
        f"{metrics['verb_mt5r']:10.4f}"
        f"{metrics['noun_mt5r']:11.4f}"
        f"{metrics['action_mt5r']:11.4f}"
    )
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()

