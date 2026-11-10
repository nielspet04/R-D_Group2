"""Audit STA annotations, splits, context and video coverage."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import (
    index_videos,
    timestamp_to_seconds,
)
from src.sta_feature_extraction import sta_segment_boundaries


DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT / "data/raw/EPIC_100_train.csv"
)
DEFAULT_TRAIN = (
    PROJECT_ROOT
    / "data/processed/EPIC_100_train_participant_split.csv"
)
DEFAULT_VALIDATION = (
    PROJECT_ROOT
    / "data/processed/EPIC_100_validation_participant_split.csv"
)
DEFAULT_RAW_DIR = PROJECT_ROOT / "data/raw"


def validate_split_integrity(
    all_annotations: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    for name, data in (
        ("all", all_annotations),
        ("train", train),
        ("validation", validation),
    ):
        required = {
            "narration_id",
            "participant_id",
            "video_id",
            "start_timestamp",
        }
        missing = required.difference(data.columns)

        if missing:
            raise ValueError(
                f"{name} annotations missing columns: "
                f"{sorted(missing)}"
            )

        if data["narration_id"].duplicated().any():
            raise ValueError(
                f"{name} annotations contain duplicate narration IDs"
            )

    train_ids = set(train["narration_id"].astype(str))
    validation_ids = set(
        validation["narration_id"].astype(str)
    )
    all_ids = set(
        all_annotations["narration_id"].astype(str)
    )

    if train_ids.intersection(validation_ids):
        raise ValueError(
            "Train and validation narration IDs overlap"
        )

    if train_ids.union(validation_ids) != all_ids:
        raise ValueError(
            "Train and validation do not exactly partition annotations"
        )

    train_participants = set(
        train["participant_id"].astype(str)
    )
    validation_participants = set(
        validation["participant_id"].astype(str)
    )

    if train_participants.intersection(
        validation_participants
    ):
        raise ValueError(
            "Train and validation participants overlap"
        )


def context_status(action_start: float) -> str:
    _, valid_mask = sta_segment_boundaries(action_start)
    valid_count = sum(valid_mask)

    if valid_count == 5:
        return "full"
    if valid_count > 0:
        return "partial"
    return "no_context"


def count_context_statuses(
    annotations: pd.DataFrame,
) -> dict[str, int]:
    counts: Counter[str] = Counter()

    for timestamp in annotations["start_timestamp"]:
        action_start = timestamp_to_seconds(str(timestamp))
        counts[context_status(action_start)] += 1

    return {
        "full": counts["full"],
        "partial": counts["partial"],
        "no_context": counts["no_context"],
    }


def validate_video_coverage(
    annotations: pd.DataFrame,
    videos_by_id: dict[str, Path],
) -> None:
    required_video_ids = set(
        annotations["video_id"]
        .astype(str)
        .str.upper()
    )
    available_video_ids = set(videos_by_id)

    missing = sorted(
        required_video_ids.difference(available_video_ids)
    )

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"{len(missing)} videos are missing. "
            f"First missing IDs: {preview}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN,
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_VALIDATION,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_annotations = pd.read_csv(args.annotations)
    train = pd.read_csv(args.train)
    validation = pd.read_csv(args.validation)

    if len(all_annotations) != 67_217:
        raise ValueError(
            f"Expected 67,217 annotations, got "
            f"{len(all_annotations):,}"
        )
    if len(train) != 59_767:
        raise ValueError(
            f"Expected 59,767 train rows, got {len(train):,}"
        )
    if len(validation) != 7_450:
        raise ValueError(
            f"Expected 7,450 validation rows, got "
            f"{len(validation):,}"
        )

    validate_split_integrity(
        all_annotations,
        train,
        validation,
    )

    unique_video_count = (
        all_annotations["video_id"]
        .astype(str)
        .str.upper()
        .nunique()
    )

    if unique_video_count != 495:
        raise ValueError(
            f"Expected 495 unique video IDs, got "
            f"{unique_video_count}"
        )

    train_counts = count_context_statuses(train)
    validation_counts = count_context_statuses(validation)

    expected_train = {
        "full": 58_895,
        "partial": 796,
        "no_context": 76,
    }
    expected_validation = {
        "full": 7_307,
        "partial": 119,
        "no_context": 24,
    }

    if train_counts != expected_train:
        raise ValueError(
            f"Unexpected train context counts: {train_counts}"
        )
    if validation_counts != expected_validation:
        raise ValueError(
            "Unexpected validation context counts: "
            f"{validation_counts}"
        )

    videos_by_id = index_videos(args.raw_dir)
    validate_video_coverage(
        all_annotations,
        videos_by_id,
    )

    print("STA data audit passed")
    print(f"Annotations:       {len(all_annotations):,}")
    print(f"Train:             {len(train):,}")
    print(f"Validation:        {len(validation):,}")
    print(f"Unique videos:     {unique_video_count:,}")
    print(f"Train contexts:    {train_counts}")
    print(f"Validation contexts: {validation_counts}")


if __name__ == "__main__":
    main()
     