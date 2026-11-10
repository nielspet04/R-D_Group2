"""Validate and summarize cached STA feature shards."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sta_features import (
    ALLOWED_SPLITS,
    STAFeatureDataset,
    load_sta_feature_shard,
)


DEFAULT_FEATURE_ROOT = (
    PROJECT_ROOT / "data/processed/sta_features"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_FEATURE_ROOT,
    )
    parser.add_argument(
        "--split",
        choices=sorted(ALLOWED_SPLITS),
        required=True,
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = STAFeatureDataset(
        args.feature_root,
        args.split,
    )

    rows_to_check = dataset.manifest_rows

    if not args.check_all:
        rows_to_check = rows_to_check[:1]

    statuses: Counter[str] = Counter()
    narration_ids: list[str] = []
    checked_events = 0

    for row in rows_to_check:
        shard_path = (
            dataset.split_directory / row["shard_file"]
        )
        shard = load_sta_feature_shard(
            shard_path,
            expected_split=args.split,
        )

        event_count = len(shard["narration_ids"])
        checked_events += event_count
        narration_ids.extend(shard["narration_ids"])
        statuses.update(shard["status"])

        if tuple(shard["features"].shape[1:]) != (5, 512):
            raise ValueError(
                f"Incorrect feature shape in {shard_path}"
            )
        if shard["features"].dtype != torch.float16:
            raise ValueError(
                f"Incorrect feature dtype in {shard_path}"
            )
        if not torch.isfinite(shard["features"]).all().item():
            raise ValueError(
                f"Non-finite features in {shard_path}"
            )
        if torch.any(
            shard["features"][~shard["valid_mask"]] != 0
        ).item():
            raise ValueError(
                f"Nonzero padded feature in {shard_path}"
            )

    if len(narration_ids) != len(set(narration_ids)):
        raise ValueError(
            "Duplicate narration IDs occur across shards"
        )

    print(f"Split:          {args.split}")
    print(f"Manifest rows:  {len(dataset.manifest_rows):,}")
    print(f"Dataset events: {len(dataset):,}")
    print(f"Checked events: {checked_events:,}")
    print(f"Full:           {statuses['full']:,}")
    print(f"Partial:        {statuses['partial']:,}")
    print(f"No context:     {statuses['no_context']:,}")
    print(f"Video errors:   {statuses['video_error']:,}")
    print("Validation passed")


if __name__ == "__main__":
    main()