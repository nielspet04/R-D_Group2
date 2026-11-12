"""Check completed labelled STA feature shards against their source CSV."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from audit_sta_data import context_status
from src.data_processing import timestamp_to_seconds
from src.sta_features import STAFeatureDataset, load_sta_feature_shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--shard-size", type=int, required=True)
    parser.add_argument("--limit", action="store_true", help="Check the first --events CSV rows.")
    args = parser.parse_args()

    annotation_path = args.annotations or (
        PROJECT_ROOT / "data" / "processed"
        / f"EPIC_100_{args.split}_participant_split.csv"
    )
    annotations = pd.read_csv(annotation_path)
    if args.limit:
        annotations = annotations.iloc[:args.events]

    if len(annotations) != args.events:
        raise SystemExit(f"Expected {args.events} CSV rows, got {len(annotations)}")

    narration_ids = annotations["narration_id"].astype(str).tolist()
    if len(set(narration_ids)) != len(narration_ids):
        raise SystemExit("The source CSV contains duplicate narration IDs")

    expected_statuses = [
        context_status(timestamp_to_seconds(str(timestamp)))
        for timestamp in annotations["start_timestamp"]
    ]
    dataset = STAFeatureDataset(args.feature_root, args.split)
    expected_sizes = [
        min(args.shard_size, args.events - start)
        for start in range(0, args.events, args.shard_size)
    ]
    actual_sizes = [row["event_count"] for row in dataset.manifest_rows]
    if len(dataset) != args.events or actual_sizes != expected_sizes:
        raise SystemExit(f"Unexpected shard sizes: {actual_sizes}; expected {expected_sizes}")

    statuses: Counter[str] = Counter()
    status_mismatches = 0
    for row in dataset.manifest_rows:
        path = dataset.split_directory / row["shard_file"]
        shard = load_sta_feature_shard(path, expected_split=args.split)
        start = row["start_index"]
        end = row["end_index_exclusive"]
        if shard["narration_ids"] != narration_ids[start:end]:
            raise SystemExit(f"Missing, extra, or out-of-order narration IDs in {path}")

        statuses.update(shard["status"])
        for offset, (actual, expected) in enumerate(
            zip(shard["status"], expected_statuses[start:end])
        ):
            if actual != expected:
                status_mismatches += 1
                print(
                    f"{path}: {shard['narration_ids'][offset]} "
                    f"video={shard['video_ids'][offset]} status={actual} expected={expected}",
                    flush=True,
                )

    print(f"{args.split}: {len(dataset):,} events in {len(actual_sizes)} shards")
    print(f"Shard sizes: {actual_sizes}")
    for status in ("full", "partial", "no_context", "video_error"):
        print(f"{status}: {statuses[status]:,}")
    if status_mismatches:
        raise SystemExit(f"{status_mismatches} events differ from their expected context status")

    for index in (0, len(dataset) - 1):
        item = dataset[index]
        if item["narration_id"] != narration_ids[index]:
            raise SystemExit(f"Dataset returned the wrong narration ID at index {index}")
        features = item["features"]
        if features.dtype != torch.float32 or tuple(features.shape) != (5, 512):
            raise SystemExit(f"Unexpected dataset feature dtype or shape at index {index}")

    print("CSV order, shard sizes, context statuses, and float32 dataset reads passed")


if __name__ == "__main__":
    main()
