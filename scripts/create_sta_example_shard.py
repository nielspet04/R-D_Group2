"""Create the deterministic eight-event shard used for STA integration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sta_features import save_sta_feature_shard


DEFAULT_OUTPUT = PROJECT_ROOT / "results/local_sta_contract/train/shard_00000.pt"


def build_example_shard() -> dict:
    """Return a small labelled shard covering full, partial and no context."""
    import torch

    generator = torch.Generator().manual_seed(42)
    event_count = 8
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [False, False, True, True, True],
            [False, True, True, True, True],
            [True, True, True, True, True],
            [False, False, False, True, True],
            [True, True, True, True, True],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    features = torch.randn(
        event_count, 5, 512, generator=generator, dtype=torch.float16
    )
    features[~valid_mask] = 0

    action_starts = torch.tensor(
        [20.0, 4.0, 5.0, 30.0, 3.0, 15.0, 0.8, 12.0], dtype=torch.float32
    )
    # A negative cutoff is valid metadata for an action before one second: it
    # explains why that event has no legal observable context.
    observation_ends = action_starts - 1.0

    return {
        "schema_version": 1,
        "split": "train",
        "features": features,
        "valid_mask": valid_mask,
        "verb_labels": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64),
        "noun_labels": torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.int64),
        "narration_ids": [f"contract_{index:04d}" for index in range(event_count)],
        "video_ids": ["P00_01"] * event_count,
        "action_start_seconds": action_starts,
        "observation_end_seconds": observation_ends,
        "status": [
            "full",
            "partial",
            "partial",
            "full",
            "partial",
            "full",
            "no_context",
            "video_error",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard = build_example_shard()
    path = save_sta_feature_shard(shard, args.output)
    print(f"Saved {len(shard['narration_ids'])} validated events to {path}")


if __name__ == "__main__":
    main()
