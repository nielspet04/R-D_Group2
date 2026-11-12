from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data_processing import index_videos, timestamp_to_seconds
from src.sta_feature_extraction import (build_r3d18_feature_extractor, extract_event_batch)
from src.sta_features import (
    SCHEMA_VERSION,
    load_sta_feature_shard,
    save_sta_feature_shard,
)

DEFAULT_RAW_DIR = PROJECT_ROOT / "data/raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/processed/sta_features"

DEFAULT_SPLIT_PATHS = {
    "train": (
        PROJECT_ROOT
        / "data/processed/EPIC_100_train_participant_split.csv"
    ),
    "validation": (
        PROJECT_ROOT
        / "data/processed/EPIC_100_validation_participant_split.csv"
    ),
}


MANIFEST_FIELDS = [
    "schema_version",
    "split",
    "shard_index",
    "shard_file",
    "start_index",
    "end_index_exclusive",
    "event_count",
]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Override the annotation CSV for the selected split.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--event-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dtype",
        choices=("float16",),
        default="float16",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def resolve_annotation_path(args: argparse.Namespace) -> Path:
    if args.annotations is not None:
        return args.annotations

    if args.split == "test":
        raise ValueError(
            "--annotations is required for the hidden-test split"
        )

    return DEFAULT_SPLIT_PATHS[args.split]


def validate_arguments(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    if args.event_batch_size <= 0:
        raise ValueError("--event-batch-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )


def require_annotation_columns(
    annotations: pd.DataFrame,
    split: str,
) -> None:
    required = {
        "narration_id",
        "video_id",
    }

    if split in {"train", "validation"}:
        required.update(
            {
                "start_timestamp",
                "verb_class",
                "noun_class",
            }
        )
    else:
        required.add("sta_observation_end_seconds")

    missing = required.difference(annotations.columns)

    if missing:
        raise ValueError(
            f"Missing annotation columns: {sorted(missing)}"
        )


def event_times(
    annotations: pd.DataFrame,
    split: str,
) -> tuple[list[float], list[float]]:
    action_starts: list[float] = []
    observation_ends: list[float] = []

    for _, row in annotations.iterrows():
        if split == "test":
            observation_end = float(
                row["sta_observation_end_seconds"]
            )
            action_start = observation_end + 1.0
        else:
            action_start = timestamp_to_seconds(
                str(row["start_timestamp"])
            )
            observation_end = action_start - 1.0

        action_starts.append(action_start)
        observation_ends.append(observation_end)

    return action_starts, observation_ends


def write_manifest(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".csv.tmp")

    with temporary_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(path)


def existing_shard_is_valid(
    shard_path: Path,
    split: str,
    expected_ids: list[str],
) -> bool:
    try:
        shard = load_sta_feature_shard(
            shard_path,
            expected_split=split,
        )
    except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        print(f"Rebuilding unreadable shard {shard_path}: {exc}", flush=True)
        return False

    if shard["narration_ids"] != expected_ids:
        print(f"Rebuilding shard with different narration IDs: {shard_path}", flush=True)
        return False

    if "video_error" in shard["status"]:
        print(f"Retrying shard with video errors: {shard_path}", flush=True)
        return False

    return True


def main() -> None:
    args = parse_args()
    validate_arguments(args)

    annotation_path = resolve_annotation_path(args)
    annotations = pd.read_csv(annotation_path)
    require_annotation_columns(annotations, args.split)

    if annotations["narration_id"].duplicated().any():
        raise ValueError("Annotation CSV contains duplicate narration IDs")

    if args.limit is not None:
        annotations = annotations.iloc[: args.limit].copy()

    annotations = annotations.reset_index(drop=True)

    action_starts, observation_ends = event_times(
        annotations,
        args.split,
    )

    videos_by_id = index_videos(args.raw_dir)

    output_directory = args.output_root / args.split
    output_directory.mkdir(parents=True, exist_ok=True)

    encoder, preprocessing = build_r3d18_feature_extractor(
        args.device
    )

    manifest_rows: list[dict] = []
    total_events = len(annotations)

    for shard_index, shard_start in enumerate(
        range(0, total_events, args.shard_size)
    ):
        shard_stop = min(
            shard_start + args.shard_size,
            total_events,
        )
        shard_annotations = annotations.iloc[
            shard_start:shard_stop
        ]
        shard_path = (
            output_directory / f"shard_{shard_index:05d}.pt"
        )

        expected_ids = (
            shard_annotations["narration_id"]
            .astype(str)
            .tolist()
        )

        if (
            args.resume
            and shard_path.exists()
            and existing_shard_is_valid(
                shard_path,
                args.split,
                expected_ids,
            )
        ):
            print(f"Skipping validated shard: {shard_path.name}")
        else:
            feature_batches = []
            mask_batches = []
            statuses: list[str] = []

            batch_starts = range(
                shard_start,
                shard_stop,
                args.event_batch_size,
            )

            for batch_start in tqdm(
                batch_starts,
                desc=f"Shard {shard_index:05d}",
            ):
                batch_stop = min(
                    batch_start + args.event_batch_size,
                    shard_stop,
                )
                batch_annotations = annotations.iloc[
                    batch_start:batch_stop
                ]

                batch_video_paths: list[Path] = []

                for video_id in batch_annotations[
                    "video_id"
                ].astype(str):
                    normalized_id = video_id.upper()
                    video_path = videos_by_id.get(normalized_id)

                    if video_path is None:
                        # A nonexistent path intentionally produces a
                        # video_error without dropping the event.
                        video_path = (
                            args.raw_dir
                            / f"__missing__{normalized_id}.mp4"
                        )

                    batch_video_paths.append(video_path)

                batch_features, batch_masks, batch_statuses = (
                    extract_event_batch(
                        video_paths=batch_video_paths,
                        action_starts=action_starts[
                            batch_start:batch_stop
                        ],
                        encoder=encoder,
                        preprocessing=preprocessing,
                        device=args.device,
                        workers=args.workers,
                    )
                )

                feature_batches.append(batch_features)
                mask_batches.append(batch_masks)
                statuses.extend(batch_statuses)

            features = torch.cat(feature_batches).to(
                dtype=torch.float16,
                device="cpu",
            )
            valid_mask = torch.cat(mask_batches).to(
                dtype=torch.bool,
                device="cpu",
            )

            stored_action_starts = torch.tensor(
                action_starts[shard_start:shard_stop],
                dtype=torch.float32,
            )
            stored_observation_ends = torch.tensor(
                observation_ends[shard_start:shard_stop],
                dtype=torch.float32,
            )
            if args.split in {"train", "validation"}:
                # Keep independently rounded timestamps within the shard contract.
                stored_observation_ends = torch.minimum(
                    stored_observation_ends, stored_action_starts - 1.0
                )

            shard = {
                "schema_version": SCHEMA_VERSION,
                "split": args.split,
                "features": features,
                "valid_mask": valid_mask,
                "narration_ids": expected_ids,
                "video_ids": (
                    shard_annotations["video_id"]
                    .astype(str)
                    .tolist()
                ),
                "action_start_seconds": stored_action_starts,
                "observation_end_seconds": stored_observation_ends,
                "status": statuses,
            }

            if args.split in {"train", "validation"}:
                shard["verb_labels"] = torch.tensor(
                    shard_annotations["verb_class"]
                    .astype(int)
                    .tolist(),
                    dtype=torch.int64,
                )
                shard["noun_labels"] = torch.tensor(
                    shard_annotations["noun_class"]
                    .astype(int)
                    .tolist(),
                    dtype=torch.int64,
                )

            save_sta_feature_shard(shard, shard_path)
            print(f"Saved {shard_path}")

        manifest_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "split": args.split,
                "shard_index": shard_index,
                "shard_file": shard_path.name,
                "start_index": shard_start,
                "end_index_exclusive": shard_stop,
                "event_count": shard_stop - shard_start,
            }
        )

    write_manifest(
        output_directory / "manifest.csv",
        manifest_rows,
    )

    print(
        f"Completed {total_events:,} events in "
        f"{len(manifest_rows):,} shards"
    )


if __name__ == "__main__":
    main()
