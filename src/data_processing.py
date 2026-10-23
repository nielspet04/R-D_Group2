"""Reusable data loading and preprocessing utilities for EPIC-KITCHENS-100.

Keep transformations deterministic and leave files in ``data/raw`` unchanged.
Run the participant split from the repository root with::

    python -m src.data_processing
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REQUIRED_ANNOTATION_COLUMNS = {
    "narration_id",
    "participant_id",
    "video_id",
    "start_timestamp",
    "stop_timestamp",
    "narration",
    "verb",
    "verb_class",
    "noun",
    "noun_class",
}


def find_annotation_file(
    raw_dir: Path = DEFAULT_RAW_DIR,
    filename: str = "EPIC_100_train.csv",
) -> Path:
    """Return the unique annotation file found recursively below ``raw_dir``."""
    candidates = sorted(raw_dir.rglob(filename))
    if not candidates:
        raise FileNotFoundError(f"{filename} was not found below {raw_dir}")
    if len(candidates) > 1:
        locations = "\n".join(f"- {path}" for path in candidates)
        raise RuntimeError(
            f"Found multiple copies of {filename}; pass one explicitly:\n{locations}"
        )
    return candidates[0]


def load_annotations(annotation_path: Path) -> pd.DataFrame:
    """Load an EPIC-KITCHENS annotation CSV and validate its schema."""
    annotations = pd.read_csv(annotation_path)
    missing = REQUIRED_ANNOTATION_COLUMNS.difference(annotations.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")
    if annotations.empty:
        raise ValueError(f"Annotation file is empty: {annotation_path}")
    return annotations


def index_videos(raw_dir: Path = DEFAULT_RAW_DIR) -> dict[str, Path]:
    """Index MP4 files below ``raw_dir`` by upper-case video ID."""
    video_files = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not video_files:
        # ``Path.rglob`` may not traverse a symlinked dataset on a cluster.
        video_files = sorted(
            Path(root) / filename
            for root, _, filenames in os.walk(raw_dir, followlinks=True)
            for filename in filenames
            if filename.lower().endswith(".mp4")
        )
    videos_by_id: dict[str, Path] = {}
    for path in video_files:
        video_id = path.stem.upper()
        if video_id in videos_by_id:
            raise RuntimeError(
                f"Duplicate video ID {video_id}: "
                f"{videos_by_id[video_id]} and {path}"
            )
        videos_by_id[video_id] = path
    return videos_by_id


def match_available_annotations(
    annotations: pd.DataFrame, videos_by_id: dict[str, Path]
) -> pd.DataFrame:
    """Keep annotation rows whose source video is locally available."""
    normalized_video_ids = annotations["video_id"].astype("string").str.upper()
    return annotations.loc[normalized_video_ids.isin(videos_by_id)].copy()


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert an ``HH:MM:SS.s`` timestamp to seconds."""
    return float(pd.to_timedelta(timestamp).total_seconds())


def video_is_readable(path: Path) -> bool:
    """Return whether OpenCV can open a video file."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    readable = capture.isOpened()
    capture.release()
    return readable


def read_frame(path: Path, time_seconds: float) -> np.ndarray:
    """Read one RGB frame at approximately ``time_seconds``."""
    import cv2

    if time_seconds < 0:
        raise ValueError("time_seconds must be non-negative")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"Could not read frame at {time_seconds:.2f}s: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_clip(
    source: Path,
    start_seconds: float,
    stop_seconds: float,
    destination: Path,
) -> Path:
    """Extract a video interval with OpenCV and return the generated MP4 path."""
    import cv2

    if start_seconds < 0 or stop_seconds <= start_seconds:
        raise ValueError(
            f"Invalid clip interval: start={start_seconds}, stop={stop_seconds}"
        )

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not fps or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video metadata for {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not initialize MP4 writer for {destination}")

    capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)
    frames_written = 0
    while capture.get(cv2.CAP_PROP_POS_MSEC) < stop_seconds * 1000:
        success, frame = capture.read()
        if not success:
            break
        writer.write(frame)
        frames_written += 1

    capture.release()
    writer.release()
    if frames_written == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError("No frames were written for the requested interval")
    return destination


def participant_split(
    annotations: pd.DataFrame,
    validation_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic train/validation splits with disjoint participants."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    participants = np.array(sorted(annotations["participant_id"].dropna().unique()))
    if len(participants) < 2:
        raise ValueError("At least two participants are required for a split")

    shuffled = np.random.default_rng(seed).permutation(participants)
    validation_count = round(len(participants) * validation_fraction)
    validation_count = min(max(1, validation_count), len(participants) - 1)
    validation_participants = set(shuffled[:validation_count])

    is_validation = annotations["participant_id"].isin(validation_participants)
    training = annotations.loc[~is_validation].copy()
    validation = annotations.loc[is_validation].copy()

    assert set(training["participant_id"]).isdisjoint(validation["participant_id"])
    assert len(training) + len(validation) == len(annotations)
    return training, validation


def save_participant_split(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
) -> tuple[Path, Path]:
    """Save split CSV files and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "EPIC_100_train_participant_split.csv"
    validation_path = output_dir / "EPIC_100_validation_participant_split.csv"
    training.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    return train_path, validation_path


def build_participant_split(
    annotation_path: Path,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    validation_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, split, and save annotations as one reproducible pipeline."""
    annotations = load_annotations(annotation_path)
    training, validation = participant_split(
        annotations,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    train_path, validation_path = save_participant_split(
        training, validation, output_dir
    )

    print(f"Loaded {len(annotations):,} rows from {annotation_path}")
    print(f"Saved {len(training):,} training rows to {train_path}")
    print(f"Saved {len(validation):,} validation rows to {validation_path}")
    print(
        "Validation participants:",
        sorted(validation["participant_id"].unique()),
    )
    return training, validation


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the participant-split pipeline."""
    parser = argparse.ArgumentParser(
        description="Create a participant-level EPIC-KITCHENS train/validation split."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Annotation CSV (default: find EPIC_100_train.csv below data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory for generated split CSV files.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the preprocessing pipeline from the command line."""
    args = parse_args()
    annotation_path = args.annotations or find_annotation_file()
    build_participant_split(
        annotation_path=annotation_path,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

