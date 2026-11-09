"""Torch-free helpers: annotation loading, timestamp parsing, clip sampling."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

_EXTS = (".MP4", ".mp4", ".MOV", ".mov")


def parse_ts(s) -> float:
    """'00:01:02.50' -> seconds (float)."""
    h, m, sec = str(s).strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def load_annotations(csv_path: str) -> pd.DataFrame:
    """Read an EPIC-KITCHENS-100 style csv and add start_s / stop_s (seconds).

    Needed columns: narration_id, start_timestamp, stop_timestamp.
    Optional: video_id (derived from narration_id if absent), verb_class, noun_class
    (set to -1 when absent, e.g. for an unlabelled test csv).
    """
    df = pd.read_csv(csv_path)
    need = ["narration_id", "start_timestamp", "stop_timestamp"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}. Found: {list(df.columns)}")
    if "video_id" not in df.columns:
        df["video_id"] = df["narration_id"].str.rsplit("_", n=1).str[0]
    df["start_s"] = df["start_timestamp"].map(parse_ts)
    df["stop_s"] = df["stop_timestamp"].map(parse_ts)
    for c in ("verb_class", "noun_class"):
        if c not in df.columns:
            df[c] = -1
        df[c] = df[c].fillna(-1).astype(int)
    return df.reset_index(drop=True)


def find_video(root: str, video_id: str) -> str:
    """Locate <root>/<P01>/<P01_01>.MP4 (also tries .mp4 and a flat layout)."""
    pid = video_id.split("_")[0]
    for base in (os.path.join(root, pid), root):
        for ext in _EXTS:
            p = os.path.join(base, video_id + ext)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(f"video {video_id} not found under {root}")


def sample_times(start: float, stop: float, n: int, train: bool, rng: np.random.Generator) -> np.ndarray:
    """TSN-style sampling: split [start, stop] in n bins, one timestamp per bin.

    train: random position inside each bin; eval: bin centre. Result is ascending.
    """
    dur = max(stop - start, 0.1)
    step = dur / n
    edges = start + step * np.arange(n)
    offs = rng.random(n) if train else np.full(n, 0.5)
    return edges + offs * step
