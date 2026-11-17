"""Full-coverage event dataset with original source-table row identities."""

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_processing import (
    index_videos,
    timestamp_to_seconds,
)

from .sampling import (
    sample_frame_indices,
    UnavailableContextError,
)
from .transforms import ClipTransform
from .video import (
    decode_requested_frames,
    VideoReadError,
)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_cutoff(row, split):
    if split == "test":
        cutoff = float(
            row["sta_observation_end_seconds"]
        )
    else:
        cutoff = (
            timestamp_to_seconds(
                str(row["start_timestamp"])
            )
            - 1.0
        )

    if not math.isfinite(cutoff):
        raise ValueError("Nonfinite cutoff")

    has_start = (
        "start_timestamp" in row
        and pd.notna(row["start_timestamp"])
    )

    has_cutoff = (
        "sta_observation_end_seconds" in row
        and pd.notna(row["sta_observation_end_seconds"])
    )

    if has_start and has_cutoff:
        expected = (
            timestamp_to_seconds(
                str(row["start_timestamp"])
            )
            - 1.0
        )

        if not math.isclose(
            expected,
            float(row["sta_observation_end_seconds"]),
            abs_tol=1e-9,
            rel_tol=0,
        ):
            raise ValueError(
                f"Inconsistent cutoff: {row['narration_id']}"
            )

    return cutoff


def load_events(
    annotations,
    *,
    split,
    train_split=None,
    validation_split=None,
):
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")

    table = pd.read_csv(annotations)

    required = {"narration_id", "video_id"}

    if split == "test":
        required.add("sta_observation_end_seconds")
    else:
        required.update({
            "participant_id",
            "start_timestamp",
            "verb_class",
            "noun_class",
        })

    missing = required - set(table.columns)

    if missing:
        raise ValueError(
            f"Missing annotation columns: {sorted(missing)}"
        )

    if table[list(required)].isna().any().any():
        raise ValueError("Null required annotation fields")

    if table["narration_id"].duplicated().any():
        raise ValueError("Duplicate narration IDs")

    table = table.copy()

    table["narration_id"] = (
        table["narration_id"].astype(str)
    )
    table["video_id"] = (
        table["video_id"].astype(str).str.upper()
    )

    # Assign before selecting the participant split.
    table["row_index"] = np.arange(
        len(table),
        dtype=np.int64,
    )

    if split != "test":
        if train_split is None or validation_split is None:
            raise ValueError(
                "Both existing participant split files are required"
            )

        from scripts.audit_sta_data import (
            validate_split_integrity,
        )

        train = pd.read_csv(train_split)
        validation = pd.read_csv(validation_split)

        validate_split_integrity(
            table,
            train,
            validation,
        )

        # Verify split payloads have not drifted from the original table.
        original_by_id = table.set_index("narration_id")

        for subset in (train, validation):
            ids = subset["narration_id"].astype(str)

            for column in (
                "participant_id",
                "video_id",
                "start_timestamp",
                "verb_class",
                "noun_class",
            ):
                if column not in subset:
                    continue

                original_values = (
                    original_by_id.loc[ids, column]
                    .astype(str)
                    .to_numpy()
                )

                split_values = (
                    subset[column]
                    .astype(str)
                    .to_numpy()
                )

                if not np.array_equal(
                    original_values,
                    split_values,
                ):
                    raise ValueError(
                        f"Split metadata differs: {column}"
                    )

        chosen = (
            train
            if split == "train"
            else validation
        )

        chosen_ids = set(
            chosen["narration_id"].astype(str)
        )

        table = table.loc[
            table["narration_id"].isin(chosen_ids)
        ].copy()

        for column, count in (
            ("verb_class", 97),
            ("noun_class", 300),
        ):
            values = pd.to_numeric(
                table[column],
                errors="raise",
            ).to_numpy()

            if (
                not np.isfinite(values).all()
                or np.any(values != values.astype(np.int64))
                or np.any(values < 0)
                or np.any(values >= count)
            ):
                raise ValueError(f"Invalid labels: {column}")

    for row in table.to_dict("records"):
        resolve_cutoff(row, split)

    return table.reset_index(drop=True)


class STAClipDataset(Dataset):
    def __init__(
        self,
        events,
        raw_dir,
        metadata_path,
        *,
        split,
        training=False,
        seed=42,
    ):
        self.events = events.to_dict("records")
        self.split = split
        self.epoch = 0

        self.transform = ClipTransform(
            training=training,
            seed=seed,
        )

        self.videos = index_videos(Path(raw_dir))
        self.metadata_path = Path(metadata_path)

        manifest = json.loads(
            self.metadata_path.read_text()
        )

        if manifest["schema_version"] != 1:
            raise ValueError("Unsupported metadata version")

        self.metadata = manifest["videos"]

        # Worker-local, bounded timing-metadata cache.
        self.timelines = {}

        # Useful for a worker-free CPU profiling pass.
        self.last_decode_seconds = 0.0
        self.last_transform_seconds = 0.0

        for video_id in {
            row["video_id"] for row in self.events
        }:
            item = self.metadata.get(video_id)
            path = self.videos.get(video_id)

            if item is None:
                raise ValueError(
                    f"Video absent from timing audit: {video_id}"
                )

            if item["status"] == "ok" and path is not None:
                stat = path.stat()

                if (
                    stat.st_size != item["size_bytes"]
                    or stat.st_mtime_ns != item["mtime_ns"]
                ):
                    raise ValueError(
                        f"Media changed since audit: {video_id}. "
                        "Rebuild its timestamp index."
                    )

    def __len__(self):
        return len(self.events)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _timeline(self, video_id):
        if video_id not in self.timelines:
            item = self.metadata[video_id]

            path = (
                self.metadata_path.parent
                / item["timeline"]
            )

            if (
                sha256_file(path)
                != item["timeline_sha256"]
            ):
                raise ValueError(
                    f"Timeline checksum mismatch: {video_id}"
                )

            with np.load(
                path,
                allow_pickle=False,
            ) as saved:
                pts = saved["pts"].copy()

            time_base = Fraction(item["time_base"])

            timestamps = (
                pts.astype(np.float64)
                * float(time_base)
            )

            self.timelines[video_id] = (
                pts,
                time_base,
                timestamps,
            )

            if len(self.timelines) > 8:
                oldest = next(iter(self.timelines))
                del self.timelines[oldest]

        return self.timelines[video_id]

    def __getitem__(self, index):
        self.last_decode_seconds = 0.0
        self.last_transform_seconds = 0.0

        row = self.events[index]
        cutoff = resolve_cutoff(row, self.split)

        sample = {
            "video": torch.zeros(
                3, 32, 384, 384,
                dtype=torch.float32,
            ),
            "has_context": torch.tensor(
                False,
                dtype=torch.bool,
            ),
            "row_index": torch.tensor(
                row["row_index"],
                dtype=torch.long,
            ),
            "narration_id": str(row["narration_id"]),
            "status": "no_context",
            "cutoff_seconds": torch.tensor(
                cutoff,
                dtype=torch.float32,
            ),
            "frame_indices": torch.full(
                (32,),
                -1,
                dtype=torch.long,
            ),
        }

        if self.split != "test":
            sample["verb_label"] = torch.tensor(
                int(row["verb_class"]),
                dtype=torch.long,
            )
            sample["noun_label"] = torch.tensor(
                int(row["noun_class"]),
                dtype=torch.long,
            )

        # No decoder or timeline read for these events.
        if cutoff <= 0:
            return sample

        video_id = row["video_id"]
        item = self.metadata[video_id]
        path = self.videos.get(video_id)

        if item["status"] != "ok" or path is None:
            sample["status"] = "video_error"
            return sample

        pts, time_base, timestamps = self._timeline(
            video_id
        )

        try:
            selection = sample_frame_indices(
                cutoff,
                timestamps,
            )
        except UnavailableContextError:
            sample["status"] = "video_error"
            return sample

        sample["status"] = selection.status
        sample["frame_indices"] = torch.from_numpy(
            selection.indices.copy()
        )

        if selection.status == "no_context":
            return sample

        started = time.perf_counter()

        try:
            frames = decode_requested_frames(
                path,
                pts,
                time_base,
                selection.indices,
                cutoff,
            )
        except VideoReadError as error:
            warnings.warn(
                f"{row['narration_id']}: {error}",
                RuntimeWarning,
            )

            sample["status"] = "video_error"
            return sample
        finally:
            self.last_decode_seconds = (
                time.perf_counter() - started
            )

        started = time.perf_counter()

        sample["video"] = self.transform(
            frames,
            epoch=self.epoch,
            row_index=int(row["row_index"]),
        )

        self.last_transform_seconds = (
            time.perf_counter() - started
        )

        sample["has_context"] = torch.tensor(
            True,
            dtype=torch.bool,
        )

        return sample


def collate_sta_clips(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    expected = set(samples[0])

    if any(set(sample) != expected for sample in samples):
        raise ValueError(
            "Mixed labelled/unlabelled sample schemas"
        )

    output = {}

    for key in samples[0]:
        if key in {"narration_id", "status"}:
            output[key] = [
                sample[key] for sample in samples
            ]
        else:
            output[key] = torch.stack([
                sample[key] for sample in samples
            ])

    return output
    