"""Shared storage contract for cached STA video features.

Feature extraction and model training meet at this module.  Keeping validation
here prevents the two pipelines from silently interpreting a shard differently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
SEGMENT_COUNT = 5
FEATURE_DIMENSION = 512
LABELLED_SPLITS = frozenset({"train", "validation"})
ALLOWED_SPLITS = LABELLED_SPLITS | {"test"}
ALLOWED_STATUSES = frozenset({"full", "partial", "no_context", "video_error"})


def _torch():
    """Import PyTorch only when shard functionality is used.

    The repository's lightweight data utilities can still be imported on a
    machine where the optional local PyTorch environment has not been created.
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for STA feature shards. Activate rd_lab (local) "
            "or rd_epic (VSC) first."
        ) from error
    return torch


def validate_sta_feature_shard(
    shard: Mapping[str, Any],
    *,
    expected_split: str | None = None,
) -> int:
    """Validate one in-memory STA feature shard and return its event count."""
    torch = _torch()

    required_keys = {
        "schema_version",
        "split",
        "features",
        "valid_mask",
        "narration_ids",
        "video_ids",
        "action_start_seconds",
        "observation_end_seconds",
        "status",
    }
    missing = required_keys.difference(shard)
    if missing:
        raise ValueError(f"Shard is missing keys: {sorted(missing)}")

    if shard["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {shard['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}"
        )

    split = shard["split"]
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}, got {split!r}")
    if expected_split is not None and split != expected_split:
        raise ValueError(f"Expected split {expected_split!r}, got {split!r}")

    features = shard["features"]
    valid_mask = shard["valid_mask"]
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("features must be a tensor with shape [N, 5, 512]")
    event_count = features.shape[0]
    expected_feature_shape = (event_count, SEGMENT_COUNT, FEATURE_DIMENSION)
    if tuple(features.shape) != expected_feature_shape:
        raise ValueError(
            f"features must have shape {expected_feature_shape}, got {tuple(features.shape)}"
        )
    if features.dtype != torch.float16:
        raise ValueError(f"features must use torch.float16, got {features.dtype}")
    if features.device.type != "cpu":
        raise ValueError("features must be stored on CPU")
    if not torch.isfinite(features).all().item():
        raise ValueError("features contain NaN or infinite values")

    if not isinstance(valid_mask, torch.Tensor):
        raise ValueError("valid_mask must be a tensor")
    if tuple(valid_mask.shape) != (event_count, SEGMENT_COUNT):
        raise ValueError(
            f"valid_mask must have shape {(event_count, SEGMENT_COUNT)}, "
            f"got {tuple(valid_mask.shape)}"
        )
    if valid_mask.dtype != torch.bool:
        raise ValueError(f"valid_mask must use torch.bool, got {valid_mask.dtype}")
    if valid_mask.device.type != "cpu":
        raise ValueError("valid_mask must be stored on CPU")
    if torch.any(features[~valid_mask] != 0).item():
        raise ValueError("Features for invalid segments must be exactly zero")

    tensor_fields = {
        "action_start_seconds": torch.float32,
        "observation_end_seconds": torch.float32,
    }
    for field, dtype in tensor_fields.items():
        value = shard[field]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (event_count,):
            raise ValueError(f"{field} must be a tensor with shape [{event_count}]")
        if value.dtype != dtype or value.device.type != "cpu":
            raise ValueError(f"{field} must be a CPU {dtype} tensor")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{field} contains NaN or infinite values")

    if torch.any(
        shard["observation_end_seconds"] > shard["action_start_seconds"] - 1.0 + 1e-6
    ).item():
        raise ValueError("observation_end_seconds crosses the one-second anticipation gap")

    list_fields = ("narration_ids", "video_ids", "status")
    for field in list_fields:
        value = shard[field]
        if not isinstance(value, list) or len(value) != event_count:
            raise ValueError(f"{field} must be a list with {event_count} entries")
    if not all(isinstance(value, str) and value for value in shard["narration_ids"]):
        raise ValueError("Every narration ID must be a non-empty string")
    if len(set(shard["narration_ids"])) != event_count:
        raise ValueError("narration_ids must be unique within a shard")
    if not all(isinstance(value, str) and value for value in shard["video_ids"]):
        raise ValueError("Every video ID must be a non-empty string")
    invalid_statuses = set(shard["status"]).difference(ALLOWED_STATUSES)
    if invalid_statuses:
        raise ValueError(f"Unknown status values: {sorted(invalid_statuses)}")

    valid_counts = valid_mask.sum(dim=1)
    for index, status in enumerate(shard["status"]):
        count = int(valid_counts[index].item())
        if status == "full" and count != SEGMENT_COUNT:
            raise ValueError(f"Event {index} has status 'full' but {count} valid segments")
        if status == "partial" and not 0 < count < SEGMENT_COUNT:
            raise ValueError(
                f"Event {index} has status 'partial' but {count} valid segments"
            )
        if status in {"no_context", "video_error"} and count != 0:
            raise ValueError(f"Event {index} has status {status!r} but a non-empty mask")

    has_verb_labels = "verb_labels" in shard
    has_noun_labels = "noun_labels" in shard
    if has_verb_labels != has_noun_labels:
        raise ValueError("verb_labels and noun_labels must either both be present or absent")
    should_have_labels = split in LABELLED_SPLITS
    if has_verb_labels != should_have_labels:
        expectation = "present" if should_have_labels else "absent"
        raise ValueError(f"Labels must be {expectation} for split {split!r}")

    if should_have_labels:
        for field, class_count in (("verb_labels", 97), ("noun_labels", 300)):
            value = shard[field]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (event_count,):
                raise ValueError(f"{field} must be a tensor with shape [{event_count}]")
            if value.dtype != torch.int64 or value.device.type != "cpu":
                raise ValueError(f"{field} must be a CPU torch.int64 tensor")
            if event_count and (
                int(value.min().item()) < 0 or int(value.max().item()) >= class_count
            ):
                raise ValueError(f"{field} contains a value outside 0..{class_count - 1}")

    return event_count


def save_sta_feature_shard(shard: Mapping[str, Any], path: Path) -> Path:
    """Validate and atomically save one shard."""
    torch = _torch()
    validate_sta_feature_shard(shard)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(shard), temporary_path)
    temporary_path.replace(path)
    return path


def load_sta_feature_shard(path: Path, *, expected_split: str | None = None) -> dict:
    """Load a shard on CPU, validate it and return its dictionary."""
    torch = _torch()
    path = Path(path)
    try:
        shard = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch 2.5 supports weights_only, but this keeps the contract usable
        # with an older local installation.
        shard = torch.load(path, map_location="cpu")
    if not isinstance(shard, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    validate_sta_feature_shard(shard, expected_split=expected_split)
    return shard
