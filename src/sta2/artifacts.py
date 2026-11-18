"""E5 adapters for E4 artifacts; inference and scoring remain in E4."""

from pathlib import Path

import numpy as np
import torch

from src.sta2.data import sha256_file
from src.sta2.evaluation import (
    FALLBACK_STATUSES, VALID_STATUSES, action_vocabulary_sha256,
    merge_prediction_records,
)


PRIMARY_SCORE = "P(verb=v) * P(noun=n)"


def comparison_identity(annotations, train_split, validation_split, pairs):
    return {
        "split": "validation",
        "split_hashes": {
            "annotations": sha256_file(annotations),
            "train_split": sha256_file(train_split),
            "validation_split": sha256_file(validation_split),
        },
        "target_events_sha256": sha256_file(annotations),
        "valid_action_classes": len(pairs),
        "valid_action_vocabulary_sha256": action_vocabulary_sha256(pairs),
        "primary_action_score": PRIMARY_SCORE,
    }


def require_compatible(metadata, expected):
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"incompatible run metadata: {key}; expected {value!r}, got {metadata.get(key)!r}")


def load_prediction_artifact(path, events, expected):
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != 1:
        raise ValueError("expected E4 prediction artifact schema_version=1")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("prediction artifact lacks metadata")
    require_compatible(metadata, expected)
    if metadata.get("fresh_reload_verified") is not True:
        raise ValueError("E4 fresh-reload verification is required")
    if metadata.get("primary_action_source") != "primary_verb_and_noun_heads":
        raise ValueError("primary predictions must come from verb and noun heads")
    configuration = metadata.get("checkpoint_configuration", {})
    mode = metadata.get("observed_only")
    if not isinstance(mode, bool) or configuration.get("experiment", {}).get("observed_only") is not mode:
        raise ValueError("observed_only disagrees with checkpoint configuration")
    head = configuration.get("head", {})
    if (head.get("num_verbs"), head.get("num_nouns")) != (97, 300):
        raise ValueError("course predictions require 97 verbs and 300 nouns")
    if not isinstance(artifact.get("predictions"), list):
        raise ValueError("prediction artifact requires a predictions list")
    records = []
    for supplied in artifact["predictions"]:
        record = dict(supplied)
        status = record.get("status")
        if status not in VALID_STATUSES or record.get("used_frequency_prior") is not (status in FALLBACK_STATUSES):
            raise ValueError("prediction status and frequency-fallback flag disagree")
        for name, size in (("verb_probabilities", 97), ("noun_probabilities", 300)):
            vector = record.get(name)
            if not isinstance(vector, torch.Tensor) or vector.shape != (size,):
                raise ValueError(f"{name} must have shape [{size}]")
            if vector.dtype != torch.float32:
                raise ValueError(f"{name} must use FP32")
            record[name] = vector.numpy().copy()
        # Ignore saved auxiliary/ranking diagnostics and recompute primary metrics.
        records.append(record)
    return metadata, merge_prediction_records(records, events, world_size=1)


def verify_fallbacks(records, priors):
    for record in records:
        if record["used_frequency_prior"]:
            for name, prior in zip(("verb_probabilities", "noun_probabilities"), priors):
                if not np.array_equal(record[name], prior):
                    raise ValueError(f"fallback differs from verified training prior: {record['narration_id']} {name}")
