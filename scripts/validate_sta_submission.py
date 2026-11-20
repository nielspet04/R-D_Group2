"""Validate the exact course STA payload without modifying it."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sta2.data import load_events


def validate_submission(payload, expected_ids):
    """Reject schema, identity, order, device and probability violations."""
    expected_ids = list(expected_ids)
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("hidden-test IDs must be nonempty and unique")
    if not isinstance(payload, list):
        raise ValueError("submission must be an outer list")
    ids = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict) or set(record) != {
            "narration_id", "verb_output", "noun_output"
        }:
            raise ValueError(f"event {index}: exact keys required: narration_id, verb_output, noun_output")
        identity = record["narration_id"]
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"event {index}: narration_id must be a nonempty string")
        ids.append(identity)
        for name, size in (("verb_output", 97), ("noun_output", 300)):
            vector = record[name]
            prefix = f"event {index} ({identity}) {name}"
            if not isinstance(vector, torch.Tensor) or vector.shape != (size,):
                raise ValueError(f"{prefix}: expected tensor of shape [{size}]")
            if vector.device.type != "cpu" or vector.dtype != torch.float32:
                raise ValueError(f"{prefix}: expected CPU FP32 tensor")
            if vector.layout != torch.strided:
                raise ValueError(f"{prefix}: expected dense tensor")
            if not torch.isfinite(vector).all().item():
                raise ValueError(f"{prefix}: NaN or infinity")
            if (vector < 0).any().item():
                raise ValueError(f"{prefix}: negative probability")
            if not torch.isclose(vector.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5).item():
                raise ValueError(f"{prefix}: probabilities must sum to one")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate narration IDs in submission")
    missing, unexpected = set(expected_ids) - set(ids), set(ids) - set(expected_ids)
    if missing or unexpected:
        raise ValueError(f"coverage mismatch: missing IDs={sorted(missing)}, unexpected IDs={sorted(unexpected)}")
    if ids != expected_ids:
        raise ValueError("submission must follow original hidden-test event order")
    return {"events": len(ids), "valid": True}


def validate_file(path, expected_ids):
    try:
        # Do not remap devices: a CUDA payload must not become valid on load.
        payload = torch.load(path, weights_only=True)
    except Exception as error:
        raise ValueError(f"cannot reload submission {path}: {error}") from error
    return validate_submission(payload, expected_ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args(argv)
    events = load_events(args.events, split="test")
    result = validate_file(args.submission, events.narration_id.tolist())
    print(f"Valid STA submission: {result['events']} events")
    return result


if __name__ == "__main__":
    main()
