"""Export a verified E4 hidden-test artifact to the exact course payload.

Run evaluate_sta2.py --split test first. No model or optimizer is constructed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_sta_submission import validate_file, validate_submission
from src.sta2.artifacts import PRIMARY_SCORE, load_prediction_artifact, verify_fallbacks
from src.sta2.data import load_events, sha256_file
from src.sta2.evaluation import (
    action_vocabulary_sha256, evaluation_action_pairs, frequency_priors,
    load_evaluation_checkpoint, verify_evaluation_inputs, write_json_atomic,
)


def save_submission_atomic(path, payload, expected_ids):
    """Validate in memory and after reload before replacing the destination."""
    validate_submission(payload, expected_ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        validate_file(temporary, expected_ids)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_submission(args):
    inputs = [args.predictions, args.checkpoint, args.backbone_checkpoint,
              args.annotations, args.train_split, args.validation_split, args.events]
    metadata_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    protected = {Path(path).resolve() for path in inputs}
    if any(path.resolve() in protected for path in (
        args.output, metadata_path, metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    )):
        raise ValueError("output paths must not overwrite source artifacts")
    before = {str(path): sha256_file(path) for path in inputs}
    checkpoint = load_evaluation_checkpoint(args.checkpoint)
    train = load_events(args.annotations, split="train", train_split=args.train_split,
                        validation_split=args.validation_split)
    events = load_events(args.events, split="test")
    verified = verify_evaluation_inputs(
        checkpoint, backbone_checkpoint=args.backbone_checkpoint,
        annotations=args.annotations, train_split=args.train_split,
        validation_split=args.validation_split, train_events=train,
    )
    pairs = evaluation_action_pairs(pd.read_csv(args.annotations), num_verbs=97, num_nouns=300)
    expected = {
        "split": "test", "target_events_sha256": sha256_file(args.events),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "backbone_sha256": verified["backbone_sha256"],
        "split_hashes": verified["split_hashes"],
        "checkpoint_configuration": checkpoint["configuration"],
        "observed_only": checkpoint["configuration"]["experiment"]["observed_only"],
        "auxiliary_action_vocabulary_sha256": verified["auxiliary_action_vocabulary_sha256"],
        "valid_action_classes": len(pairs),
        "valid_action_vocabulary_sha256": action_vocabulary_sha256(pairs),
        "primary_action_score": PRIMARY_SCORE,
    }
    head = checkpoint["configuration"]["head"]
    if (head["num_verbs"], head["num_nouns"]) != (97, 300):
        raise ValueError("course export requires 97 verbs and 300 nouns")
    metadata, records = load_prediction_artifact(args.predictions, events, expected)
    verify_fallbacks(records, frequency_priors(train, num_verbs=97, num_nouns=300))
    payload = [{
        "narration_id": record["narration_id"],
        "verb_output": torch.from_numpy(record["verb_probabilities"].copy()),
        "noun_output": torch.from_numpy(record["noun_probabilities"].copy()),
    } for record in records]
    if before != {str(path): sha256_file(path) for path in inputs}:
        raise ValueError("source artifacts changed during export; rerun with immutable inputs")
    save_submission_atomic(args.output, payload, events.narration_id.tolist())
    provenance = {
        "schema_version": 1, "evaluation_metadata": metadata,
        "source_sha256": before, "submission_sha256": sha256_file(args.output),
        "events": len(payload), "observed_only": expected["observed_only"],
    }
    write_json_atomic(metadata_path, provenance)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "checkpoint", "backbone-checkpoint", "annotations",
                 "train-split", "validation-split", "events", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = export_submission(parser.parse_args(argv))
    print(f"Exported and validated {result['events']} hidden-test events")
    return result


if __name__ == "__main__":
    main()
