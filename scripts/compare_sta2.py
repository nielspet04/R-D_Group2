"""Compare five STA systems on one verified internal validation protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_sta_model import config_from_checkpoint
from src.sta2.artifacts import (
    comparison_identity, load_prediction_artifact, verify_fallbacks,
)
from src.sta2.data import load_events, resolve_cutoff, sha256_file
from src.sta2.evaluation import (
    evaluation_action_pairs, frequency_priors, score_prediction_records,
    write_json_atomic,
)
from src.sta_features import STAFeatureDataset
from src.sta_model_0 import FrequencyPriorSTA
from src.sta_training import build_model, load_checkpoint, validate_one_epoch


SYSTEMS = ("frequency_prior", "r3d18_mean_pool", "r3d18_gru",
           "vjepa_observed_only", "vjepa_observed_plus_future")
METRICS = tuple(f"{branch}_{metric}" for branch in ("verb", "noun", "action")
                for metric in ("top1_accuracy", "top5_accuracy", "mt5r"))


def metric_row(system, metrics, metadata):
    for key in METRICS:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{system}: {key} must be a finite metric on [0,1]")
    return {
        "system": system, "available": True, "metadata": metadata,
        "metrics": {key: float(metrics[key]) for key in METRICS},
        "counts": {
            "events": metrics["validation_events"],
            "usable_context": metrics.get("validation_usable_events"),
            "fallback": metrics.get("prior_fallback_count", metrics.get("validation_fallback_events")),
            "valid_actions": metadata["valid_action_classes"],
        },
    }


def unavailable(system, reason):
    return {"system": system, "available": False, "reason": reason}


def audit_feature_events(dataset, events):
    """Verify cached event identities, labels and legal cutoffs before scoring."""
    expected = {row["narration_id"]: row for row in events.to_dict("records")}
    seen = set()
    for index in range(len(dataset)):
        item = dataset[index]
        identity = item["narration_id"]
        if identity not in expected or identity in seen:
            raise ValueError(f"{dataset.split} cache contains unexpected/duplicate ID: {identity}")
        seen.add(identity)
        row = expected[identity]
        for cached, source in (("verb_label", "verb_class"), ("noun_label", "noun_class")):
            if item[cached] != int(row[source]):
                raise ValueError(f"cache label differs for {identity}: {cached}")
        if str(item["video_id"]).upper() != row["video_id"]:
            raise ValueError(f"cache video differs for {identity}")
        if not np.isclose(item["observation_end_seconds"], resolve_cutoff(row, dataset.split), atol=1e-4, rtol=0):
            raise ValueError(f"cache anticipation cutoff differs for {identity}")
    if seen != set(expected):
        raise ValueError(f"{dataset.split} cache is missing {len(set(expected) - seen)} events")


def r3d_result(path, feature_root, model_name, train, events, pairs, identity):
    if path is None:
        raise FileNotFoundError(f"no {model_name} checkpoint supplied; legacy metrics lack split/vocabulary provenance")
    checkpoint = load_checkpoint(path, map_location="cpu")
    config = config_from_checkpoint(checkpoint)
    if config.model != model_name or (config.num_verbs, config.num_nouns) != (97, 300):
        raise ValueError(f"checkpoint architecture must be {model_name} with 97 verbs and 300 nouns")
    if feature_root is None:
        raise FileNotFoundError("no --feature-root supplied for R3D reevaluation")
    train_cache = STAFeatureDataset(feature_root, "train")
    validation_cache = STAFeatureDataset(feature_root, "validation")
    audit_feature_events(train_cache, train)
    audit_feature_events(validation_cache, events)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    prior = FrequencyPriorSTA.fit(train.verb_class.to_numpy(), train.noun_class.to_numpy())
    metrics = validate_one_epoch(model, DataLoader(validation_cache, batch_size=config.validation_batch_size,
                                                  shuffle=False), torch.device("cpu"), prior, pairs)
    metadata = {
        **identity, "checkpoint_sha256": sha256_file(path),
        "configuration": checkpoint["configuration"],
        "training_lineage": "legacy checkpoint has no training-data hashes; evaluation cache and labels verified",
    }
    return metric_row("r3d18_" + model_name, metrics, metadata)


def presentation_table(rows):
    lines = ["| System | Verb T1/T5/MT5R (%) | Noun T1/T5/MT5R (%) | Action T1/T5/MT5R (%) | Events/usable/fallback/actions |",
             "|---|---|---|---|---|"]
    for row in rows:
        if not row["available"]:
            reason = row["reason"].replace("|", "/").replace("\n", " ")
            lines.append(f"| {row['system']} | Unavailable: {reason} | — | — | — |")
            continue
        cells = [row["system"]]
        for branch in ("verb", "noun", "action"):
            cells.append(" / ".join(f"{100 * row['metrics'][f'{branch}_{metric}']:.2f}"
                                    for metric in ("top1_accuracy", "top5_accuracy", "mt5r")))
        cells.append(" / ".join(str(value) if value is not None else "n/a" for value in row["counts"].values()))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def compare(args):
    sources = [args.annotations, args.train_split, args.validation_split,
               args.mean_checkpoint, args.gru_checkpoint, args.observed_only, args.observed_plus_future]
    protected = {path.resolve() for path in sources if path is not None}
    for path in (args.output, args.output.with_suffix(args.output.suffix + ".tmp"),
                 args.output.with_suffix(args.output.suffix + ".md")):
        if path.resolve() in protected:
            raise ValueError("comparison output must not overwrite source artifacts")
    train = load_events(args.annotations, split="train", train_split=args.train_split,
                        validation_split=args.validation_split)
    events = load_events(args.annotations, split="validation", train_split=args.train_split,
                         validation_split=args.validation_split)
    pairs = evaluation_action_pairs(pd.read_csv(args.annotations), num_verbs=97, num_nouns=300)
    identity = comparison_identity(args.annotations, args.train_split, args.validation_split, pairs)
    priors = frequency_priors(train, num_verbs=97, num_nouns=300)
    records = [{
        "row_index": int(row["row_index"]), "narration_id": row["narration_id"],
        "verb_label": int(row["verb_class"]), "noun_label": int(row["noun_class"]),
        "status": "no_context", "used_frequency_prior": True,
        "verb_probabilities": priors[0], "noun_probabilities": priors[1],
    } for row in events.to_dict("records")]
    metrics, _ = score_prediction_records(records, action_pairs=pairs, auxiliary_action_vocabulary=())
    # A nonvisual baseline has no measured video availability/fallback counts.
    metrics["validation_usable_events"] = None
    metrics["prior_fallback_count"] = None
    rows = [metric_row(SYSTEMS[0], metrics, identity)]
    for name, path in (("mean_pool", args.mean_checkpoint), ("gru", args.gru_checkpoint)):
        try:
            row = r3d_result(path, args.feature_root, name, train, events, pairs, identity)
        except FileNotFoundError as error:
            row = unavailable("r3d18_" + name, str(error))
        rows.append(row)
    for system, path, observed_only in (
        (SYSTEMS[3], args.observed_only, True), (SYSTEMS[4], args.observed_plus_future, False)
    ):
        if path is None:
            rows.append(unavailable(system, "no E4 validation predictions supplied"))
            continue
        try:
            metadata, predictions = load_prediction_artifact(path, events, {**identity, "observed_only": observed_only})
        except FileNotFoundError as error:
            rows.append(unavailable(system, str(error)))
            continue
        verify_fallbacks(predictions, priors)
        metrics, _ = score_prediction_records(predictions, action_pairs=pairs, auxiliary_action_vocabulary=())
        rows.append(metric_row(system, metrics, metadata))
    result = {"schema_version": 1, "protocol": identity, "metric_scale": "[0,1]",
              "metric_definitions": "src.evaluation: top_k_accuracy(k=1,5), class_mean_top_k_recall(k=5)",
              "systems": rows}
    write_json_atomic(args.output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "train-split", "validation-split", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("feature-root", "mean-checkpoint", "gru-checkpoint", "observed-only", "observed-plus-future"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    result = compare(args)
    table = presentation_table(result["systems"])
    args.output.with_suffix(args.output.suffix + ".md").write_text(table, encoding="utf-8")
    print(table)
    return result


if __name__ == "__main__":
    main()
