"""Independent, course-compatible evaluation for STA2 checkpoints."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data.distributed import DistributedSampler

from src.evaluation import (
    action_labels_from_pairs,
    build_action_pairs,
    class_mean_top_k_recall,
    joint_action_scores,
    top_k_accuracy,
)
from src.sta2.data import sha256_file
from src.sta2.head import AnticipationHead
from src.sta2.model import (
    ActionVocabulary,
    STA2Model,
    build_action_vocabulary,
)
from src.sta2.training import CHECKPOINT_SCHEMA_VERSION


VALID_STATUSES = frozenset({
    "full",
    "partial",
    "no_context",
    "video_error",
})
FALLBACK_STATUSES = frozenset({"no_context", "video_error"})


def action_vocabulary_sha256(
    action_pairs: Sequence[Sequence[int]] | np.ndarray,
) -> str:
    """Hash an ordered action vocabulary using a stable JSON encoding."""

    pairs = [
        [int(pair[0]), int(pair[1])]
        for pair in action_pairs
    ]
    payload = json.dumps(
        pairs,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")

    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error

    if converted <= 0 or converted != value:
        raise ValueError(f"{name} must be a positive integer")

    return converted


def _checkpoint_action_vocabulary(
    value: Any,
) -> ActionVocabulary:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("checkpoint action_vocabulary must be a sequence")

    pairs: list[tuple[int, int]] = []
    for pair in value:
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes))
            or len(pair) != 2
        ):
            raise ValueError("checkpoint action pairs must contain two labels")

        if (
            isinstance(pair[0], bool)
            or not isinstance(pair[0], int)
            or pair[0] < 0
            or isinstance(pair[1], bool)
            or not isinstance(pair[1], int)
            or pair[1] < 0
        ):
            raise ValueError(
                "checkpoint action labels must be non-negative integers"
            )
        verb = pair[0]
        noun = pair[1]
        pairs.append((verb, noun))

    vocabulary = tuple(pairs)
    if not vocabulary:
        raise ValueError("checkpoint action_vocabulary must not be empty")
    if vocabulary != tuple(sorted(set(vocabulary))):
        raise ValueError(
            "checkpoint action_vocabulary must be sorted and unique"
        )
    return vocabulary


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a completed E3 checkpoint without restoring training state."""

    required = {
        "schema_version",
        "head_state",
        "optimizer_state",
        "scheduler_state",
        "epoch",
        "optimizer_step",
        "sampler_epoch",
        "rng_states",
        "configuration",
        "backbone_provenance",
        "split_hashes",
        "action_vocabulary",
        "best_metric",
        "epochs_without_improvement",
        "git_commit",
        "git_dirty",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(
            f"checkpoint is missing fields: {sorted(missing)}"
        )

    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")

    epoch = _positive_int("checkpoint epoch", checkpoint["epoch"])
    if checkpoint["sampler_epoch"] != epoch:
        raise ValueError("checkpoint sampler_epoch must equal epoch")

    optimizer_step = checkpoint["optimizer_step"]
    if (
        isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step < 0
    ):
        raise ValueError("checkpoint optimizer_step must be non-negative")

    patience = checkpoint["epochs_without_improvement"]
    if (
        isinstance(patience, bool)
        or not isinstance(patience, int)
        or patience < 0
    ):
        raise ValueError(
            "checkpoint epochs_without_improvement must be non-negative"
        )

    candidate_configuration = checkpoint.get("configuration")
    final_training = (
        candidate_configuration.get("final_training")
        if isinstance(candidate_configuration, Mapping) else None
    )
    if final_training:
        if not isinstance(final_training, Mapping):
            raise ValueError("final_training must be a mapping")
        if checkpoint["best_metric"] is not None:
            raise ValueError("final checkpoint must not claim a validation metric")
        target_epoch = _positive_int("final_training.epochs", final_training.get("epochs"))
        if epoch != target_epoch:
            raise ValueError("final retraining is incomplete; resume to its selected epoch before export")
    else:
        best_metric = float(checkpoint["best_metric"])
        if not math.isfinite(best_metric):
            raise ValueError("checkpoint best_metric must be finite")

    if not isinstance(checkpoint["head_state"], Mapping):
        raise ValueError("checkpoint head_state must be a mapping")
    if not isinstance(checkpoint["optimizer_state"], Mapping):
        raise ValueError("checkpoint optimizer_state must be a mapping")
    if not isinstance(checkpoint["scheduler_state"], Mapping):
        raise ValueError("checkpoint scheduler_state must be a mapping")
    if not isinstance(checkpoint["rng_states"], Sequence):
        raise ValueError("checkpoint rng_states must be a sequence")

    configuration = checkpoint["configuration"]
    if not isinstance(configuration, Mapping):
        raise ValueError("checkpoint configuration must be a mapping")
    if configuration.get("schema_version") != 1:
        raise ValueError("unsupported resolved configuration version")

    for section in ("head", "training", "experiment", "paths"):
        if not isinstance(configuration.get(section), Mapping):
            raise ValueError(
                f"checkpoint configuration.{section} must be a mapping"
            )

    head = configuration["head"]
    required_head = {
        "num_verbs",
        "num_nouns",
        "num_heads",
        "mlp_ratio",
        "depth",
        "use_activation_checkpointing",
    }
    missing_head = required_head.difference(head)
    if missing_head:
        raise ValueError(
            "checkpoint head configuration is missing: "
            f"{sorted(missing_head)}"
        )

    for name in ("num_verbs", "num_nouns", "num_heads", "depth"):
        _positive_int(f"head.{name}", head[name])
    mlp_ratio = float(head["mlp_ratio"])
    if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
        raise ValueError("head.mlp_ratio must be positive")
    if not isinstance(head["use_activation_checkpointing"], bool):
        raise ValueError("head.use_activation_checkpointing must be Boolean")

    training = configuration["training"]
    if "seed" not in training or "bf16" not in training:
        raise ValueError("checkpoint training configuration lacks seed or bf16")
    if isinstance(training["seed"], bool) or not isinstance(
        training["seed"], int
    ):
        raise ValueError("training.seed must be an integer")
    if not isinstance(training["bf16"], bool):
        raise ValueError("training.bf16 must be Boolean")

    experiment = configuration["experiment"]
    if not isinstance(experiment.get("observed_only"), bool):
        raise ValueError("experiment.observed_only must be Boolean")
    if experiment.get("selection_metric") != "action_mt5r":
        raise ValueError("checkpoint selection metric is not action_mt5r")

    world_size = _positive_int(
        "configuration.world_size",
        configuration.get("world_size"),
    )
    if len(checkpoint["rng_states"]) != world_size:
        raise ValueError(
            "checkpoint RNG-state count differs from training world size"
        )

    backbone = checkpoint["backbone_provenance"]
    if not isinstance(backbone, Mapping):
        raise ValueError("checkpoint backbone_provenance must be a mapping")
    checksum = backbone.get("checkpoint_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("checkpoint lacks a valid backbone SHA-256")

    split_hashes = checkpoint["split_hashes"]
    if not isinstance(split_hashes, Mapping):
        raise ValueError("checkpoint split_hashes must be a mapping")
    expected_hashes = {"annotations", "train_split", "validation_split"}
    if expected_hashes.difference(split_hashes):
        raise ValueError("checkpoint lacks required split hashes")
    for name in expected_hashes:
        value = split_hashes[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"checkpoint split hash is invalid: {name}")

    vocabulary = _checkpoint_action_vocabulary(
        checkpoint["action_vocabulary"]
    )

    return {
        "configuration": dict(configuration),
        "head_configuration": dict(head),
        "action_vocabulary": vocabulary,
        "observed_only": experiment["observed_only"],
        "bf16": training["bf16"],
        "seed": training["seed"],
        "training_world_size": world_size,
    }


def load_evaluation_checkpoint(path: Path) -> dict[str, Any]:
    """Read and validate checkpoint metadata without touching optimizer/RNG."""

    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    validate_checkpoint_metadata(checkpoint)
    return dict(checkpoint)


def build_head_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    embed_dim: int,
) -> AnticipationHead:
    """Reconstruct the exact trainable architecture and load it strictly."""

    metadata = validate_checkpoint_metadata(checkpoint)
    config = metadata["head_configuration"]
    head = AnticipationHead(
        embed_dim=_positive_int("backbone embed_dim", embed_dim),
        num_actions=len(metadata["action_vocabulary"]),
        num_verbs=_positive_int("head.num_verbs", config["num_verbs"]),
        num_nouns=_positive_int("head.num_nouns", config["num_nouns"]),
        num_heads=_positive_int("head.num_heads", config["num_heads"]),
        mlp_ratio=float(config["mlp_ratio"]),
        depth=_positive_int("head.depth", config["depth"]),
        use_activation_checkpointing=bool(
            config["use_activation_checkpointing"]
        ),
    )
    try:
        head.load_state_dict(checkpoint["head_state"], strict=True)
    except RuntimeError as error:
        raise ValueError(
            "checkpoint head state does not match its architecture"
        ) from error
    head.eval()
    return head


def verify_evaluation_inputs(
    checkpoint: Mapping[str, Any],
    *,
    backbone_checkpoint: Path,
    annotations: Path,
    train_split: Path,
    validation_split: Path,
    train_events: pd.DataFrame,
) -> dict[str, Any]:
    """Verify immutable data/model identities and the auxiliary vocabulary."""

    metadata = validate_checkpoint_metadata(checkpoint)
    actual_hashes = {
        "annotations": sha256_file(annotations),
        "train_split": sha256_file(train_split),
        "validation_split": sha256_file(validation_split),
    }
    expected_hashes = dict(checkpoint["split_hashes"])
    if actual_hashes != expected_hashes:
        changed = sorted(
            name
            for name, digest in actual_hashes.items()
            if expected_hashes.get(name) != digest
        )
        raise ValueError(
            f"evaluation inputs differ from checkpoint: {changed}"
        )

    backbone_sha256 = sha256_file(backbone_checkpoint)
    expected_backbone = checkpoint["backbone_provenance"][
        "checkpoint_sha256"
    ]
    if backbone_sha256 != expected_backbone:
        raise ValueError("backbone checkpoint checksum differs from training")

    rebuilt_vocabulary = build_action_vocabulary(
        torch.tensor(
            train_events["verb_class"].to_numpy(),
            dtype=torch.long,
        ),
        torch.tensor(
            train_events["noun_class"].to_numpy(),
            dtype=torch.long,
        ),
    )
    if rebuilt_vocabulary != metadata["action_vocabulary"]:
        raise ValueError(
            "checkpoint action vocabulary differs from verified train split"
        )

    return {
        "split_hashes": actual_hashes,
        "backbone_sha256": backbone_sha256,
        "auxiliary_action_vocabulary_sha256": (
            action_vocabulary_sha256(rebuilt_vocabulary)
        ),
    }


def frequency_priors(
    train_events: pd.DataFrame,
    *,
    num_verbs: int,
    num_nouns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit training-only categorical priors for invalid video events."""

    priors: list[np.ndarray] = []
    for name, count in (
        ("verb_class", num_verbs),
        ("noun_class", num_nouns),
    ):
        labels = train_events[name].to_numpy(dtype=np.int64)
        if not len(labels):
            raise ValueError("cannot fit frequency priors from no events")
        if np.any(labels < 0) or np.any(labels >= count):
            raise ValueError(f"training labels outside range: {name}")
        frequencies = np.bincount(labels, minlength=count).astype(np.float64)
        priors.append((frequencies / frequencies.sum()).astype(np.float32))
    return priors[0], priors[1]


def evaluation_action_pairs(
    complete_annotations: pd.DataFrame,
    *,
    num_verbs: int,
    num_nouns: int,
) -> np.ndarray:
    """Build the course action vocabulary from every labelled annotation."""

    required = {"verb_class", "noun_class"}
    missing = required.difference(complete_annotations.columns)
    if missing:
        raise ValueError(f"complete annotations lack: {sorted(missing)}")

    verbs = complete_annotations["verb_class"].to_numpy(dtype=np.int64)
    nouns = complete_annotations["noun_class"].to_numpy(dtype=np.int64)
    if (
        np.any(verbs < 0)
        or np.any(verbs >= num_verbs)
        or np.any(nouns < 0)
        or np.any(nouns >= num_nouns)
    ):
        raise ValueError("complete annotation labels are outside class range")
    return build_action_pairs(verbs, nouns)


def _validate_prior(name: str, prior: Tensor, classes: int) -> None:
    if prior.shape != (classes,):
        raise ValueError(f"{name} prior must have shape [{classes}]")
    if not torch.isfinite(prior).all().item() or torch.any(prior < 0):
        raise ValueError(f"{name} prior must be finite and non-negative")
    if not torch.isclose(
        prior.float().sum(),
        torch.tensor(1.0, device=prior.device),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError(f"{name} prior must sum to one")


def collect_local_predictions(
    model: STA2Model,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    verb_prior: Tensor,
    noun_prior: Tensor,
    bf16: bool,
) -> list[dict[str, Any]]:
    """Run read-only inference and return one record per sampler item."""

    model.eval()
    records: list[dict[str, Any]] = []
    verb_classes = model.head.verb_classifier.out_features
    noun_classes = model.head.noun_classifier.out_features
    _validate_prior("verb", verb_prior, verb_classes)
    _validate_prior("noun", noun_prior, noun_classes)

    def autocast_context():
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if bf16 and device.type == "cuda"
            else nullcontext()
        )

    with torch.inference_mode():
        for cpu_batch in loader:
            video = cpu_batch["video"].to(device, non_blocking=True)
            has_context = cpu_batch["has_context"].to(
                device,
                dtype=torch.bool,
                non_blocking=True,
            )
            row_indices = cpu_batch["row_index"].to(
                dtype=torch.long
            )
            batch_size = len(row_indices)
            narration_ids = cpu_batch.get("narration_id")
            statuses = cpu_batch.get("status")
            if (
                not isinstance(narration_ids, Sequence)
                or len(narration_ids) != batch_size
                or not isinstance(statuses, Sequence)
                or len(statuses) != batch_size
            ):
                raise ValueError(
                    "evaluation batches require narration_id and status lists"
                )

            with autocast_context():
                outputs = model(video, has_context)

            required = {"verb_logits", "noun_logits", "action_logits"}
            if required.difference(outputs):
                raise ValueError("model omitted required prediction heads")
            for name in required:
                if not torch.isfinite(outputs[name]).all().item():
                    raise ValueError(f"model produced nonfinite {name}")

            verb_scores = torch.softmax(
                outputs["verb_logits"].float(), dim=1
            )
            noun_scores = torch.softmax(
                outputs["noun_logits"].float(), dim=1
            )

            labelled = "verb_label" in cpu_batch or "noun_label" in cpu_batch
            if labelled != (
                "verb_label" in cpu_batch and "noun_label" in cpu_batch
            ):
                raise ValueError("evaluation batch has incomplete labels")

            for index in range(batch_size):
                status = str(statuses[index])
                if status not in VALID_STATUSES:
                    raise ValueError(f"unknown evaluation status: {status}")
                expected_context = status not in FALLBACK_STATUSES
                actual_context = bool(has_context[index].item())
                if actual_context != expected_context:
                    raise ValueError(
                        "status and has_context disagree for row_index "
                        f"{int(row_indices[index])}"
                    )

                if status in FALLBACK_STATUSES:
                    verb = verb_prior
                    noun = noun_prior
                else:
                    verb = verb_scores[index]
                    noun = noun_scores[index]

                if (
                    not torch.isfinite(verb).all().item()
                    or not torch.isfinite(noun).all().item()
                ):
                    raise ValueError("nonfinite primary prediction")

                record: dict[str, Any] = {
                    "row_index": int(row_indices[index].item()),
                    "narration_id": str(narration_ids[index]),
                    "status": status,
                    "used_frequency_prior": status in FALLBACK_STATUSES,
                    "verb_probabilities": (
                        verb.detach().cpu().numpy().astype(
                            np.float32, copy=True
                        )
                    ),
                    "noun_probabilities": (
                        noun.detach().cpu().numpy().astype(
                            np.float32, copy=True
                        )
                    ),
                }
                if labelled:
                    record["verb_label"] = int(
                        cpu_batch["verb_label"][index].item()
                    )
                    record["noun_label"] = int(
                        cpu_batch["noun_label"][index].item()
                    )
                records.append(record)

    return records


def distributed_sampler_occurrences(
    events: pd.DataFrame,
    world_size: int,
) -> Counter[int]:
    """Return exact row-index multiplicities produced by all sampler ranks."""

    _positive_int("world_size", world_size)
    if not len(events):
        raise ValueError("evaluation event table must not be empty")
    if "row_index" not in events:
        raise ValueError("evaluation events lack row_index")

    row_indices = [int(value) for value in events["row_index"]]
    counts: Counter[int] = Counter()
    dataset = list(range(len(events)))
    for rank in range(world_size):
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        counts.update(row_indices[position] for position in sampler)
    return counts


def _predictions_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        if isinstance(left[key], np.ndarray):
            if not isinstance(right[key], np.ndarray) or not np.array_equal(
                left[key], right[key]
            ):
                return False
        elif left[key] != right[key]:
            return False
    return True


def merge_prediction_records(
    records: Sequence[Mapping[str, Any]],
    events: pd.DataFrame,
    *,
    world_size: int,
) -> list[dict[str, Any]]:
    """Validate exact event coverage and remove only sampler padding."""

    required_columns = {"row_index", "narration_id"}
    missing = required_columns.difference(events.columns)
    if missing:
        raise ValueError(f"evaluation event table lacks: {sorted(missing)}")
    if events["row_index"].duplicated().any():
        raise ValueError("evaluation event table contains duplicate row indices")
    if events["narration_id"].astype(str).duplicated().any():
        raise ValueError("evaluation event table contains duplicate narration IDs")

    expected_order = [int(value) for value in events["row_index"]]
    expected = {
        int(row["row_index"]): row
        for row in events.to_dict("records")
    }
    allowed_counts = distributed_sampler_occurrences(events, world_size)
    actual_counts: Counter[int] = Counter()
    unique: dict[int, dict[str, Any]] = {}

    for supplied in records:
        record = dict(supplied)
        if "row_index" not in record:
            raise ValueError("prediction record lacks row_index")
        row_index = int(record["row_index"])
        if row_index not in expected:
            raise ValueError(f"unexpected prediction row_index: {row_index}")
        actual_counts[row_index] += 1

        expected_row = expected[row_index]
        if str(record.get("narration_id")) != str(
            expected_row["narration_id"]
        ):
            raise ValueError(
                f"prediction narration ID conflicts at row_index {row_index}"
            )
        for record_field, event_field in (
            ("verb_label", "verb_class"),
            ("noun_label", "noun_class"),
        ):
            if event_field in expected_row:
                if record_field not in record or int(
                    record[record_field]
                ) != int(
                    expected_row[event_field]
                ):
                    raise ValueError(
                        f"prediction {record_field} conflicts at row_index "
                        f"{row_index}"
                    )
            elif record_field in record:
                raise ValueError("unlabelled evaluation record contains labels")

        previous = unique.get(row_index)
        if previous is not None and not _predictions_equal(previous, record):
            raise ValueError(
                f"conflicting sampler duplicate at row_index {row_index}"
            )
        unique.setdefault(row_index, record)

    missing_indices = sorted(set(expected) - set(unique))
    if missing_indices:
        raise ValueError(f"missing prediction row indices: {missing_indices}")
    if actual_counts != allowed_counts:
        raise ValueError(
            "prediction duplicates do not match DistributedSampler padding"
        )

    ordered = [unique[row_index] for row_index in expected_order]
    for record in ordered:
        for name in ("verb_probabilities", "noun_probabilities"):
            scores = np.asarray(record[name])
            if scores.ndim != 1 or not np.isfinite(scores).all():
                raise ValueError(f"invalid per-event {name}")
            if np.any(scores < 0) or not np.isclose(
                scores.sum(), 1.0, atol=1e-5, rtol=1e-5
            ):
                raise ValueError(f"invalid probability vector: {name}")
    return ordered


def assert_reloaded_predictions_equal(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> None:
    """Require bit-identical event predictions after a fresh head reload."""

    if len(first) != len(second):
        raise ValueError("fresh checkpoint reload changed event coverage")
    for left, right in zip(first, second):
        if not _predictions_equal(left, right):
            raise ValueError(
                "fresh checkpoint reload changed validation prediction at "
                f"row_index {left.get('row_index')}"
            )


def score_prediction_records(
    records: Sequence[Mapping[str, Any]],
    *,
    action_pairs: np.ndarray,
    auxiliary_action_vocabulary: ActionVocabulary,
) -> tuple[dict[str, int | float | dict[str, int]], list[dict[str, Any]]]:
    """Compute course metrics and attach primary joint-action rankings."""

    if not records:
        raise ValueError("at least one prediction record is required")
    if "verb_label" not in records[0] or "noun_label" not in records[0]:
        raise ValueError("labelled records are required for metric scoring")

    verb_labels = np.asarray(
        [int(record["verb_label"]) for record in records],
        dtype=np.int64,
    )
    noun_labels = np.asarray(
        [int(record["noun_label"]) for record in records],
        dtype=np.int64,
    )
    verb_scores = np.stack([
        np.asarray(record["verb_probabilities"], dtype=np.float64)
        for record in records
    ])
    noun_scores = np.stack([
        np.asarray(record["noun_probabilities"], dtype=np.float64)
        for record in records
    ])
    if not np.isfinite(verb_scores).all() or not np.isfinite(noun_scores).all():
        raise ValueError("nonfinite primary predictions")

    action_labels = action_labels_from_pairs(
        verb_labels, noun_labels, action_pairs
    )
    action_scores = joint_action_scores(
        verb_scores, noun_scores, action_pairs
    )
    if not np.isfinite(action_scores).all():
        raise ValueError("nonfinite joint action predictions")
    if action_scores.shape[1] < 5:
        raise ValueError("course Top-5 metrics require at least five actions")

    status_counts = Counter(str(record["status"]) for record in records)
    auxiliary_pairs = set(auxiliary_action_vocabulary)
    unsupported = sum(
        (int(verb), int(noun)) not in auxiliary_pairs
        for verb, noun in zip(verb_labels, noun_labels)
    )
    metrics: dict[str, int | float | dict[str, int]] = {
        "validation_events": len(records),
        "validation_usable_events": sum(
            not bool(record["used_frequency_prior"])
            for record in records
        ),
        "prior_fallback_count": sum(
            bool(record["used_frequency_prior"])
            for record in records
        ),
        "unsupported_auxiliary_action_label_count": unsupported,
        "event_counts_by_status": {
            status: int(status_counts.get(status, 0))
            for status in sorted(VALID_STATUSES)
        },
        "verb_top1_accuracy": top_k_accuracy(
            verb_labels, verb_scores, k=1
        ),
        "verb_top5_accuracy": top_k_accuracy(
            verb_labels, verb_scores, k=5
        ),
        "verb_mt5r": class_mean_top_k_recall(
            verb_labels, verb_scores, k=5
        ),
        "noun_top1_accuracy": top_k_accuracy(
            noun_labels, noun_scores, k=1
        ),
        "noun_top5_accuracy": top_k_accuracy(
            noun_labels, noun_scores, k=5
        ),
        "noun_mt5r": class_mean_top_k_recall(
            noun_labels, noun_scores, k=5
        ),
        "action_top1_accuracy": top_k_accuracy(
            action_labels, action_scores, k=1
        ),
        "action_top5_accuracy": top_k_accuracy(
            action_labels, action_scores, k=5
        ),
        "action_mt5r": class_mean_top_k_recall(
            action_labels, action_scores, k=5
        ),
    }

    enriched: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        ranking = np.argsort(
            -action_scores[index], kind="stable"
        )[:5]
        value = dict(record)
        value["primary_action_top5_ids"] = ranking.astype(np.int64)
        value["primary_action_top5_pairs"] = action_pairs[ranking].astype(
            np.int64, copy=True
        )
        value["primary_action_top5_scores"] = action_scores[
            index, ranking
        ].astype(np.float32, copy=True)
        enriched.append(value)
    return metrics, enriched


def summarize_unlabelled_records(
    records: Sequence[Mapping[str, Any]],
    *,
    action_pairs: np.ndarray,
) -> tuple[dict[str, int | dict[str, int]], list[dict[str, Any]]]:
    """Attach primary rankings and coverage counts for hidden-test events."""

    status_counts = Counter(str(record["status"]) for record in records)
    enriched: list[dict[str, Any]] = []
    for record in records:
        scores = joint_action_scores(
            np.asarray(record["verb_probabilities"])[None, :],
            np.asarray(record["noun_probabilities"])[None, :],
            action_pairs,
        )[0]
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite hidden-test action prediction")
        count = min(5, len(scores))
        ranking = np.argsort(-scores, kind="stable")[:count]
        value = dict(record)
        value["primary_action_top5_ids"] = ranking.astype(np.int64)
        value["primary_action_top5_pairs"] = action_pairs[ranking].astype(
            np.int64, copy=True
        )
        value["primary_action_top5_scores"] = scores[ranking].astype(
            np.float32, copy=True
        )
        enriched.append(value)
    summary: dict[str, int | dict[str, int]] = {
        "test_events": len(records),
        "test_usable_events": sum(
            not bool(record["used_frequency_prior"])
            for record in records
        ),
        "prior_fallback_count": sum(
            bool(record["used_frequency_prior"])
            for record in records
        ),
        "event_counts_by_status": {
            status: int(status_counts.get(status, 0))
            for status in sorted(VALID_STATUSES)
        },
    }
    return summary, enriched


def _serializable_record(record: Mapping[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in record.items():
        if isinstance(item, np.ndarray):
            value[key] = torch.from_numpy(item.copy())
        else:
            value[key] = item
    return value


def save_predictions_atomic(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Path:
    """Save ordered CPU predictions without modifying source checkpoints."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "metadata": dict(metadata),
            "predictions": [
                _serializable_record(record) for record in records
            ],
        },
        temporary,
    )
    temporary.replace(path)
    return path


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)
    return path
