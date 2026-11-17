"""Independently evaluate a completed STA2 checkpoint with course metrics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sta2.backbone import FrozenAnticipationBackbone  # noqa: E402
from src.sta2.data import (  # noqa: E402
    STAClipDataset,
    collate_sta_clips,
    load_events,
    sha256_file,
)
from src.sta2.evaluation import (  # noqa: E402
    action_vocabulary_sha256,
    assert_reloaded_predictions_equal,
    build_head_from_checkpoint,
    collect_local_predictions,
    evaluation_action_pairs,
    frequency_priors,
    load_evaluation_checkpoint,
    merge_prediction_records,
    save_predictions_atomic,
    score_prediction_records,
    summarize_unlabelled_records,
    verify_evaluation_inputs,
    write_json_atomic,
)
from src.sta2.model import STA2Model  # noqa: E402
from src.sta2.training import (  # noqa: E402
    close_distributed,
    initialize_distributed,
    seed_everything,
)


DEFAULT_BACKBONE_CONFIG = PROJECT_ROOT / "src/sta2/data_backbone.yaml"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data/raw/EPIC_100_train.csv"
DEFAULT_TRAIN_SPLIT = (
    PROJECT_ROOT / "data/processed/EPIC_100_train_participant_split.csv"
)
DEFAULT_VALIDATION_SPLIT = (
    PROJECT_ROOT
    / "data/processed/EPIC_100_validation_participant_split.csv"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--backbone-checkpoint", type=Path, required=True
    )
    parser.add_argument(
        "--backbone-config", type=Path, default=DEFAULT_BACKBONE_CONFIG
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help=(
            "Complete labelled annotation table used during training and "
            "to define the course-compatible action vocabulary."
        ),
    )
    parser.add_argument(
        "--train-split", type=Path, default=DEFAULT_TRAIN_SPLIT
    )
    parser.add_argument(
        "--validation-split", type=Path, default=DEFAULT_VALIDATION_SPLIT
    )
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument(
        "--events",
        type=Path,
        help=(
            "Hidden-test event table. Required for --split test; validation "
            "uses --annotations plus the checkpointed participant split."
        ),
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int)
    return parser.parse_args(argv)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _gather_records(
    local_records: list[dict[str, Any]],
    world_size: int,
) -> list[dict[str, Any]]:
    if not dist.is_initialized():
        return local_records
    gathered: list[list[dict[str, Any]] | None] = [
        None for _ in range(world_size)
    ]
    dist.all_gather_object(gathered, local_records)
    return [
        record
        for rank_records in gathered
        for record in rank_records or []
    ]


def _broadcast_rank_zero_result(
    value: Any,
    *,
    is_main: bool,
) -> Any:
    if not dist.is_initialized():
        return value
    payload = [value if is_main else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _rank_zero_verify_inputs(
    checkpoint: dict[str, Any],
    *,
    is_main: bool,
    backbone_checkpoint: Path,
    annotations: Path,
    train_split: Path,
    validation_split: Path,
    train_events: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    error: str | None = None
    if is_main:
        try:
            result = verify_evaluation_inputs(
                checkpoint,
                backbone_checkpoint=backbone_checkpoint,
                annotations=annotations,
                train_split=train_split,
                validation_split=validation_split,
                train_events=train_events,
            )
        except Exception as caught:  # broadcast before every rank exits
            error = f"{type(caught).__name__}: {caught}"

    payload = _broadcast_rank_zero_result(
        {"result": result, "error": error},
        is_main=is_main,
    )
    if payload["error"] is not None:
        raise ValueError(payload["error"])
    assert payload["result"] is not None
    return payload["result"]


def _loader(
    dataset: STAClipDataset,
    sampler: DistributedSampler,
    *,
    batch_size: int,
    workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if workers < 0:
        raise ValueError("workers must be non-negative")
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
        "collate_fn": collate_sta_clips,
        "drop_last": False,
    }
    if workers:
        options["prefetch_factor"] = prefetch_factor
    return DataLoader(**options)


def _run_prediction_pass(
    *,
    model: STA2Model,
    loader: DataLoader,
    events: pd.DataFrame,
    device: torch.device,
    verb_prior: torch.Tensor,
    noun_prior: torch.Tensor,
    bf16: bool,
    world_size: int,
) -> list[dict[str, Any]]:
    local_records = collect_local_predictions(
        model,
        loader,
        device=device,
        verb_prior=verb_prior,
        noun_prior=noun_prior,
        bf16=bf16,
    )
    gathered = _gather_records(local_records, world_size)
    return merge_prediction_records(
        gathered,
        events,
        world_size=world_size,
    )


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Validate provenance, independently infer, score, and save outputs."""

    if args.split == "test" and args.events is None:
        raise ValueError("--events is required for --split test")
    if args.split == "validation" and args.events is not None:
        raise ValueError("--events is only valid with --split test")

    context = initialize_distributed(require_cuda=True)
    try:
        checkpoint_stat = args.checkpoint.stat() if context.is_main else None
        checkpoint_sha256_before = (
            sha256_file(args.checkpoint) if context.is_main else None
        )
        checkpoint = load_evaluation_checkpoint(args.checkpoint)
        configuration = checkpoint["configuration"]
        checkpoint_epoch = int(checkpoint["epoch"])
        checkpoint_git_commit = checkpoint["git_commit"]
        checkpoint_git_dirty = bool(checkpoint["git_dirty"])
        auxiliary_vocabulary = tuple(
            (int(pair[0]), int(pair[1]))
            for pair in checkpoint["action_vocabulary"]
        )
        head_configuration = configuration["head"]
        if (
            int(head_configuration["num_verbs"]) != 97
            or int(head_configuration["num_nouns"]) != 300
        ):
            raise ValueError(
                "course STA2 evaluation requires 97 verbs and 300 nouns"
            )
        training_configuration = configuration["training"]
        experiment_configuration = configuration["experiment"]
        data_configuration = configuration.get("data", {})
        seed_everything(int(training_configuration["seed"]), context.rank)

        train_events = load_events(
            args.annotations,
            split="train",
            train_split=args.train_split,
            validation_split=args.validation_split,
        )
        target_annotations = (
            args.annotations if args.split == "validation" else args.events
        )
        assert target_annotations is not None
        events = load_events(
            target_annotations,
            split=args.split,
            train_split=args.train_split,
            validation_split=args.validation_split,
        )

        verified = _rank_zero_verify_inputs(
            checkpoint,
            is_main=context.is_main,
            backbone_checkpoint=args.backbone_checkpoint,
            annotations=args.annotations,
            train_split=args.train_split,
            validation_split=args.validation_split,
            train_events=train_events,
        )
        complete_annotations = pd.read_csv(args.annotations)
        action_pairs = evaluation_action_pairs(
            complete_annotations,
            num_verbs=int(head_configuration["num_verbs"]),
            num_nouns=int(head_configuration["num_nouns"]),
        )
        verb_prior_array, noun_prior_array = frequency_priors(
            train_events,
            num_verbs=int(head_configuration["num_verbs"]),
            num_nouns=int(head_configuration["num_nouns"]),
        )
        verb_prior = torch.tensor(
            verb_prior_array, device=context.device
        )
        noun_prior = torch.tensor(
            noun_prior_array, device=context.device
        )

        dataset = STAClipDataset(
            events,
            args.raw_dir,
            args.metadata,
            split=args.split,
            training=False,
            seed=int(training_configuration["seed"]),
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
            drop_last=False,
        )
        workers = (
            int(args.workers)
            if args.workers is not None
            else int(data_configuration.get("workers_per_rank", 0))
        )
        loader = _loader(
            dataset,
            sampler,
            batch_size=args.batch_size,
            workers=workers,
            pin_memory=bool(data_configuration.get("pin_memory", True)),
            prefetch_factor=int(data_configuration.get("prefetch_factor", 1)),
        )

        backbone = FrozenAnticipationBackbone.from_checkpoint(
            args.backbone_checkpoint,
            _read_yaml(args.backbone_config),
            observed_only=bool(
                experiment_configuration["observed_only"]
            ),
            bf16=bool(training_configuration["bf16"]),
        ).to(context.device)
        head = build_head_from_checkpoint(
            checkpoint, embed_dim=backbone.embed_dim
        ).to(context.device)
        # The optimizer, scheduler, and RNG payloads are intentionally not
        # restored. Release their CPU storage as soon as the head is loaded.
        del checkpoint
        model = STA2Model(backbone, head).to(context.device)
        first = _run_prediction_pass(
            model=model,
            loader=loader,
            events=events,
            device=context.device,
            verb_prior=verb_prior,
            noun_prior=noun_prior,
            bf16=bool(training_configuration["bf16"]),
            world_size=context.world_size,
        )

        del model
        del head
        reloaded_checkpoint = load_evaluation_checkpoint(args.checkpoint)
        reloaded_head = build_head_from_checkpoint(
            reloaded_checkpoint, embed_dim=backbone.embed_dim
        ).to(context.device)
        del reloaded_checkpoint
        reloaded_model = STA2Model(backbone, reloaded_head).to(
            context.device
        )
        second = _run_prediction_pass(
            model=reloaded_model,
            loader=loader,
            events=events,
            device=context.device,
            verb_prior=verb_prior,
            noun_prior=noun_prior,
            bf16=bool(training_configuration["bf16"]),
            world_size=context.world_size,
        )
        assert_reloaded_predictions_equal(first, second)

        if args.split == "validation":
            summary, predictions = score_prediction_records(
                first,
                action_pairs=action_pairs,
                auxiliary_action_vocabulary=auxiliary_vocabulary,
            )
        else:
            summary, predictions = summarize_unlabelled_records(
                first,
                action_pairs=action_pairs,
            )

        result: dict[str, Any] | None = None
        if context.is_main:
            checkpoint_sha256 = sha256_file(args.checkpoint)
            final_stat = args.checkpoint.stat()
            if (
                checkpoint_stat is None
                or checkpoint_sha256_before is None
                or checkpoint_sha256 != checkpoint_sha256_before
                or final_stat.st_size != checkpoint_stat.st_size
                or final_stat.st_mtime_ns != checkpoint_stat.st_mtime_ns
            ):
                raise RuntimeError("evaluation modified the source checkpoint")

            target_sha256 = sha256_file(target_annotations)
            run_metadata = {
                "schema_version": 1,
                "split": args.split,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_epoch": checkpoint_epoch,
                "checkpoint_git_commit": checkpoint_git_commit,
                "checkpoint_git_dirty": checkpoint_git_dirty,
                "backbone_checkpoint": str(
                    args.backbone_checkpoint.resolve()
                ),
                "backbone_sha256": verified["backbone_sha256"],
                "split_hashes": verified["split_hashes"],
                "target_events_sha256": target_sha256,
                "observed_only": bool(
                    experiment_configuration["observed_only"]
                ),
                "checkpoint_configuration": configuration,
                "auxiliary_action_classes": len(auxiliary_vocabulary),
                "auxiliary_action_vocabulary_sha256": verified[
                    "auxiliary_action_vocabulary_sha256"
                ],
                "valid_action_classes": len(action_pairs),
                "valid_action_vocabulary_sha256": (
                    action_vocabulary_sha256(action_pairs)
                ),
                "world_size": context.world_size,
                "fresh_reload_verified": True,
                "primary_action_score": "P(verb=v) * P(noun=n)",
                "primary_action_source": "primary_verb_and_noun_heads",
            }
            predictions_path = args.output_dir / "predictions.pt"
            metrics_path = args.output_dir / "metrics.json"
            save_predictions_atomic(
                predictions_path, predictions, run_metadata
            )
            result = {
                **run_metadata,
                **summary,
                "predictions_path": str(predictions_path.resolve()),
                "metrics_path": str(metrics_path.resolve()),
            }
            write_json_atomic(metrics_path, result)

        result = _broadcast_rank_zero_result(
            result, is_main=context.is_main
        )
        assert result is not None
        return result
    finally:
        close_distributed()


def print_results(result: dict[str, Any]) -> None:
    print(f"Split:              {result['split']}")
    print(f"Checkpoint epoch:   {result['checkpoint_epoch']}")
    event_key = (
        "validation_events" if result["split"] == "validation" else "test_events"
    )
    print(f"Events:             {result[event_key]:,}")
    print(f"Prior fallbacks:    {result['prior_fallback_count']:,}")
    print(f"Valid actions:      {result['valid_action_classes']:,}")
    if result["split"] == "validation":
        print()
        print("Metric                 Verb       Noun     Action")
        print(
            "Top-1 accuracy     "
            f"{result['verb_top1_accuracy']:10.4f}"
            f"{result['noun_top1_accuracy']:11.4f}"
            f"{result['action_top1_accuracy']:11.4f}"
        )
        print(
            "Top-5 accuracy     "
            f"{result['verb_top5_accuracy']:10.4f}"
            f"{result['noun_top5_accuracy']:11.4f}"
            f"{result['action_top5_accuracy']:11.4f}"
        )
        print(
            "Class-mean Top-5  "
            f"{result['verb_mt5r']:10.4f}"
            f"{result['noun_mt5r']:11.4f}"
            f"{result['action_mt5r']:11.4f}"
        )
    print(f"\nSaved predictions to {result['predictions_path']}")
    print(f"Saved metadata to    {result['metrics_path']}")


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = evaluate_checkpoint(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print_results(result)
    return result


if __name__ == "__main__":
    main()
