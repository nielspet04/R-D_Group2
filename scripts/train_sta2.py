"""Train the V-JEPA-based STA2 model with torchrun."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import build_action_pairs  # noqa: E402
from src.sta2.backbone import (  # noqa: E402
    FrozenAnticipationBackbone,
)
from src.sta2.data import (  # noqa: E402
    STAClipDataset,
    collate_sta_clips,
    load_events,
    sha256_file,
)
from src.sta2.head import AnticipationHead  # noqa: E402
from src.sta2.model import (  # noqa: E402
    STA2Model,
    build_action_vocabulary,
)
from src.sta2.training import (  # noqa: E402
    STA2TrainingConfig,
    accumulation_steps,
    build_optimizer,
    build_scheduler,
    close_distributed,
    fit,
    get_git_provenance,
    initialize_distributed,
    seed_everything,
    wrap_head_for_distributed,
    write_json_atomic,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/sta2/main.yaml"
DEFAULT_BACKBONE_CONFIG = PROJECT_ROOT / "src/sta2/data_backbone.yaml"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data/raw/EPIC_100_train.csv"
DEFAULT_TRAIN_SPLIT = (
    PROJECT_ROOT
    / "data/processed/EPIC_100_train_participant_split.csv"
)
DEFAULT_VALIDATION_SPLIT = (
    PROJECT_ROOT
    / "data/processed/EPIC_100_validation_participant_split.csv"
)
DEFAULT_SOURCE_PROVENANCE = (
    PROJECT_ROOT / "configs/sta2/source_provenance.json"
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--backbone-config",
        type=Path,
        default=DEFAULT_BACKBONE_CONFIG,
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
    )
    parser.add_argument(
        "--train-split",
        type=Path,
        default=DEFAULT_TRAIN_SPLIT,
    )
    parser.add_argument(
        "--validation-split",
        type=Path,
        default=DEFAULT_VALIDATION_SPLIT,
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-provenance",
        type=Path,
        default=DEFAULT_SOURCE_PROVENANCE,
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--final-epochs", type=int,
        help="Final retraining duration selected on held-out validation. Requires all labelled events in train and an empty validation CSV.",
    )
    parser.add_argument("--require-complete-metadata", action="store_true")
    parser.add_argument(
        "--observed-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args(argv)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())

    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")

    return value


def _class_prior(
    labels: np.ndarray,
    class_count: int,
    device: torch.device,
) -> torch.Tensor:
    counts = np.bincount(
        labels.astype(np.int64),
        minlength=class_count,
    ).astype(np.float64)

    if counts.sum() == 0:
        raise ValueError("cannot fit a prior from no labels")

    return torch.tensor(
        counts / counts.sum(),
        dtype=torch.float32,
        device=device,
    )


def _rank_zero_metadata(
    context,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, str]]:
    payload = None

    if context.is_main:
        upstream = json.loads(
            args.source_provenance.read_text()
        )
        backbone_provenance = {
            "checkpoint_path": str(
                args.backbone_checkpoint.resolve()
            ),
            "checkpoint_sha256": sha256_file(
                args.backbone_checkpoint
            ),
            "upstream": upstream,
        }
        split_hashes = {
            "annotations": sha256_file(args.annotations),
            "train_split": sha256_file(args.train_split),
            "validation_split": sha256_file(
                args.validation_split
            ),
        }
        payload = (backbone_provenance, split_hashes)

    if dist.is_initialized():
        values = [payload]
        dist.broadcast_object_list(values, src=0)
        payload = values[0]

    assert payload is not None
    return payload


def _loader(
    dataset: STAClipDataset,
    sampler: DistributedSampler,
    *,
    config: STA2TrainingConfig,
    data_config: dict[str, Any],
) -> DataLoader:
    workers = int(data_config["workers_per_rank"])
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.microbatch_size,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(data_config["pin_memory"]),
        # Dataset.set_epoch must reach worker copies on every new iterator.
        "persistent_workers": False,
        "collate_fn": collate_sta_clips,
        "drop_last": False,
    }

    if workers:
        options["prefetch_factor"] = int(
            data_config["prefetch_factor"]
        )

    return DataLoader(**options)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    context = initialize_distributed(require_cuda=True)

    try:
        resolved = _read_yaml(args.config)

        if resolved.get("schema_version") != 1:
            raise ValueError("unsupported STA2 configuration version")

        config = STA2TrainingConfig.from_mapping(
            resolved["training"]
        )
        if args.final_epochs is not None:
            if not 1 <= args.final_epochs <= config.max_epochs:
                raise ValueError("final epochs must be between 1 and the configured max_epochs")
            resolved["final_training"] = {"epochs": args.final_epochs}
        backbone_config = _read_yaml(args.backbone_config)
        seed_everything(config.seed, context.rank)

        observed_only = resolved["experiment"][
            "observed_only"
        ]
        if args.observed_only is not None:
            observed_only = args.observed_only

        resolved = {
            **resolved,
            "experiment": {
                **resolved["experiment"],
                "observed_only": observed_only,
            },
            "paths": {
                "annotations": str(args.annotations.resolve()),
                "train_split": str(args.train_split.resolve()),
                "validation_split": str(
                    args.validation_split.resolve()
                ),
                "raw_dir": str(args.raw_dir.resolve()),
                "metadata": str(args.metadata.resolve()),
                "backbone_checkpoint": str(
                    args.backbone_checkpoint.resolve()
                ),
            },
            "world_size": context.world_size,
            "backbone_config": backbone_config,
            "metadata_sha256": sha256_file(args.metadata),
        }

        train_events = load_events(
            args.annotations,
            split="train",
            train_split=args.train_split,
            validation_split=args.validation_split,
        )
        validation_events = load_events(
            args.annotations,
            split="validation",
            train_split=args.train_split,
            validation_split=args.validation_split,
        )
        if args.final_epochs is not None:
            if len(validation_events) or len(train_events) != len(pd.read_csv(args.annotations)):
                raise ValueError("final retraining requires all labelled events in train and empty validation")
        elif not len(train_events) or not len(validation_events):
            raise ValueError("development training requires nonempty train and validation splits")
        train_dataset = STAClipDataset(
            train_events,
            args.raw_dir,
            args.metadata,
            split="train",
            training=True,
            seed=config.seed,
        )
        validation_dataset = STAClipDataset(
            validation_events,
            args.raw_dir,
            args.metadata,
            split="validation",
            training=False,
            seed=config.seed,
        )
        if args.require_complete_metadata:
            required_videos = set(train_events["video_id"]) | set(validation_events["video_id"])
            invalid = sorted(
                video_id for video_id in required_videos
                if train_dataset.metadata.get(video_id, {}).get("status") != "ok"
                or video_id not in train_dataset.videos
            )
            if invalid:
                raise ValueError(f"Full run requires audited, present videos: {invalid}")
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )
        validation_sampler = DistributedSampler(
            validation_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
            drop_last=False,
        )
        train_loader = _loader(
            train_dataset,
            train_sampler,
            config=config,
            data_config=resolved["data"],
        )
        validation_loader = _loader(
            validation_dataset,
            validation_sampler,
            config=config,
            data_config=resolved["data"],
        )

        train_verbs = torch.tensor(
            train_events["verb_class"].to_numpy(),
            dtype=torch.long,
        )
        train_nouns = torch.tensor(
            train_events["noun_class"].to_numpy(),
            dtype=torch.long,
        )
        action_vocabulary = build_action_vocabulary(
            train_verbs,
            train_nouns,
        )
        complete_annotations = pd.read_csv(args.annotations)
        evaluation_action_pairs = build_action_pairs(
            complete_annotations["verb_class"].to_numpy(),
            complete_annotations["noun_class"].to_numpy(),
        )
        backbone_provenance, split_hashes = _rank_zero_metadata(context, args)
        if context.is_main:
            known_pairs = set(action_vocabulary)
            validation_pairs = list(zip(
                validation_events["verb_class"],
                validation_events["noun_class"],
            ))
            run_manifest = {
                "configuration": resolved,
                "split_hashes": split_hashes,
                "backbone_provenance": backbone_provenance,
                **get_git_provenance(PROJECT_ROOT),
                "train_events": len(train_events),
                "validation_events": len(validation_events),
                "train_participants": sorted(train_events["participant_id"].unique().tolist()),
                "validation_participants": sorted(validation_events["participant_id"].unique().tolist()),
                "training_action_classes": len(action_vocabulary),
                "evaluation_action_classes": len(evaluation_action_pairs),
                "validation_events_in_training_action_vocabulary": sum(
                    pair in known_pairs for pair in validation_pairs
                ),
                "validation_events_outside_training_action_vocabulary": sum(
                    pair not in known_pairs for pair in validation_pairs
                ),
                "accumulation_steps": accumulation_steps(config, context.world_size),
                "resume": str(args.resume) if args.resume else None,
            }
            write_json_atomic(args.output_dir / f"run-manifest-{os.environ.get('SLURM_JOB_ID', 'local')}.json", run_manifest)
            print(json.dumps({"event": "training_setup", **run_manifest}), flush=True)
        head_config = resolved["head"]
        verb_prior = _class_prior(
            train_verbs.numpy(),
            int(head_config["num_verbs"]),
            context.device,
        )
        noun_prior = _class_prior(
            train_nouns.numpy(),
            int(head_config["num_nouns"]),
            context.device,
        )
        backbone = FrozenAnticipationBackbone.from_checkpoint(
            args.backbone_checkpoint,
            backbone_config,
            observed_only=observed_only,
            bf16=config.bf16,
        ).to(context.device)
        head = AnticipationHead(
            embed_dim=backbone.embed_dim,
            num_actions=len(action_vocabulary),
            num_verbs=int(head_config["num_verbs"]),
            num_nouns=int(head_config["num_nouns"]),
            num_heads=int(head_config["num_heads"]),
            mlp_ratio=float(head_config["mlp_ratio"]),
            depth=int(head_config["depth"]),
            use_activation_checkpointing=bool(
                head_config["use_activation_checkpointing"]
            ),
        ).to(context.device)
        model = STA2Model(backbone, head)
        wrap_head_for_distributed(model, context)
        optimizer = build_optimizer(model.head, config)
        steps = accumulation_steps(config, context.world_size)
        windows_per_epoch = math.ceil(len(train_loader) / steps)
        scheduler = build_scheduler(
            optimizer,
            windows_per_epoch=windows_per_epoch,
            config=config,
        )
        return fit(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            train_dataset=train_dataset,
            action_vocabulary=action_vocabulary,
            evaluation_action_pairs=evaluation_action_pairs,
            verb_prior=verb_prior,
            noun_prior=noun_prior,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            context=context,
            output_dir=args.output_dir,
            resolved_config=resolved,
            backbone_provenance=backbone_provenance,
            split_hashes=split_hashes,
            repository_root=PROJECT_ROOT,
            resume=args.resume,
            final_epochs=args.final_epochs,
        )
    finally:
        close_distributed()


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = run_training(args)

    if int(os.environ.get("RANK", "0")) == 0:
        if args.final_epochs is None:
            print(f"Best Action MT5R: {result['best_action_mt5r']:.4f}")
        else:
            print("Final retraining: no validation metric; export last.pt after the requested epochs finish.")
        print(f"Completed epoch:  {result['completed_epoch']}")
        print(f"Best checkpoint: {result['best_checkpoint']}")
        print(f"Last checkpoint: {result['last_checkpoint']}")

    return result


if __name__ == "__main__":
    main()
