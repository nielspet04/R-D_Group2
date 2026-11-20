"""Distributed training and checkpoint utilities for STA2."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

from src.evaluation import (
    action_labels_from_pairs,
    class_mean_top_k_recall,
    joint_action_scores,
    top_k_accuracy,
)
from src.sta2.model import (
    ActionVocabulary,
    STA2Model,
    compute_sta2_losses,
    encode_action_labels,
)


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class STA2TrainingConfig:
    """Resolved E3 training settings."""

    seed: int = 42
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999

    max_epochs: int = 20
    warmup_epochs: int = 1
    early_stopping_patience: int = 5
    gradient_clip_norm: float = 1.0

    microbatch_size: int = 1
    effective_global_batch_size: int = 16
    bf16: bool = True

    def __post_init__(self) -> None:
        positive_values = {
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "warmup_epochs": self.warmup_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "gradient_clip_norm": self.gradient_clip_norm,
            "microbatch_size": self.microbatch_size,
            "effective_global_batch_size": (
                self.effective_global_batch_size
            ),
        }

        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        for name, value in (
            ("beta1", self.beta1),
            ("beta2", self.beta2),
        ):
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "STA2TrainingConfig":
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistributedContext:
    """Rank and device information shared by E3 helpers."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(
    *,
    require_cuda: bool = True,
) -> DistributedContext:
    """Initialize one-node torchrun/NCCL execution."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("STA2 production training requires CUDA")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        if device.type != "cuda":
            raise RuntimeError(
                "Distributed STA2 training requires NCCL/CUDA"
            )

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def close_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    """Use deterministic but rank-distinct random streams."""

    resolved_seed = seed + rank
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def accumulation_steps(
    config: STA2TrainingConfig,
    world_size: int,
) -> int:
    denominator = config.microbatch_size * world_size

    if config.effective_global_batch_size % denominator:
        raise ValueError(
            "effective_global_batch_size must be divisible by "
            "microbatch_size * world_size"
        )

    return config.effective_global_batch_size // denominator


def unwrap_head(head: nn.Module) -> nn.Module:
    if isinstance(head, DistributedDataParallel):
        return head.module
    return head


def wrap_head_for_distributed(
    model: STA2Model,
    context: DistributedContext,
) -> None:
    """Wrap only the trainable head; keep the backbone outside DDP."""

    if not context.is_distributed:
        return

    model.head = DistributedDataParallel(
        model.head,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
    )


def build_optimizer(
    head: nn.Module,
    config: STA2TrainingConfig,
) -> torch.optim.AdamW:
    """Create the head-only AdamW optimizer."""

    parameters = [
        parameter
        for parameter in head.parameters()
        if parameter.requires_grad
    ]

    if not parameters:
        raise ValueError("head has no trainable parameters")

    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    windows_per_epoch: int,
    config: STA2TrainingConfig,
) -> LambdaLR:
    """Create one-epoch warm-up followed by cosine decay."""

    if windows_per_epoch <= 0:
        raise ValueError("windows_per_epoch must be positive")

    total_steps = windows_per_epoch * config.max_epochs
    warmup_steps = windows_per_epoch * config.warmup_epochs

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        remaining = max(total_steps - warmup_steps, 1)
        progress = min(
            max((step - warmup_steps) / remaining, 0.0),
            1.0,
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


def _all_reduce_sum(
    value: int | float,
    device: torch.device,
) -> float:
    tensor = torch.tensor(
        float(value),
        dtype=torch.float64,
        device=device,
    )

    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return float(tensor.item())


def _autocast_context(
    device: torch.device,
    enabled: bool,
):
    if enabled and device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    return nullcontext()


def _move_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        **batch,
        "video": batch["video"].to(
            device=device,
            non_blocking=True,
        ),
        "has_context": batch["has_context"].to(
            device=device,
            non_blocking=True,
        ),
        "verb_label": batch["verb_label"].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ),
        "noun_label": batch["noun_label"].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ),
        "row_index": batch["row_index"].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ),
    }


def _windows(
    loader: Iterable[Mapping[str, Any]],
    size: int,
) -> Iterable[list[Mapping[str, Any]]]:
    window: list[Mapping[str, Any]] = []

    for batch in loader:
        window.append(batch)

        if len(window) == size:
            yield window
            window = []

    if window:
        yield window


def train_one_epoch(
    model: STA2Model,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    action_vocabulary: ActionVocabulary,
    config: STA2TrainingConfig,
    context: DistributedContext,
) -> dict[str, int | float | None]:
    """Train using globally normalized accumulation windows."""

    model.train()
    steps_per_window = accumulation_steps(
        config,
        context.world_size,
    )

    local_events = 0
    local_usable = 0
    local_actions = 0
    local_verb_sum = 0.0
    local_noun_sum = 0.0
    local_action_sum = 0.0
    optimizer_steps = 0
    skipped_windows = 0
    started = time.monotonic()

    for window in _windows(loader, steps_per_window):
        local_window_usable = sum(
            int(batch["has_context"].sum().item())
            for batch in window
        )
        local_window_actions = 0

        for batch in window:
            action_labels = encode_action_labels(
                batch["verb_label"],
                batch["noun_label"],
                action_vocabulary,
            )
            action_mask = batch["has_context"] & action_labels.ge(0)
            local_window_actions += int(action_mask.sum().item())

        global_window_usable = int(
            _all_reduce_sum(
                local_window_usable,
                context.device,
            )
        )
        global_window_actions = int(
            _all_reduce_sum(
                local_window_actions,
                context.device,
            )
        )

        if global_window_usable == 0:
            skipped_windows += 1
            local_events += sum(
                int(batch["video"].shape[0])
                for batch in window
            )
            continue

        optimizer.zero_grad(set_to_none=True)

        for microbatch_index, cpu_batch in enumerate(window):
            batch = _move_batch(cpu_batch, context.device)
            action_labels = encode_action_labels(
                batch["verb_label"],
                batch["noun_label"],
                action_vocabulary,
            )

            is_last_microbatch = (
                microbatch_index == len(window) - 1
            )

            if (
                isinstance(model.head, DistributedDataParallel)
                and not is_last_microbatch
            ):
                synchronization = model.head.no_sync()
            else:
                synchronization = nullcontext()

            with synchronization:
                with _autocast_context(
                    context.device,
                    config.bf16,
                ):
                    outputs = model(
                        batch["video"],
                        batch["has_context"],
                    )

                losses = compute_sta2_losses(
                    outputs,
                    batch["verb_label"],
                    batch["noun_label"],
                    action_labels,
                    batch["has_context"],
                )

                scaled_loss = context.world_size * (
                    losses.verb_loss
                    * losses.usable_count
                    / global_window_usable
                    + losses.noun_loss
                    * losses.usable_count
                    / global_window_usable
                )

                if global_window_actions:
                    scaled_loss = scaled_loss + (
                        context.world_size
                        * losses.action_loss
                        * losses.action_count
                        / global_window_actions
                    )

                if not torch.isfinite(scaled_loss).item():
                    raise FloatingPointError(
                        "nonfinite STA2 training loss"
                    )

                scaled_loss.backward()

            local_events += int(batch["video"].shape[0])
            local_usable += losses.usable_count
            local_actions += losses.action_count
            local_verb_sum += (
                losses.verb_loss.detach().item()
                * losses.usable_count
            )
            local_noun_sum += (
                losses.noun_loss.detach().item()
                * losses.usable_count
            )
            local_action_sum += (
                losses.action_loss.detach().item()
                * losses.action_count
            )

        nn.utils.clip_grad_norm_(
            unwrap_head(model.head).parameters(),
            max_norm=config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        optimizer_steps += 1
        if context.is_main and (optimizer_steps == 1 or optimizer_steps % 100 == 0):
            print(json.dumps({
                "event": "training_progress",
                "optimizer_steps": optimizer_steps,
                "rank0_events": local_events,
                "elapsed_seconds": time.monotonic() - started,
            }), flush=True)

    global_events = int(
        _all_reduce_sum(local_events, context.device)
    )
    global_usable = int(
        _all_reduce_sum(local_usable, context.device)
    )
    global_actions = int(
        _all_reduce_sum(local_actions, context.device)
    )
    global_verb_sum = _all_reduce_sum(
        local_verb_sum,
        context.device,
    )
    global_noun_sum = _all_reduce_sum(
        local_noun_sum,
        context.device,
    )
    global_action_sum = _all_reduce_sum(
        local_action_sum,
        context.device,
    )

    if global_events == 0:
        raise ValueError("training loader contains no events")

    verb_loss = (
        global_verb_sum / global_usable
        if global_usable
        else None
    )
    noun_loss = (
        global_noun_sum / global_usable
        if global_usable
        else None
    )
    action_loss = (
        global_action_sum / global_actions
        if global_actions
        else 0.0
    )
    train_loss = (
        verb_loss + noun_loss + action_loss
        if verb_loss is not None and noun_loss is not None
        else None
    )

    return {
        "train_loss": train_loss,
        "train_verb_loss": verb_loss,
        "train_noun_loss": noun_loss,
        "train_action_loss": action_loss,
        "train_events": global_events,
        "train_usable_events": global_usable,
        "train_action_events": global_actions,
        "train_excluded_events": global_events - global_usable,
        "train_optimizer_steps": optimizer_steps,
        "train_skipped_windows": skipped_windows,
    }


def validate_one_epoch(
    model: STA2Model,
    loader: Iterable[Mapping[str, Any]],
    action_vocabulary: ActionVocabulary,
    evaluation_action_pairs: np.ndarray,
    verb_prior: Tensor,
    noun_prior: Tensor,
    expected_events: int,
    config: STA2TrainingConfig,
    context: DistributedContext,
) -> dict[str, int | float]:
    """Gather, deduplicate, and score full validation output."""

    model.eval()
    records: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader):
            if context.is_main and batch_index % 1000 == 0:
                print(json.dumps({"event": "validation_progress", "rank0_batches": batch_index}), flush=True)
            batch = _move_batch(cpu_batch, context.device)
            action_labels = encode_action_labels(
                batch["verb_label"],
                batch["noun_label"],
                action_vocabulary,
            )

            with _autocast_context(
                context.device,
                config.bf16,
            ):
                outputs = model(
                    batch["video"],
                    batch["has_context"],
                )

            verb_scores = torch.softmax(
                outputs["verb_logits"].float(),
                dim=1,
            )
            noun_scores = torch.softmax(
                outputs["noun_logits"].float(),
                dim=1,
            )
            usable = batch["has_context"]
            verb_scores = torch.where(
                usable[:, None],
                verb_scores,
                verb_prior[None, :],
            )
            noun_scores = torch.where(
                usable[:, None],
                noun_scores,
                noun_prior[None, :],
            )
            # Keep per-event losses with the predictions so sampler padding
            # is removed from loss denominators as well as ranking metrics.
            per_event_losses = {
                "verb_loss": F.cross_entropy(outputs["verb_logits"].float(), batch["verb_label"], reduction="none"),
                "noun_loss": F.cross_entropy(outputs["noun_logits"].float(), batch["noun_label"], reduction="none"),
                "action_loss": F.cross_entropy(outputs["action_logits"].float(), action_labels, ignore_index=-1, reduction="none"),
            }

            for index in range(len(batch["row_index"])):
                records.append({
                    "row_index": int(
                        batch["row_index"][index].item()
                    ),
                    "verb_label": int(
                        batch["verb_label"][index].item()
                    ),
                    "noun_label": int(
                        batch["noun_label"][index].item()
                    ),
                    "has_context": bool(usable[index].item()),
                    "known_action": bool(action_labels[index].ge(0).item()),
                    **{name: float(values[index].item()) for name, values in per_event_losses.items()},
                    "verb_scores": (
                        verb_scores[index].cpu().numpy()
                    ),
                    "noun_scores": (
                        noun_scores[index].cpu().numpy()
                    ),
                })

    if dist.is_initialized():
        gathered: list[list[dict[str, Any]] | None] = [
            None for _ in range(context.world_size)
        ]
        dist.all_gather_object(gathered, records)
        records = [
            record
            for rank_records in gathered
            for record in rank_records or []
        ]

    metrics: dict[str, int | float] | None = None

    if context.is_main:
        deduplicated: dict[int, dict[str, Any]] = {}

        for record in records:
            row_index = record["row_index"]
            previous = deduplicated.get(row_index)

            if previous is not None:
                if (
                    previous["verb_label"] != record["verb_label"]
                    or previous["noun_label"] != record["noun_label"]
                ):
                    raise ValueError(
                        "conflicting validation sampler duplicate"
                    )
                continue

            deduplicated[row_index] = record

        ordered = [
            deduplicated[index]
            for index in sorted(deduplicated)
        ]

        if len(ordered) != expected_events:
            raise ValueError(
                "validation gathering changed event coverage: "
                f"expected {expected_events}, got {len(ordered)}"
            )

        verb_labels = np.asarray(
            [record["verb_label"] for record in ordered],
            dtype=np.int64,
        )
        noun_labels = np.asarray(
            [record["noun_label"] for record in ordered],
            dtype=np.int64,
        )
        verb_scores = np.stack([
            record["verb_scores"] for record in ordered
        ])
        noun_scores = np.stack([
            record["noun_scores"] for record in ordered
        ])
        action_labels = action_labels_from_pairs(
            verb_labels,
            noun_labels,
            evaluation_action_pairs,
        )
        action_scores = joint_action_scores(
            verb_scores,
            noun_scores,
            evaluation_action_pairs,
        )
        usable_records = [record for record in ordered if record["has_context"]]
        action_records = [record for record in usable_records if record["known_action"]]
        global_usable = len(usable_records)
        global_actions = len(action_records)
        global_verb_sum = sum(record["verb_loss"] for record in usable_records)
        global_noun_sum = sum(record["noun_loss"] for record in usable_records)
        global_action_sum = sum(record["action_loss"] for record in action_records)
        verb_loss = (
            global_verb_sum / global_usable
            if global_usable
            else 0.0
        )
        noun_loss = (
            global_noun_sum / global_usable
            if global_usable
            else 0.0
        )
        action_loss = (
            global_action_sum / global_actions
            if global_actions
            else 0.0
        )

        metrics = {
            "validation_loss": (
                verb_loss + noun_loss + action_loss
            ),
            "validation_events": len(ordered),
            "validation_usable_events": sum(
                record["has_context"] for record in ordered
            ),
            "validation_fallback_events": sum(
                not record["has_context"] for record in ordered
            ),
            "validation_action_loss_events": global_actions,
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

    if dist.is_initialized():
        payload = [metrics]
        dist.broadcast_object_list(payload, src=0)
        metrics = payload[0]

    assert metrics is not None
    return metrics


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            # Torch 2.5 cannot serialize a torch.uint32 storage. Preserve
            # NumPy's MT19937 keys losslessly as signed 64-bit integers.
            "keys": torch.from_numpy(
                numpy_state[1].astype(np.int64)
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["name"],
        numpy_state["keys"].cpu().numpy().astype(np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())

    if torch.cuda.is_available() and state["cuda"]:
        # torch.load(map_location=cuda) also relocates RNG byte tensors;
        # generator.set_state requires them on CPU.
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def get_git_provenance(
    repository_root: Path,
) -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "git_commit": commit.stdout.strip(),
        "git_dirty": bool(status.stdout.strip()),
    }


def save_checkpoint(
    path: Path,
    *,
    model: STA2Model,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    optimizer_step: int,
    resolved_config: Mapping[str, Any],
    backbone_provenance: Mapping[str, Any],
    split_hashes: Mapping[str, str],
    action_vocabulary: ActionVocabulary,
    best_metric: float,
    epochs_without_improvement: int,
    rng_states: Sequence[Mapping[str, Any]],
    repository_root: Path,
) -> Path:
    """Atomically save trainable state without the 2B backbone."""

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "head_state": unwrap_head(model.head).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "optimizer_step": optimizer_step,
        "sampler_epoch": epoch,
        "rng_states": list(rng_states),
        "configuration": dict(resolved_config),
        "backbone_provenance": dict(backbone_provenance),
        "split_hashes": dict(split_hashes),
        "action_vocabulary": [
            [verb, noun]
            for verb, noun in action_vocabulary
        ],
        "best_metric": best_metric,
        "epochs_without_improvement": (
            epochs_without_improvement
        ),
        **get_git_provenance(repository_root),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(
    path: Path,
    *,
    model: STA2Model,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    rank: int,
    map_location: torch.device,
) -> dict[str, Any]:
    """Restore a completed-epoch checkpoint."""

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )
    required = {
        "schema_version",
        "head_state",
        "optimizer_state",
        "scheduler_state",
        "epoch",
        "optimizer_step",
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

    unwrap_head(model.head).load_state_dict(
        checkpoint["head_state"]
    )
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    rng_states = checkpoint["rng_states"]

    if rank >= len(rng_states):
        raise ValueError("checkpoint lacks RNG state for this rank")

    restore_rng_state(rng_states[rank])
    return checkpoint


def gather_rng_states(
    context: DistributedContext,
) -> list[Mapping[str, Any]]:
    state = capture_rng_state()

    if not dist.is_initialized():
        return [state]

    gathered: list[Mapping[str, Any] | None] = [
        None for _ in range(context.world_size)
    ]
    dist.all_gather_object(gathered, state)
    return [item for item in gathered if item is not None]


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True)
    )
    temporary.replace(path)


def fit(
    *,
    model: STA2Model,
    train_loader: Any,
    validation_loader: Any,
    train_sampler: Any,
    validation_sampler: Any,
    train_dataset: Any,
    action_vocabulary: ActionVocabulary,
    evaluation_action_pairs: np.ndarray,
    verb_prior: Tensor,
    noun_prior: Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    config: STA2TrainingConfig,
    context: DistributedContext,
    output_dir: Path,
    resolved_config: Mapping[str, Any],
    backbone_provenance: Mapping[str, Any],
    split_hashes: Mapping[str, str],
    repository_root: Path,
    resume: Path | None = None,
) -> dict[str, Any]:
    """Train, validate, early-stop, checkpoint, and resume at epochs."""

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    history_path = output_dir / "history.json"
    start_epoch = 1
    optimizer_step = 0
    best_metric = float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    if resume is not None:
        checkpoint = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=context.rank,
            map_location=context.device,
        )
        saved_vocabulary = tuple(
            tuple(pair)
            for pair in checkpoint["action_vocabulary"]
        )

        if saved_vocabulary != action_vocabulary:
            raise ValueError(
                "resume action vocabulary differs from this run"
            )

        if checkpoint["split_hashes"] != dict(split_hashes):
            raise ValueError(
                "resume split hashes differ from this run"
            )

        if (
            checkpoint["backbone_provenance"]
            != dict(backbone_provenance)
        ):
            raise ValueError(
                "resume backbone provenance differs from this run"
            )

        if checkpoint["configuration"] != dict(resolved_config):
            raise ValueError(
                "resume configuration differs from this run"
            )

        start_epoch = int(checkpoint["epoch"]) + 1
        optimizer_step = int(checkpoint["optimizer_step"])
        best_metric = float(checkpoint["best_metric"])
        epochs_without_improvement = int(
            checkpoint["epochs_without_improvement"]
        )

        if history_path.is_file():
            saved_history = json.loads(history_path.read_text())

            if not isinstance(saved_history, list):
                raise ValueError("training history must contain a list")

            # Discard records newer than the checkpoint when recovering files.
            history = [
                item for item in saved_history
                if int(item["epoch"]) <= int(checkpoint["epoch"])
            ]

    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, config.max_epochs + 1):
        if epochs_without_improvement >= config.early_stopping_patience:
            break
        epoch_started = time.monotonic()
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        if context.is_main:
            print(json.dumps({"event": "epoch_start", "epoch": epoch}), flush=True)
        train_sampler.set_epoch(epoch)
        validation_sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            action_vocabulary,
            config,
            context,
        )
        optimizer_step += int(
            train_metrics["train_optimizer_steps"]
        )
        train_seconds = time.monotonic() - epoch_started
        validation_started = time.monotonic()
        validation_metrics = validate_one_epoch(
            model,
            validation_loader,
            action_vocabulary,
            evaluation_action_pairs,
            verb_prior,
            noun_prior,
            len(validation_loader.dataset),
            config,
            context,
        )
        validation_seconds = time.monotonic() - validation_started
        score = float(validation_metrics["action_mt5r"])
        improved = score > best_metric

        if improved:
            best_metric = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch,
            **train_metrics,
            **validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best_action_mt5r": best_metric,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "train_events_per_second": train_metrics["train_events"] / train_seconds,
            "validation_events_per_second": validation_metrics["validation_events"] / validation_seconds,
            "rank_peak_reserved_gpu_bytes": (
                torch.cuda.max_memory_reserved(context.device)
                if context.device.type == "cuda" else 0
            ),
        }
        history.append(epoch_record)
        completed_epoch = epoch
        rng_states = gather_rng_states(context)

        if context.is_main:
            save_arguments = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "epoch": epoch,
                "optimizer_step": optimizer_step,
                "resolved_config": resolved_config,
                "backbone_provenance": backbone_provenance,
                "split_hashes": split_hashes,
                "action_vocabulary": action_vocabulary,
                "best_metric": best_metric,
                "epochs_without_improvement": (
                    epochs_without_improvement
                ),
                "rng_states": rng_states,
                "repository_root": repository_root,
            }
            save_checkpoint(
                checkpoint_dir / "last.pt",
                **save_arguments,
            )

            if improved:
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    **save_arguments,
                )

            write_json_atomic(history_path, history)
            print(json.dumps({"event": "epoch_complete", **epoch_record}), flush=True)

        if dist.is_initialized():
            dist.barrier()

        # End cleanly at an epoch boundary if another epoch is unlikely to fit.
        # No mid-epoch checkpoint is claimed; a first epoch must fit the job limit.
        stop_for_walltime = False
        if context.is_main and os.environ.get("SLURM_JOB_END_TIME"):
            remaining = float(os.environ["SLURM_JOB_END_TIME"]) - time.time()
            stop_for_walltime = remaining < 1.2 * (time.monotonic() - epoch_started) + 600
        if dist.is_initialized():
            payload = [stop_for_walltime]
            dist.broadcast_object_list(payload, src=0)
            stop_for_walltime = payload[0]
        if stop_for_walltime:
            if context.is_main:
                print("Stopping at completed epoch before Slurm walltime; resume last.pt", flush=True)
            break

        if (
            epochs_without_improvement
            >= config.early_stopping_patience
        ):
            break

    return {
        "best_action_mt5r": best_metric,
        "completed_epoch": completed_epoch,
        "optimizer_step": optimizer_step,
        "history": history,
        "best_checkpoint": checkpoint_dir / "best.pt",
        "last_checkpoint": checkpoint_dir / "last.pt",
    }
