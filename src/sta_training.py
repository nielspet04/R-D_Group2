"""Shared training utilities for STA Models 1 and 2."""

from __future__ import annotations

import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.evaluation import (
    action_labels_from_pairs,
    class_mean_top_k_recall,
    joint_action_scores,
    top_k_accuracy,
)
from src.sta_model_0 import FrequencyPriorSTA
from src.sta_model_1 import MaskedMeanPoolSTA
from src.sta_model_2 import MaskedGRUSTA


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainingConfig:
    """Fixed shared configuration for the initial STA neural baselines."""

    model: str = "mean_pool"
    seed: int = 42

    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    train_batch_size: int = 256
    validation_batch_size: int = 512

    max_epochs: int = 30
    early_stopping_patience: int = 5
    gradient_clip_norm: float = 1.0

    feature_size: int = 512
    hidden_size: int = 512
    num_verbs: int = 97
    num_nouns: int = 300
    dropout: float = 0.3

    def __post_init__(self) -> None:
        if self.model not in {"mean_pool", "gru"}:
            raise ValueError("model must be 'mean_pool' or 'gru'")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""
        return asdict(self)


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch before creating models or loaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(config: TrainingConfig) -> nn.Module:
    """Construct the requested model with shared classification settings."""
    common = {
        "feature_size": config.feature_size,
        "num_verbs": config.num_verbs,
        "num_nouns": config.num_nouns,
        "dropout": config.dropout,
    }

    if config.model == "mean_pool":
        return MaskedMeanPoolSTA(**common)

    return MaskedGRUSTA(
        **common,
        hidden_size=config.hidden_size,
    )


def build_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> torch.optim.AdamW:
    """Create the shared AdamW optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def masked_classification_loss(
    verb_logits: Tensor,
    noun_logits: Tensor,
    verb_labels: Tensor,
    noun_labels: Tensor,
    valid_mask: Tensor,
) -> Tensor | None:
    """Return combined cross-entropy over events with usable context.

    All events remain in the batch, but events whose masks are entirely false
    are excluded from both the verb and noun losses. If the whole batch lacks
    usable context, ``None`` signals that no optimizer update should occur.
    """
    has_context = valid_mask.any(dim=1)

    if not has_context.any().item():
        return None

    verb_loss = F.cross_entropy(
        verb_logits[has_context],
        verb_labels[has_context],
    )
    noun_loss = F.cross_entropy(
        noun_logits[has_context],
        noun_labels[has_context],
    )

    return verb_loss + noun_loss


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip_norm: float = 1.0,
) -> dict[str, int | float | None]:
    """Train for one epoch and report context-aware event accounting."""
    model.train()

    total_events = 0
    usable_events = 0
    skipped_batches = 0
    optimizer_steps = 0
    loss_sum = 0.0

    for batch in loader:
        features = batch["features"].to(
            device=device,
            dtype=torch.float32,
        )
        valid_mask = batch["valid_mask"].to(device=device)
        verb_labels = batch["verb_label"].to(
            device=device,
            dtype=torch.long,
        )
        noun_labels = batch["noun_label"].to(
            device=device,
            dtype=torch.long,
        )

        batch_events = features.shape[0]
        batch_usable = int(valid_mask.any(dim=1).sum().item())

        total_events += batch_events
        usable_events += batch_usable

        optimizer.zero_grad(set_to_none=True)

        if batch_usable == 0:
            skipped_batches += 1
            continue

        verb_logits, noun_logits = model(features, valid_mask)
        loss = masked_classification_loss(
            verb_logits,
            noun_logits,
            verb_labels,
            noun_labels,
            valid_mask,
        )

        assert loss is not None

        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite STA training loss")

        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()

        optimizer_steps += 1
        loss_sum += loss.detach().item() * batch_usable

    if total_events == 0:
        raise ValueError("Training loader contains no events")

    return {
        "train_loss": loss_sum / usable_events if usable_events else None,
        "train_events": total_events,
        "train_usable_events": usable_events,
        "train_excluded_events": total_events - usable_events,
        "train_skipped_batches": skipped_batches,
        "train_optimizer_steps": optimizer_steps,
    }


def validate_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    frequency_prior: FrequencyPriorSTA,
    action_pairs: np.ndarray,
) -> dict[str, int | float | None]:
    """Evaluate all events, using Model 0 when video context is absent."""
    model.eval()

    total_events = 0
    usable_events = 0
    fallback_events = 0
    loss_sum = 0.0

    all_verb_labels: list[np.ndarray] = []
    all_noun_labels: list[np.ndarray] = []
    all_verb_scores: list[np.ndarray] = []
    all_noun_scores: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(
                device=device,
                dtype=torch.float32,
            )
            valid_mask = batch["valid_mask"].to(device=device)
            verb_labels = batch["verb_label"].to(
                device=device,
                dtype=torch.long,
            )
            noun_labels = batch["noun_label"].to(
                device=device,
                dtype=torch.long,
            )

            batch_events = features.shape[0]
            has_context = valid_mask.any(dim=1)
            batch_usable = int(has_context.sum().item())
            batch_fallback = batch_events - batch_usable

            total_events += batch_events
            usable_events += batch_usable
            fallback_events += batch_fallback

            # Start with Model 0 predictions for the complete batch. Neural
            # probabilities replace them only where usable context exists.
            verb_scores, noun_scores = frequency_prior.predict(batch_events)

            if batch_usable:
                verb_logits, noun_logits = model(features, valid_mask)
                loss = masked_classification_loss(
                    verb_logits,
                    noun_logits,
                    verb_labels,
                    noun_labels,
                    valid_mask,
                )
                assert loss is not None

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        "Non-finite STA validation loss"
                    )

                loss_sum += loss.item() * batch_usable

                neural_verb_scores = torch.softmax(
                    verb_logits[has_context],
                    dim=1,
                ).cpu().numpy()
                neural_noun_scores = torch.softmax(
                    noun_logits[has_context],
                    dim=1,
                ).cpu().numpy()

                has_context_cpu = has_context.cpu().numpy()
                verb_scores[has_context_cpu] = neural_verb_scores
                noun_scores[has_context_cpu] = neural_noun_scores

            all_verb_labels.append(verb_labels.cpu().numpy())
            all_noun_labels.append(noun_labels.cpu().numpy())
            all_verb_scores.append(verb_scores)
            all_noun_scores.append(noun_scores)

    if total_events == 0:
        raise ValueError("Validation loader contains no events")

    verb_labels_array = np.concatenate(all_verb_labels)
    noun_labels_array = np.concatenate(all_noun_labels)
    verb_scores_array = np.concatenate(all_verb_scores)
    noun_scores_array = np.concatenate(all_noun_scores)

    action_labels = action_labels_from_pairs(
        verb_labels_array,
        noun_labels_array,
        action_pairs,
    )
    action_scores = joint_action_scores(
        verb_scores_array,
        noun_scores_array,
        action_pairs,
    )

    return {
        "validation_loss": (
            loss_sum / usable_events if usable_events else None
        ),
        "validation_events": total_events,
        "validation_usable_events": usable_events,
        "validation_fallback_events": fallback_events,
        "verb_top1_accuracy": top_k_accuracy(
            verb_labels_array, verb_scores_array, k=1
        ),
        "verb_top5_accuracy": top_k_accuracy(
            verb_labels_array, verb_scores_array, k=5
        ),
        "verb_mt5r": class_mean_top_k_recall(
            verb_labels_array, verb_scores_array, k=5
        ),
        "noun_top1_accuracy": top_k_accuracy(
            noun_labels_array, noun_scores_array, k=1
        ),
        "noun_top5_accuracy": top_k_accuracy(
            noun_labels_array, noun_scores_array, k=5
        ),
        "noun_mt5r": class_mean_top_k_recall(
            noun_labels_array, noun_scores_array, k=5
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


def get_git_provenance(
    repository_root: Path | None = None,
) -> dict[str, str | bool]:
    """Return the current Git commit and working-tree state."""
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[1]

    repository_root = Path(repository_root)
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )

    return {
        "git_commit": commit_result.stdout.strip(),
        "git_dirty": bool(status_result.stdout.strip()),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainingConfig,
    validation_metrics: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
) -> Path:
    """Atomically save a reproducible STA training checkpoint."""
    if epoch < 1:
        raise ValueError("epoch must be at least 1")

    # Read Git state before creating output files so the checkpoint cannot
    # affect its own recorded dirty-tree status.
    provenance = get_git_provenance(repository_root)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "seed": config.seed,
        "configuration": config.to_dict(),
        "validation_metrics": dict(validation_metrics),
        **provenance,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)
    return path


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore model and optimizer state."""
    path = Path(path)
    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=map_location,
        )

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")

    required_fields = {
        "schema_version",
        "model_state",
        "optimizer_state",
        "epoch",
        "seed",
        "configuration",
        "validation_metrics",
        "git_commit",
        "git_dirty",
    }
    missing_fields = required_fields.difference(checkpoint)
    if missing_fields:
        raise ValueError(
            f"Checkpoint is missing fields: {sorted(missing_fields)}"
        )

    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported checkpoint schema version "
            f"{checkpoint['schema_version']!r}"
        )

    if model is not None:
        model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        if model is None:
            raise ValueError(
                "A model must be provided when restoring an optimizer"
            )
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint
