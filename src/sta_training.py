"""Shared training utilities for STA Models 1 and 2."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
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
