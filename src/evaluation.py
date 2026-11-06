"""Evaluation utilities shared by EPIC-KITCHENS AR and STA models."""

from __future__ import annotations

import numpy as np


def _validate_classification_inputs(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize classification labels and score arrays."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores)

    if labels.ndim != 1:
        raise ValueError(f"labels must be one-dimensional, got {labels.shape}")
    if scores.ndim != 2:
        raise ValueError(f"scores must be two-dimensional, got {scores.shape}")
    if len(labels) != len(scores):
        raise ValueError(
            f"labels and scores contain different sample counts: "
            f"{len(labels)} and {len(scores)}"
        )
    if not 1 <= k <= scores.shape[1]:
        raise ValueError(f"k must be between 1 and {scores.shape[1]}, got {k}")
    if labels.size and (labels.min() < 0 or labels.max() >= scores.shape[1]):
        raise ValueError("A label is outside the score-vector class range")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinite values")
    return labels, scores


def top_k_hits(labels: np.ndarray, scores: np.ndarray, k: int = 5) -> np.ndarray:
    """Return one Boolean indicating a Top-k hit for every sample."""
    labels, scores = _validate_classification_inputs(labels, scores, k)
    top_k = np.argpartition(scores, -k, axis=1)[:, -k:]
    return np.any(top_k == labels[:, None], axis=1)


def top_k_accuracy(labels: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    """Return ordinary Top-k accuracy across samples."""
    hits = top_k_hits(labels, scores, k)
    if not len(hits):
        raise ValueError("At least one sample is required")
    return float(hits.mean())


def class_mean_top_k_recall(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int = 5,
) -> float:
    """Return mean per-class Top-k recall over classes present in ``labels``.

    This is the MT5R definition used for the course STA task when ``k=5``.
    Each represented class contributes equally, even if some classes have many
    more validation examples than others.
    """
    labels, scores = _validate_classification_inputs(labels, scores, k)
    if not len(labels):
        raise ValueError("At least one sample is required")

    hits = top_k_hits(labels, scores, k)
    recalls = [hits[labels == class_id].mean() for class_id in np.unique(labels)]
    return float(np.mean(recalls))


def build_action_pairs(
    verb_labels: np.ndarray,
    noun_labels: np.ndarray,
) -> np.ndarray:
    """Return the sorted unique verb-noun pairs defining valid actions."""
    verb_labels = np.asarray(verb_labels, dtype=np.int64)
    noun_labels = np.asarray(noun_labels, dtype=np.int64)
    if verb_labels.ndim != 1 or noun_labels.ndim != 1:
        raise ValueError("verb_labels and noun_labels must be one-dimensional")
    if len(verb_labels) != len(noun_labels):
        raise ValueError("verb_labels and noun_labels must have equal length")
    if not len(verb_labels):
        raise ValueError("At least one action label pair is required")
    return np.unique(np.column_stack((verb_labels, noun_labels)), axis=0)


def action_labels_from_pairs(
    verb_labels: np.ndarray,
    noun_labels: np.ndarray,
    action_pairs: np.ndarray,
) -> np.ndarray:
    """Map each verb-noun label pair to its row in ``action_pairs``."""
    verb_labels = np.asarray(verb_labels, dtype=np.int64)
    noun_labels = np.asarray(noun_labels, dtype=np.int64)
    action_pairs = np.asarray(action_pairs, dtype=np.int64)
    if action_pairs.ndim != 2 or action_pairs.shape[1] != 2:
        raise ValueError("action_pairs must have shape [action_count, 2]")
    if len(verb_labels) != len(noun_labels):
        raise ValueError("verb_labels and noun_labels must have equal length")

    pair_to_id = {tuple(pair): index for index, pair in enumerate(action_pairs)}
    try:
        return np.asarray(
            [
                pair_to_id[(int(verb), int(noun))]
                for verb, noun in zip(verb_labels, noun_labels)
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(f"Unknown verb-noun action pair: {error.args[0]}") from error


def joint_action_scores(
    verb_probabilities: np.ndarray,
    noun_probabilities: np.ndarray,
    action_pairs: np.ndarray,
) -> np.ndarray:
    """Combine verb and noun probabilities for the supplied valid actions.

    The score for action ``(verb, noun)`` is ``P(verb) * P(noun)``. Restricting
    the output to known action pairs avoids ranking impossible combinations.
    """
    verb_probabilities = np.asarray(verb_probabilities, dtype=np.float64)
    noun_probabilities = np.asarray(noun_probabilities, dtype=np.float64)
    action_pairs = np.asarray(action_pairs, dtype=np.int64)

    if verb_probabilities.ndim != 2 or noun_probabilities.ndim != 2:
        raise ValueError("Verb and noun probabilities must be two-dimensional")
    if len(verb_probabilities) != len(noun_probabilities):
        raise ValueError("Verb and noun probabilities need equal sample counts")
    if action_pairs.ndim != 2 or action_pairs.shape[1] != 2:
        raise ValueError("action_pairs must have shape [action_count, 2]")
    if np.any(verb_probabilities < 0) or np.any(noun_probabilities < 0):
        raise ValueError("Probabilities must be non-negative")
    if not np.isfinite(verb_probabilities).all() or not np.isfinite(
        noun_probabilities
    ).all():
        raise ValueError("Probabilities contain NaN or infinite values")
    if action_pairs[:, 0].max() >= verb_probabilities.shape[1]:
        raise ValueError("An action pair references an unavailable verb class")
    if action_pairs[:, 1].max() >= noun_probabilities.shape[1]:
        raise ValueError("An action pair references an unavailable noun class")

    return (
        verb_probabilities[:, action_pairs[:, 0]]
        * noun_probabilities[:, action_pairs[:, 1]]
    )

