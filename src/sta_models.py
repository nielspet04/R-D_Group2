"""Simple Short-Term Action Anticipation baseline models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalized_class_frequencies(labels: np.ndarray, class_count: int) -> np.ndarray:
    """Convert integer labels into a probability vector."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("labels must be a non-empty one-dimensional array")
    if class_count <= 0:
        raise ValueError("class_count must be positive")
    if labels.min() < 0 or labels.max() >= class_count:
        raise ValueError(f"A label is outside the range 0..{class_count - 1}")

    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    return counts / counts.sum()


@dataclass
class FrequencyPriorSTA:
    """Predict the training-set class frequencies for every STA event."""

    verb_probabilities: np.ndarray
    noun_probabilities: np.ndarray

    @classmethod
    def fit(
        cls,
        verb_labels: np.ndarray,
        noun_labels: np.ndarray,
        num_verbs: int = 97,
        num_nouns: int = 300,
    ) -> "FrequencyPriorSTA":
        """Estimate verb and noun probabilities from training labels."""
        return cls(
            verb_probabilities=_normalized_class_frequencies(
                verb_labels, num_verbs
            ),
            noun_probabilities=_normalized_class_frequencies(
                noun_labels, num_nouns
            ),
        )

    def predict(self, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
        """Return independent copies of the priors for ``sample_count`` events."""
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        return (
            np.tile(self.verb_probabilities, (sample_count, 1)),
            np.tile(self.noun_probabilities, (sample_count, 1)),
        )

