"""PyTorch models for Short-Term Action Anticipation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

class MaskedMeanPoolSTA(nn.Module):
    """predict STA verbs and nouns from maskrf temporal video features"""

    def __init__(self, feature_size: int = 512, num_verbs: int = 97, num_nouns: int = 300, dropout: float = 0.3,
    )-> None:
        super().__init__()

        if feature_size <= 0:
            raise ValueError("feature_size must be positive")
        if num_verbs <= 0:
            raise ValueError("num_verbs must be positive")
        if num_nouns <= 0:
            raise ValueError("num_nouns must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        self.feature_size = feature_size
        self.dropout = nn.Dropout(dropout)
        self.verb_head = nn.Linear(feature_size, num_verbs)
        self.noun_head = nn.Linear(feature_size, num_nouns)

    def pool_features(self, features: Tensor, valid_mask: Tensor,
        ) -> Tensor:
        """average valid temporal features while ignoring padded segments"""
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, segments, feature_size]")

        if features.shape[-1] != self.feature_size:
            raise ValueError(
                f"expected feature size {self.feature_size}, "
                f"received {features.shape[-1]}"
            )

        if valid_mask.shape != features.shape[:2]:
            raise ValueError(
                "valid_mask must match the batch and segment dimensions"
            )

        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")

        if not features.is_floating_point():
            raise TypeError("features must be a floating-point tensor")

        mask = valid_mask.unsqueeze(-1).to(
            device = features.device, dtype = features.dtype
        )

        feature_sum = (features * mask).sum(dim=1)
        valid_count = mask.sum(dim=1).clamp(min=1)
        return feature_sum / valid_count

    def forward(self, features: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        """return verb and noun logits for every event"""
        pooled = self.pool_features(features, valid_mask)
        representation = self.dropout(pooled)

        verb_logits = self.verb_head(representation)
        noun_logits = self.noun_head(representation)

        return verb_logits, noun_logits
