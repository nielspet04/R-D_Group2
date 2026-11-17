"""STA2 backbone/head wrapper, action vocabulary, and masked losses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.sta2.head import AnticipationHead


UNKNOWN_ACTION_LABEL = -1
ActionVocabulary = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class STA2Losses:
    """Masked STA2 losses and the counts used to normalize training."""

    total_loss: Tensor
    verb_loss: Tensor
    noun_loss: Tensor
    action_loss: Tensor
    usable_count: int
    action_count: int

    @property
    def should_update(self) -> bool:
        """Whether this local batch contains supervised primary events."""

        return self.usable_count > 0


def _validate_label_vector(
    name: str,
    labels: Tensor,
    batch_size: int,
    device: torch.device,
) -> None:
    if labels.ndim != 1 or labels.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [{batch_size}]")

    if labels.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long")

    if labels.device != device:
        raise ValueError(f"{name} must be on the logits device")


def build_action_vocabulary(
    verb_labels: Tensor,
    noun_labels: Tensor,
) -> ActionVocabulary:
    """Return unique training action pairs in lexicographic order."""

    if verb_labels.ndim != 1 or noun_labels.ndim != 1:
        raise ValueError("verb_labels and noun_labels must be one-dimensional")

    if verb_labels.shape != noun_labels.shape:
        raise ValueError(
            "verb_labels and noun_labels must have the same shape"
        )

    if verb_labels.dtype != torch.long:
        raise TypeError("verb_labels must have dtype torch.long")

    if noun_labels.dtype != torch.long:
        raise TypeError("noun_labels must have dtype torch.long")

    if verb_labels.numel() == 0:
        raise ValueError("at least one training action pair is required")

    if (verb_labels < 0).any().item() or (noun_labels < 0).any().item():
        raise ValueError("training labels must be non-negative")

    pairs = zip(
        verb_labels.detach().cpu().tolist(),
        noun_labels.detach().cpu().tolist(),
    )
    return tuple(sorted(set(pairs)))


def _validate_action_vocabulary(
    action_vocabulary: Sequence[tuple[int, int]],
) -> ActionVocabulary:
    vocabulary = tuple(
        (int(verb), int(noun))
        for verb, noun in action_vocabulary
    )

    if not vocabulary:
        raise ValueError("action_vocabulary must not be empty")

    if any(verb < 0 or noun < 0 for verb, noun in vocabulary):
        raise ValueError("action vocabulary labels must be non-negative")

    if vocabulary != tuple(sorted(set(vocabulary))):
        raise ValueError(
            "action_vocabulary must contain unique pairs in sorted order"
        )

    return vocabulary


def encode_action_labels(
    verb_labels: Tensor,
    noun_labels: Tensor,
    action_vocabulary: Sequence[tuple[int, int]],
) -> Tensor:
    """Map verb/noun pairs to action indices, using -1 when unknown."""

    if verb_labels.ndim != 1 or noun_labels.ndim != 1:
        raise ValueError("verb_labels and noun_labels must be one-dimensional")

    if verb_labels.shape != noun_labels.shape:
        raise ValueError(
            "verb_labels and noun_labels must have the same shape"
        )

    if verb_labels.dtype != torch.long:
        raise TypeError("verb_labels must have dtype torch.long")

    if noun_labels.dtype != torch.long:
        raise TypeError("noun_labels must have dtype torch.long")

    if verb_labels.device != noun_labels.device:
        raise ValueError(
            "verb_labels and noun_labels must be on the same device"
        )

    vocabulary = _validate_action_vocabulary(action_vocabulary)
    pair_to_index = {
        pair: index
        for index, pair in enumerate(vocabulary)
    }

    verb_values = verb_labels.detach().cpu().tolist()
    noun_values = noun_labels.detach().cpu().tolist()

    encoded = [
        pair_to_index.get(
            (int(verb), int(noun)),
            UNKNOWN_ACTION_LABEL,
        )
        for verb, noun in zip(verb_values, noun_values)
    ]

    return torch.tensor(
        encoded,
        dtype=torch.long,
        device=verb_labels.device,
    )


def _masked_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> Tensor:
    """Compute FP32 mean CE, or a differentiable zero if empty."""

    if mask.any().item():
        return F.cross_entropy(
            logits[mask].float(),
            labels[mask],
            reduction="mean",
        )

    # Keep the zero connected to the logits so an invalid-only distributed
    # rank can participate when another rank still performs backpropagation.
    return logits.float().sum() * 0.0


def compute_sta2_losses(
    outputs: Mapping[str, Tensor],
    verb_labels: Tensor,
    noun_labels: Tensor,
    action_labels: Tensor,
    has_context: Tensor,
) -> STA2Losses:
    """Compute equally weighted masked verb, noun, and action CE."""

    required_keys = {
        "verb_logits",
        "noun_logits",
        "action_logits",
    }
    missing_keys = required_keys.difference(outputs)

    if missing_keys:
        raise KeyError(
            f"missing model outputs: {sorted(missing_keys)}"
        )

    verb_logits = outputs["verb_logits"]
    noun_logits = outputs["noun_logits"]
    action_logits = outputs["action_logits"]

    for name, logits in (
        ("verb_logits", verb_logits),
        ("noun_logits", noun_logits),
        ("action_logits", action_logits),
    ):
        if logits.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, classes]")

        if not logits.is_floating_point():
            raise TypeError(f"{name} must be floating point")

    batch_size = verb_logits.shape[0]
    device = verb_logits.device

    if noun_logits.shape[0] != batch_size:
        raise ValueError("noun_logits batch size does not match verb_logits")

    if action_logits.shape[0] != batch_size:
        raise ValueError(
            "action_logits batch size does not match verb_logits"
        )

    if noun_logits.device != device or action_logits.device != device:
        raise ValueError("all logits must be on the same device")

    _validate_label_vector(
        "verb_labels",
        verb_labels,
        batch_size,
        device,
    )
    _validate_label_vector(
        "noun_labels",
        noun_labels,
        batch_size,
        device,
    )
    _validate_label_vector(
        "action_labels",
        action_labels,
        batch_size,
        device,
    )

    if has_context.ndim != 1 or has_context.shape[0] != batch_size:
        raise ValueError(f"has_context must have shape [{batch_size}]")

    if has_context.dtype != torch.bool:
        raise TypeError("has_context must have dtype torch.bool")

    if has_context.device != device:
        raise ValueError("has_context must be on the logits device")

    usable_count = int(has_context.sum().item())

    if usable_count:
        usable_verbs = verb_labels[has_context]
        usable_nouns = noun_labels[has_context]
        usable_actions = action_labels[has_context]

        if (
            (usable_verbs < 0).any().item()
            or (usable_verbs >= verb_logits.shape[1]).any().item()
        ):
            raise ValueError("usable verb label is outside the class range")

        if (
            (usable_nouns < 0).any().item()
            or (usable_nouns >= noun_logits.shape[1]).any().item()
        ):
            raise ValueError("usable noun label is outside the class range")

        if (
            (usable_actions < UNKNOWN_ACTION_LABEL).any().item()
            or (usable_actions >= action_logits.shape[1]).any().item()
        ):
            raise ValueError(
                "usable action labels must be -1 or a valid action index"
            )

    action_mask = has_context & (
        action_labels != UNKNOWN_ACTION_LABEL
    )
    action_count = int(action_mask.sum().item())

    verb_loss = _masked_cross_entropy(
        verb_logits,
        verb_labels,
        has_context,
    )
    noun_loss = _masked_cross_entropy(
        noun_logits,
        noun_labels,
        has_context,
    )
    action_loss = _masked_cross_entropy(
        action_logits,
        action_labels,
        action_mask,
    )

    total_loss = verb_loss + noun_loss + action_loss

    return STA2Losses(
        total_loss=total_loss,
        verb_loss=verb_loss,
        noun_loss=noun_loss,
        action_loss=action_loss,
        usable_count=usable_count,
        action_count=action_count,
    )


class STA2Model(nn.Module):
    """Connect a frozen anticipation backbone to the trainable head."""

    def __init__(
        self,
        backbone: nn.Module | None,
        head: AnticipationHead,
    ) -> None:
        super().__init__()

        embed_dim = getattr(head, "embed_dim", None)
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError("head must expose a positive embed_dim")

        if backbone is not None:
            backbone_embed_dim = getattr(backbone, "embed_dim", None)
            if backbone_embed_dim != embed_dim:
                raise ValueError(
                    "backbone and head embedding dimensions must match"
                )

        self.backbone = backbone
        self.head = head
        self.embed_dim = embed_dim

        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        if self.backbone is None:
            return

        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "STA2Model":
        """Train the head while keeping the backbone in evaluation mode."""

        super().train(mode)
        self.head.train(mode)

        if self.backbone is not None:
            self.backbone.eval()

        return self

    def _head_reference_parameter(self) -> Tensor:
        try:
            return next(self.head.parameters())
        except StopIteration as error:
            raise ValueError(
                "head must contain at least one parameter"
            ) from error

    def _prepare_usable_tokens(
        self,
        backbone_outputs: Mapping[str, Tensor],
        has_context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        missing = {
            "observed_tokens",
            "future_tokens",
        }.difference(backbone_outputs)

        if missing:
            raise KeyError(
                f"missing backbone outputs: {sorted(missing)}"
            )

        observed = backbone_outputs["observed_tokens"]
        future = backbone_outputs["future_tokens"]

        for name, tokens in (
            ("observed_tokens", observed),
            ("future_tokens", future),
        ):
            if tokens.ndim != 3:
                raise ValueError(
                    f"{name} must have shape [batch, tokens, embed_dim]"
                )

            if tokens.shape[-1] != self.embed_dim:
                raise ValueError(
                    f"{name} embedding dimension does not match the head"
                )

            if not tokens.is_floating_point():
                raise TypeError(f"{name} must be floating point")

        if observed.shape[0] != future.shape[0]:
            raise ValueError(
                "observed and future token batch sizes must match"
            )

        if observed.device != future.device:
            raise ValueError(
                "observed and future tokens must be on the same device"
            )

        if observed.dtype != future.dtype:
            raise ValueError(
                "observed and future tokens must have the same dtype"
            )

        full_batch_size = has_context.shape[0]
        usable_count = int(has_context.sum().item())
        output_batch_size = observed.shape[0]

        if output_batch_size == full_batch_size:
            observed = observed[has_context]
            future = future[has_context]
        elif output_batch_size != usable_count:
            raise ValueError(
                "backbone outputs must contain either the full batch "
                "or only usable events"
            )

        if observed.shape[1] + future.shape[1] == 0:
            raise ValueError(
                "usable events require at least one observed or future token"
            )

        if not torch.isfinite(observed).all().item():
            raise ValueError("usable observed tokens contain nonfinite values")

        if not torch.isfinite(future).all().item():
            raise ValueError("usable future tokens contain nonfinite values")

        return observed.detach(), future.detach()

    def forward(
        self,
        video: Tensor,
        has_context: Tensor,
        *,
        backbone_outputs: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Run the backbone on usable events and the head on all events.

        ``backbone_outputs`` permits CPU development with the synthetic
        fixture before the real frozen backbone is available. It may contain
        either the full batch or only the usable rows.
        """

        if video.ndim != 5:
            raise ValueError(
                "video must have shape [batch, channels, frames, height, width]"
            )

        if not video.is_floating_point():
            raise TypeError("video must be floating point")

        batch_size = video.shape[0]
        if batch_size == 0:
            raise ValueError("video batch must not be empty")

        if has_context.shape != (batch_size,):
            raise ValueError(
                f"has_context must have shape [{batch_size}]"
            )

        if has_context.dtype != torch.bool:
            raise TypeError("has_context must have dtype torch.bool")

        if has_context.device != video.device:
            raise ValueError(
                "has_context and video must be on the same device"
            )

        head_reference = self._head_reference_parameter()
        if head_reference.device != video.device:
            raise ValueError(
                "video and head must be on the same device"
            )

        usable_count = int(has_context.sum().item())

        if usable_count:
            if backbone_outputs is None:
                if self.backbone is None:
                    raise ValueError(
                        "a backbone or synthetic backbone_outputs is required"
                    )

                usable_video = video[has_context]
                anticipation_seconds = torch.ones(
                    usable_count,
                    dtype=torch.float32,
                    device=video.device,
                )

                with torch.no_grad():
                    backbone_outputs = self.backbone(
                        usable_video,
                        anticipation_seconds,
                    )

            if not isinstance(backbone_outputs, Mapping):
                raise TypeError(
                    "backbone must return a mapping of token tensors"
                )

            usable_observed, usable_future = (
                self._prepare_usable_tokens(
                    backbone_outputs,
                    has_context,
                )
            )

            if usable_observed.device != head_reference.device:
                raise ValueError(
                    "backbone tokens and head must be on the same device"
                )

            observed_tokens = usable_observed.new_zeros(
                batch_size,
                usable_observed.shape[1],
                self.embed_dim,
            )
            future_tokens = usable_future.new_zeros(
                batch_size,
                usable_future.shape[1],
                self.embed_dim,
            )

            observed_tokens[has_context] = usable_observed
            future_tokens[has_context] = usable_future
        else:
            # One finite observed token keeps the head input legal. The
            # empty future tensor also represents observed-only operation.
            observed_tokens = torch.zeros(
                batch_size,
                1,
                self.embed_dim,
                dtype=head_reference.dtype,
                device=head_reference.device,
            )
            future_tokens = torch.empty(
                batch_size,
                0,
                self.embed_dim,
                dtype=head_reference.dtype,
                device=head_reference.device,
            )

        return self.head(
            observed_tokens,
            future_tokens,
        )
