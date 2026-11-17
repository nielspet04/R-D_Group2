"""Frozen V-JEPA 2.1 encoder/predictor adapter."""

from contextlib import nullcontext

import torch
from torch import nn


def load_strict(module, supplied):
    cleaned = {}

    for original, tensor in supplied.items():
        name = original

        # Only remove documented leading namespace prefixes.
        while name.startswith(("module.", "backbone.")):
            name = name.split(".", 1)[1]

        if name in cleaned:
            raise ValueError(
                f"Prefix collision: {original} -> {name}"
            )

        cleaned[name] = tensor

    expected = module.state_dict()

    missing = sorted(
        expected.keys() - cleaned.keys()
    )
    unexpected = sorted(
        cleaned.keys() - expected.keys()
    )

    shape_errors = {
        name: {
            "expected": tuple(expected[name].shape),
            "actual": tuple(cleaned[name].shape),
        }
        for name in expected.keys() & cleaned.keys()
        if expected[name].shape != cleaned[name].shape
    }

    if missing or unexpected or shape_errors:
        raise ValueError(
            "Checkpoint mismatch:\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}\n"
            f"shape_errors={shape_errors}"
        )

    module.load_state_dict(cleaned, strict=True)


class FrozenAnticipationBackbone(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        *,
        observed_only=False,
        bf16=True,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor

        self.embed_dim = int(encoder.embed_dim)
        self.observed_only = observed_only
        self.bf16 = bf16

        self.grid_size = (
            384 // int(encoder.patch_size)
        )
        self.tubelet_size = int(
            encoder.tubelet_size
        )

        if self.tubelet_size != 2:
            raise ValueError(
                "This adapter expects tubelet size two"
            )

        # Predictor consumes hierarchical features when required.
        self.encoder.return_hierarchical = (
            len(
                getattr(
                    predictor,
                    "hierarchical_layers",
                    [],
                )
            )
            > 1
        )

        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        config,
        *,
        observed_only=False,
        bf16=True,
    ):
        from .vendor.vjepa21.models.vision_transformer import (
            vit_gigantic_xformers,
        )
        from .vendor.vjepa21.models.predictor import (
            vit_predictor,
        )

        settings = config["backbone"]

        encoder = vit_gigantic_xformers(
            **settings["encoder"]
        )

        predictor = vit_predictor(
            embed_dim=encoder.embed_dim,
            **settings["predictor"],
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        load_strict(
            encoder,
            checkpoint[settings["encoder_key"]],
        )

        load_strict(
            predictor,
            checkpoint[settings["predictor_key"]],
        )

        return cls(
            encoder,
            predictor,
            observed_only=observed_only,
            bf16=bf16,
        )

    def train(self, mode=True):
        # An enclosing model.train() must not unfreeze behavior.
        super().train(False)
        return self

    def target_positions(
        self,
        observed_count,
        anticipation_seconds,
    ):
        anticipation_steps = (
            anticipation_seconds
            * 8
            / self.tubelet_size
        ).to(torch.long)

        tokens_per_tubelet = self.grid_size ** 2

        first_positions = (
            observed_count
            + tokens_per_tubelet * anticipation_steps
        )

        offsets = torch.arange(
            tokens_per_tubelet,
            device=anticipation_seconds.device,
        )

        positions = (
            first_positions[:, None]
            + offsets[None, :]
        )

        capacity = (
            self.predictor.num_frames
            // self.tubelet_size
            * tokens_per_tubelet
        )

        if torch.any(positions >= capacity):
            raise ValueError(
                "Horizon exceeds predictor temporal capacity"
            )

        return positions

    @torch.no_grad()
    def forward(
        self,
        video,
        anticipation_seconds,
    ):
        if (
            video.ndim != 5
            or tuple(video.shape[1:])
            != (3, 32, 384, 384)
        ):
            raise ValueError(
                "Expected [B,3,32,384,384]"
            )

        if tuple(anticipation_seconds.shape) != (
            len(video),
        ):
            raise ValueError(
                "anticipation_seconds must have shape [B]"
            )

        if (
            not torch.isfinite(
                anticipation_seconds
            ).all()
            or torch.any(anticipation_seconds < 0)
        ):
            raise ValueError(
                "Horizon must be finite and nonnegative"
            )

        if video.device != anticipation_seconds.device:
            raise ValueError(
                "Video and horizon must share a device"
            )

        if video.is_cuda and self.bf16:
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "BF16 requested but unsupported"
                )

            autocast = torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            )
        else:
            autocast = nullcontext()

        with autocast:
            hierarchy = self.encoder(video)

            if (
                hierarchy.ndim != 3
                or hierarchy.shape[-1] % self.embed_dim
            ):
                raise ValueError(
                    "Unexpected encoder representation"
                )

            observed = hierarchy[
                ..., -self.embed_dim:
            ]

            if self.observed_only:
                future = observed.new_empty(
                    len(video),
                    0,
                    self.embed_dim,
                )

            else:
                context_positions = torch.arange(
                    hierarchy.shape[1],
                    device=video.device,
                )[None].expand(len(video), -1)

                future_positions = self.target_positions(
                    hierarchy.shape[1],
                    anticipation_seconds,
                )

                prediction = self.predictor(
                    hierarchy,
                    masks_x=context_positions,
                    masks_y=future_positions,
                )

                full_future = (
                    prediction[0]
                    if isinstance(prediction, tuple)
                    else prediction
                )

                if (
                    full_future.ndim != 3
                    or full_future.shape[-1] % self.embed_dim
                ):
                    raise ValueError(
                        "Unexpected predictor representation"
                    )

                future = full_future[
                    ..., -self.embed_dim:
                ]

                if tuple(future.shape[:2]) != tuple(
                    future_positions.shape
                ):
                    raise ValueError(
                        "Wrong predicted token count"
                    )

        if (
            not torch.isfinite(observed).all()
            or not torch.isfinite(future).all()
        ):
            raise RuntimeError(
                "Nonfinite backbone representation"
            )

        return {
            "observed_tokens": observed,
            "future_tokens": future,
        }
        