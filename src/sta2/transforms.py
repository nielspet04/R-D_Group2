"""Reproducible spatial preprocessing shared across every frame."""

import math

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

CROP_SIZE = 384
VALIDATION_SHORT_SIDE = 438


def resolve_preprocessing(settings=None):
    """Resolve supported settings so recorded configuration matches execution."""
    resolved = {
        "crop_size": CROP_SIZE,
        "validation_short_side": VALIDATION_SHORT_SIDE,
        "crop_scale": [0.5, 1.0],
        "crop_ratio": [3 / 4, 4 / 3],
        "flip_probability": 0.5,
        "mean": list(MEAN),
        "std": list(STD),
        "interpolation": "bilinear",
        "antialias": True,
    }
    if settings is not None:
        unknown = set(settings) - set(resolved)
        if unknown:
            raise ValueError(f"Unknown preprocessing settings: {sorted(unknown)}")
        resolved.update(settings)
    if resolved["crop_size"] != 384:
        raise ValueError("The production backbone contract requires crop_size=384")
    short_side = resolved["validation_short_side"]
    if not isinstance(short_side, int) or short_side < 384:
        raise ValueError("validation_short_side must be an integer >= 384")
    for name in ("crop_scale", "crop_ratio"):
        values = list(resolved[name])
        if len(values) != 2 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Invalid {name}")
        if not 0 < values[0] <= values[1]:
            raise ValueError(f"Invalid {name}")
        if name == "crop_scale" and values[1] > 1:
            raise ValueError("Crop scale cannot exceed image area")
        resolved[name] = values
    for name in ("mean", "std"):
        values = list(resolved[name])
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Invalid {name}")
        if name == "std" and any(v <= 0 for v in values):
            raise ValueError("Standard deviations must be positive")
        resolved[name] = values
    if not 0 <= resolved["flip_probability"] <= 1:
        raise ValueError("flip_probability must be in [0,1]")
    if resolved["interpolation"] != "bilinear":
        raise ValueError("Only bilinear interpolation is currently supported")
    if not isinstance(resolved["antialias"], bool):
        raise ValueError("antialias must be a boolean")
    return resolved


class ClipTransform:
    def __init__(
        self,
        *,
        training: bool,
        seed: int = 42,
        preprocessing=None,
    ):
        self.training = training
        self.seed = seed
        self.preprocessing = resolve_preprocessing(preprocessing)

    def __call__(
        self,
        rgb: np.ndarray,
        *,
        epoch: int,
        row_index: int,
    ) -> torch.Tensor:
        if (
            rgb.ndim != 4
            or rgb.shape[0] != 32
            or rgb.shape[-1] != 3
            or rgb.dtype != np.uint8
        ):
            raise ValueError("Expected uint8 RGB [32,H,W,3]")

        # [T,H,W,C] -> [T,C,H,W], with pixels in [0,1].
        x = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
        )

        if self.training:
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, epoch, row_index]
                )
            )

            height, width = x.shape[-2:]
            rectangle = None
            scale_min, scale_max = self.preprocessing["crop_scale"]
            ratio_min, ratio_max = self.preprocessing["crop_ratio"]

            for _ in range(10):
                area = height * width * rng.uniform(scale_min, scale_max)

                ratio = math.exp(
                    rng.uniform(
                        math.log(ratio_min),
                        math.log(ratio_max),
                    )
                )

                crop_width = round(math.sqrt(area * ratio))
                crop_height = round(math.sqrt(area / ratio))

                if (
                    0 < crop_width <= width
                    and 0 < crop_height <= height
                ):
                    top = int(
                        rng.integers(height - crop_height + 1)
                    )
                    left = int(
                        rng.integers(width - crop_width + 1)
                    )

                    rectangle = (
                        top,
                        left,
                        crop_height,
                        crop_width,
                    )
                    break

            # Standard center-crop fallback if ten random attempts fail.
            if rectangle is None:
                source_ratio = width / height

                if source_ratio < ratio_min:
                    crop_width = width
                    crop_height = round(width / ratio_min)
                elif source_ratio > ratio_max:
                    crop_height = height
                    crop_width = round(height * ratio_max)
                else:
                    crop_height = height
                    crop_width = width

                rectangle = (
                    (height - crop_height) // 2,
                    (width - crop_width) // 2,
                    crop_height,
                    crop_width,
                )

            # One rectangle applied to the entire clip.
            x = F.resized_crop(
                x,
                *rectangle,
                size=[CROP_SIZE, CROP_SIZE],
                interpolation=InterpolationMode.BILINEAR,
                antialias=self.preprocessing["antialias"],
            )

            # One flip decision applied to the entire clip.
            if rng.random() < self.preprocessing["flip_probability"]:
                x = F.hflip(x)

        else:
            x = F.resize(
                x,
                self.preprocessing["validation_short_side"],
                interpolation=InterpolationMode.BILINEAR,
                antialias=self.preprocessing["antialias"],
            )

            x = F.center_crop(
                x,
                [CROP_SIZE, CROP_SIZE],
            )

        x = F.normalize(
            x,
            mean=self.preprocessing["mean"],
            std=self.preprocessing["std"],
        )

        # Final contract: [C,T,H,W].
        return x.permute(1, 0, 2, 3).contiguous()
