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


class ClipTransform:
    def __init__(
        self,
        *,
        training: bool,
        seed: int = 42,
    ):
        self.training = training
        self.seed = seed

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

            for _ in range(10):
                area = height * width * rng.uniform(0.5, 1.0)

                ratio = math.exp(
                    rng.uniform(
                        math.log(3 / 4),
                        math.log(4 / 3),
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

                if source_ratio < 3 / 4:
                    crop_width = width
                    crop_height = round(width / (3 / 4))
                elif source_ratio > 4 / 3:
                    crop_height = height
                    crop_width = round(height * (4 / 3))
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
                antialias=True,
            )

            # One flip decision applied to the entire clip.
            if rng.random() < 0.5:
                x = F.hflip(x)

        else:
            x = F.resize(
                x,
                VALIDATION_SHORT_SIDE,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

            x = F.center_crop(
                x,
                [CROP_SIZE, CROP_SIZE],
            )

        x = F.normalize(x, mean=MEAN, std=STD)

        # Final contract: [C,T,H,W].
        return x.permute(1, 0, 2, 3).contiguous()