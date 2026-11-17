import unittest

import numpy as np
import torch

from src.sta2.transforms import ClipTransform, resolve_preprocessing


class TransformTests(unittest.TestCase):
    def test_configuration_is_applied(self):
        rgb = np.full((32, 20, 30, 3), 255, dtype=np.uint8)
        transform = ClipTransform(training=False, preprocessing={
            "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
            "validation_short_side": 400,
        })
        output = transform(rgb, epoch=0, row_index=0)
        torch.testing.assert_close(output, torch.ones_like(output))
        self.assertEqual(transform.preprocessing["validation_short_side"], 400)

    def test_unsupported_configuration_is_rejected(self):
        for settings in (
            {"crop_size": 224}, {"validation_short_side": 200},
            {"std": [0, 1, 1]}, {"interpolation": "nearest"},
            {"crop_scale": [1.0, 0.5]}, {"unknown": True},
            {"flip_probability": 2},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                resolve_preprocessing(settings)

    def test_validation_shape_and_normalization(self):
        rgb = np.zeros(
            (32, 40, 60, 3),
            dtype=np.uint8,
        )
        rgb[..., 0] = 255

        transform = ClipTransform(training=False)

        output = transform(
            rgb,
            epoch=0,
            row_index=10,
        )

        self.assertEqual(
            output.shape,
            (3, 32, 384, 384),
        )

        expected = torch.tensor([
            (1 - 0.485) / 0.229,
            (0 - 0.456) / 0.224,
            (0 - 0.406) / 0.225,
        ])

        torch.testing.assert_close(
            output[:, 0, 0, 0],
            expected,
        )

    def test_training_is_clip_consistent_and_repeatable(self):
        rng = np.random.default_rng(1)

        frame = rng.integers(
            0,
            256,
            size=(60, 90, 3),
            dtype=np.uint8,
        )

        rgb = np.repeat(
            frame[None],
            32,
            axis=0,
        )

        transform = ClipTransform(
            training=True,
            seed=42,
        )

        first = transform(
            rgb,
            epoch=3,
            row_index=20,
        )
        second = transform(
            rgb,
            epoch=3,
            row_index=20,
        )

        torch.testing.assert_close(
            first,
            second,
            rtol=0,
            atol=0,
        )

        for index in range(1, 32):
            torch.testing.assert_close(
                first[:, 0],
                first[:, index],
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
