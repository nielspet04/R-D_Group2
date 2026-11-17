import unittest

import numpy as np

from src.sta2.sampling import (
    sample_frame_indices,
    UnavailableContextError,
)


class SamplingTests(unittest.TestCase):
    def test_full_context(self):
        times = np.arange(100, dtype=np.float64) / 8

        result = sample_frame_indices(6.0, times)

        self.assertEqual(result.status, "full")

        np.testing.assert_array_equal(
            result.indices,
            np.arange(16, 48),
        )

        self.assertTrue(
            np.all(result.timestamps < 6.0)
        )

    def test_partial_context(self):
        times = np.arange(100, dtype=np.float64) / 8

        result = sample_frame_indices(0.5, times)

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.indices.shape, (32,))

        np.testing.assert_array_equal(
            result.indices[:28],
            np.zeros(28, dtype=np.int64),
        )

        np.testing.assert_array_equal(
            result.indices[-4:],
            [0, 1, 2, 3],
        )

    def test_tiny_positive_cutoff(self):
        times = np.arange(100, dtype=np.float64) / 8

        result = sample_frame_indices(0.01, times)

        self.assertEqual(result.status, "partial")

        np.testing.assert_array_equal(
            result.indices,
            np.zeros(32, dtype=np.int64),
        )

    def test_no_context(self):
        times = np.arange(100, dtype=np.float64) / 8

        for cutoff in (0.0, -0.5):
            result = sample_frame_indices(cutoff, times)

            self.assertEqual(
                result.status,
                "no_context",
            )
            self.assertTrue(
                np.all(result.indices == -1)
            )

    def test_fractional_frame_rates(self):
        for fps in (
            29.97,
            30000 / 1001,
            60000 / 1001,
            50.0,
        ):
            times = (
                np.arange(20000, dtype=np.float64)
                / fps
            )

            for cutoff in (6.0, 163.8, 293.16):
                result = sample_frame_indices(
                    cutoff,
                    times,
                )

                desired = (
                    cutoff
                    - 4
                    + np.arange(32) / 8
                )

                self.assertTrue(
                    np.all(result.timestamps < cutoff)
                )

                self.assertTrue(
                    np.all(result.timestamps <= desired)
                )

                next_indices = result.indices + 1

                self.assertTrue(
                    np.all(times[next_indices] > desired)
                )

    def test_truncated_recording(self):
        times = np.arange(8, dtype=np.float64) / 8

        with self.assertRaises(
            UnavailableContextError
        ):
            sample_frame_indices(10.0, times)


if __name__ == "__main__":
    unittest.main()