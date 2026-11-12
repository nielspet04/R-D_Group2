"""Regression checks for floating-point segment boundaries."""

import unittest

import numpy as np

from src.sta_feature_extraction import (
    sample_segment_indices,
    sta_segment_boundaries,
)


class STASamplingRegressionTests(unittest.TestCase):
    def test_fractional_segment_endpoints_at_fifty_fps(self):
        for start, stop in (
            (162.8, 163.8),
            (292.16, 293.16),
        ):
            with self.subTest(start=start, stop=stop):
                indices = sample_segment_indices(
                    fps=50.0,
                    segment_start=start,
                    segment_stop=stop,
                )
                times = indices.astype(np.float64) / 50.0

                self.assertEqual(indices.shape, (16,))
                self.assertTrue(np.all(times >= start))
                self.assertTrue(np.all(times < stop))

    def test_reported_events_respect_exact_anticipation_cutoff(self):
        for action_start in (168.8, 298.16, 1.7):
            with self.subTest(action_start=action_start):
                cutoff = action_start - 1.0
                boundaries, mask = sta_segment_boundaries(action_start)

                self.assertEqual(boundaries[-1][1], cutoff)

                for (start, stop), valid in zip(boundaries, mask):
                    if not valid:
                        continue

                    indices = sample_segment_indices(
                        fps=50.0,
                        segment_start=start,
                        segment_stop=stop,
                    )
                    times = indices.astype(np.float64) / 50.0

                    self.assertTrue(np.all(times >= start))
                    self.assertTrue(np.all(times < stop))
                    self.assertTrue(np.all(times < cutoff))


if __name__ == "__main__":
    unittest.main()
