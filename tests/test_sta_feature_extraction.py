import unittest
import numpy as np
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_VIDEO = PROJECT_ROOT / "data/raw/P01_03.mp4"
from src.sta_feature_extraction import sample_segment_frames
from src.sta_feature_extraction import sta_segment_boundaries
from src.sta_feature_extraction import sample_segment_indices
from src.sta_feature_extraction import build_r3d18_feature_extractor 
from src.sta_feature_extraction import encode_segment_batch
from src.sta_feature_extraction import extract_event_features
from src.sta_feature_extraction import extract_event_batch 


class TestStaSegmentBoundaries(unittest.TestCase):
    def test_full_context(self):
        boundaries, valid_mask = sta_segment_boundaries(20.0)
        self.assertEqual(boundaries,[
                (14.0, 15.0),
                (15.0, 16.0),
                (16.0, 17.0),
                (17.0, 18.0),
                (18.0, 19.0),
            ], )
        self.assertEqual(valid_mask, [True, True, True, True, True])

    def test_partial_context(self):
        boundaries, valid_mask = sta_segment_boundaries(4.0)
        self.assertEqual(boundaries, [
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 2.0),
                (2.0, 3.0),
            ])
        self.assertEqual(valid_mask, [False, False, True, True, True])
    def test_partially_clipped_segment(self):
        boundaries, valid_mask = sta_segment_boundaries(4.5)
        self.assertEqual(boundaries[1], (0.0, 0.5))
        self.assertEqual(valid_mask, [False, True, True, True, True])

    def test_no_context(self):
        boundaries, valid_mask = sta_segment_boundaries(0.8)
        self.assertEqual(boundaries, [(0.0, 0.0)] * 5)
        self.assertEqual(valid_mask, [False] * 5)
    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            sta_segment_boundaries(float('nan'))
        with self.assertRaises(ValueError):
            sta_segment_boundaries(10.0, context_seconds=-1.0)
        with self.assertRaises(ValueError):
            sta_segment_boundaries(10.0, gap_seconds=-1.0)
        with self.assertRaises(ValueError):
            sta_segment_boundaries(20.0, segment_seconds=0)

class STAFrameIndexTests(unittest.TestCase):
    def test_selects_sixteen_frames(self):
        indices = sample_segment_indices(
            fps=30.0,
            segment_start=14.0,
            segment_stop=15.0,
        )

        self.assertEqual(indices.shape, (16,))
        self.assertTrue((indices / 30.0 < 15.0).all())

    def test_repeats_short_interval_frames(self):
        indices = sample_segment_indices(
            fps=10.0,
            segment_start=0.0,
            segment_stop=0.3,
            frame_count=16,
        )

        self.assertEqual(indices.shape, (16,))
        self.assertLess(len(np.unique(indices)), 16)

    def test_never_crosses_anticipation_cutoff(self):
        action_start = 20.0
        cutoff = action_start - 1.0

        boundaries, mask = sta_segment_boundaries(action_start)

        for boundary, valid in zip(boundaries, mask):
            if not valid:
                continue

            indices = sample_segment_indices(
                fps=29.97,
                segment_start=boundary[0],
                segment_stop=boundary[1],
            )
            self.assertTrue((indices / 29.97 < cutoff).all())

@unittest.skipUnless(LOCAL_VIDEO.exists(), "Local video file not found")
class TestSTAFrameExtraction(unittest.TestCase):
    def test_reads_rgb_frames(self):
        frames, indices, fps = sample_segment_frames(
            LOCAL_VIDEO,
            segment_start=0.0,
            segment_stop=1.0,
        )
        self.assertEqual(frames.shape[0], 16)
        self.assertEqual(frames.shape[-1], 3)
        self.assertEqual(frames.dtype, np.uint8)
        self.assertEqual(indices.shape, (16,))
        self.assertGreater(fps, 0)
        self.assertTrue((indices / fps < 1.0).all())

@unittest.skipIf(torch is None, "PyTorch is not installed")
@unittest.skipUnless(LOCAL_VIDEO.exists(), "Local test video is unavailable")
class STAR3D18EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        cls.encoder, cls.preprocessing = build_r3d18_feature_extractor(cls.device)

    def test_encoder_is_frozen_and_in_evaluation_mode(self):
        self.assertFalse(self.encoder.training)

        for parameter in self.encoder.parameters():
            self.assertFalse(parameter.requires_grad)

    def test_real_segment_produces_finite_512_features(self):
        frames, indices, fps = sample_segment_frames(
            video_path=LOCAL_VIDEO,
            segment_start=0.0,
            segment_stop=1.0,
            frame_count=16,
        )

        features = encode_segment_batch(
            frame_batches=[frames],
            encoder=self.encoder,
            preprocessing=self.preprocessing,
            device=self.device,
        )

        self.assertEqual(tuple(features.shape), (1, 512))
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(features.device.type, "cpu")
        self.assertTrue(torch.isfinite(features).all().item())

        # Confirm the sampled frames remain inside the requested interval.
        self.assertTrue((indices.astype(np.float64) / fps < 1.0).all())

    def test_encoder_is_deterministic(self):
        frames, _, _ = sample_segment_frames(
            video_path=LOCAL_VIDEO,
            segment_start=0.0,
            segment_stop=1.0,
            frame_count=16,
        )

        first = encode_segment_batch(
            frame_batches=[frames],
            encoder=self.encoder,
            preprocessing=self.preprocessing,
            device=self.device,
        )
        second = encode_segment_batch(
            frame_batches=[frames],
            encoder=self.encoder,
            preprocessing=self.preprocessing,
            device=self.device,
        )
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    def test_full_event_features(self):
        features, valid_mask, status = extract_event_features(
            video_path=LOCAL_VIDEO,
            action_start=20.0,
            encoder=self.encoder,
            preprocessing=self.preprocessing,
            device=self.device,
        )

        self.assertEqual(tuple(features.shape), (5, 512))
        self.assertEqual(tuple(valid_mask.shape), (5,))
        self.assertEqual(valid_mask.dtype, torch.bool)
        self.assertEqual(status, "full")
        self.assertTrue(valid_mask.all().item())
        self.assertTrue(torch.isfinite(features).all().item())
        self.assertTrue(torch.any(features != 0).item())


    def test_partial_event_has_zero_padded_segments(self):
        features, valid_mask, status = extract_event_features(
            video_path=LOCAL_VIDEO,
            action_start=4.0,
            encoder=self.encoder,
            preprocessing=self.preprocessing,
            device=self.device,
        )

        expected_mask = torch.tensor(
            [False, False, True, True, True],
            dtype=torch.bool,
        )

        self.assertEqual(tuple(features.shape), (5, 512))
        self.assertTrue(torch.equal(valid_mask, expected_mask))
        self.assertEqual(status, "partial")

        # Invalid segments must be exactly zero.
        self.assertTrue(torch.all(features[~valid_mask] == 0).item())

        # At least one valid feature must be nonzero.
        self.assertTrue(torch.any(features[valid_mask] != 0).item())


    def test_no_context_event_is_all_zero(self):
        features, valid_mask, status = extract_event_features(
            video_path=LOCAL_VIDEO,
            action_start=0.8,
            encoder=self.encoder,
            preprocessing=self.preprocessing,
            device=self.device,
        )

        self.assertEqual(tuple(features.shape), (5, 512))
        self.assertFalse(valid_mask.any().item())
        self.assertTrue(torch.all(features == 0).item())
        self.assertEqual(status, "no_context")


    def test_missing_video_becomes_video_error(self):
        missing_video = PROJECT_ROOT / "data/raw/DOES_NOT_EXIST.mp4"

        features, valid_mask, status = extract_event_features(
            video_path=missing_video,
            action_start=20.0,
            encoder=self.encoder,
            preprocessing=self.preprocessing,
            device=self.device,
        )

        self.assertEqual(tuple(features.shape), (5, 512))
        self.assertFalse(valid_mask.any().item())
        self.assertTrue(torch.all(features == 0).item())
        self.assertEqual(status, "video_error")
    def test_event_batch_preserves_every_event(self):
        missing_video = PROJECT_ROOT / "data/raw/DOES_NOT_EXIST.mp4"

        features, valid_masks, statuses = extract_event_batch(
            video_paths=[LOCAL_VIDEO, missing_video],
            action_starts=[20.0, 20.0],
            encoder=self.encoder,
            preprocessing=self.preprocessing,
            device=self.device,
            workers=0,
        )

        self.assertEqual(tuple(features.shape), (2, 5, 512))
        self.assertEqual(tuple(valid_masks.shape), (2, 5))
        self.assertEqual(statuses, ["full", "video_error"])

        self.assertTrue(valid_masks[0].all().item())
        self.assertFalse(valid_masks[1].any().item())
        self.assertTrue(torch.all(features[1] == 0).item())

        

if __name__ == "__main__":
    unittest.main()