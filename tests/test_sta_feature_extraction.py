import unittest
from unittest.mock import patch
import tempfile
import numpy as np
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_VIDEO = PROJECT_ROOT / "data/raw/P01_03.mp4"
from src.sta_feature_extraction import prepare_event_frames
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


    def test_pretrained_batch_matches_legacy_reader_exactly(self):
        def legacy_prepare(path, action_start):
            boundaries, mask = sta_segment_boundaries(action_start)
            positions = [i for i, valid in enumerate(mask) if valid]
            frames = [sample_segment_frames(path, *boundaries[i])[0] for i in positions]
            status = "full" if all(mask) else ("partial" if any(mask) else "no_context")
            return frames, positions, mask, status

        arguments = dict(
            video_paths=[LOCAL_VIDEO] * 3,
            action_starts=[20.0, 1.7, 0.8],
            encoder=self.encoder,
            preprocessing=self.preprocessing,
            device=self.device,
            workers=2,
        )
        expected_inputs = []
        actual_inputs = []
        def record_preprocessing(target):
            def preprocess(tensor):
                target.append(tensor.clone())
                return self.preprocessing(tensor)
            return preprocess

        with patch("src.sta_feature_extraction.prepare_event_frames", side_effect=legacy_prepare):
            expected = extract_event_batch(**{**arguments, "preprocessing": record_preprocessing(expected_inputs)})
        actual = extract_event_batch(**{**arguments, "preprocessing": record_preprocessing(actual_inputs)})
        self.assertEqual(len(actual_inputs), len(expected_inputs))
        for actual_input, expected_input in zip(actual_inputs, expected_inputs):
            self.assertTrue(torch.equal(actual_input, expected_input))
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertEqual(actual[2], expected[2])

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

        


class FakeCapture:
    """Deterministic decoder with observable opens/seeks/reads and injected failures."""

    def __init__(self, fps=29.97, opened=True, fail_at=None, seek_ok=True):
        self.fps = fps
        self.opened = opened
        self.fail_at = fail_at
        self.seek_ok = seek_ok
        self.cursor = 0
        self.seeks = []
        self.reads = []
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        import cv2
        return self.fps if prop == cv2.CAP_PROP_FPS else 100000

    def set(self, prop, index):
        self.seeks.append(index)
        self.cursor = index
        return self.seek_ok

    def read(self):
        index = self.cursor
        self.cursor += 1
        self.reads.append(index)
        if index == self.fail_at:
            return False, None
        # Encode the frame number in BGR, so RGB conversion is checked too.
        return True, np.full((2, 3, 3), [index % 256, index // 256, 17], dtype=np.uint8)

    def release(self):
        self.released = True


class STAEventReaderTests(unittest.TestCase):
    def test_matches_legacy_reader_with_one_open_and_seek(self):
        for fps in (10.0, 29.97, 50.0):
            for action_start in (1.1, 4.5, 20.0, 298.16):
                with self.subTest(fps=fps, action_start=action_start):
                    capture = FakeCapture(fps=fps)
                    with patch("cv2.VideoCapture", return_value=capture) as factory:
                        frames, positions, mask, status = prepare_event_frames(Path("video"), action_start)
                    factory.assert_called_once_with("video")
                    self.assertEqual(len(capture.seeks), 1)
                    self.assertTrue(capture.released)
                    self.assertEqual(capture.reads, list(range(capture.reads[0], capture.reads[-1] + 1)))
                    boundaries, expected_mask = sta_segment_boundaries(action_start)
                    self.assertEqual(mask, expected_mask)
                    self.assertEqual(status, "full" if all(mask) else "partial")
                    self.assertEqual(positions, [i for i, valid in enumerate(mask) if valid])
                    for segment, position in zip(frames, positions):
                        with patch("cv2.VideoCapture", return_value=FakeCapture(fps=fps)):
                            expected, indices, _ = sample_segment_frames(Path("video"), *boundaries[position])
                        np.testing.assert_array_equal(segment, expected)
                        self.assertTrue(np.all(indices / fps < action_start - 1.0))

    def test_no_context_does_not_open_video(self):
        with patch("cv2.VideoCapture") as factory:
            self.assertEqual(prepare_event_frames(Path("missing"), 0.8),
                             ([], [], [False] * 5, "no_context"))
        factory.assert_not_called()

    def test_failures_release_capture(self):
        # 422 is an intermediate frame, not one of the requested 16 positions.
        for capture in (FakeCapture(opened=False), FakeCapture(fail_at=422),
                        FakeCapture(seek_ok=False), FakeCapture(fps=0)):
            with self.subTest(capture=capture):
                with patch("cv2.VideoCapture", return_value=capture):
                    with self.assertRaises((RuntimeError, ValueError)):
                        prepare_event_frames(Path("video"), 20.0)
                self.assertTrue(capture.released)

    def test_real_fractional_fps_video_matches_legacy(self):
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fractional.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 29.97, (32, 24))
            self.assertTrue(writer.isOpened())
            try:
                rng = np.random.default_rng(42)
                for _ in range(330):
                    writer.write(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8))
            finally:
                writer.release()
            for action_start in (1.1, 4.5, 10.8):
                frames, positions, _, _ = prepare_event_frames(path, action_start)
                boundaries, _ = sta_segment_boundaries(action_start)
                for segment, position in zip(frames, positions):
                    expected, _, _ = sample_segment_frames(path, *boundaries[position])
                    np.testing.assert_array_equal(segment, expected)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_batch_workers_preserve_order_features_and_errors(self):
        def factory(path):
            return FakeCapture(opened=path != "missing", fail_at=422 if path == "broken" else None)

        # Small deterministic encoder exercises the real tensor preparation and
        # feature scatter without requiring pretrained weights or a GPU.
        class Encoder:
            def __call__(self, batch):
                return batch.float().mean(dim=(1, 2, 3, 4)).unsqueeze(1).repeat(1, 512)

        paths = list(map(Path, ["video", "missing", "video", "broken", "missing"]))
        starts = [20.0, 20.0, 4.5, 20.0, 0.8]
        def legacy_prepare(path, start):
            boundaries, mask = sta_segment_boundaries(start)
            positions = [i for i, valid in enumerate(mask) if valid]
            frames = [sample_segment_frames(path, *boundaries[i])[0] for i in positions]
            return frames, positions, mask, "full" if all(mask) else ("partial" if any(mask) else "no_context")

        for workers in (0, 3):
            with patch("cv2.VideoCapture", side_effect=factory):
                actual = extract_event_batch(paths, starts, Encoder(), lambda x: x, "cpu", workers=workers)
                with patch("src.sta_feature_extraction.prepare_event_frames", side_effect=legacy_prepare):
                    # Use the first three and last event: the legacy reader can
                    # skip the broken intermediate frame, the new reader must fail.
                    expected = extract_event_batch(paths[:3] + paths[4:], starts[:3] + starts[4:],
                                                   Encoder(), lambda x: x, "cpu", workers=workers)
                for index, (path, start) in enumerate(zip(paths, starts)):
                    single = extract_event_features(path, start, Encoder(), lambda x: x, "cpu")
                    torch.testing.assert_close(actual[0][index], single[0], rtol=0, atol=0)
                    self.assertTrue(torch.equal(actual[1][index], single[1]))
                    self.assertEqual(actual[2][index], single[2])
            torch.testing.assert_close(actual[0][[0, 1, 2, 4]], expected[0], rtol=0, atol=0)
            self.assertTrue(torch.equal(actual[1][[0, 1, 2, 4]], expected[1]))
            self.assertEqual(actual[2], ["full", "video_error", "partial", "video_error", "no_context"])
            self.assertTrue(torch.all(actual[0][~actual[1]] == 0))

if __name__ == "__main__":
    unittest.main()