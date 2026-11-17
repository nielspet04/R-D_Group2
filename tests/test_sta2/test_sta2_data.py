import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.sta2.data import (
    STAClipDataset, collate_sta_clips, load_events, resolve_cutoff, sha256_file,
)
from src.sta2.video import VideoReadError, decode_requested_frames

try:
    import av
except ImportError:
    av = None
try:
    from .video_fixtures import write_video
except ImportError:
    from video_fixtures import write_video


class AnnotationTests(unittest.TestCase):
    def test_hidden_cutoff_is_authoritative_and_conflicts_are_rejected(self):
        row = {"narration_id": "hidden", "sta_observation_end_seconds": 0.5}
        self.assertEqual(resolve_cutoff(row, "test"), 0.5)
        with self.assertRaises(ValueError):
            resolve_cutoff({**row, "start_timestamp": "00:00:03"}, "test")

    def test_original_row_indices_survive_split_selection(self):
        table = pd.DataFrame({
            "narration_id": ["a", "b", "c"],
            "participant_id": ["P01", "P02", "P01"],
            "video_id": ["P01_01", "P02_01", "P01_01"],
            "start_timestamp": ["00:00:07"] * 3,
            "verb_class": [0, 1, 2], "noun_class": [1, 2, 3],
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table.to_csv(root / "all.csv", index=False)
            table.iloc[[2, 0]].to_csv(root / "train.csv", index=False)
            table.iloc[[1]].to_csv(root / "validation.csv", index=False)
            selected = load_events(
                root / "all.csv", split="train", train_split=root / "train.csv",
                validation_split=root / "validation.csv",
            )
        self.assertEqual(selected.row_index.tolist(), [0, 2])
        self.assertEqual(selected.narration_id.tolist(), ["a", "c"])


@unittest.skipIf(av is None, "PyAV is required for real media tests")
class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "P01_01.mp4"
        self.pts, self.base = write_video(self.path)
        timeline = self.root / "P01_01.npz"
        np.savez_compressed(timeline, pts=self.pts)
        stat = self.path.stat()
        manifest = {"schema_version": 1, "videos": {
            "P01_01": {
                "status": "ok", "timeline": timeline.name,
                "timeline_sha256": sha256_file(timeline), "time_base": str(self.base),
                "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            },
            "P02_01": {"status": "video_error", "error": "missing_file"},
        }}
        self.metadata = self.root / "manifest.json"
        self.metadata.write_text(json.dumps(manifest))
        self.events = pd.DataFrame({
            "narration_id": ["full", "early", "none", "missing"],
            "video_id": ["P01_01", "P01_01", "P01_01", "P02_01"],
            "start_timestamp": ["00:00:07", "00:00:01.5", "00:00:01", "00:00:07"],
            "verb_class": [1, 2, 3, 4], "noun_class": [2, 3, 4, 5],
            "row_index": [100, 8, 21, 3],
        })

    def dataset(self, events=None, split="train", **kwargs):
        return STAClipDataset(
            self.events if events is None else events, self.root, self.metadata,
            split=split, **kwargs,
        )

    def test_coverage_statuses_padding_and_collation(self):
        dataset = self.dataset()
        samples = [dataset[index] for index in range(len(dataset))]
        batch = collate_sta_clips(samples)
        self.assertEqual(batch["status"], ["full", "partial", "no_context", "video_error"])
        self.assertEqual(batch["row_index"].tolist(), [100, 8, 21, 3])
        self.assertEqual(batch["has_context"].tolist(), [True, True, False, False])
        self.assertEqual(tuple(batch["video"].shape), (4, 3, 32, 384, 384))
        self.assertEqual(batch["video"].dtype, torch.float32)
        self.assertEqual(batch["frame_indices"].dtype, torch.int64)
        self.assertEqual(torch.count_nonzero(batch["video"][2:]).item(), 0)
        self.assertTrue(torch.all(batch["frame_indices"][1, :28] == 0))

    def test_no_context_never_reads_timeline_or_decoder(self):
        dataset = self.dataset()
        with patch.object(dataset, "_timeline", side_effect=AssertionError("timeline read")), \
                patch("src.sta2.data.decode_requested_frames", side_effect=AssertionError("decode")):
            sample = dataset[2]
        self.assertEqual(sample["status"], "no_context")

    def test_requested_read_failure_retains_event_and_attempted_indices(self):
        dataset = self.dataset()
        with patch("src.sta2.data.decode_requested_frames", side_effect=VideoReadError("failed read")):
            with self.assertWarns(RuntimeWarning):
                sample = dataset[0]
        self.assertEqual(sample["status"], "video_error")
        self.assertEqual(sample["row_index"].item(), 100)
        self.assertFalse(sample["has_context"].item())
        self.assertTrue(torch.all(sample["frame_indices"] >= 0))
        self.assertEqual(torch.count_nonzero(sample["video"]).item(), 0)

    def test_decoder_rgb_order_repeats_and_strict_cutoff(self):
        indices = np.array([0, 0, 1], dtype=np.int64)
        frames = decode_requested_frames(self.path, self.pts, self.base, indices, 0.5)
        self.assertGreater(int(frames[0, 0, 0, 0]), 200)
        self.assertLess(int(frames[0, 0, 0, 2]), 20)
        np.testing.assert_array_equal(frames[0], frames[1])
        with self.assertRaises(VideoReadError):
            decode_requested_frames(self.path, self.pts, self.base, np.array([0]), 0.0)

    def test_missing_open_and_seek_failures_are_video_errors(self):
        with self.assertRaises(VideoReadError):
            decode_requested_frames(self.root / "absent.mp4", self.pts, self.base, np.array([0]), 1)
        with patch("av.open") as opened:
            container = opened.return_value.__enter__.return_value
            container.streams.video = [type("Stream", (), {"time_base": self.base})()]
            container.seek.side_effect = OSError("seek failure")
            with self.assertRaises(VideoReadError):
                decode_requested_frames(self.path, self.pts, self.base, np.array([0]), 1)

    def test_hidden_labels_omitted_and_yaml_normalization_reaches_dataset(self):
        events = self.events.iloc[[0]].drop(columns=["verb_class", "noun_class", "start_timestamp"])
        events["sta_observation_end_seconds"] = 6.0
        dataset = self.dataset(events, split="test", preprocessing={"mean": [0, 0, 0], "std": [1, 1, 1]})
        sample = dataset[0]
        self.assertNotIn("verb_label", sample)
        self.assertNotIn("noun_label", sample)
        self.assertGreater(sample["video"][0].mean().item(), 0.9)
        self.assertEqual(sample["row_index"].item(), 100)

    def test_subsetting_keeps_original_identity(self):
        dataset = self.dataset(self.events.iloc[[1, 0]])
        self.assertEqual(dataset[0]["row_index"].item(), 8)
        self.assertEqual(dataset[1]["row_index"].item(), 100)

    def test_training_epoch_resume_is_deterministic(self):
        dataset = self.dataset(training=True)
        dataset.set_epoch(3)
        first = dataset[0]["video"]
        resumed = self.dataset(training=True)
        resumed.set_epoch(3)
        torch.testing.assert_close(first, resumed[0]["video"], rtol=0, atol=0)
