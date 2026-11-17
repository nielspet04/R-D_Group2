import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import av
    from scripts import build_sta2_metadata as metadata
except ImportError:
    av = None

try:
    from .video_fixtures import write_video
except ImportError:
    from video_fixtures import write_video


@unittest.skipIf(av is None, "PyAV is required for timeline generation tests")
class MetadataResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        write_video(self.raw / "P01_01.mp4", frames=16)
        self.annotations = self.root / "annotations.csv"
        self.annotations.write_text("video_id\nP01_01\nP02_01\n")
        self.info = self.root / "video_info.csv"
        self.info.write_text("video_id,fps\nP01_01,8\nP02_01,8\n")
        self.output = self.root / "timelines"

    def run_builder(self):
        argv = ["build_sta2_metadata.py", "--annotations", str(self.annotations),
                "--video-info", str(self.info), "--raw-dir", str(self.raw),
                "--output-dir", str(self.output)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            metadata.main()

    def manifest(self):
        return json.loads((self.output / "manifest.json").read_text())

    def test_resume_skips_success_and_retries_repaired_missing_video(self):
        self.run_builder()
        original = self.manifest()["videos"]["P01_01"]
        write_video(self.raw / "P02_01.mp4", frames=16)
        with patch.object(av, "open", wraps=av.open) as opened:
            self.run_builder()
        self.assertEqual(opened.call_count, 1)
        report = self.manifest()
        self.assertEqual(report["videos"]["P01_01"], original)
        self.assertEqual(report["videos"]["P02_01"]["status"], "ok")
        self.assertTrue(report["complete"])

    def test_corrupt_timeline_is_rebuilt(self):
        self.run_builder()
        (self.output / "P01_01.npz").write_bytes(b"broken")
        with patch.object(av, "open", wraps=av.open) as opened:
            self.run_builder()
        self.assertEqual(opened.call_count, 1)
        item = self.manifest()["videos"]["P01_01"]
        self.assertEqual(item["status"], "ok")
        self.assertEqual(metadata.sha256_file(self.output / item["timeline"]), item["timeline_sha256"])

    def test_changed_source_is_reindexed(self):
        self.run_builder()
        write_video(self.raw / "P01_01.mp4", frames=24)
        with patch.object(av, "open", wraps=av.open) as opened:
            self.run_builder()
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(self.manifest()["videos"]["P01_01"]["frame_count"], 24)

    def test_different_annotations_cannot_overwrite_prior_progress(self):
        self.run_builder()
        before = (self.output / "manifest.json").read_bytes()
        self.annotations.write_text("video_id\nP01_01\n")
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.run_builder()
        self.assertEqual((self.output / "manifest.json").read_bytes(), before)

    def test_interrupted_scan_resumes_completed_videos(self):
        write_video(self.raw / "P02_01.mp4", frames=16)
        original_open = av.open
        count = 0

        def interrupted_open(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt
            return original_open(*args, **kwargs)

        with patch.object(av, "open", side_effect=interrupted_open):
            with self.assertRaises(KeyboardInterrupt):
                self.run_builder()
        self.assertEqual(set(self.manifest()["videos"]), {"P01_01"})
        self.assertFalse(self.manifest()["complete"])
        with patch.object(av, "open", wraps=av.open) as opened:
            self.run_builder()
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(len(self.manifest()["videos"]), 2)
