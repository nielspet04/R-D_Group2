"""Downloader retries must never leave a partial video at its final path."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.download_kitchen_mango import download_video


class MangoDownloadTests(unittest.TestCase):
    def test_existing_file_is_skipped_unless_overwrite_is_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'P25_107.mp4'
            path.write_bytes(b'old')
            session = SimpleNamespace(data_objects=Mock())
            self.assertFalse(download_video(session, '/remote/video', path, 3))
            session.data_objects.get.assert_not_called()
            self.assertEqual(path.read_bytes(), b'old')

    def test_successful_refresh_replaces_old_file_after_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'P25_107.mp4'
            path.write_bytes(b'old')
            def transfer(remote, destination):
                self.assertEqual(path.read_bytes(), b'old')
                Path(destination).write_bytes(b'fixed')
            session = SimpleNamespace(data_objects=SimpleNamespace(get=transfer))
            self.assertTrue(download_video(session, '/remote/video', path, 5, overwrite=True))
            self.assertEqual(path.read_bytes(), b'fixed')
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_or_truncated_transfer_preserves_old_file_and_can_retry(self):
        for failure in ('exception', 'short'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'P25_107.mp4'
                path.write_bytes(b'old')
                def transfer(remote, destination):
                    Path(destination).write_bytes(b'part')
                    if failure == 'exception':
                        raise OSError('connection lost')
                session = SimpleNamespace(data_objects=SimpleNamespace(get=transfer))
                with self.assertRaises((OSError, RuntimeError)):
                    download_video(session, '/remote/video', path, 5, overwrite=True)
                self.assertEqual(path.read_bytes(), b'old')
                self.assertEqual(list(Path(directory).iterdir()), [path])
                session.data_objects.get = lambda remote, destination: Path(destination).write_bytes(b'fixed')
                self.assertTrue(download_video(session, '/remote/video', path, 5, overwrite=True))
                self.assertEqual(path.read_bytes(), b'fixed')

    def test_new_download_creates_participant_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'P25' / 'P25_107.mp4'
            session = SimpleNamespace(data_objects=SimpleNamespace(
                get=lambda remote, destination: Path(destination).write_bytes(b'fixed')
            ))
            self.assertTrue(download_video(session, '/remote/video', path, 5))
            self.assertEqual(path.read_bytes(), b'fixed')
