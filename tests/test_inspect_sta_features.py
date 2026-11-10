"""Integration test for the feature inspection command."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.create_sta_example_shard import build_example_shard
from src.sta_features import save_sta_feature_shard


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InspectSTAFeaturesTests(unittest.TestCase):
    def test_valid_example_passes_inspection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            feature_root = Path(temporary_directory)
            split_directory = feature_root / "train"
            split_directory.mkdir(parents=True)

            save_sta_feature_shard(
                build_example_shard(),
                split_directory / "shard_00000.pt",
            )

            with (split_directory / "manifest.csv").open(
                "w",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "schema_version",
                        "split",
                        "shard_index",
                        "shard_file",
                        "start_index",
                        "end_index_exclusive",
                        "event_count",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "schema_version": 1,
                        "split": "train",
                        "shard_index": 0,
                        "shard_file": "shard_00000.pt",
                        "start_index": 0,
                        "end_index_exclusive": 8,
                        "event_count": 8,
                    }
                )

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/inspect_sta_features.py"),
                    "--feature-root",
                    str(feature_root),
                    "--split",
                    "train",
                    "--check-all",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )

            self.assertIn("8", result.stdout)