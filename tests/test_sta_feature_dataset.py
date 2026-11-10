"""Tests for loading cached STA feature shards as a dataset."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from scripts.create_sta_example_shard import build_example_shard
from src.sta_features import STAFeatureDataset, save_sta_feature_shard


def write_manifest(
    path: Path,
    *,
    start_index: int = 0,
    end_index_exclusive: int = 8,
    event_count: int = 8,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
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
                "start_index": start_index,
                "end_index_exclusive": end_index_exclusive,
                "event_count": event_count,
            }
        )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class STAFeatureDatasetTests(unittest.TestCase):
    def create_valid_feature_root(self, root: Path) -> dict:
        split_directory = root / "train"
        shard = build_example_shard()

        save_sta_feature_shard(
            shard,
            split_directory / "shard_00000.pt",
        )
        write_manifest(split_directory / "manifest.csv")

        return shard

    def test_dataset_length_matches_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            feature_root = Path(temporary_directory)
            self.create_valid_feature_root(feature_root)

            dataset = STAFeatureDataset(feature_root, "train")

            self.assertEqual(len(dataset), 8)

    def test_dataset_returns_expected_item(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            feature_root = Path(temporary_directory)
            shard = self.create_valid_feature_root(feature_root)

            dataset = STAFeatureDataset(feature_root, "train")
            item = dataset[0]

            self.assertEqual(tuple(item["features"].shape), (5, 512))
            self.assertEqual(item["features"].dtype, torch.float32)
            self.assertEqual(tuple(item["valid_mask"].shape), (5,))
            self.assertEqual(item["valid_mask"].dtype, torch.bool)

            self.assertEqual(
                item["narration_id"],
                shard["narration_ids"][0],
            )
            self.assertEqual(
                item["video_id"],
                shard["video_ids"][0],
            )
            self.assertEqual(
                item["verb_label"],
                int(shard["verb_labels"][0].item()),
            )
            self.assertEqual(
                item["noun_label"],
                int(shard["noun_labels"][0].item()),
            )
            self.assertEqual(
                item["status"],
                shard["status"][0],
            )

    def test_invalid_segments_remain_zero_after_conversion(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            feature_root = Path(temporary_directory)
            self.create_valid_feature_root(feature_root)

            dataset = STAFeatureDataset(feature_root, "train")
            partial_item = dataset[1]

            invalid = ~partial_item["valid_mask"]

            self.assertTrue(
                torch.all(partial_item["features"][invalid] == 0).item()
            )

    def test_negative_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            feature_root = Path(temporary_directory)
            self.create_valid_feature_root(feature_root)

            dataset = STAFeatureDataset(feature_root, "train")

            with self.assertRaises(IndexError):
                _ = dataset[-1]

    def test_out_of_range_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            feature_root = Path(temporary_directory)
            self.create_valid_feature_root(feature_root)

            dataset = STAFeatureDataset(feature_root, "train")

            with self.assertRaises(IndexError):
                _ = dataset[8]

    def test_manifest_gap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            feature_root = Path(temporary_directory)
            split_directory = feature_root / "train"

            save_sta_feature_shard(
                build_example_shard(),
                split_directory / "shard_00000.pt",
            )

            # A first shard cannot begin at source index 1.
            write_manifest(
                split_directory / "manifest.csv",
                start_index=1,
                end_index_exclusive=9,
                event_count=8,
            )

            with self.assertRaises(ValueError):
                STAFeatureDataset(feature_root, "train")


if __name__ == "__main__":
    unittest.main()