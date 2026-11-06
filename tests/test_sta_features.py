"""Tests for the shared STA feature-shard contract."""

import unittest

try:
    import torch
except ImportError:
    torch = None

from scripts.create_sta_example_shard import build_example_shard
from src.sta_features import validate_sta_feature_shard


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class STAFeatureShardTests(unittest.TestCase):
    def test_example_shard_satisfies_contract(self):
        shard = build_example_shard()
        self.assertEqual(validate_sta_feature_shard(shard), 8)

    def test_invalid_segments_must_be_zero(self):
        shard = build_example_shard()
        shard["features"][1, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            validate_sta_feature_shard(shard)

    def test_test_split_must_not_contain_labels(self):
        shard = build_example_shard()
        shard["split"] = "test"
        with self.assertRaisesRegex(ValueError, "absent"):
            validate_sta_feature_shard(shard)

    def test_status_and_mask_must_agree(self):
        shard = build_example_shard()
        shard["status"][0] = "partial"
        with self.assertRaisesRegex(ValueError, "status 'partial'"):
            validate_sta_feature_shard(shard)


if __name__ == "__main__":
    unittest.main()
