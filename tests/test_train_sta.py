"""Synthetic end-to-end correctness checks for the STA training entry point."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from scripts.evaluate_sta_model import main as evaluate_main
    from scripts.train_sta import ShardAwareBatchSampler, main
    from src.sta_features import STAFeatureDataset, save_sta_feature_shard
    from src.sta_training import load_checkpoint


MANIFEST_FIELDS = [
    "schema_version",
    "split",
    "shard_index",
    "shard_file",
    "start_index",
    "end_index_exclusive",
    "event_count",
]


def make_shard(
    split: str,
    shard_index: int,
    masks: list[list[bool]],
    statuses: list[str],
) -> dict:
    event_count = len(masks)
    generator = torch.Generator().manual_seed(100 + shard_index)
    valid_mask = torch.tensor(masks, dtype=torch.bool)
    features = torch.randn(
        event_count,
        5,
        512,
        generator=generator,
        dtype=torch.float16,
    )
    features[~valid_mask] = 0
    label_offset = shard_index * event_count
    labels = torch.arange(
        label_offset,
        label_offset + event_count,
        dtype=torch.int64,
    )
    return {
        "schema_version": 1,
        "split": split,
        "features": features,
        "valid_mask": valid_mask,
        "verb_labels": labels.remainder(6),
        "noun_labels": labels.remainder(6),
        "narration_ids": [
            f"{split}_{shard_index}_{index}"
            for index in range(event_count)
        ],
        "video_ids": [f"video_{shard_index}"] * event_count,
        "action_start_seconds": torch.full(
            (event_count,),
            10.0,
            dtype=torch.float32,
        ),
        "observation_end_seconds": torch.full(
            (event_count,),
            9.0,
            dtype=torch.float32,
        ),
        "status": statuses,
    }


def write_split(
    feature_root: Path,
    split: str,
    shards: list[dict],
) -> None:
    split_directory = feature_root / split
    rows = []
    start_index = 0
    for shard_index, shard in enumerate(shards):
        shard_file = f"shard_{shard_index:05d}.pt"
        save_sta_feature_shard(shard, split_directory / shard_file)
        event_count = len(shard["narration_ids"])
        rows.append(
            {
                "schema_version": 1,
                "split": split,
                "shard_index": shard_index,
                "shard_file": shard_file,
                "start_index": start_index,
                "end_index_exclusive": start_index + event_count,
                "event_count": event_count,
            }
        )
        start_index += event_count

    with (split_directory / "manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrainSTAIntegrationTests(unittest.TestCase):
    def test_shard_aware_sampler_keeps_batches_within_shards(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            feature_root = Path(temporary_directory)
            full_masks = [[True] * 5] * 3
            write_split(
                feature_root,
                "train",
                [
                    make_shard("train", 0, full_masks, ["full"] * 3),
                    make_shard("train", 1, full_masks, ["full"] * 3),
                ],
            )
            dataset = STAFeatureDataset(feature_root, "train")
            sampler = ShardAwareBatchSampler(dataset, batch_size=2, seed=42)

            batches = list(sampler)
            self.assertEqual(sorted(index for batch in batches for index in batch), list(range(6)))
            for batch in batches:
                shard_ids = {
                    dataset._find_manifest_row(index)["shard_index"]
                    for index in batch
                }
                self.assertEqual(len(shard_ids), 1)

    def test_synthetic_training_entry_point_writes_reloadable_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            feature_root = root / "features"
            output_root = root / "results"
            annotation_path = root / "annotations.csv"

            full_masks = [[True] * 5] * 3
            write_split(
                feature_root,
                "train",
                [
                    make_shard("train", 0, full_masks, ["full"] * 3),
                    make_shard("train", 1, full_masks, ["full"] * 3),
                ],
            )
            write_split(
                feature_root,
                "validation",
                [
                    make_shard(
                        "validation",
                        0,
                        [
                            [True, True, True, True, True],
                            [False, False, True, True, True],
                            [False, False, False, False, False],
                        ],
                        ["full", "partial", "no_context"],
                    )
                ],
            )
            pd.DataFrame(
                {
                    "narration_id": [f"annotation_{index}" for index in range(6)],
                    "verb_class": list(range(6)),
                    "noun_class": list(range(6)),
                }
            ).to_csv(annotation_path, index=False)

            for model_name in ("mean_pool", "gru"):
                with self.subTest(model=model_name):
                    result = main(
                        [
                            "--model",
                            model_name,
                            "--feature-root",
                            str(feature_root),
                            "--annotations",
                            str(annotation_path),
                            "--output-root",
                            str(output_root),
                            "--device",
                            "cpu",
                            "--smoke-test-epochs",
                            "1",
                        ]
                    )

                    expected_paths = [
                        output_root
                        / f"checkpoints/{model_name}/best.pt",
                        output_root
                        / f"logs/{model_name}/history.csv",
                        output_root
                        / f"tables/{model_name}_validation.json",
                        output_root / f"configs/{model_name}.json",
                    ]
                    for path in expected_paths:
                        self.assertTrue(path.is_file())

                    self.assertEqual(result["best_epoch"], 1)
                    self.assertEqual(
                        result["best_validation_metrics"][
                            "validation_events"
                        ],
                        3,
                    )
                    self.assertEqual(
                        result["best_validation_metrics"][
                            "validation_usable_events"
                        ],
                        2,
                    )
                    self.assertEqual(
                        result["best_validation_metrics"][
                            "validation_fallback_events"
                        ],
                        1,
                    )

                    checkpoint = load_checkpoint(expected_paths[0])
                    self.assertEqual(checkpoint["epoch"], 1)
                    self.assertEqual(
                        checkpoint["configuration"]["model"],
                        model_name,
                    )

                    evaluation_path = (
                        output_root
                        / f"tables/{model_name}_independent_evaluation.json"
                    )
                    evaluation = evaluate_main(
                        [
                            "--checkpoint",
                            str(expected_paths[0]),
                            "--feature-root",
                            str(feature_root),
                            "--annotations",
                            str(annotation_path),
                            "--output",
                            str(evaluation_path),
                            "--device",
                            "cpu",
                        ]
                    )
                    self.assertTrue(evaluation_path.is_file())
                    stored_evaluation = json.loads(
                        evaluation_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(evaluation["model"], model_name)
                    self.assertEqual(evaluation["checkpoint_epoch"], 1)
                    self.assertEqual(evaluation["validation_events"], 3)
                    self.assertEqual(
                        evaluation["validation_fallback_events"],
                        1,
                    )
                    self.assertEqual(
                        stored_evaluation["action_mt5r"],
                        checkpoint["validation_metrics"]["action_mt5r"],
                    )


if __name__ == "__main__":
    unittest.main()
