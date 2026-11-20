"""Final retraining uses all allowed labels without invented validation scores."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.prepare_sta2_final import prepare
from src.evaluation import build_action_pairs
from src.sta2.data import load_events, sha256_file
from src.sta2.evaluation import validate_checkpoint_metadata
from src.sta2.training import fit
from tests.test_sta2 import test_sta2_training as fixtures
from tests.test_sta2.test_sta2_evaluation import make_checkpoint


class FinalTrainingTests(unittest.TestCase):
    def test_fixed_epoch_training_skips_validation_and_resumes(self):
        helper = fixtures.STA2TrainingTests()
        helper.setUp()
        model = helper.make_model()
        optimizer, scheduler = helper.make_optimizer_scheduler(model)
        batches = fixtures.TinyLoader(fixtures.make_batch(i, verb=i, noun=i) for i in range(2))
        with tempfile.TemporaryDirectory() as directory:
            arguments = dict(
                model=model, train_loader=batches, validation_loader=fixtures.TinyLoader(),
                train_sampler=Mock(), validation_sampler=Mock(), train_dataset=Mock(),
                action_vocabulary=helper.vocabulary,
                evaluation_action_pairs=build_action_pairs(np.arange(6), np.arange(6)),
                verb_prior=torch.full((6,), 1 / 6), noun_prior=torch.full((6,), 1 / 6),
                optimizer=optimizer, scheduler=scheduler, config=helper.config,
                context=helper.context, output_dir=Path(directory),
                resolved_config={"world_size": 1, "final_training": {"epochs": 2}},
                backbone_provenance={}, split_hashes={}, repository_root=fixtures.PROJECT_ROOT,
                final_epochs=2,
            )
            with patch("src.sta2.training.validate_one_epoch", side_effect=AssertionError("validation called")):
                with patch.dict("os.environ", {"SLURM_JOB_END_TIME": "1"}):
                    first = fit(**arguments)
                self.assertEqual(first["completed_epoch"], 1)
                result = fit(**arguments, resume=Path(directory) / "checkpoints/last.pt")
            self.assertEqual(result["completed_epoch"], 2)
            self.assertIsNone(result["best_action_mt5r"])
            self.assertFalse((Path(directory) / "checkpoints/best.pt").exists())
            self.assertTrue(all("action_mt5r" not in row for row in result["history"]))
            saved = torch.load(result["last_checkpoint"], weights_only=True)
            self.assertIsNone(saved["best_metric"])
            self.assertEqual(saved["epochs_without_improvement"], 0)

    def test_prepare_preserves_source_and_freezes_all_training_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = pd.DataFrame({
                "narration_id": ["a", "b"], "participant_id": ["P01", "P02"],
                "video_id": ["P01_01", "P02_01"], "start_timestamp": ["00:00:07"] * 2,
                "verb_class": [0, 1], "noun_class": [0, 1],
            })
            paths = {name: root / (name + ".csv") for name in ("annotations", "train_split", "validation_split")}
            table.to_csv(paths["annotations"], index=False)
            table.iloc[:1].to_csv(paths["train_split"], index=False)
            table.iloc[1:].to_csv(paths["validation_split"], index=False)
            checkpoint = make_checkpoint()
            checkpoint["configuration"]["training"]["max_epochs"] = 20
            checkpoint["configuration"]["backbone_config"] = {"fixed": True}
            checkpoint["split_hashes"] = {name: sha256_file(path) for name, path in paths.items()}
            selection = root / "best.pt"
            torch.save(checkpoint, selection)
            original_hashes = copy.deepcopy(checkpoint["split_hashes"])
            output = root / "final"
            result = prepare(selection, **paths, output_dir=output)
            self.assertEqual(result["selected_epochs"], 3)
            self.assertEqual(sha256_file(output / "train.csv"), sha256_file(paths["annotations"]))
            full = load_events(paths["annotations"], split="train", train_split=output / "train.csv",
                               validation_split=output / "validation.csv")
            self.assertEqual(full.row_index.tolist(), [0, 1])
            self.assertEqual(len(pd.read_csv(output / "validation.csv")), 0)
            self.assertEqual(yaml.safe_load((output / "config.yaml").read_text())["training"]["max_epochs"], 20)
            self.assertEqual(original_hashes, {name: sha256_file(path) for name, path in paths.items()})
            with self.assertRaises(FileExistsError):
                prepare(selection, **paths, output_dir=output)

    def test_final_checkpoint_must_be_complete_and_have_no_validation_metric(self):
        checkpoint = make_checkpoint()
        checkpoint["configuration"]["final_training"] = {"epochs": 3}
        checkpoint["best_metric"] = None
        validate_checkpoint_metadata(checkpoint)
        checkpoint["configuration"]["final_training"]["epochs"] = 5
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_checkpoint_metadata(checkpoint)
        checkpoint["best_metric"] = 0.3
        with self.assertRaisesRegex(ValueError, "must not claim"):
            validate_checkpoint_metadata(checkpoint)
