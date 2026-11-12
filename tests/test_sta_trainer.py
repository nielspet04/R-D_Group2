"""Tests for full shared STA training orchestration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.sta_model_0 import FrequencyPriorSTA
    from src.sta_trainer import train_model
    from src.sta_training import (
        TrainingConfig,
        build_model,
        build_optimizer,
        load_checkpoint,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class STATrainerTests(unittest.TestCase):
    def test_selects_best_epoch_and_stops_early(self):
        config = TrainingConfig(
            model="mean_pool",
            feature_size=4,
            hidden_size=4,
            num_verbs=6,
            num_nouns=6,
            dropout=0.0,
            max_epochs=10,
            early_stopping_patience=2,
        )
        model = build_model(config)
        optimizer = build_optimizer(model, config)

        validation_scores = iter([0.10, 0.20, 0.15, 0.14])
        epoch_counter = {"value": 0}

        def fake_train(*args, **kwargs):
            epoch_counter["value"] += 1
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(float(epoch_counter["value"]))

            return {
                "train_loss": 1.0,
                "train_events": 4,
                "train_usable_events": 4,
                "train_excluded_events": 0,
                "train_skipped_batches": 0,
                "train_optimizer_steps": 1,
            }

        def fake_validate(*args, **kwargs):
            score = next(validation_scores)
            return {
                "validation_loss": 1.0,
                "validation_events": 6,
                "validation_usable_events": 5,
                "validation_fallback_events": 1,
                "verb_top1_accuracy": 0.1,
                "verb_top5_accuracy": 0.5,
                "verb_mt5r": 0.4,
                "noun_top1_accuracy": 0.1,
                "noun_top5_accuracy": 0.5,
                "noun_mt5r": 0.4,
                "action_top1_accuracy": 0.05,
                "action_top5_accuracy": 0.25,
                "action_mt5r": score,
            }

        labels = np.zeros(6, dtype=np.int64)
        frequency_prior = FrequencyPriorSTA.fit(
            labels,
            labels,
            num_verbs=6,
            num_nouns=6,
        )
        action_pairs = np.column_stack(
            (np.arange(6), np.arange(6))
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch(
                    "src.sta_trainer.train_one_epoch",
                    side_effect=fake_train,
                ),
                patch(
                    "src.sta_trainer.validate_one_epoch",
                    side_effect=fake_validate,
                ),
            ):
                result = train_model(
                    model,
                    optimizer,
                    train_loader=[],
                    validation_loader=[],
                    frequency_prior=frequency_prior,
                    action_pairs=action_pairs,
                    device=torch.device("cpu"),
                    config=config,
                    output_root=Path(temporary_directory),
                )

            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(result["best_action_mt5r"], 0.20)
            self.assertEqual(result["epochs_completed"], 4)
            self.assertTrue(result["stopped_early"])

            # Later epochs set every parameter to 3 and then 4. The returned
            # model must have reloaded the selected epoch-2 state.
            for parameter in model.parameters():
                self.assertTrue(torch.all(parameter == 2.0))

            self.assertTrue(result["checkpoint_path"].is_file())
            self.assertTrue(result["history_path"].is_file())
            self.assertTrue(result["validation_path"].is_file())
            self.assertTrue(result["configuration_path"].is_file())

            checkpoint = load_checkpoint(result["checkpoint_path"])
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(
                checkpoint["validation_metrics"]["action_mt5r"],
                0.20,
            )


if __name__ == "__main__":
    unittest.main()
