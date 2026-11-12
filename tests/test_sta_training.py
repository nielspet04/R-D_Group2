"""Tests for the shared STA Model 1 and Model 2 training utilities."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

if torch is not None:
    from src.sta_model_0 import FrequencyPriorSTA
    from src.sta_training import (
        TrainingConfig,
        build_model,
        build_optimizer,
        masked_classification_loss,
        train_one_epoch,
        validate_one_epoch,
    )


def make_train_batch(*, has_context: bool = True):
    """Return a deterministic synthetic batch used only for training tests."""
    generator = torch.Generator().manual_seed(42)
    mask_value = 1 if has_context else 0
    return {
        "features": torch.randn(
            4, 5, 4, generator=generator, dtype=torch.float16
        ),
        "valid_mask": torch.full(
            (4, 5), mask_value, dtype=torch.bool
        ),
        "verb_label": torch.tensor([0, 1, 2, 3]),
        "noun_label": torch.tensor([1, 2, 3, 4]),
    }


def make_validation_batch():
    """Return a separate all-false synthetic validation fixture."""
    generator = torch.Generator().manual_seed(123)
    return {
        "features": torch.randn(
            6, 5, 4, generator=generator, dtype=torch.float16
        ),
        "valid_mask": torch.zeros(6, 5, dtype=torch.bool),
        "verb_label": torch.zeros(6, dtype=torch.long),
        "noun_label": torch.zeros(6, dtype=torch.long),
    }


if torch is not None:
    class DtypeRecordingModel(nn.Module):
        """Small differentiable model that records its input dtype."""

        def __init__(self) -> None:
            super().__init__()
            self.verb_bias = nn.Parameter(torch.zeros(6))
            self.noun_bias = nn.Parameter(torch.zeros(6))
            self.seen_dtype = None

        def forward(self, features, valid_mask):
            self.seen_dtype = features.dtype
            batch_size = features.shape[0]
            return (
                self.verb_bias.expand(batch_size, -1),
                self.noun_bias.expand(batch_size, -1),
            )


    class NoContextModel(nn.Module):
        """Fail if validation needlessly calls a model without context."""

        def forward(self, features, valid_mask):
            raise AssertionError(
                "The neural model must not run for an all-false batch"
            )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class STATrainingTests(unittest.TestCase):
    def small_config(self, model: str) -> TrainingConfig:
        return TrainingConfig(
            model=model,
            feature_size=4,
            hidden_size=4,
            num_verbs=6,
            num_nouns=6,
            dropout=0.0,
        )

    def test_masked_loss_ignores_labels_without_context(self):
        generator = torch.Generator().manual_seed(7)
        verb_logits = torch.randn(3, 6, generator=generator)
        noun_logits = torch.randn(3, 6, generator=generator)
        valid_mask = torch.tensor(
            [
                [True, True, True, True, True],
                [False, False, True, True, True],
                [False, False, False, False, False],
            ]
        )
        verb_labels = torch.tensor([0, 1, 2])
        noun_labels = torch.tensor([1, 2, 3])

        original = masked_classification_loss(
            verb_logits,
            noun_logits,
            verb_labels,
            noun_labels,
            valid_mask,
        )

        changed_verb_labels = verb_labels.clone()
        changed_noun_labels = noun_labels.clone()
        changed_verb_labels[-1] = 5
        changed_noun_labels[-1] = 5
        changed = masked_classification_loss(
            verb_logits,
            noun_logits,
            changed_verb_labels,
            changed_noun_labels,
            valid_mask,
        )

        self.assertIsNotNone(original)
        self.assertIsNotNone(changed)
        self.assertTrue(torch.equal(original, changed))

    def test_empty_context_batch_skips_parameter_updates(self):
        for architecture in ("mean_pool", "gru"):
            with self.subTest(model=architecture):
                torch.manual_seed(42)
                config = self.small_config(architecture)
                model = build_model(config)
                optimizer = build_optimizer(model, config)
                parameters_before = [
                    parameter.detach().clone()
                    for parameter in model.parameters()
                ]

                statistics = train_one_epoch(
                    model,
                    [make_train_batch(has_context=False)],
                    optimizer,
                    torch.device("cpu"),
                )

                self.assertEqual(statistics["train_events"], 4)
                self.assertEqual(statistics["train_usable_events"], 0)
                self.assertEqual(statistics["train_excluded_events"], 4)
                self.assertEqual(statistics["train_skipped_batches"], 1)
                self.assertEqual(statistics["train_optimizer_steps"], 0)
                self.assertIsNone(statistics["train_loss"])
                for before, after in zip(
                    parameters_before, model.parameters()
                ):
                    self.assertTrue(torch.equal(before, after))

    def test_usable_context_updates_both_architectures(self):
        for architecture in ("mean_pool", "gru"):
            with self.subTest(model=architecture):
                torch.manual_seed(42)
                config = self.small_config(architecture)
                model = build_model(config)
                optimizer = build_optimizer(model, config)
                parameters_before = [
                    parameter.detach().clone()
                    for parameter in model.parameters()
                ]

                statistics = train_one_epoch(
                    model,
                    [make_train_batch(has_context=True)],
                    optimizer,
                    torch.device("cpu"),
                )

                self.assertEqual(statistics["train_optimizer_steps"], 1)
                self.assertIsNotNone(statistics["train_loss"])
                self.assertTrue(
                    any(
                        not torch.equal(before, after)
                        for before, after in zip(
                            parameters_before, model.parameters()
                        )
                    )
                )

    def test_training_converts_float16_features_to_float32(self):
        model = DtypeRecordingModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        train_one_epoch(
            model,
            [make_train_batch(has_context=True)],
            optimizer,
            torch.device("cpu"),
        )

        self.assertEqual(model.seen_dtype, torch.float32)

    def test_validation_keeps_all_events_and_uses_frequency_fallback(self):
        labels = np.zeros(6, dtype=np.int64)
        frequency_prior = FrequencyPriorSTA.fit(
            labels,
            labels,
            num_verbs=6,
            num_nouns=6,
        )
        action_pairs = np.column_stack(
            (np.arange(6, dtype=np.int64),) * 2
        )

        metrics = validate_one_epoch(
            NoContextModel(),
            [make_validation_batch()],
            torch.device("cpu"),
            frequency_prior,
            action_pairs,
        )

        self.assertEqual(metrics["validation_events"], 6)
        self.assertEqual(metrics["validation_usable_events"], 0)
        self.assertEqual(metrics["validation_fallback_events"], 6)
        self.assertIsNone(metrics["validation_loss"])
        for metric_name in (
            "verb_top1_accuracy",
            "verb_top5_accuracy",
            "verb_mt5r",
            "noun_top1_accuracy",
            "noun_top5_accuracy",
            "noun_mt5r",
            "action_top1_accuracy",
            "action_top5_accuracy",
            "action_mt5r",
        ):
            with self.subTest(metric=metric_name):
                self.assertEqual(metrics[metric_name], 1.0)


if __name__ == "__main__":
    unittest.main()
