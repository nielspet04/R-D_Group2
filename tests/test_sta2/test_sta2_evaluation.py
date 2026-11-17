"""Focused tests for independent STA2 checkpoint evaluation."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.evaluation import build_action_pairs, joint_action_scores
from src.sta2.data import sha256_file
from src.sta2.evaluation import (
    action_vocabulary_sha256,
    assert_reloaded_predictions_equal,
    build_head_from_checkpoint,
    collect_local_predictions,
    evaluation_action_pairs,
    frequency_priors,
    load_evaluation_checkpoint,
    merge_prediction_records,
    save_predictions_atomic,
    score_prediction_records,
    summarize_unlabelled_records,
    validate_checkpoint_metadata,
    verify_evaluation_inputs,
)
from src.sta2.head import AnticipationHead
from src.sta2.model import STA2Model


def make_checkpoint(
    *,
    observed_only: bool = False,
    embed_dim: int = 8,
    action_vocabulary: tuple[tuple[int, int], ...] | None = None,
) -> dict:
    vocabulary = action_vocabulary or tuple(
        (index, index) for index in range(5)
    )
    head = AnticipationHead(
        embed_dim=embed_dim,
        num_actions=len(vocabulary),
        num_verbs=6,
        num_nouns=6,
        num_heads=2,
        mlp_ratio=2.0,
        depth=2,
        use_activation_checkpointing=False,
    )
    return {
        "schema_version": 1,
        "head_state": head.state_dict(),
        "optimizer_state": {},
        "scheduler_state": {},
        "epoch": 3,
        "optimizer_step": 12,
        "sampler_epoch": 3,
        "rng_states": [{}],
        "configuration": {
            "schema_version": 1,
            "training": {"seed": 42, "bf16": False},
            "head": {
                "num_verbs": 6,
                "num_nouns": 6,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "depth": 2,
                "use_activation_checkpointing": False,
            },
            "data": {
                "workers_per_rank": 0,
                "pin_memory": False,
                "prefetch_factor": 1,
            },
            "experiment": {
                "observed_only": observed_only,
                "selection_metric": "action_mt5r",
            },
            "paths": {},
            "world_size": 1,
        },
        "backbone_provenance": {
            "checkpoint_sha256": "a" * 64,
            "upstream": {},
        },
        "split_hashes": {
            "annotations": "b" * 64,
            "train_split": "c" * 64,
            "validation_split": "d" * 64,
        },
        "action_vocabulary": [list(pair) for pair in vocabulary],
        "best_metric": 0.25,
        "epochs_without_improvement": 2,
        "git_commit": "deadbeef",
        "git_dirty": False,
    }


class TinyBackbone(nn.Module):
    embed_dim = 6

    def __init__(self, observed_only: bool = False) -> None:
        super().__init__()
        self.observed_only = observed_only

    def forward(self, video, anticipation_seconds):
        summary = video.mean(dim=(1, 2, 3, 4)).reshape(-1, 1, 1)
        observed = summary.expand(-1, 2, self.embed_dim)
        if self.observed_only:
            future = observed.new_empty(len(video), 0, self.embed_dim)
        else:
            future = (summary + 1).expand(-1, 1, self.embed_dim)
        return {
            "observed_tokens": observed,
            "future_tokens": future,
        }


class FixedHead(nn.Module):
    embed_dim = 6

    def __init__(self, *, nonfinite_auxiliary: bool = False) -> None:
        super().__init__()
        self.verb_classifier = nn.Linear(6, 6)
        self.noun_classifier = nn.Linear(6, 6)
        self.action_classifier = nn.Linear(6, 5)
        self.nonfinite_auxiliary = nonfinite_auxiliary

    def forward(self, observed_tokens, future_tokens):
        values = observed_tokens[:, 0, 0]
        batch_size = len(values)
        verbs = torch.zeros(batch_size, 6, device=values.device)
        nouns = torch.zeros(batch_size, 6, device=values.device)
        indices = values.round().long().remainder(6)
        verbs.scatter_(1, indices[:, None], 5.0)
        nouns.scatter_(1, indices[:, None], 4.0)
        actions = torch.zeros(batch_size, 5, device=values.device)
        # Deliberately rank auxiliary action zero highest. Primary rankings
        # must remain determined only by the verb and noun branches.
        actions[:, 0] = 1000.0
        if self.nonfinite_auxiliary:
            actions[0, 0] = float("nan")
        return {
            "verb_logits": verbs,
            "noun_logits": nouns,
            "action_logits": actions,
        }


def make_batch() -> dict:
    statuses = [
        "full",
        "no_context",
        "video_error",
        "partial",
        "full",
        "full",
    ]
    has_context = torch.tensor(
        [status not in {"no_context", "video_error"} for status in statuses]
    )
    return {
        "video": torch.stack([
            torch.full((3, 2, 2, 2), float(index))
            for index in range(6)
        ]),
        "has_context": has_context,
        "row_index": torch.tensor([2, 7, 10, 20, 30, 40]),
        "narration_id": [f"event-{index}" for index in range(6)],
        "status": statuses,
        "verb_label": torch.tensor([0, 1, 2, 3, 4, 5]),
        "noun_label": torch.tensor([0, 1, 2, 3, 4, 0]),
    }


def probability_record(
    row_index: int,
    narration_id: str,
    verb: int,
    noun: int,
) -> dict:
    verb_scores = np.full(6, 0.02, dtype=np.float32)
    noun_scores = np.full(6, 0.02, dtype=np.float32)
    verb_scores[verb] = 0.9
    noun_scores[noun] = 0.9
    verb_scores /= verb_scores.sum()
    noun_scores /= noun_scores.sum()
    return {
        "row_index": row_index,
        "narration_id": narration_id,
        "status": "full",
        "used_frequency_prior": False,
        "verb_label": verb,
        "noun_label": noun,
        "verb_probabilities": verb_scores,
        "noun_probabilities": noun_scores,
    }


class STA2EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_checkpoint_reconstructs_exact_head_for_both_modes(self):
        for observed_only in (False, True):
            checkpoint = make_checkpoint(observed_only=observed_only)
            metadata = validate_checkpoint_metadata(checkpoint)
            rebuilt = build_head_from_checkpoint(checkpoint, embed_dim=8)

            self.assertEqual(metadata["observed_only"], observed_only)
            self.assertEqual(rebuilt.embed_dim, 8)
            self.assertEqual(rebuilt.num_actions, 5)
            for name, expected in checkpoint["head_state"].items():
                self.assertTrue(
                    torch.equal(rebuilt.state_dict()[name], expected)
                )

        corrupt = make_checkpoint()
        corrupt["head_state"]["verb_classifier.weight"] = torch.zeros(7, 8)
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_head_from_checkpoint(corrupt, embed_dim=8)

        wrong_configuration = make_checkpoint()
        wrong_configuration["configuration"]["experiment"][
            "selection_metric"
        ] = "auxiliary_action_accuracy"
        with self.assertRaisesRegex(ValueError, "selection metric"):
            validate_checkpoint_metadata(wrong_configuration)

    def test_checkpoint_loader_is_read_only_and_reload_is_deterministic(self):
        checkpoint = make_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save(checkpoint, path)
            before_hash = sha256_file(path)
            before_stat = path.stat()

            first_checkpoint = load_evaluation_checkpoint(path)
            first_head = build_head_from_checkpoint(
                first_checkpoint, embed_dim=8
            )
            tokens = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
            empty = torch.empty(1, 0, 8)
            first_output = first_head(tokens, empty)

            second_checkpoint = load_evaluation_checkpoint(path)
            second_head = build_head_from_checkpoint(
                second_checkpoint, embed_dim=8
            )
            second_output = second_head(tokens, empty)

            records_one = [{
                "row_index": 1,
                "narration_id": "one",
                "verb_probabilities": torch.softmax(
                    first_output["verb_logits"], 1
                ).detach().numpy()[0],
            }]
            records_two = [{
                "row_index": 1,
                "narration_id": "one",
                "verb_probabilities": torch.softmax(
                    second_output["verb_logits"], 1
                ).detach().numpy()[0],
            }]
            assert_reloaded_predictions_equal(records_one, records_two)

            after_stat = path.stat()
            self.assertEqual(before_hash, sha256_file(path))
            self.assertEqual(before_stat.st_size, after_stat.st_size)
            self.assertEqual(before_stat.st_mtime_ns, after_stat.st_mtime_ns)

    def test_provenance_and_action_vocabulary_are_verified(self):
        vocabulary = ((0, 0), (1, 1), (2, 2))
        checkpoint = make_checkpoint(action_vocabulary=vocabulary)
        train_events = pd.DataFrame({
            "verb_class": [0, 1, 2, 0],
            "noun_class": [0, 1, 2, 0],
        })

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "backbone": root / "backbone.pt",
                "annotations": root / "annotations.csv",
                "train_split": root / "train.csv",
                "validation_split": root / "validation.csv",
            }
            for name, path in paths.items():
                path.write_text(f"{name}\n")
            checkpoint["backbone_provenance"]["checkpoint_sha256"] = (
                sha256_file(paths["backbone"])
            )
            checkpoint["split_hashes"] = {
                name: sha256_file(paths[name])
                for name in ("annotations", "train_split", "validation_split")
            }

            result = verify_evaluation_inputs(
                checkpoint,
                backbone_checkpoint=paths["backbone"],
                annotations=paths["annotations"],
                train_split=paths["train_split"],
                validation_split=paths["validation_split"],
                train_events=train_events,
            )
            self.assertEqual(
                result["auxiliary_action_vocabulary_sha256"],
                action_vocabulary_sha256(vocabulary),
            )

            wrong_vocabulary = make_checkpoint(
                action_vocabulary=((0, 0), (1, 1), (2, 3))
            )
            wrong_vocabulary["backbone_provenance"] = checkpoint[
                "backbone_provenance"
            ]
            wrong_vocabulary["split_hashes"] = checkpoint["split_hashes"]
            with self.assertRaisesRegex(ValueError, "action vocabulary"):
                verify_evaluation_inputs(
                    wrong_vocabulary,
                    backbone_checkpoint=paths["backbone"],
                    annotations=paths["annotations"],
                    train_split=paths["train_split"],
                    validation_split=paths["validation_split"],
                    train_events=train_events,
                )

            paths["backbone"].write_text("changed backbone\n")
            with self.assertRaisesRegex(ValueError, "backbone checkpoint"):
                verify_evaluation_inputs(
                    checkpoint,
                    backbone_checkpoint=paths["backbone"],
                    annotations=paths["annotations"],
                    train_split=paths["train_split"],
                    validation_split=paths["validation_split"],
                    train_events=train_events,
                )
            paths["backbone"].write_text("backbone\n")

            paths["validation_split"].write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "validation_split"):
                verify_evaluation_inputs(
                    checkpoint,
                    backbone_checkpoint=paths["backbone"],
                    annotations=paths["annotations"],
                    train_split=paths["train_split"],
                    validation_split=paths["validation_split"],
                    train_events=train_events,
                )

    def test_fallback_primary_metrics_and_unknown_auxiliary_pair(self):
        model = STA2Model(TinyBackbone(), FixedHead())
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        verb_prior = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.2, 0.3])
        noun_prior = torch.tensor([0.3, 0.2, 0.2, 0.15, 0.1, 0.05])
        records = collect_local_predictions(
            model,
            [make_batch()],
            device=torch.device("cpu"),
            verb_prior=verb_prior,
            noun_prior=noun_prior,
            bf16=False,
        )

        np.testing.assert_array_equal(
            records[1]["verb_probabilities"], verb_prior.numpy()
        )
        np.testing.assert_array_equal(
            records[2]["noun_probabilities"], noun_prior.numpy()
        )
        self.assertFalse(records[0]["used_frequency_prior"])
        self.assertFalse(records[3]["used_frequency_prior"])

        action_pairs = build_action_pairs(
            np.array([0, 1, 2, 3, 4, 5]),
            np.array([0, 1, 2, 3, 4, 0]),
        )
        metrics, enriched = score_prediction_records(
            records,
            action_pairs=action_pairs,
            auxiliary_action_vocabulary=tuple(
                (index, index) for index in range(5)
            ),
        )
        self.assertEqual(metrics["validation_events"], 6)
        self.assertEqual(metrics["prior_fallback_count"], 2)
        self.assertEqual(
            metrics["unsupported_auxiliary_action_label_count"], 1
        )
        self.assertEqual(metrics["event_counts_by_status"]["partial"], 1)
        for name in (
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
            self.assertTrue(np.isfinite(metrics[name]))

        expected_scores = joint_action_scores(
            records[5]["verb_probabilities"][None, :],
            records[5]["noun_probabilities"][None, :],
            action_pairs,
        )[0]
        expected_top = int(np.argmax(expected_scores))
        self.assertEqual(
            int(enriched[5]["primary_action_top5_ids"][0]), expected_top
        )
        # The fixed auxiliary head always ranks ID 0 highest; event five's
        # primary action is a different valid pair.
        self.assertNotEqual(expected_top, 0)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, parameters_before[name]))

    def test_exact_sampler_padding_is_deduplicated_and_bad_coverage_rejected(self):
        events = pd.DataFrame({
            "row_index": [2, 7, 10],
            "narration_id": ["two", "seven", "ten"],
            "verb_class": [0, 1, 2],
            "noun_class": [0, 1, 2],
        })
        records = [
            probability_record(2, "two", 0, 0),
            probability_record(7, "seven", 1, 1),
            probability_record(10, "ten", 2, 2),
            probability_record(2, "two", 0, 0),
        ]
        merged = merge_prediction_records(records, events, world_size=2)
        self.assertEqual(
            [record["row_index"] for record in merged], [2, 7, 10]
        )

        with self.assertRaisesRegex(ValueError, "padding"):
            merge_prediction_records(records[:-1], events, world_size=2)
        with self.assertRaisesRegex(ValueError, "missing prediction"):
            merge_prediction_records(
                [records[0], records[1], records[3]],
                events,
                world_size=2,
            )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            conflicting = [dict(record) for record in records]
            conflicting[-1]["status"] = "partial"
            merge_prediction_records(conflicting, events, world_size=2)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            unexpected = records + [
                probability_record(99, "other", 0, 0)
            ]
            merge_prediction_records(unexpected, events, world_size=2)

        duplicate_events = pd.concat([events, events.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "duplicate row"):
            merge_prediction_records(records, duplicate_events, world_size=2)

    def test_nonfinite_predictions_are_rejected(self):
        model = STA2Model(
            TinyBackbone(), FixedHead(nonfinite_auxiliary=True)
        )
        prior = torch.full((6,), 1 / 6)
        with self.assertRaisesRegex(ValueError, "nonfinite action_logits"):
            collect_local_predictions(
                model,
                [make_batch()],
                device=torch.device("cpu"),
                verb_prior=prior,
                noun_prior=prior,
                bf16=False,
            )

        events = pd.DataFrame({
            "row_index": [2],
            "narration_id": ["two"],
            "verb_class": [0],
            "noun_class": [0],
        })
        record = probability_record(2, "two", 0, 0)
        record["verb_probabilities"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid per-event"):
            merge_prediction_records([record], events, world_size=1)

    def test_priors_action_identity_hidden_coverage_and_saved_order(self):
        train = pd.DataFrame({
            "verb_class": [0, 0, 1],
            "noun_class": [2, 2, 3],
        })
        verb_prior, noun_prior = frequency_priors(
            train, num_verbs=6, num_nouns=6
        )
        self.assertAlmostEqual(float(verb_prior.sum()), 1.0)
        self.assertAlmostEqual(float(noun_prior.sum()), 1.0)

        annotations = pd.DataFrame({
            "verb_class": [0, 1, 2, 3, 4, 5],
            "noun_class": [1, 2, 3, 4, 5, 0],
        })
        action_pairs = evaluation_action_pairs(
            annotations, num_verbs=6, num_nouns=6
        )
        hidden = []
        for row_index, narration_id in ((5, "five"), (1, "one")):
            value = probability_record(row_index, narration_id, 0, 1)
            value.pop("verb_label")
            value.pop("noun_label")
            hidden.append(value)
        summary, enriched = summarize_unlabelled_records(
            hidden, action_pairs=action_pairs
        )
        self.assertEqual(summary["test_events"], 2)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.pt"
            save_predictions_atomic(output, enriched, {"split": "test"})
            payload = torch.load(output, weights_only=True)
        self.assertEqual(
            [item["narration_id"] for item in payload["predictions"]],
            ["five", "one"],
        )
        self.assertEqual(payload["metadata"]["split"], "test")
        digest = action_vocabulary_sha256(action_pairs)
        self.assertEqual(len(digest), hashlib.sha256().digest_size * 2)


if __name__ == "__main__":
    unittest.main()
