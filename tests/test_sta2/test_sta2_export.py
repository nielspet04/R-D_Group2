"""E5 contract tests, using real files and synthetic CPU predictions."""

import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.compare_sta2 import (
    SYSTEMS, audit_feature_events, compare, metric_row, presentation_table,
)
from scripts.export_sta2 import export_submission, save_submission_atomic
from scripts.validate_sta_submission import validate_file, validate_submission
from src.sta2.artifacts import comparison_identity, require_compatible
from src.sta2.data import load_events, sha256_file
from src.sta2.evaluation import (
    action_vocabulary_sha256, evaluation_action_pairs, frequency_priors,
    save_predictions_atomic,
)
from src.sta2.head import AnticipationHead
from src.sta_features import save_sta_feature_shard
from src.sta_training import TrainingConfig, build_model
from tests.test_sta2.test_sta2_evaluation import make_checkpoint


def payload():
    return [{"narration_id": name, "verb_output": torch.full((97,), 1 / 97),
             "noun_output": torch.full((300,), 1 / 300)} for name in ("z", "a")]


class SubmissionValidatorTests(unittest.TestCase):
    def test_valid_payload_and_reload_are_read_only(self):
        """Validating in memory and on disk preserves every tensor and byte."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.pt"
            value = payload()
            original = copy.deepcopy(value)
            torch.save(value, path)
            digest = sha256_file(path)
            self.assertEqual(validate_file(path, ["z", "a"]), {"events": 2, "valid": True})
            validate_submission(value, ["z", "a"])
            self.assertEqual(digest, sha256_file(path))
            for left, right in zip(value, original):
                for key in ("verb_output", "noun_output"):
                    self.assertTrue(torch.equal(left[key], right[key]))

    def test_rejects_structure_ids_coverage_and_order(self):
        """Reject outer types, key drift, bad IDs, missing/extra/duplicate IDs and reordering."""
        cases = [({}, "outer list"), ([None], "exact keys"),
                 (payload()[:1], "missing IDs"), (payload()[::-1], "order")]
        for field in ("action_output", "extra"):
            value = payload()
            value[0][field] = torch.ones(5)
            cases.append((value, "exact keys"))
        value = payload()
        del value[0]["noun_output"]
        cases.append((value, "exact keys"))
        for bad_id, message in ((123, "string"), ("", "string"), ("unknown", "unexpected IDs"), ("a", "duplicate")):
            value = payload()
            value[0]["narration_id"] = bad_id
            cases.append((value, message))
        for value, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_submission(value, ["z", "a"])
        for expected in ([], ["z", "z"]):
            with self.assertRaisesRegex(ValueError, "nonempty and unique"):
                validate_submission(payload(), expected)

    def test_rejects_every_invalid_vector_condition_for_both_heads(self):
        """Both heads reject non-tensors, shape/dtype/device errors, NaN/Inf, negatives and bad sums."""
        for key, size in (("verb_output", 97), ("noun_output", 300)):
            cases = [([1 / size] * size, "tensor"),
                     (torch.ones(size - 1), "shape"), (torch.ones(1, size), "shape"),
                     (torch.ones(size, dtype=torch.float64), "CPU FP32"),
                     (torch.ones(size, dtype=torch.float16), "CPU FP32"),
                     (torch.ones(size, dtype=torch.int64), "CPU FP32"),
                     (torch.ones(size, device="meta"), "CPU FP32"),
                     (torch.ones(size).to_sparse(), "dense tensor"),
                     (torch.ones(size), "sum to one")]
            for number, message in ((float("nan"), "NaN"), (float("inf"), "infinity"), (-0.01, "negative")):
                vector = torch.full((size,), 1 / size)
                vector[0] = number
                cases.append((vector, message))
            for vector, message in cases:
                with self.subTest(key=key, message=message), self.assertRaisesRegex(ValueError, message):
                    value = payload()
                    value[0][key] = vector
                    validate_submission(value, ["z", "a"])

    def test_corrupt_and_missing_files_have_actionable_errors(self):
        """File load failures identify the submission that could not be reloaded."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.pt"
            with self.assertRaisesRegex(ValueError, "cannot reload submission"):
                validate_file(path, ["z", "a"])
            path.write_bytes(b"not a torch file")
            with self.assertRaisesRegex(ValueError, "cannot reload submission"):
                validate_file(path, ["z", "a"])

    def test_atomic_save_validates_reload_and_preserves_destination_on_failure(self):
        """A failed save leaves the old file intact and removes the temporary file."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.pt"
            save_submission_atomic(path, payload(), ["z", "a"])
            digest = sha256_file(path)
            with patch("scripts.export_sta2.torch.save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    save_submission_atomic(path, payload(), ["z", "a"])
            self.assertEqual(digest, sha256_file(path))
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertTrue(validate_file(path, ["z", "a"])["valid"])


class E5ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = argparse.Namespace(**{name: self.root / (name + ".csv") for name in
                                         ("annotations", "train_split", "validation_split", "events")})
        rows = [{"narration_id": f"{participant}-{index}", "participant_id": participant,
                 "video_id": f"{participant}_01", "start_timestamp": "00:00:06.000",
                 "verb_class": index, "noun_class": index}
                for participant in ("P01", "P02") for index in range(5)]
        table = pd.DataFrame(rows)
        table.to_csv(self.args.annotations, index=False)
        table.iloc[:5].to_csv(self.args.train_split, index=False)
        table.iloc[5:].to_csv(self.args.validation_split, index=False)
        pd.DataFrame({"narration_id": ["z", "a", "m", "b"], "video_id": ["P03_01"] * 4,
                      "sta_observation_end_seconds": [5.0, -0.5, 2.0, 5.0]}).to_csv(self.args.events, index=False)
        self.train = load_events(self.args.annotations, split="train", train_split=self.args.train_split,
                                 validation_split=self.args.validation_split)
        self.pairs = evaluation_action_pairs(table, num_verbs=97, num_nouns=300)
        self.priors = frequency_priors(self.train, num_verbs=97, num_nouns=300)
        self.args.backbone_checkpoint = self.root / "backbone.pt"
        self.args.backbone_checkpoint.write_bytes(b"synthetic backbone; no model constructed")
        self.args.checkpoint = self.root / "head.pt"
        self.args.predictions = self.root / "predictions.pt"
        self.args.output = self.root / "submission.pt"

    def artifact(self, observed_only=False, split="test"):
        """Build a genuine E4-schema artifact with verified hashes and both fallback statuses."""
        checkpoint = make_checkpoint(observed_only=observed_only)
        checkpoint["configuration"]["head"].update(num_verbs=97, num_nouns=300)
        checkpoint["head_state"] = AnticipationHead(
            embed_dim=8, num_actions=5, num_verbs=97, num_nouns=300,
            num_heads=2, mlp_ratio=2.0, depth=2, use_activation_checkpointing=False,
        ).state_dict()
        checkpoint["backbone_provenance"]["checkpoint_sha256"] = sha256_file(self.args.backbone_checkpoint)
        checkpoint["split_hashes"] = {key: sha256_file(getattr(self.args, key)) for key in
                                      ("annotations", "train_split", "validation_split")}
        torch.save(checkpoint, self.args.checkpoint)
        source = self.args.events if split == "test" else self.args.annotations
        events = load_events(source, split=split, train_split=self.args.train_split,
                             validation_split=self.args.validation_split)
        metadata = {
            **comparison_identity(self.args.annotations, self.args.train_split, self.args.validation_split, self.pairs),
            "split": split, "target_events_sha256": sha256_file(source),
            "checkpoint_sha256": sha256_file(self.args.checkpoint),
            "backbone_sha256": sha256_file(self.args.backbone_checkpoint),
            "checkpoint_configuration": checkpoint["configuration"], "observed_only": observed_only,
            "auxiliary_action_vocabulary_sha256": action_vocabulary_sha256(checkpoint["action_vocabulary"]),
            "fresh_reload_verified": True, "primary_action_source": "primary_verb_and_noun_heads",
        }
        records = []
        for index, row in enumerate(events.to_dict("records")):
            status = ("full", "no_context", "video_error", "partial")[index % 4]
            record = {"row_index": int(row["row_index"]), "narration_id": row["narration_id"],
                      "status": status, "used_frequency_prior": status in {"no_context", "video_error"},
                      "verb_probabilities": self.priors[0].copy(), "noun_probabilities": self.priors[1].copy()}
            if split == "validation":
                record.update(verb_label=int(row["verb_class"]), noun_label=int(row["noun_class"]))
            records.append(record)
        # Scrambled input must be restored to annotation order through E4 coverage validation.
        save_predictions_atomic(self.args.predictions, records[::-1], metadata)
        return metadata, records

    def test_export_exact_schema_fallback_both_modes_and_read_only_inputs(self):
        """Both modes record provenance, preserve priors/order, reload, and leave inputs untouched."""
        for mode in (True, False):
            with self.subTest(observed_only=mode):
                _, records = self.artifact(mode)
                digest = sha256_file(self.args.checkpoint)
                result = export_submission(self.args)
                self.assertEqual(result["observed_only"], mode)
                self.assertEqual(digest, sha256_file(self.args.checkpoint))
                self.assertTrue(self.args.output.with_suffix(".pt.provenance.json").is_file())
                value = torch.load(self.args.output, weights_only=True)
                validate_submission(value, ["z", "a", "m", "b"])
                for exported, original in zip(value, records):
                    self.assertEqual(set(exported), {"narration_id", "verb_output", "noun_output"})
                    np.testing.assert_array_equal(exported["verb_output"].numpy(), original["verb_probabilities"])
                    np.testing.assert_array_equal(exported["noun_output"].numpy(), original["noun_probabilities"])

    def test_export_rejects_provenance_drift_and_fallback_replacement(self):
        """Wrong checkpoint/config/data/vocabulary/mode metadata or changed fallback blocks export."""
        for field in ("checkpoint_sha256", "backbone_sha256", "checkpoint_configuration", "split_hashes",
                      "target_events_sha256", "valid_action_vocabulary_sha256", "observed_only",
                      "auxiliary_action_vocabulary_sha256", "fresh_reload_verified", "primary_action_source"):
            with self.subTest(field=field):
                metadata, records = self.artifact()
                metadata[field] = "changed"
                save_predictions_atomic(self.args.predictions, records, metadata)
                with self.assertRaises(ValueError):
                    export_submission(self.args)
        metadata, records = self.artifact()
        records[1]["verb_probabilities"] = np.full(97, 1 / 97, dtype=np.float32)
        save_predictions_atomic(self.args.predictions, records, metadata)
        with self.assertRaisesRegex(ValueError, "fallback differs"):
            export_submission(self.args)

    def test_export_rejects_source_overwrite(self):
        """Destination checks protect checkpoint and source artifacts from accidental replacement."""
        self.artifact()
        self.args.output = self.args.checkpoint
        with self.assertRaisesRegex(ValueError, "overwrite"):
            export_submission(self.args)

    def comparison_args(self):
        return argparse.Namespace(annotations=self.args.annotations, train_split=self.args.train_split,
                                  validation_split=self.args.validation_split, output=self.root / "comparison.json",
                                  feature_root=None, mean_checkpoint=None, gru_checkpoint=None,
                                  observed_only=None, observed_plus_future=None)

    def test_comparison_generates_prior_marks_missing_runs_and_is_deterministic(self):
        """Local annotations generate all nine prior metrics; missing models have reasons and no invented scores."""
        args = self.comparison_args()
        first = compare(args)
        encoded = args.output.read_bytes()
        second = compare(args)
        self.assertEqual(first, second)
        self.assertEqual(encoded, args.output.read_bytes())
        self.assertEqual([row["system"] for row in first["systems"]], list(SYSTEMS))
        self.assertEqual(len(first["systems"][0]["metrics"]), 9)
        for row in first["systems"][1:]:
            self.assertFalse(row["available"])
            self.assertTrue(row["reason"])
            self.assertNotIn("metrics", row)
        self.assertIn("Unavailable", presentation_table(first["systems"]))

    def test_comparison_rescores_e4_and_rejects_split_or_vocabulary_mismatch(self):
        """Use E4 primary scoring and full coverage; incompatible supplied runs fail instead of disappearing."""
        metadata, records = self.artifact(True, "validation")
        args = self.comparison_args()
        args.observed_only = self.args.predictions
        result = compare(args)
        row = result["systems"][3]
        self.assertTrue(row["available"])
        self.assertEqual(row["counts"]["events"], 5)
        for key in comparison_identity(args.annotations, args.train_split, args.validation_split, self.pairs):
            with self.subTest(key=key):
                changed = {**metadata, key: "wrong"}
                save_predictions_atomic(self.args.predictions, records, changed)
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    compare(args)

    def test_metric_scale_and_missing_metric_rejected(self):
        """Percent-scale, nonfinite and missing metrics cannot enter the comparison JSON."""
        row = compare(self.comparison_args())["systems"][0]
        for value in (50, float("nan"), -0.01, None, True):
            metrics = {**row["metrics"], "action_mt5r": value, "validation_events": 5}
            with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
                metric_row("bad", metrics, row["metadata"])

    def test_r3d_baselines_generated_with_existing_evaluator(self):
        """Real mean/GRU checkpoints and shards are audited and reevaluated with complete metric/count output."""
        args = self.comparison_args()
        args.feature_root = self.root / "features"
        for split, source in (("train", args.train_split), ("validation", args.validation_split)):
            table = pd.read_csv(source)
            directory = args.feature_root / split
            shard = {
                "schema_version": 1, "split": split,
                "features": torch.zeros(5, 5, 512, dtype=torch.float16),
                "valid_mask": torch.tensor([[True] * 5] * 3 + [[False] * 5] * 2),
                "narration_ids": table.narration_id.tolist(), "video_ids": table.video_id.tolist(),
                "status": ["full"] * 3 + ["no_context", "video_error"],
                "action_start_seconds": torch.full((5,), 6.0),
                "observation_end_seconds": torch.full((5,), 5.0),
                "verb_labels": torch.arange(5), "noun_labels": torch.arange(5),
            }
            save_sta_feature_shard(shard, directory / "shard_00000.pt")
            pd.DataFrame([{
                "schema_version": 1, "split": split, "shard_index": 0,
                "shard_file": "shard_00000.pt", "start_index": 0,
                "end_index_exclusive": 5, "event_count": 5,
            }]).to_csv(directory / "manifest.csv", index=False)
        for model, field in (("mean_pool", "mean_checkpoint"), ("gru", "gru_checkpoint")):
            config = TrainingConfig(model=model, hidden_size=8)
            path = self.root / (model + ".pt")
            torch.save({
                "schema_version": 1, "configuration": config.to_dict(),
                "model_state": build_model(config).state_dict(), "optimizer_state": {},
                "epoch": 1, "seed": 42, "validation_metrics": {},
                "git_commit": "fixture", "git_dirty": False,
            }, path)
            setattr(args, field, path)
        result = compare(args)
        for row in result["systems"][1:3]:
            self.assertTrue(row["available"])
            self.assertEqual(row["counts"], {"events": 5, "usable_context": 3, "fallback": 2, "valid_actions": 5})
            self.assertEqual(len(row["metrics"]), 9)

    def test_cache_audit_rejects_wrong_ids_labels_cutoffs_and_coverage(self):
        """Legacy caches must match current validation identities, labels, videos and cutoffs."""
        events = self.train
        items = [{"narration_id": row.narration_id, "video_id": row.video_id,
                  "verb_label": row.verb_class, "noun_label": row.noun_class,
                  "observation_end_seconds": 5.0} for row in events.itertuples()]
        class Cache(list):
            split = "train"
        audit_feature_events(Cache(items), events)
        for key, value in (("narration_id", "unknown"), ("verb_label", 96),
                           ("noun_label", 299), ("video_id", "P99_01"),
                           ("observation_end_seconds", 5.1)):
            changed = copy.deepcopy(items)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit_feature_events(Cache(changed), events)
        with self.assertRaisesRegex(ValueError, "missing"):
            audit_feature_events(Cache(items[:-1]), events)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_feature_events(Cache(items + [items[0]]), events)


if __name__ == "__main__":
    unittest.main()
