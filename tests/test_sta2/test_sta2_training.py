"""Tests for STA2 accumulation, validation, and checkpoints."""

import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from src.evaluation import build_action_pairs
from src.sta2.head import AnticipationHead
from src.sta2.model import STA2Model
from src.sta2.training import (
    DistributedContext,
    STA2TrainingConfig,
    accumulation_steps,
    build_optimizer,
    build_scheduler,
    capture_rng_state,
    fit,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EpochProbe(Dataset):
    epoch = 0

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return self.epoch


def epoch_collate(items):
    return items


class TinyLoader(list):
    @property
    def dataset(self):
        return self


class TinyBackbone(nn.Module):
    embed_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, video, anticipation_seconds):
        self.calls += 1
        batch_size = len(video)
        summary = video.mean(
            dim=(1, 2, 3, 4)
        ).reshape(batch_size, 1, 1)
        return {
            "observed_tokens": (
                summary * self.weight
            ).expand(batch_size, 2, self.embed_dim),
            "future_tokens": (
                (summary + 1.0) * self.weight
            ).expand(batch_size, 1, self.embed_dim),
        }


def make_batch(
    row_index: int,
    *,
    usable: bool = True,
    verb: int = 0,
    noun: int = 0,
):
    return {
        "video": torch.full(
            (1, 3, 2, 2, 2),
            float(row_index + 1),
        ),
        "has_context": torch.tensor([usable]),
        "verb_label": torch.tensor([verb]),
        "noun_label": torch.tensor([noun]),
        "row_index": torch.tensor([row_index]),
    }


class STA2TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.context = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )
        self.config = STA2TrainingConfig(
            learning_rate=1e-3,
            max_epochs=2,
            warmup_epochs=1,
            microbatch_size=1,
            effective_global_batch_size=2,
            bf16=False,
        )
        self.vocabulary = tuple(
            (index, index) for index in range(6)
        )

    def make_model(self):
        backbone = TinyBackbone()
        head = AnticipationHead(
            embed_dim=8,
            num_actions=6,
            num_verbs=6,
            num_nouns=6,
            num_heads=2,
            depth=2,
            use_activation_checkpointing=False,
        )
        return STA2Model(backbone, head)

    def make_optimizer_scheduler(self, model, windows=2):
        optimizer = build_optimizer(model.head, self.config)
        scheduler = build_scheduler(
            optimizer,
            windows_per_epoch=windows,
            config=self.config,
        )
        return optimizer, scheduler

    def test_accumulation_steps_preserve_global_batch(self):
        self.assertEqual(accumulation_steps(self.config, 1), 2)
        self.assertEqual(accumulation_steps(self.config, 2), 1)

        with self.assertRaisesRegex(ValueError, "divisible"):
            accumulation_steps(self.config, 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for mapped RNG regression")
    def test_cuda_resume_restores_rng_tensors_mapped_to_gpu(self):
        state = capture_rng_state()
        expected = torch.rand(4, device="cuda")
        # Reproduce torch.load(map_location=context.device) on RNG tensors.
        state["torch"] = state["torch"].cuda()
        state["cuda"] = [value.cuda() for value in state["cuda"]]
        restore_rng_state(state)
        self.assertTrue(torch.equal(expected, torch.rand(4, device="cuda")))

    def test_training_handles_remainder_and_updates_head_only(self):
        model = self.make_model()
        optimizer, scheduler = self.make_optimizer_scheduler(model)
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.head.named_parameters()
        }
        batches = [
            make_batch(0, verb=0, noun=0),
            make_batch(1, verb=1, noun=1),
            make_batch(2, verb=2, noun=2),
        ]

        metrics = train_one_epoch(
            model,
            batches,
            optimizer,
            scheduler,
            self.vocabulary,
            self.config,
            self.context,
        )

        self.assertEqual(metrics["train_events"], 3)
        self.assertEqual(metrics["train_usable_events"], 3)
        self.assertEqual(metrics["train_optimizer_steps"], 2)
        self.assertTrue(np.isfinite(metrics["train_loss"]))
        self.assertIsNone(model.backbone.weight.grad)
        self.assertTrue(any(
            not torch.equal(before[name], parameter)
            for name, parameter in model.head.named_parameters()
        ))

    def test_entirely_invalid_window_skips_update(self):
        model = self.make_model()
        optimizer, scheduler = self.make_optimizer_scheduler(model)
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.head.named_parameters()
        }
        metrics = train_one_epoch(
            model,
            [
                make_batch(0, usable=False),
                make_batch(1, usable=False),
            ],
            optimizer,
            scheduler,
            self.vocabulary,
            self.config,
            self.context,
        )

        self.assertEqual(metrics["train_optimizer_steps"], 0)
        self.assertEqual(metrics["train_skipped_windows"], 1)
        self.assertEqual(metrics["train_usable_events"], 0)
        self.assertEqual(model.backbone.calls, 0)

        for name, parameter in model.head.named_parameters():
            self.assertTrue(torch.equal(before[name], parameter))

    def test_validation_keeps_invalid_events_with_prior(self):
        model = self.make_model()
        batches = [
            make_batch(0, verb=0, noun=0),
            make_batch(1, usable=False, verb=1, noun=1),
            make_batch(2, verb=2, noun=2),
            make_batch(3, verb=3, noun=3),
            make_batch(4, verb=4, noun=4),
            make_batch(5, verb=5, noun=5),
        ]
        prior = torch.full((6,), 1.0 / 6.0)
        pairs = build_action_pairs(
            np.arange(6),
            np.arange(6),
        )
        metrics = validate_one_epoch(
            model,
            batches,
            self.vocabulary[:-1],
            pairs,
            prior,
            prior,
            6,
            self.config,
            self.context,
        )

        self.assertEqual(metrics["validation_events"], 6)
        self.assertEqual(metrics["validation_usable_events"], 5)
        self.assertEqual(metrics["validation_fallback_events"], 1)
        # Five usable rows, of which one pair is unseen in training.
        # All six rows still participate in the evaluation metrics.
        self.assertEqual(metrics["validation_action_loss_events"], 4)
        self.assertTrue(np.isfinite(metrics["validation_loss"]))
        self.assertTrue(np.isfinite(metrics["action_mt5r"]))

    def test_workers_receive_each_new_epoch(self):
        from scripts.train_sta2 import _loader

        dataset = EpochProbe()
        loader = _loader(
            dataset, DistributedSampler(dataset, num_replicas=1, rank=0),
            config=self.config,
            data_config={"workers_per_rank": 1, "pin_memory": False, "prefetch_factor": 1},
        )
        loader.collate_fn = epoch_collate
        dataset.epoch = 1
        self.assertEqual(list(loader), [[1], [1]])
        dataset.epoch = 2
        self.assertEqual(list(loader), [[2], [2]])

    def test_validation_padding_does_not_change_loss_denominators(self):
        model = self.make_model()
        batches = [make_batch(i, verb=i, noun=i) for i in range(6)]
        prior = torch.full((6,), 1 / 6)
        args = (model, batches, self.vocabulary[:-1],
                build_action_pairs(np.arange(6), np.arange(6)),
                prior, prior, 6, self.config, self.context)
        expected = validate_one_epoch(*args)

        def gather_with_padding(gathered, records):
            gathered[0] = records + [records[0]]

        with patch("src.sta2.training.dist.is_initialized", return_value=True), \
             patch("src.sta2.training.dist.all_gather_object", side_effect=gather_with_padding), \
             patch("src.sta2.training.dist.broadcast_object_list"):
            padded = validate_one_epoch(*args)
        self.assertEqual(expected, padded)

    def test_walltime_stop_can_resume_the_next_epoch(self):
        model = self.make_model()
        optimizer, scheduler = self.make_optimizer_scheduler(model)
        batches = TinyLoader(make_batch(i, verb=i, noun=i) for i in range(6))
        with tempfile.TemporaryDirectory() as directory:
            arguments = dict(
                model=model, train_loader=batches, validation_loader=batches,
                train_sampler=Mock(), validation_sampler=Mock(), train_dataset=Mock(),
                action_vocabulary=self.vocabulary,
                evaluation_action_pairs=build_action_pairs(np.arange(6), np.arange(6)),
                verb_prior=torch.full((6,), 1 / 6), noun_prior=torch.full((6,), 1 / 6),
                optimizer=optimizer, scheduler=scheduler, config=self.config,
                context=self.context, output_dir=Path(directory),
                resolved_config={"world_size": 1}, backbone_provenance={},
                split_hashes={}, repository_root=PROJECT_ROOT,
            )
            with patch.dict("os.environ", {"SLURM_JOB_END_TIME": "1"}):
                stopped = fit(**arguments)
            self.assertEqual(stopped["completed_epoch"], 1)
            resumed = fit(**arguments, resume=Path(directory) / "checkpoints/last.pt")
            self.assertEqual(resumed["completed_epoch"], 2)
            self.assertEqual(len(resumed["history"]), 2)

    def test_fit_writes_history_and_resumes_completed_run(self):
        model = self.make_model()
        optimizer, scheduler = self.make_optimizer_scheduler(model)
        batches = TinyLoader(make_batch(i, verb=i, noun=i) for i in range(6))
        with tempfile.TemporaryDirectory() as directory:
            arguments = dict(
                model=model, train_loader=batches, validation_loader=batches,
                train_sampler=Mock(), validation_sampler=Mock(), train_dataset=Mock(),
                action_vocabulary=self.vocabulary,
                evaluation_action_pairs=build_action_pairs(np.arange(6), np.arange(6)),
                verb_prior=torch.full((6,), 1 / 6), noun_prior=torch.full((6,), 1 / 6),
                optimizer=optimizer, scheduler=scheduler, config=self.config,
                context=self.context, output_dir=Path(directory),
                resolved_config={"world_size": 1}, backbone_provenance={},
                split_hashes={}, repository_root=PROJECT_ROOT,
            )
            result = fit(**arguments)
            self.assertEqual(result["completed_epoch"], 2)
            self.assertEqual(len(result["history"]), 2)
            self.assertGreater(result["history"][0]["train_events_per_second"], 0)
            resumed = fit(**arguments, resume=Path(directory) / "checkpoints/last.pt")
            self.assertEqual(resumed["completed_epoch"], 2)
            self.assertEqual(resumed["optimizer_step"], result["optimizer_step"])

    def test_checkpoint_round_trip_excludes_backbone(self):
        model = self.make_model()
        optimizer, scheduler = self.make_optimizer_scheduler(model)
        expected = {
            name: parameter.detach().clone()
            for name, parameter in model.head.named_parameters()
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=2,
                optimizer_step=3,
                resolved_config={"schema_version": 1},
                backbone_provenance={"sha256": "abc"},
                split_hashes={"train": "def"},
                action_vocabulary=self.vocabulary,
                best_metric=0.4,
                epochs_without_improvement=1,
                rng_states=[capture_rng_state()],
                repository_root=PROJECT_ROOT,
            )
            raw = torch.load(path, weights_only=True)
            self.assertNotIn("backbone_state", raw)
            self.assertNotIn("model_state", raw)

            with torch.no_grad():
                for parameter in model.head.parameters():
                    parameter.add_(10.0)

            checkpoint = load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                rank=0,
                map_location=torch.device("cpu"),
            )

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["optimizer_step"], 3)

        for name, parameter in model.head.named_parameters():
            self.assertTrue(torch.equal(expected[name], parameter))


if __name__ == "__main__":
    unittest.main()
