"""Tests for the STA2 model wrapper and masked losses."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from src.sta2.head import AnticipationHead
from src.sta2.model import (
    STA2Model,
    build_action_vocabulary,
    compute_sta2_losses,
    encode_action_labels,
)


class RecordingBackbone(nn.Module):
    """Small synthetic backbone that records the videos it receives."""

    embed_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.call_count = 0
        self.last_video = None
        self.last_anticipation_seconds = None

    def forward(
        self,
        video: torch.Tensor,
        anticipation_seconds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.call_count += 1
        self.last_video = video.detach().clone()
        self.last_anticipation_seconds = (
            anticipation_seconds.detach().clone()
        )

        batch_size = video.shape[0]
        summary = video.mean(
            dim=(1, 2, 3, 4),
            keepdim=False,
        ).reshape(batch_size, 1, 1)

        observed = (
            summary * self.scale
        ).expand(batch_size, 3, self.embed_dim)
        future = (
            (summary + 1.0) * self.scale
        ).expand(batch_size, 2, self.embed_dim)

        return {
            "observed_tokens": observed,
            "future_tokens": future,
        }


class RecordingHead(nn.Module):
    """Head double used to inspect tokens prepared by the wrapper."""

    embed_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.observed_tokens = None
        self.future_tokens = None

    def forward(
        self,
        observed_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.observed_tokens = observed_tokens.detach().clone()
        self.future_tokens = future_tokens.detach().clone()

        batch_size = observed_tokens.shape[0]
        base = self.anchor.expand(batch_size, 1)

        return {
            "verb_logits": base.expand(batch_size, 4),
            "noun_logits": base.expand(batch_size, 5),
            "action_logits": base.expand(batch_size, 3),
        }


class STA2ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def make_video(self, batch_size: int = 4) -> torch.Tensor:
        return torch.arange(
            batch_size * 3 * 2 * 2 * 2,
            dtype=torch.float32,
        ).reshape(batch_size, 3, 2, 2, 2)

    def test_wrapper_excludes_invalid_events_from_backbone(self) -> None:
        backbone = RecordingBackbone()
        head = RecordingHead()
        model = STA2Model(backbone, head)

        video = self.make_video()
        has_context = torch.tensor(
            [True, False, True, False]
        )

        outputs = model(video, has_context)

        self.assertEqual(backbone.call_count, 1)
        self.assertTrue(
            torch.equal(
                backbone.last_video,
                video[has_context],
            )
        )
        self.assertTrue(
            torch.equal(
                backbone.last_anticipation_seconds,
                torch.ones(2),
            )
        )

        self.assertEqual(head.observed_tokens.shape, (4, 3, 8))
        self.assertEqual(head.future_tokens.shape, (4, 2, 8))

        self.assertTrue(
            torch.equal(
                head.observed_tokens[~has_context],
                torch.zeros(2, 3, 8),
            )
        )
        self.assertTrue(
            torch.equal(
                head.future_tokens[~has_context],
                torch.zeros(2, 2, 8),
            )
        )

        for logits in outputs.values():
            self.assertTrue(torch.isfinite(logits).all())

    def test_synthetic_full_batch_outputs_work_without_backbone(
        self,
    ) -> None:
        head = RecordingHead()
        model = STA2Model(None, head)

        video = self.make_video()
        has_context = torch.tensor(
            [True, False, True, False]
        )

        observed = torch.randn(4, 3, 8)
        future = torch.randn(4, 2, 8)

        # Nonfinite values in invalid rows must never reach the head.
        observed[~has_context] = float("nan")
        future[~has_context] = float("nan")

        outputs = model(
            video,
            has_context,
            backbone_outputs={
                "observed_tokens": observed,
                "future_tokens": future,
            },
        )

        self.assertTrue(
            torch.equal(
                head.observed_tokens[has_context],
                observed[has_context],
            )
        )
        self.assertTrue(
            torch.equal(
                head.future_tokens[has_context],
                future[has_context],
            )
        )
        self.assertTrue(
            torch.equal(
                head.observed_tokens[~has_context],
                torch.zeros(2, 3, 8),
            )
        )
        self.assertTrue(
            torch.equal(
                head.future_tokens[~has_context],
                torch.zeros(2, 2, 8),
            )
        )

        for logits in outputs.values():
            self.assertTrue(torch.isfinite(logits).all())

    def test_entirely_invalid_batch_is_finite_and_has_zero_loss(
        self,
    ) -> None:
        backbone = RecordingBackbone()
        head = RecordingHead()
        model = STA2Model(backbone, head)

        video = self.make_video(batch_size=3)
        has_context = torch.zeros(3, dtype=torch.bool)

        outputs = model(video, has_context)

        self.assertEqual(backbone.call_count, 0)
        self.assertEqual(head.observed_tokens.shape, (3, 1, 8))
        self.assertEqual(head.future_tokens.shape, (3, 0, 8))
        self.assertTrue(torch.isfinite(head.observed_tokens).all())

        losses = compute_sta2_losses(
            outputs,
            verb_labels=torch.tensor([-99, -99, -99]),
            noun_labels=torch.tensor([-99, -99, -99]),
            action_labels=torch.tensor([-99, -99, -99]),
            has_context=has_context,
        )

        self.assertEqual(losses.usable_count, 0)
        self.assertEqual(losses.action_count, 0)
        self.assertFalse(losses.should_update)
        self.assertTrue(torch.isfinite(losses.total_loss))
        self.assertEqual(losses.total_loss.item(), 0.0)

        losses.total_loss.backward()
        self.assertIsNotNone(head.anchor.grad)
        self.assertTrue(torch.isfinite(head.anchor.grad))

    def test_invalid_events_are_masked_from_every_loss(self) -> None:
        verb_logits = torch.randn(3, 4, requires_grad=True)
        noun_logits = torch.randn(3, 5, requires_grad=True)
        action_logits = torch.randn(3, 3, requires_grad=True)

        outputs = {
            "verb_logits": verb_logits,
            "noun_logits": noun_logits,
            "action_logits": action_logits,
        }

        has_context = torch.tensor([True, False, True])
        verb_labels = torch.tensor([1, 999, 2])
        noun_labels = torch.tensor([3, 999, 0])
        action_labels = torch.tensor([0, 999, 2])

        losses = compute_sta2_losses(
            outputs,
            verb_labels,
            noun_labels,
            action_labels,
            has_context,
        )

        expected_verb = F.cross_entropy(
            verb_logits[has_context].float(),
            verb_labels[has_context],
        )
        expected_noun = F.cross_entropy(
            noun_logits[has_context].float(),
            noun_labels[has_context],
        )
        expected_action = F.cross_entropy(
            action_logits[has_context].float(),
            action_labels[has_context],
        )

        self.assertTrue(
            torch.allclose(losses.verb_loss, expected_verb)
        )
        self.assertTrue(
            torch.allclose(losses.noun_loss, expected_noun)
        )
        self.assertTrue(
            torch.allclose(losses.action_loss, expected_action)
        )
        self.assertTrue(
            torch.allclose(
                losses.total_loss,
                expected_verb + expected_noun + expected_action,
            )
        )

        losses.total_loss.backward()

        self.assertTrue(
            torch.equal(
                verb_logits.grad[1],
                torch.zeros_like(verb_logits.grad[1]),
            )
        )
        self.assertTrue(
            torch.equal(
                noun_logits.grad[1],
                torch.zeros_like(noun_logits.grad[1]),
            )
        )
        self.assertTrue(
            torch.equal(
                action_logits.grad[1],
                torch.zeros_like(action_logits.grad[1]),
            )
        )

    def test_unknown_action_keeps_verb_and_noun_supervision(
        self,
    ) -> None:
        verb_logits = torch.randn(2, 4, requires_grad=True)
        noun_logits = torch.randn(2, 5, requires_grad=True)
        action_logits = torch.randn(2, 3, requires_grad=True)

        verb_labels = torch.tensor([1, 2])
        noun_labels = torch.tensor([3, 4])
        action_labels = torch.tensor([1, -1])
        has_context = torch.tensor([True, True])

        losses = compute_sta2_losses(
            {
                "verb_logits": verb_logits,
                "noun_logits": noun_logits,
                "action_logits": action_logits,
            },
            verb_labels,
            noun_labels,
            action_labels,
            has_context,
        )

        self.assertEqual(losses.usable_count, 2)
        self.assertEqual(losses.action_count, 1)

        self.assertTrue(
            torch.allclose(
                losses.verb_loss,
                F.cross_entropy(verb_logits, verb_labels),
            )
        )
        self.assertTrue(
            torch.allclose(
                losses.noun_loss,
                F.cross_entropy(noun_logits, noun_labels),
            )
        )
        self.assertTrue(
            torch.allclose(
                losses.action_loss,
                F.cross_entropy(
                    action_logits[:1],
                    action_labels[:1],
                ),
            )
        )

        losses.total_loss.backward()

        self.assertGreater(verb_logits.grad[1].abs().sum().item(), 0)
        self.assertGreater(noun_logits.grad[1].abs().sum().item(), 0)
        self.assertEqual(
            action_logits.grad[1].abs().sum().item(),
            0.0,
        )

    def test_action_vocabulary_is_sorted_and_unknowns_are_minus_one(
        self,
    ) -> None:
        train_verbs = torch.tensor([1, 0, 1, 1, 0])
        train_nouns = torch.tensor([3, 2, 0, 3, 2])

        vocabulary = build_action_vocabulary(
            train_verbs,
            train_nouns,
        )

        self.assertEqual(
            vocabulary,
            ((0, 2), (1, 0), (1, 3)),
        )

        encoded = encode_action_labels(
            torch.tensor([1, 0, 2, 1]),
            torch.tensor([3, 2, 9, 0]),
            vocabulary,
        )

        self.assertTrue(
            torch.equal(
                encoded,
                torch.tensor([2, 0, -1, 1]),
            )
        )

    def test_backbone_stays_frozen_and_gradients_reach_head(
        self,
    ) -> None:
        backbone = RecordingBackbone()
        head = AnticipationHead(
            embed_dim=8,
            num_actions=3,
            num_verbs=4,
            num_nouns=5,
            num_heads=2,
            depth=2,
            use_activation_checkpointing=False,
        )
        model = STA2Model(backbone, head)
        model.train()

        self.assertFalse(backbone.training)
        self.assertTrue(head.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in backbone.parameters()
            )
        )

        video = self.make_video(batch_size=3)
        has_context = torch.tensor([True, False, True])

        outputs = model(video, has_context)
        losses = compute_sta2_losses(
            outputs,
            verb_labels=torch.tensor([1, 999, 2]),
            noun_labels=torch.tensor([3, 999, 4]),
            action_labels=torch.tensor([0, 999, 2]),
            has_context=has_context,
        )
        losses.total_loss.backward()

        for parameter in backbone.parameters():
            self.assertIsNone(parameter.grad)

        for name, parameter in head.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(
                    torch.isfinite(parameter.grad).all()
                )


if __name__ == "__main__":
    unittest.main()
