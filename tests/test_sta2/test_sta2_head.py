"""Tests for the STA2 attentive prediction head."""

import unittest

import torch

from src.sta2.head import AnticipationHead


class AnticipationHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

        self.embed_dim = 32
        self.num_actions = 11

        self.head = AnticipationHead(
            embed_dim=self.embed_dim,
            num_actions=self.num_actions,
            num_heads=4,
            use_activation_checkpointing=False,
        )
        self.head.eval()

    def test_output_shapes_and_finite_values(self) -> None:
        observed = torch.randn(3, 8, self.embed_dim)
        future = torch.randn(3, 2, self.embed_dim)

        with torch.inference_mode():
            outputs = self.head(observed, future)

        self.assertEqual(
            set(outputs),
            {
                "verb_logits",
                "noun_logits",
                "action_logits",
            },
        )
        self.assertEqual(outputs["verb_logits"].shape, (3, 97))
        self.assertEqual(outputs["noun_logits"].shape, (3, 300))
        self.assertEqual(
            outputs["action_logits"].shape,
            (3, self.num_actions),
        )

        for output in outputs.values():
            self.assertTrue(torch.isfinite(output).all())

    def test_pooler_has_three_queries_and_four_block_depth(
        self,
    ) -> None:
        self.assertEqual(
            self.head.pooler.query_tokens.shape,
            (1, 3, self.embed_dim),
        )

        # Depth four consists of three self-attention blocks
        # followed by one cross-attention block.
        self.assertEqual(len(self.head.pooler.blocks), 3)
        self.assertIsNotNone(
            self.head.pooler.cross_attention_block
        )

    def test_observed_only_mode_accepts_empty_future_tokens(
        self,
    ) -> None:
        observed = torch.randn(2, 8, self.embed_dim)
        future = observed.new_empty(2, 0, self.embed_dim)

        with torch.inference_mode():
            outputs = self.head(observed, future)

        self.assertEqual(outputs["verb_logits"].shape, (2, 97))
        self.assertEqual(outputs["noun_logits"].shape, (2, 300))
        self.assertEqual(
            outputs["action_logits"].shape,
            (2, self.num_actions),
        )

    def test_future_tokens_affect_predictions(self) -> None:
        observed = torch.randn(2, 8, self.embed_dim)
        empty_future = observed.new_empty(2, 0, self.embed_dim)
        future = torch.full((2, 2, self.embed_dim), 5.0)

        with torch.inference_mode():
            observed_only = self.head(
                observed,
                empty_future,
            )
            observed_and_future = self.head(
                observed,
                future,
            )

        changed = any(
            not torch.allclose(
                observed_only[name],
                observed_and_future[name],
            )
            for name in observed_only
        )
        self.assertTrue(changed)

    def test_backward_produces_finite_gradients(self) -> None:
        self.head.train()

        observed = torch.randn(
            2,
            8,
            self.embed_dim,
            requires_grad=True,
        )
        future = torch.randn(
            2,
            2,
            self.embed_dim,
            requires_grad=True,
        )

        outputs = self.head(observed, future)
        loss = sum(
            output.square().mean()
            for output in outputs.values()
        )
        loss.backward()

        self.assertIsNotNone(observed.grad)
        self.assertIsNotNone(future.grad)
        self.assertTrue(torch.isfinite(observed.grad).all())
        self.assertTrue(torch.isfinite(future.grad).all())

        for name, parameter in self.head.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(
                    torch.isfinite(parameter.grad).all()
                )

    def test_activation_checkpointing_backward(self) -> None:
        head = AnticipationHead(
            embed_dim=self.embed_dim,
            num_actions=self.num_actions,
            num_heads=4,
            use_activation_checkpointing=True,
        )
        head.train()

        observed = torch.randn(2, 4, self.embed_dim)
        future = torch.randn(2, 1, self.embed_dim)

        outputs = head(observed, future)
        loss = sum(output.mean() for output in outputs.values())
        loss.backward()

        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in head.parameters()
            )
        )

    def test_rejects_embedding_dimension_mismatch(self) -> None:
        observed = torch.randn(2, 8, self.embed_dim)
        future = torch.randn(2, 2, 16)

        with self.assertRaisesRegex(
            ValueError,
            "embedding dimension",
        ):
            self.head(observed, future)

    def test_rejects_different_batch_sizes(self) -> None:
        observed = torch.randn(2, 8, self.embed_dim)
        future = torch.randn(3, 2, self.embed_dim)

        with self.assertRaisesRegex(
            ValueError,
            "same batch size",
        ):
            self.head(observed, future)

    def test_rejects_non_floating_tokens(self) -> None:
        observed = torch.ones(
            2,
            8,
            self.embed_dim,
            dtype=torch.int64,
        )
        future = torch.ones(
            2,
            2,
            self.embed_dim,
            dtype=torch.int64,
        )

        with self.assertRaisesRegex(
            TypeError,
            "floating point",
        ):
            self.head(observed, future)

    def test_rejects_completely_empty_input(self) -> None:
        observed = torch.empty(2, 0, self.embed_dim)
        future = torch.empty(2, 0, self.embed_dim)

        with self.assertRaisesRegex(
            ValueError,
            "at least one",
        ):
            self.head(observed, future)

    def test_rejects_incompatible_head_count(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "divisible",
        ):
            AnticipationHead(
                embed_dim=30,
                num_actions=10,
                num_heads=16,
            )

    def test_rejects_invalid_class_counts(self) -> None:
        for argument in (
            "num_actions",
            "num_verbs",
            "num_nouns",
        ):
            with self.subTest(argument=argument):
                arguments = {
                    "embed_dim": 32,
                    "num_actions": 10,
                    "num_heads": 4,
                }
                arguments[argument] = 0

                with self.assertRaisesRegex(
                    ValueError,
                    f"{argument} must be positive",
                ):
                    AnticipationHead(**arguments)

if __name__ == "__main__":
    unittest.main()