"""Tests for STA Model 2: masked chronological GRU."""

import unittest

try:
    import torch
    from torch.nn import functional as F
except ImportError:
    torch = None
    F = None

if torch is not None:
    from src.sta_model_2 import MaskedGRUSTA


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MaskedGRUSTATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = MaskedGRUSTA(dropout=0.0)
        self.model.eval()

    def test_output_shapes_and_finite_values(self):
        features = torch.randn(4, 5, 512)
        mask = torch.ones(4, 5, dtype=torch.bool)

        with torch.inference_mode():
            hidden = self.model.final_hidden(features, mask)
            verbs, nouns = self.model(features, mask)

        self.assertEqual(hidden.shape, (4, 512))
        self.assertEqual(verbs.shape, (4, 97))
        self.assertEqual(nouns.shape, (4, 300))
        for tensor in (hidden, verbs, nouns):
            self.assertTrue(torch.isfinite(tensor).all())

    def test_all_false_mask_produces_exactly_zero_hidden(self):
        features = torch.randn(2, 5, 512)
        mask = torch.zeros(2, 5, dtype=torch.bool)

        with torch.inference_mode():
            hidden = self.model.final_hidden(features, mask)

        self.assertTrue(torch.equal(hidden, torch.zeros_like(hidden)))
        self.assertTrue(torch.isfinite(hidden).all())

    def test_leading_padding_does_not_update_hidden(self):
        features = torch.randn(2, 5, 512)
        mask = torch.tensor([[False, False, True, True, True]] * 2)

        with torch.inference_mode():
            leading = self.model.final_hidden(features[:, :2], mask[:, :2])
            actual = self.model.final_hidden(features, mask)
            without_padding = self.model.final_hidden(
                features[:, 2:], mask[:, 2:]
            )

        self.assertTrue(torch.equal(leading, torch.zeros_like(leading)))
        self.assertTrue(torch.equal(actual, without_padding))

    def test_invalid_segments_cannot_change_hidden_or_predictions(self):
        features = torch.randn(4, 5, 512)
        mask = torch.tensor([
            [True, True, True, True, True],
            [False, False, True, True, True],
            [True, False, True, False, True],
            [False, False, False, False, False],
        ])
        modified = features.clone()
        modified[~mask] = 1_000_000.0

        with torch.inference_mode():
            original_hidden = self.model.final_hidden(features, mask)
            modified_hidden = self.model.final_hidden(modified, mask)
            original_outputs = self.model(features, mask)
            modified_outputs = self.model(modified, mask)

        self.assertTrue(torch.equal(original_hidden, modified_hidden))
        for original, changed in zip(original_outputs, modified_outputs):
            self.assertTrue(torch.equal(original, changed))

    def test_middle_padding_preserves_previous_hidden(self):
        features = torch.randn(2, 3, 512)
        mask = torch.tensor([[True, False, True]] * 2)

        with torch.inference_mode():
            before_gap = self.model.final_hidden(features[:, :1], mask[:, :1])
            after_gap = self.model.final_hidden(features[:, :2], mask[:, :2])
            final = self.model.final_hidden(features, mask)
            expected_final = self.model.gru_cell(features[:, 2], before_gap)

        self.assertTrue(torch.equal(before_gap, after_gap))
        self.assertTrue(torch.equal(final, expected_final))

    def test_chronological_order_affects_hidden(self):
        features = torch.arange(5 * 512, dtype=torch.float32).reshape(1, 5, 512)
        features = features / features.numel()
        mask = torch.ones(1, 5, dtype=torch.bool)

        with torch.inference_mode():
            chronological = self.model.final_hidden(features, mask)
            reversed_order = self.model.final_hidden(features.flip(1), mask)

        self.assertFalse(torch.allclose(chronological, reversed_order))

    def test_backward_produces_finite_gradients_excluding_empty_events(self):
        self.model.train()
        features = torch.randn(4, 5, 512, requires_grad=True)
        mask = torch.tensor([
            [True, True, True, True, True],
            [False, False, True, True, True],
            [True, False, True, False, True],
            [False, False, False, False, False],
        ])
        has_context = mask.any(dim=1)
        verbs, nouns = self.model(features, mask)
        loss = F.cross_entropy(
            verbs[has_context], torch.tensor([0, 1, 2, 3])[has_context]
        ) + F.cross_entropy(
            nouns[has_context], torch.tensor([10, 11, 12, 13])[has_context]
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(torch.all(features.grad[~mask] == 0))

    def test_rejects_wrong_feature_size(self):
        with self.assertRaisesRegex(ValueError, "expected feature size 512"):
            self.model(torch.randn(2, 5, 256), torch.ones(2, 5, dtype=torch.bool))

    def test_rejects_wrong_feature_rank(self):
        with self.assertRaisesRegex(ValueError, "features must have shape"):
            self.model(torch.randn(2, 512), torch.ones(2, 5, dtype=torch.bool))

    def test_rejects_wrong_mask_shape(self):
        with self.assertRaisesRegex(ValueError, "valid_mask must match"):
            self.model(torch.randn(2, 5, 512), torch.ones(2, 4, dtype=torch.bool))

    def test_rejects_non_boolean_mask(self):
        with self.assertRaisesRegex(TypeError, "torch.bool"):
            self.model(torch.randn(2, 5, 512), torch.ones(2, 5))

    def test_rejects_integer_features(self):
        with self.assertRaisesRegex(TypeError, "floating-point"):
            self.model(
                torch.ones(2, 5, 512, dtype=torch.int64),
                torch.ones(2, 5, dtype=torch.bool),
            )

    def test_rejects_invalid_constructor_dimensions(self):
        for name in ("feature_size", "hidden_size", "num_verbs", "num_nouns"):
            for value in (0, -1):
                with self.subTest(argument=name, value=value):
                    with self.assertRaisesRegex(ValueError, f"{name} must be positive"):
                        MaskedGRUSTA(**{name: value})

    def test_rejects_invalid_dropout(self):
        for value in (-0.1, 1.0, 1.1, float("nan")):
            with self.subTest(dropout=value):
                with self.assertRaisesRegex(ValueError, "dropout must be"):
                    MaskedGRUSTA(dropout=value)


if __name__ == "__main__":
    unittest.main()
