"""Tests for STA Model 1: masked mean pooling."""

import unittest

try:
    import torch
    from torch.nn import functional as F
except ImportError:
    torch = None
    F = None

if torch is not None:
    from src.sta_model_1 import MaskedMeanPoolSTA


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MaskedMeanPoolSTATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = MaskedMeanPoolSTA(dropout=0.0)

    def test_output_shapes(self):
        features = torch.randn(4, 5, 512)
        valid_mask = torch.ones(4, 5, dtype=torch.bool)

        verb_logits, noun_logits = self.model(features, valid_mask)

        self.assertEqual(verb_logits.shape, (4, 97))
        self.assertEqual(noun_logits.shape, (4, 300))

    def test_pooling_averages_only_valid_segments(self):
        features = torch.randn(2, 5, 512)
        valid_mask = torch.tensor(
            [
                [True, True, True, True, True],
                [False, False, True, True, True],
            ]
        )

        expected = torch.stack(
            [
                features[0].mean(dim=0),
                features[1, 2:].mean(dim=0),
            ]
        )
        actual = self.model.pool_features(features, valid_mask)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_invalid_segments_cannot_change_predictions(self):
        features = torch.randn(2, 5, 512)
        valid_mask = torch.tensor(
            [
                [True, True, True, True, True],
                [False, False, True, True, True],
            ]
        )
        modified_features = features.clone()
        modified_features[~valid_mask] = 1_000_000

        self.model.eval()
        with torch.inference_mode():
            original_outputs = self.model(features, valid_mask)
            modified_outputs = self.model(modified_features, valid_mask)

        self.assertTrue(
            torch.allclose(original_outputs[0], modified_outputs[0])
        )
        self.assertTrue(
            torch.allclose(original_outputs[1], modified_outputs[1])
        )

    def test_all_false_mask_produces_zero_pool(self):
        features = torch.randn(2, 5, 512)
        valid_mask = torch.zeros(2, 5, dtype=torch.bool)

        pooled = self.model.pool_features(features, valid_mask)

        self.assertTrue(torch.all(pooled == 0))
        self.assertTrue(torch.isfinite(pooled).all())

    def test_backward_pass_produces_finite_gradients(self):
        features = torch.randn(4, 5, 512)
        valid_mask = torch.ones(4, 5, dtype=torch.bool)
        verb_labels = torch.tensor([0, 1, 2, 3])
        noun_labels = torch.tensor([10, 11, 12, 13])

        verb_logits, noun_logits = self.model(features, valid_mask)
        loss = F.cross_entropy(
            verb_logits, verb_labels
        ) + F.cross_entropy(noun_logits, noun_labels)
        loss.backward()

        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_rejects_wrong_feature_size(self):
        features = torch.randn(2, 5, 256)
        valid_mask = torch.ones(2, 5, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "expected feature size 512"):
            self.model(features, valid_mask)

    def test_rejects_non_boolean_mask(self):
        features = torch.randn(2, 5, 512)
        valid_mask = torch.ones(2, 5)

        with self.assertRaisesRegex(TypeError, "torch.bool"):
            self.model(features, valid_mask)


if __name__ == "__main__":
    unittest.main()
