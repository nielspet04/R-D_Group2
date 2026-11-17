import unittest

import torch
from torch import nn

from src.sta2.backbone import (
    FrozenAnticipationBackbone,
    load_strict,
)


class FakeEncoder(nn.Module):
    embed_dim = 8
    patch_size = 16
    tubelet_size = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(16))

    def forward(self, video):
        return self.weight[None, None].expand(
            len(video),
            16,
            -1,
        )


class FakePredictor(nn.Module):
    num_frames = 64
    hierarchical_layers = [0, 1]

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(16))
        self.calls = 0

    def forward(self, x, masks_x, masks_y):
        self.calls += 1

        assert x.shape[-1] == 16

        output = self.weight[None, None].expand(
            len(x),
            masks_y.shape[1],
            -1,
        )

        return output, None


class BackboneTests(unittest.TestCase):
    def make_backbone(self, observed_only=False):
        return FrozenAnticipationBackbone(
            FakeEncoder(),
            FakePredictor(),
            observed_only=observed_only,
            bf16=False,
        )

    def input_video(self):
        # Expanded view avoids allocating a full video for a fake model.
        return torch.zeros(1).expand(
            1, 3, 32, 384, 384
        )

    def test_outputs_and_freezing(self):
        backbone = self.make_backbone()
        backbone.train()

        self.assertFalse(backbone.training)
        self.assertFalse(backbone.encoder.training)
        self.assertFalse(backbone.predictor.training)

        before = {
            name: value.detach().clone()
            for name, value in backbone.state_dict().items()
        }

        tokens = backbone(
            self.input_video(),
            torch.ones(1),
        )

        self.assertEqual(
            tokens["observed_tokens"].shape,
            (1, 16, 8),
        )
        self.assertEqual(
            tokens["future_tokens"].shape,
            (1, 576, 8),
        )

        head = nn.Linear(8, 3)
        optimizer = torch.optim.SGD(
            head.parameters(),
            lr=0.1,
        )

        head_before = head.weight.detach().clone()

        logits = head(
            tokens["observed_tokens"].mean(dim=1)
        )

        loss = nn.functional.cross_entropy(
            logits,
            torch.tensor([1]),
        )

        loss.backward()
        optimizer.step()

        self.assertFalse(
            torch.equal(
                head_before,
                head.weight,
            )
        )

        for parameter in backbone.parameters():
            self.assertIsNone(parameter.grad)
            self.assertFalse(
                parameter.requires_grad
            )

        for name, value in backbone.state_dict().items():
            self.assertTrue(
                torch.equal(before[name], value)
            )

    def test_observed_only_bypasses_predictor(self):
        backbone = self.make_backbone(
            observed_only=True
        )

        output = backbone(
            self.input_video(),
            torch.ones(1),
        )

        self.assertEqual(
            output["future_tokens"].shape,
            (1, 0, 8),
        )

        self.assertEqual(
            backbone.predictor.calls,
            0,
        )

    def test_horizon_changes_positions(self):
        backbone = self.make_backbone()

        one_second = backbone.target_positions(
            9216,
            torch.tensor([1.0]),
        )

        two_seconds = backbone.target_positions(
            9216,
            torch.tensor([2.0]),
        )

        self.assertTrue(
            torch.all(
                two_seconds - one_second
                == 4 * 576
            )
        )

    def test_strict_loading(self):
        layer = nn.Linear(2, 3)

        with self.assertRaises(ValueError):
            load_strict(layer, {})

        wrong_shape = {
            "weight": torch.zeros(4, 2),
            "bias": torch.zeros(3),
        }

        with self.assertRaises(ValueError):
            load_strict(layer, wrong_shape)

        collision = {
            "weight": torch.zeros(3, 2),
            "module.weight": torch.zeros(3, 2),
            "bias": torch.zeros(3),
        }

        with self.assertRaises(ValueError):
            load_strict(layer, collision)


if __name__ == "__main__":
    unittest.main()