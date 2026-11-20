import unittest

import torch
from torch import nn

from src.sta2.backbone import (
    FrozenAnticipationBackbone,
    load_strict,
    state_sha256,
)
from src.sta2.head import AnticipationHead
from src.sta2.model import STA2Model


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


class BF16FakePredictor(FakePredictor):
    def forward(self, x, masks_x, masks_y):
        output, context = super().forward(x, masks_x, masks_y)
        return output.to(torch.bfloat16), context


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
        digest_before = state_sha256(backbone)

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
        self.assertEqual(digest_before, state_sha256(backbone))

    def test_mixed_encoder_predictor_dtypes_reach_head_and_backward(self):
        # Reproduce AMP returning FP32 encoder norms and BF16 predictor projection.
        backbone = FrozenAnticipationBackbone(
            FakeEncoder(), BF16FakePredictor(), bf16=False,
        )
        tokens = backbone(self.input_video(), torch.ones(1))
        self.assertEqual(tokens["observed_tokens"].dtype, torch.float32)
        self.assertEqual(tokens["future_tokens"].dtype, torch.float32)
        head = AnticipationHead(
            embed_dim=8, num_actions=3, num_verbs=4, num_nouns=5,
            num_heads=2, depth=2, use_activation_checkpointing=True,
        )
        model = STA2Model(backbone, head)
        model.train()
        before = head.verb_classifier.weight.detach().clone()
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        output = model(self.input_video(), torch.tensor([True]))
        loss = sum(logits.float().square().mean() for logits in output.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, head.verb_classifier.weight))
        self.assertTrue(all(parameter.grad is None for parameter in backbone.parameters()))

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA BF16 required for vendored AMP integration",
    )
    def test_vendored_cuda_bf16_encoder_predictor_and_head_update(self):
        from src.sta2.vendor.vjepa21.models.vision_transformer import VisionTransformer
        from src.sta2.vendor.vjepa21.models.predictor import VisionTransformerPredictor

        # Small real vendor modules exercise AMP/residual/norm/projection code
        # without requiring the production checkpoint or allocating ViT-G.
        encoder = VisionTransformer(
            img_size=16, patch_size=16, num_frames=32, embed_dim=32,
            depth=12, num_heads=2, use_rope=True, n_output_distillation=4,
        )
        predictor = VisionTransformerPredictor(
            img_size=16, patch_size=16, num_frames=64, embed_dim=32,
            predictor_embed_dim=32, depth=4, num_heads=2,
            use_rope=True, use_mask_tokens=True, return_all_tokens=True,
            n_output_distillation=4,
        )
        backbone = FrozenAnticipationBackbone(encoder, predictor, bf16=True).cuda()
        # Keep the public video shape; the fixture uses one spatial patch.
        encoder.register_forward_pre_hook(
            lambda module, args: (args[0][..., :16, :16],)
        )
        backbone.grid_size = 1
        raw_dtypes = {}

        def record_encoder(module, args, output):
            raw_dtypes["encoder"] = output.dtype

        def record_predictor(module, args, output):
            raw_dtypes["predictor"] = output[0].dtype

        encoder.register_forward_hook(record_encoder)
        predictor.register_forward_hook(record_predictor)
        head = AnticipationHead(
            embed_dim=32, num_actions=3, num_verbs=4, num_nouns=5,
            num_heads=2, depth=2, use_activation_checkpointing=True,
        ).cuda()
        model = STA2Model(backbone, head).train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        before = head.verb_classifier.weight.detach().clone()
        backbone_before = state_sha256(backbone)
        video = torch.zeros(1, device="cuda").expand(1, 3, 32, 384, 384)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(video, torch.tensor([True], device="cuda"))
        loss = sum(logits.float().square().mean() for logits in output.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, head.verb_classifier.weight))
        self.assertEqual(backbone_before, state_sha256(backbone))
        self.assertEqual(raw_dtypes["encoder"], torch.float32)
        self.assertEqual(raw_dtypes["predictor"], torch.bfloat16)
        print("Vendored CUDA AMP output dtypes:", raw_dtypes)

    def test_state_digest_detects_parameter_and_buffer_changes(self):
        module = nn.Linear(2, 2)
        module.register_buffer("counter", torch.tensor(0))
        before = state_sha256(module)
        with torch.no_grad():
            module.weight.add_(1)
        self.assertNotEqual(before, state_sha256(module))
        before = state_sha256(module)
        module.counter.add_(1)
        self.assertNotEqual(before, state_sha256(module))

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
