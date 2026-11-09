"""Shared video encoder (VideoMAE) + separate verb (97) and noun (300) heads."""
from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"


class VideoMAEVerbNoun(nn.Module):
    """name='tiny-random' builds a very small randomly initialised model (no download),
    only meant for smoke-testing the pipeline."""

    def __init__(self, name=DEFAULT_MODEL, num_frames=16, n_verb=97, n_noun=300,
                 dropout=0.1, freeze_backbone=False, grad_ckpt=False):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification

        if name == "tiny-random":
            cfg = VideoMAEConfig(image_size=224, patch_size=32, num_frames=num_frames,
                                 tubelet_size=2, hidden_size=64, num_hidden_layers=2,
                                 num_attention_heads=2, intermediate_size=128,
                                 use_mean_pooling=True)
            full = VideoMAEForVideoClassification(cfg)
        else:
            # num_frames can differ from 16: VideoMAE uses fixed sinusoidal position embeddings
            full = VideoMAEForVideoClassification.from_pretrained(name, num_frames=num_frames)
        self.backbone = full.videomae
        self.fc_norm = full.fc_norm  # None when the checkpoint uses CLS pooling
        hidden = full.config.hidden_size
        del full  # drops the original 400-class Kinetics head

        if grad_ckpt:
            self.backbone.gradient_checkpointing_enable()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.drop = nn.Dropout(dropout)
        self.verb_head = nn.Linear(hidden, n_verb)
        self.noun_head = nn.Linear(hidden, n_noun)
        for h in (self.verb_head, self.noun_head):
            nn.init.normal_(h.weight, std=0.02)
            nn.init.zeros_(h.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W) normalised -> (B, hidden). Reusable by other tasks (e.g. STA)."""
        h = self.backbone(pixel_values=x).last_hidden_state
        return self.fc_norm(h.mean(1)) if self.fc_norm is not None else h[:, 0]

    def forward(self, x: torch.Tensor):
        f = self.drop(self.features(x))
        return self.verb_head(f), self.noun_head(f)

    def backbone_params(self):
        ps = list(self.backbone.parameters())
        if self.fc_norm is not None:
            ps += list(self.fc_norm.parameters())
        return ps

    def head_params(self):
        return list(self.verb_head.parameters()) + list(self.noun_head.parameters())
