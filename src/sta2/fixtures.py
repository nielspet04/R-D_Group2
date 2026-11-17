"""Eight synthetic events, independent of pretrained weights."""

import torch


def make_fixture(
    *,
    size=32,
    embed_dim=64,
    observed_tokens=16,
    future_tokens=4,
    labelled=True,
):
    generator = torch.Generator().manual_seed(42)

    statuses = [
        "full",
        "partial",
        "no_context",
        "video_error",
    ] * 2

    valid = torch.tensor([
        status in {"full", "partial"}
        for status in statuses
    ])

    video = torch.rand(
        8, 3, 32, size, size,
        generator=generator,
    )
    video[~valid] = 0

    frame_indices = torch.arange(32).repeat(8, 1)

    for index, status in enumerate(statuses):
        if status == "partial":
            frame_indices[index, :16] = 0
        elif status in {"no_context", "video_error"}:
            frame_indices[index] = -1

    batch = {
        "video": video,
        "has_context": valid,
        "row_index": torch.tensor(
            [80, 3, 20, 1, 70, 19, 11, 5]
        ),
        "narration_id": [
            f"synthetic_{index}" for index in range(8)
        ],
        "status": statuses,
        "cutoff_seconds": torch.tensor(
            [6.0, 0.5, 0.0, 6.0] * 2
        ),
        "frame_indices": frame_indices,
    }

    if labelled:
        batch["verb_label"] = torch.arange(8)
        batch["noun_label"] = torch.arange(8) * 3

    return {
        "synthetic": True,
        "production_video_shape": size == 384,
        "batch": batch,
        "backbone_outputs": {
            "observed_tokens": torch.randn(
                8,
                observed_tokens,
                embed_dim,
                generator=generator,
            ),
            "future_tokens": torch.randn(
                8,
                future_tokens,
                embed_dim,
                generator=generator,
            ),
        },
    }