"""Experiment orchestration for shared STA Model 1 and Model 2 training."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from src.sta_model_0 import FrequencyPriorSTA
from src.sta_training import (
    TrainingConfig,
    get_git_provenance,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)


def write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write JSON through a temporary file to avoid partial results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(dict(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def write_history(
    path: Path,
    history: list[dict[str, Any]],
) -> Path:
    """Write one CSV row for every completed epoch."""
    if not history:
        raise ValueError("Training history cannot be empty")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0]),
        )
        writer.writeheader()
        writer.writerows(history)

    temporary_path.replace(path)
    return path


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: Iterable[Mapping[str, Any]],
    validation_loader: Iterable[Mapping[str, Any]],
    frequency_prior: FrequencyPriorSTA,
    action_pairs: np.ndarray,
    device: torch.device,
    config: TrainingConfig,
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Train one STA model and restore its best validation checkpoint."""
    output_root = Path(output_root)
    checkpoint_path = (
        output_root / "checkpoints" / config.model / "best.pt"
    )
    history_path = (
        output_root / "logs" / config.model / "history.csv"
    )
    validation_path = (
        output_root / "tables" / f"{config.model}_validation.json"
    )
    configuration_path = (
        output_root / "configs" / f"{config.model}.json"
    )

    write_json_atomic(
        configuration_path,
        {
            "configuration": config.to_dict(),
            **get_git_provenance(),
        },
    )

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch: int | None = None
    best_validation_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    stopped_early = False

    model.to(device)

    for epoch in range(1, config.max_epochs + 1):
        training_statistics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        if training_statistics["train_usable_events"] == 0:
            raise RuntimeError(
                "Training split contains no events with usable context"
            )

        validation_metrics = validate_one_epoch(
            model,
            validation_loader,
            device,
            frequency_prior,
            action_pairs,
        )
        action_mt5r = float(validation_metrics["action_mt5r"])
        if not np.isfinite(action_mt5r):
            raise FloatingPointError(
                "Validation Action MT5R is not finite"
            )

        improved = action_mt5r > best_score
        if improved:
            best_score = action_mt5r
            best_epoch = epoch
            best_validation_metrics = dict(validation_metrics)
            epochs_without_improvement = 0

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                config,
                validation_metrics,
            )
            write_json_atomic(
                validation_path,
                {
                    "model": config.model,
                    "best_epoch": epoch,
                    **validation_metrics,
                },
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "is_best": improved,
                "epochs_without_improvement": (
                    epochs_without_improvement
                ),
                **training_statistics,
                **validation_metrics,
            }
        )
        write_history(history_path, history)

        if (
            epochs_without_improvement
            >= config.early_stopping_patience
        ):
            stopped_early = True
            break

    if best_epoch is None or best_validation_metrics is None:
        raise RuntimeError(
            "Training completed without a valid checkpoint"
        )

    # The final epoch may not be the selected epoch. Return the model and
    # optimizer in the state chosen by validation Action MT5R.
    load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        map_location=device,
    )

    return {
        "best_epoch": best_epoch,
        "best_action_mt5r": best_score,
        "best_validation_metrics": best_validation_metrics,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "validation_path": validation_path,
        "configuration_path": configuration_path,
        "history": history,
    }
