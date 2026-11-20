"""Freeze final-retraining inputs from a selected development best.pt."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sta2.data import load_events, sha256_file
from src.sta2.evaluation import validate_checkpoint_metadata


def prepare(selection_checkpoint, annotations, train_split, validation_split, output_dir,
            backbone_checkpoint=None, metadata=None):
    selection_sha256 = sha256_file(selection_checkpoint)
    checkpoint = torch.load(selection_checkpoint, map_location="cpu", weights_only=True)
    if sha256_file(selection_checkpoint) != selection_sha256:
        raise ValueError("selection checkpoint changed while reading; use a stable best.pt")
    validate_checkpoint_metadata(checkpoint)
    configuration = checkpoint["configuration"]
    if configuration.get("final_training"):
        raise ValueError("select a development checkpoint with held-out validation")
    if backbone_checkpoint is not None and sha256_file(backbone_checkpoint) != checkpoint["backbone_provenance"]["checkpoint_sha256"]:
        raise ValueError("backbone differs from the selected development model")
    if metadata is not None and sha256_file(metadata) != configuration["metadata_sha256"]:
        raise ValueError("timing metadata differs from the selected development model")
    paths = {"annotations": annotations, "train_split": train_split,
             "validation_split": validation_split}
    if {key: sha256_file(path) for key, path in paths.items()} != checkpoint["split_hashes"]:
        raise ValueError("selection checkpoint does not match the supplied development data")
    train = load_events(annotations, split="train", train_split=train_split,
                        validation_split=validation_split)
    validation = load_events(annotations, split="validation", train_split=train_split,
                             validation_split=validation_split)
    if not len(train) or not len(validation):
        raise ValueError("selection must come from nonempty development train/validation splits")
    epoch = int(checkpoint["epoch"])
    if not 1 <= epoch <= configuration["training"]["max_epochs"]:
        raise ValueError("selected epoch is outside the training schedule")
    output_dir = Path(output_dir)
    # Never overwrite another run's frozen inputs.
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(annotations, output_dir / "train.csv")
    pd.read_csv(annotations).iloc[:0].to_csv(output_dir / "validation.csv", index=False)
    config = {key: configuration[key] for key in (
        "schema_version", "training", "head", "data", "experiment"
    )}
    # Keep the original LR schedule horizon; stop at the selected epoch.
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config))
    (output_dir / "backbone.yaml").write_text(yaml.safe_dump(configuration["backbone_config"]))
    (output_dir / "source_provenance.json").write_text(
        json.dumps(checkpoint["backbone_provenance"]["upstream"], indent=2) + "\n"
    )
    selection = {
        "selection_checkpoint": str(Path(selection_checkpoint).resolve()),
        "selection_checkpoint_sha256": selection_sha256,
        "selected_epochs": epoch,
        "development_best_action_mt5r": checkpoint["best_metric"],
        "training_events": len(train) + len(validation),
        "development_split_hashes": checkpoint["split_hashes"],
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection), flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/EPIC_100_train.csv"))
    parser.add_argument("--train-split", type=Path, default=Path("data/processed/EPIC_100_train_participant_split.csv"))
    parser.add_argument("--validation-split", type=Path, default=Path("data/processed/EPIC_100_validation_participant_split.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
