"""CPU-only preflight for Emiel's selected STA2 checkpoint, job 62771307.

Run from the repository root. PREP names the generated-report directory.
Never constructs a model or allocates a GPU.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path.cwd()))
import pandas as pd
import yaml
from src.sta2.artifacts import PRIMARY_SCORE, load_prediction_artifact, verify_fallbacks
from src.sta2.data import STAClipDataset, load_events, sha256_file
from src.sta2.evaluation import (
    action_vocabulary_sha256, evaluation_action_pairs, frequency_priors,
    load_evaluation_checkpoint, verify_evaluation_inputs,
)
from src.sta2.sampling import sample_frame_indices, UnavailableContextError
from src.sta2.transforms import resolve_preprocessing

EXPECTED_SHA = "9890d287eb4e9a7b6da61c470adcff955859b5e1d372e71821ca026e11a12610"
EXPECTED_SCORE = 0.08774957568731417
report = {"schema_version": 1, "gpu_work": False, "blocking_inputs": []}
destination = Path(os.environ["PREP"]) / "preflight.json"
destination.parent.mkdir(parents=True, exist_ok=True)

def save():
    destination.write_text(json.dumps(report, indent=2) + "\n")

try:
    keys = ("STA_CKPT", "BACKBONE", "BACKBONE_CONFIG", "ANNOTATIONS",
            "TRAIN_SPLIT", "VAL_SPLIT", "METADATA", "RAW_DIR")
    paths = {key: Path(os.environ[key]).resolve() for key in keys}
    report["paths"] = {key: str(path) for key, path in paths.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    checkpoint_hash = sha256_file(paths["STA_CKPT"])
    if checkpoint_hash != EXPECTED_SHA:
        raise ValueError("Selected checkpoint SHA-256 differs from Emiel's verified checkpoint")
    checkpoint = load_evaluation_checkpoint(paths["STA_CKPT"])
    configuration = checkpoint["configuration"]
    if int(checkpoint["epoch"]) != 5 or not math.isclose(
        float(checkpoint["best_metric"]), EXPECTED_SCORE, rel_tol=0, abs_tol=1e-14
    ):
        raise ValueError("Selected checkpoint epoch/metric differs from the frozen selection")
    history_path = paths["STA_CKPT"].parent.parent / "history.json"
    history = json.loads(history_path.read_text())
    if [int(row["epoch"]) for row in history] != list(range(1, 11)):
        raise ValueError("History must contain exactly completed epochs 1 through 10")
    best = max(history, key=lambda row: float(row["action_mt5r"]))
    if int(best["epoch"]) != 5 or not math.isclose(
        float(best["action_mt5r"]), EXPECTED_SCORE, rel_tol=0, abs_tol=1e-14
    ):
        raise ValueError("History does not agree with the selected best checkpoint")
    if configuration.get("final_training"):
        raise ValueError("This task selects the development checkpoint, not an all-label retrain")
    train = load_events(paths["ANNOTATIONS"], split="train",
                        train_split=paths["TRAIN_SPLIT"], validation_split=paths["VAL_SPLIT"])
    validation = load_events(paths["ANNOTATIONS"], split="validation",
                             train_split=paths["TRAIN_SPLIT"], validation_split=paths["VAL_SPLIT"])
    if (len(train), len(validation)) != (59767, 7450):
        raise ValueError("Unexpected development split counts")
    if sorted(validation.participant_id.unique().tolist()) != ["P08", "P20", "P21", "P28", "P29", "P37"]:
        raise ValueError("Unexpected held-out participants")
    verified = verify_evaluation_inputs(
        checkpoint, backbone_checkpoint=paths["BACKBONE"], annotations=paths["ANNOTATIONS"],
        train_split=paths["TRAIN_SPLIT"], validation_split=paths["VAL_SPLIT"], train_events=train,
    )
    backbone_config = yaml.safe_load(paths["BACKBONE_CONFIG"].read_text())
    if backbone_config != configuration["backbone_config"]:
        raise ValueError("Backbone YAML differs from the exact configuration saved during training")
    if resolve_preprocessing(backbone_config.get("preprocessing")) != resolve_preprocessing():
        raise ValueError("Evaluator dataset defaults differ from saved preprocessing; investigate before GPU work")
    report.update(
        checkpoint_sha256=checkpoint_hash, checkpoint_epoch=5, best_action_mt5r=EXPECTED_SCORE,
        completed_epochs=len(history), verified=verified,
        backbone_provenance=checkpoint["backbone_provenance"],
        checkpoint_git_commit=checkpoint["git_commit"], checkpoint_git_dirty=checkpoint["git_dirty"],
        inspection_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        backbone_config_sha256=sha256_file(paths["BACKBONE_CONFIG"]),
        metadata_sha256=sha256_file(paths["METADATA"]),
        training_metadata_sha256=configuration.get("metadata_sha256"),
        history_sha256=sha256_file(history_path), train_events=len(train), validation_events=len(validation),
    )
    print("PASS checkpoint: epoch 5, 10 completed epochs, Action MT5R 8.774957568731417%")
    print("PASS backbone, annotation, split hashes, action vocabulary, and backbone configuration")
    save()

    course_value = os.environ.get("COURSE_TEST", "")
    if not course_value or not Path(course_value).is_file():
        raise FileNotFoundError("COURSE_TEST must identify the actual course-supplied EPIC_100_test.csv")
    course = Path(course_value).resolve()
    table = pd.read_csv(course)
    for column in ("verb_class", "noun_class", "action_class"):
        if column in table and (pd.to_numeric(table[column], errors="coerce").dropna() >= 0).any():
            raise ValueError(f"Course test must be unlabeled; found populated {column}")
    events = load_events(course, split="test")
    if not len(events) or events.narration_id.str.strip().eq("").any():
        raise ValueError("Course table must contain nonempty, unique event IDs")
    report["course_test"] = {"path": str(course), "sha256": sha256_file(course), "events": len(events)}
    print(f"PASS course table: {len(events)} unique events; SHA-256 {sha256_file(course)}")

    # Match repository export identity; inspect only the selected run and named final outputs.
    pairs = evaluation_action_pairs(pd.read_csv(paths["ANNOTATIONS"]), num_verbs=97, num_nouns=300)
    expected = {
        "split": "test", "target_events_sha256": sha256_file(course),
        "checkpoint_sha256": checkpoint_hash, "backbone_sha256": verified["backbone_sha256"],
        "split_hashes": verified["split_hashes"], "checkpoint_configuration": configuration,
        "observed_only": configuration["experiment"]["observed_only"],
        "auxiliary_action_vocabulary_sha256": verified["auxiliary_action_vocabulary_sha256"],
        "valid_action_classes": len(pairs), "valid_action_vocabulary_sha256": action_vocabulary_sha256(pairs),
        "primary_action_score": PRIMARY_SCORE,
    }
    candidates = set(paths["STA_CKPT"].parent.parent.rglob("metrics.json"))
    for directory in Path("results/sta2").glob("competition-final-*"):
        candidates.update(directory.rglob("metrics.json"))
    if os.environ.get("PREDICTIONS"):
        candidates.add(Path(os.environ["PREDICTIONS"]).parent / "metrics.json")
    report["reusable_predictions"] = []
    report["candidate_checks"] = []
    for metrics_path in sorted(candidates):
        prediction_path = metrics_path.parent / "predictions.pt"
        try:
            metrics = json.loads(metrics_path.read_text())
            if any(metrics.get(key) != value for key, value in expected.items()):
                raise ValueError("identity differs from selected checkpoint/course inputs")
            _, records = load_prediction_artifact(prediction_path, events, expected)
            verify_fallbacks(records, frequency_priors(train, num_verbs=97, num_nouns=300))
            report["reusable_predictions"].append(str(prediction_path.resolve()))
            report["candidate_checks"].append({"path": str(prediction_path), "valid": True,
                "predictions_sha256": sha256_file(prediction_path), "metrics_sha256": sha256_file(metrics_path)})
            print(f"REUSE {prediction_path.resolve()}")
        except Exception as error:
            report["candidate_checks"].append({"path": str(prediction_path), "valid": False, "reason": str(error)})
            print(f"NOT REUSABLE {prediction_path}: {error}")
    save()

    # Constructor checks audit coverage and media identity without decoding frames.
    dataset = STAClipDataset(events, paths["RAW_DIR"], paths["METADATA"], split="test", training=False)
    counts = {"full": 0, "partial": 0, "no_context": 0, "video_error": 0}
    for row in events.to_dict("records"):
        cutoff = float(row["sta_observation_end_seconds"])
        vid = row["video_id"]
        if cutoff <= 0:
            counts["no_context"] += 1
        elif dataset.metadata[vid]["status"] != "ok" or vid not in dataset.videos:
            counts["video_error"] += 1
        else:
            _, _, timestamps = dataset._timeline(vid)
            try:
                selection = sample_frame_indices(cutoff, timestamps)
                counts[selection.status] += 1
            except UnavailableContextError:
                counts["video_error"] += 1
    report["predecode_context_counts"] = counts
    report["preflight_passed"] = True
    print("PASS timing audit, media identities, timeline hashes, and legal observation sampling:", counts)
    print("Predecode counts exclude failures that might occur during actual decoding.")
    if not report["reusable_predictions"]:
        print("NO MATCHING VERIFIED TEST ARTIFACT: obtain a live GPU quote before inference")
except Exception as error:
    report["blocking_inputs"].append(f"{type(error).__name__}: {error}")
    report["preflight_passed"] = False
    print("BLOCKED:", report["blocking_inputs"][-1], file=sys.stderr)
    save()
    raise SystemExit(1)
save()
print(f"Saved {destination}")
