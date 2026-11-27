"""Validate both course tasks and create their ZIP, per runbook section 8.

Usage: python scripts/validate_codabench_bundle.py EVENTS AR_FILE STA_FILE ZIP_FILE
"""
import csv
import math
import re
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch

events_path, ar_path, sta_path, zip_path = map(Path, sys.argv[1:])
with events_path.open(newline="", encoding="utf-8-sig") as handle:
    event_ids = [row["narration_id"] for row in csv.DictReader(handle)]
expected_ids = set(event_ids)
if not event_ids or "" in expected_ids or len(expected_ids) != len(event_ids):
    raise ValueError("The course CSV must contain nonempty, unique narration IDs")

required = {"narration_id", "verb_output", "noun_output"}
optional = {"action_output"}
pair_pattern = re.compile(r"([0-9]+),([0-9]+)")

def validate(path):
    rows = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError(f"{path}: expected a list of {len(expected_ids)} predictions")
    seen = set()
    advanced_rows = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) not in (required, required | optional):
            raise ValueError(f"{path}: invalid prediction keys")
        identity = row["narration_id"]
        if not isinstance(identity, str) or identity not in expected_ids or identity in seen:
            raise ValueError(f"{path}: duplicate, unexpected, or invalid ID {identity!r}")
        seen.add(identity)
        for field, size in (("verb_output", 97), ("noun_output", 300)):
            scores = torch.as_tensor(row[field])
            if scores.shape != (size,) or not scores.is_floating_point():
                raise ValueError(f"{path}: {identity} needs {size} float scores for {field}")
            if not torch.isfinite(scores).all().item():
                raise ValueError(f"{path}: {identity} has a nonfinite {field} score")
        if "action_output" in row:
            advanced_rows += 1
            scores = row["action_output"]
            if not isinstance(scores, dict) or len(scores) != 100:
                raise ValueError(f"{path}: {identity} needs exactly 100 action scores")
            numeric_pairs = set()
            for key, score in scores.items():
                match = pair_pattern.fullmatch(key) if isinstance(key, str) else None
                if match is None:
                    raise ValueError(f"{path}: {identity} has invalid action key {key!r}")
                verb, noun = map(int, match.groups())
                if not (0 <= verb < 97 and 0 <= noun < 300) or (verb, noun) in numeric_pairs:
                    raise ValueError(f"{path}: {identity} has invalid or repeated action pair {key!r}")
                numeric_pairs.add((verb, noun))
                if type(score) is not float or not math.isfinite(score):
                    raise ValueError(f"{path}: {identity} needs a finite Python float for {key!r}")
    if seen != expected_ids or advanced_rows not in (0, len(rows)):
        raise ValueError(f"{path}: incomplete coverage or mixed action scoring modes")
    print(f"{path}: {len(rows)} events, {advanced_rows} with explicit action scores")

for path in (ar_path, sta_path):
    validate(path)

entries = ("submission.pt", "submission_sta.pt")
with ZipFile(zip_path, "x", compression=ZIP_DEFLATED) as archive:
    for path, entry in zip((ar_path, sta_path), entries):
        archive.write(path, arcname=entry)
with ZipFile(zip_path) as archive:
    if archive.namelist() != list(entries) or archive.testzip() is not None:
        raise ValueError(f"{zip_path}: archive root entries or integrity check failed")
print(f"Validated {zip_path}")
