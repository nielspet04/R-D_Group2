#!/bin/bash
# CPU-only checks for Emiel's selected checkpoint and the completed course test audit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STA2_PYTHON=/data/leuven/397/vsc39706/conda_envs/rd_sta2/bin/python
if [[ ! -x "$STA2_PYTHON" ]]; then
    printf 'Missing STA2 Python: %s\n' "$STA2_PYTHON" >&2
    exit 1
fi
export PREP="$PWD/results/sta2/codabench-first-62771307/preparation"
export STA_CKPT="$PWD/results/sta2/full-h100-62771307/checkpoints/best.pt"
export BACKBONE=/data/leuven/397/vsc39706/sta2/checkpoints/vjepa2_1_vitG_384.pt
export BACKBONE_CONFIG="$PWD/configs/sta2/data_backbone.yaml"
export ANNOTATIONS="$PWD/data/raw/EPIC_100_train.csv"
export TRAIN_SPLIT="$PWD/data/processed/EPIC_100_train_participant_split.csv"
export VAL_SPLIT="$PWD/data/processed/EPIC_100_validation_participant_split.csv"
export RAW_DIR=/scratch/leuven/397/vsc39706/epic_sta/raw_fixed
export METADATA=/scratch/leuven/397/vsc39706/epic_sta/sta2_test_timelines_62771307/manifest.json
export COURSE_TEST="$PWD/data/raw/EPIC_100_test.csv"
printf 'CPU preflight; test audit: %s\n' "$METADATA"
exec "$STA2_PYTHON" -u scripts/sta2_submission_preflight.py
