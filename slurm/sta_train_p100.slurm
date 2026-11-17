#!/bin/bash -l
#SBATCH --job-name=sta-train
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --time=04:00:00
#SBATCH --output=logs/sta-train-%j.out

set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s {mean_pool|gru}\n' "$0" >&2
    exit 2
fi

MODEL=$1
case "$MODEL" in
    mean_pool|gru) ;;
    *)
        printf 'Invalid model: %s. Use mean_pool or gru.\n' "$MODEL" >&2
        exit 2
        ;;
esac

module load cluster/wice/gpu_p100
PYTHON="$VSC_DATA/conda_envs/rd_epic_cuda/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    printf 'Missing rd_epic_cuda Python: %s\n' "$PYTHON" >&2
    exit 1
fi

export TORCH_HOME="$VSC_DATA/torch-cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$VSC_DATA/R-D_Group2"

FEATURE_ROOT=data/processed/sta_features
ANNOTATIONS=data/raw/EPIC_100_train.csv
OUTPUT_ROOT=${RND_OUTPUT_ROOT:-results}
CHECKPOINT="$OUTPUT_ROOT/checkpoints/$MODEL/best.pt"
EVALUATION="$OUTPUT_ROOT/tables/${MODEL}_evaluation.json"

printf 'Job ID: %s\n' "$SLURM_JOB_ID"
printf 'Model: %s\n' "$MODEL"
printf 'Host: %s\n' "$(hostname)"
printf 'Python: %s\n' "$PYTHON"
printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
nvidia-smi

"$PYTHON" - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

test -s "$FEATURE_ROOT/train/manifest.csv"
test -s "$FEATURE_ROOT/validation/manifest.csv"
test -s "$ANNOTATIONS"

"$PYTHON" scripts/verify_sta_extraction.py \
  --feature-root "$FEATURE_ROOT" \
  --split train \
  --events 59767 \
  --shard-size 2048

"$PYTHON" scripts/verify_sta_extraction.py \
  --feature-root "$FEATURE_ROOT" \
  --split validation \
  --events 7450 \
  --shard-size 2048

started=$SECONDS
"$PYTHON" scripts/train_sta.py \
  --model "$MODEL" \
  --feature-root "$FEATURE_ROOT" \
  --annotations "$ANNOTATIONS" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda
printf 'Training seconds: %d\n' "$((SECONDS - started))"

"$PYTHON" scripts/evaluate_sta_model.py \
  --checkpoint "$CHECKPOINT" \
  --feature-root "$FEATURE_ROOT" \
  --annotations "$ANNOTATIONS" \
  --output "$EVALUATION" \
  --device cuda

"$PYTHON" - "$EVALUATION" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
metrics = json.loads(path.read_text(encoding="utf-8"))
if metrics["validation_events"] != 7450:
    raise SystemExit(
        f"Expected 7,450 validation events, got {metrics['validation_events']}"
    )
if metrics["validation_fallback_events"] != 24:
    raise SystemExit(
        "Expected 24 validation fallback events, got "
        f"{metrics['validation_fallback_events']}"
    )
for name in (
    "validation_loss",
    "verb_top1_accuracy",
    "verb_top5_accuracy",
    "verb_mt5r",
    "noun_top1_accuracy",
    "noun_top5_accuracy",
    "noun_mt5r",
    "action_top1_accuracy",
    "action_top5_accuracy",
    "action_mt5r",
):
    value = metrics[name]
    if value is None or not math.isfinite(value):
        raise SystemExit(f"Metric {name} is not finite: {value}")
print("Evaluation coverage and finite-metric checks passed")
PY

printf 'STA %s training and evaluation passed.\n' "$MODEL"
