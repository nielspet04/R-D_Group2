#!/bin/bash -l

#SBATCH --job-name=sta2-smoke
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/sta2-smoke-%j.out

set -euo pipefail

# The P100 wrapper overrides the module and precision explicitly.
module load "${STA2_CLUSTER_MODULE:-cluster/wice/gpu_a100}"

# Use the private Miniconda installation.
# Do not load the Miniforge3 module.
source /data/leuven/397/vsc39706/miniconda3/etc/profile.d/conda.sh
conda activate /data/leuven/397/vsc39706/conda_envs/rd_sta2

cd "$VSC_DATA/R-D_Group2"

# Allow explicit overrides while providing the verified defaults.
STA2_RAW_DIR="${STA2_RAW_DIR:-$VSC_SCRATCH/epic_sta/raw_fixed}"
STA2_METADATA_DIR="$VSC_SCRATCH/epic_sta/sta2_timelines_local"
STA2_CHECKPOINT="${STA2_CHECKPOINT:-$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt}"

STA2_CONFIG="${STA2_CONFIG:-configs/sta2/data_backbone.yaml}"
export STA2_PRECISION="${STA2_PRECISION:-bf16}"
case "$STA2_PRECISION" in
    bf16|fp32) ;;
    *) printf 'STA2_PRECISION must be bf16 or fp32\n' >&2; exit 2 ;;
esac

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

MODE="${1:-future}"
EXTRA=()

case "$MODE" in
    future)
        ;;
    observed-only)
        EXTRA+=(--observed-only)
        ;;
    *)
        printf 'Usage: sbatch %s {future|observed-only}\n' "$0" >&2
        exit 2
        ;;
esac

# Optional integration supplied by Emiel: use the real head and loss adapter.
if [[ -n "${STA2_PROFILE_ADAPTER:-}" ]]; then
    EXTRA+=(--adapter "$STA2_PROFILE_ADAPTER")
fi

require_file() {
    if [[ ! -s "$1" ]]; then
        printf 'Missing or empty file: %s\n' "$1" >&2
        exit 1
    fi
}

if [[ ! -d "$STA2_RAW_DIR" ]]; then
    printf 'Video directory does not exist: %s\n' "$STA2_RAW_DIR" >&2
    exit 1
fi

require_file scripts/profile_sta2.py
require_file "$STA2_CONFIG"
require_file "$STA2_METADATA_DIR/manifest.json"
require_file "$STA2_CHECKPOINT"

require_file data/raw/EPIC_100_train.csv
require_file data/processed/EPIC_100_train_participant_split.csv
require_file data/processed/EPIC_100_validation_participant_split.csv

printf 'Job ID: %s\n' "$SLURM_JOB_ID"
printf 'Host: %s\n' "$(hostname)"
printf 'Mode: %s\n' "$MODE"
printf 'Python: %s\n' "$(command -v python)"
printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
printf 'Video directory: %s\n' "$STA2_RAW_DIR"
printf 'Metadata directory: %s\n' "$STA2_METADATA_DIR"
printf 'Checkpoint: %s\n' "$STA2_CHECKPOINT"
printf 'Configuration: %s\n' "$STA2_CONFIG"
printf 'Precision: %s\n' "$STA2_PRECISION"

nvidia-smi

python - <<'PY'
import sys
import os

import av
import pandas
import torch

expected_prefix = (
    "/data/leuven/397/vsc39706/conda_envs/rd_sta2"
)

print("Python executable:", sys.executable)
print("Environment prefix:", sys.prefix)
print("pandas:", pandas.__version__)
print("PyAV:", av.__version__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)

assert sys.prefix == expected_prefix, (
    f"Wrong Python environment: {sys.prefix}"
)
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, (
    "This smoke job expects exactly one visible GPU"
)
if os.environ["STA2_PRECISION"] == "bf16":
    assert torch.cuda.is_bf16_supported(), (
        "BF16 is not supported on the allocated GPU"
    )

properties = torch.cuda.get_device_properties(0)

print("GPU:", properties.name)
print("GPU memory GiB:", properties.total_memory / 1024**3)
PY

OUTPUT_DIR="results/sta2/pilot-${MODE}-${SLURM_JOB_ID}"

started=$SECONDS

python scripts/profile_sta2.py \
  --annotations data/raw/EPIC_100_train.csv \
  --train-split \
    data/processed/EPIC_100_train_participant_split.csv \
  --validation-split \
    data/processed/EPIC_100_validation_participant_split.csv \
  --raw-dir "$STA2_RAW_DIR" \
  --metadata "$STA2_METADATA_DIR/manifest.json" \
  --checkpoint "$STA2_CHECKPOINT" \
  --config "$STA2_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --events 64 \
  --warmup-steps 10 \
  --measured-steps 100 \
  --workers 2 \
  --seed 42 \
  --precision "$STA2_PRECISION" \
  "${EXTRA[@]}"

printf 'Elapsed seconds: %s\n' "$((SECONDS - started))"
printf 'Profile results: %s\n' "$OUTPUT_DIR"
