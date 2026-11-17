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

module load cluster/wice/gpu_h100
module load Miniforge3/25.3.0-3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rd_sta2

cd "$VSC_DATA/R-D_Group2"

: "${STA2_RAW_DIR:?Set STA2_RAW_DIR}"
: "${STA2_CHECKPOINT:?Set STA2_CHECKPOINT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

nvidia-smi

MODE="${1:-future}"

EXTRA=()

case "$MODE" in
    future)
        ;;
    observed-only)
        EXTRA+=(--observed-only)
        ;;
    *)
        printf 'Use future or observed-only\n' >&2
        exit 2
        ;;
esac

python scripts/profile_sta2.py \
  --annotations data/raw/EPIC_100_train.csv \
  --train-split \
    data/processed/EPIC_100_train_participant_split.csv \
  --validation-split \
    data/processed/EPIC_100_validation_participant_split.csv \
  --raw-dir "$STA2_RAW_DIR" \
  --metadata data/processed/sta2_timelines/manifest.json \
  --checkpoint "$STA2_CHECKPOINT" \
  --config configs/sta2/data_backbone.yaml \
  --output-dir "results/sta2/pilot-${MODE}-${SLURM_JOB_ID}" \
  --events 64 \
  --warmup-steps 10 \
  --measured-steps 100 \
  --workers 2 \
  "${EXTRA[@]}"