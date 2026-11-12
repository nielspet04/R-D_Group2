#!/bin/bash -l
#SBATCH --job-name=sta-a100-extract
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/sta-a100-extract-%j.out

set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s {train|validation}\n' "$0" >&2
    exit 2
fi

SPLIT=$1
case "$SPLIT" in
    train) EVENTS=59767 ;;
    validation) EVENTS=7450 ;;
    *)
        printf 'Invalid split: %s. Use train or validation.\n' "$SPLIT" >&2
        exit 2
        ;;
esac

module load cluster/wice/gpu_a100
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda activate rd_epic
export TORCH_HOME="$VSC_DATA/torch-cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$VSC_DATA/R-D_Group2"

printf 'Job ID: %s\n' "$SLURM_JOB_ID"
printf 'Split: %s\n' "$SPLIT"
printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
nvidia-smi

started=$SECONDS
python scripts/extract_sta_features.py \
  --raw-dir "$VSC_SCRATCH/epic_sta/raw" \
  --split "$SPLIT" \
  --output-root data/processed/sta_features \
  --device cuda \
  --event-batch-size 8 \
  --workers 8 \
  --shard-size 2048 \
  --dtype float16 \
  --resume
printf 'Extraction seconds: %d\n' "$((SECONDS - started))"

python scripts/inspect_sta_features.py \
  --feature-root data/processed/sta_features \
  --split "$SPLIT" \
  --check-all

python scripts/verify_sta_extraction.py \
  --feature-root data/processed/sta_features \
  --split "$SPLIT" \
  --events "$EVENTS" \
  --shard-size 2048

printf 'A100 %s extraction passed.\n' "$SPLIT"
