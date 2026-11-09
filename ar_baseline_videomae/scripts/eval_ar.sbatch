#!/bin/bash
# Usage: sbatch scripts/eval_ar.sbatch [run_name]    (default: the baseline)
#
#SBATCH --job-name=ar_eval
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

source /data/leuven/394/vsc39410/miniconda3/etc/profile.d/conda.sh
conda activate rd_lab

export HF_HOME=$VSC_DATA/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

REPO=/data/leuven/394/vsc39410/epic-kitchens/R-D_Group2
AR="$REPO/ar_baseline_videomae"
RUN=${1:-videomae_b_f16_seed1}
OUT="$VSC_SCRATCH/runs/$RUN"

cd "$AR"
python -m src.ar.train --eval_only --export --skip_missing \
  --val_csv "$REPO/data/processed/EPIC_100_validation_participant_split.csv" \
  --video_root "$VSC_SCRATCH/epic-kitchens/videos_640x360" \
  --out_dir "$OUT" --workers "$SLURM_CPUS_PER_TASK"

mkdir -p "$AR/results/$RUN"
cp "$OUT"/preds_val.csv "$OUT"/metrics_eval.json "$AR/results/$RUN/"