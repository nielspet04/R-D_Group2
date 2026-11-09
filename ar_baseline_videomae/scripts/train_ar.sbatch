#!/bin/bash
#
#SBATCH --job-name=ar_videomae
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

source /data/leuven/394/vsc39410/miniconda3/etc/profile.d/conda.sh
conda activate rd_lab

# compute nodes usually have no internet: the model must be cached beforehand
export HF_HOME=$VSC_DATA/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

REPO=/data/leuven/394/vsc39410/epic-kitchens/R-D_Group2
AR="$REPO/ar_baseline_videomae"
RUN=${1:?usage: sbatch scripts/train_ar.sbatch <run_name> [train.py args]}
shift

cd "$AR"
python -m src.ar.train \
  --train_csv "$REPO/data/processed/EPIC_100_train_participant_split.csv" \
  --val_csv   "$REPO/data/processed/EPIC_100_validation_participant_split.csv" \
  --video_root "$VSC_SCRATCH/epic-kitchens/videos_640x360" \
  --out_dir "$VSC_SCRATCH/runs/$RUN" \
  --frames 16 --batch_size 8 --accum 4 --epochs 10 --seed 1 \
  --workers "$SLURM_CPUS_PER_TASK" --skip_missing "$@"

# small results + light checkpoint to /data (never fails the job)
OUT="$VSC_SCRATCH/runs/$RUN"
mkdir -p "$AR/results/$RUN"
cp "$OUT"/{config.json,metrics.jsonl,best_metrics.json} "$AR/results/$RUN/" || true
OUT="$OUT" DST="$AR/results/$RUN/best_small.pt" python - <<'EOF' || true
import os, torch
c = torch.load(os.environ["OUT"] + "/best.pt", map_location="cpu", weights_only=False)
torch.save({"model": c["model"], "args": c["args"], "epoch": c["epoch"]}, os.environ["DST"])
EOF