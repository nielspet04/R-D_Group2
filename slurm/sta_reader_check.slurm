#!/bin/bash -l
#SBATCH --job-name=sta-reader-check
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96000M
#SBATCH --time=00:30:00
#SBATCH --output=logs/sta-reader-check-%j.out

set -euo pipefail

module load cluster/wice/gpu_a100
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda activate rd_epic_cuda

export TORCH_HOME="$VSC_DATA/torch-cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$VSC_DATA/R-D_Group2"

git rev-parse HEAD
nvidia-smi

python -m unittest \
  tests.test_sta_feature_extraction \
  tests.test_sta_sampling_regressions -v

CHECK_ROOT="results/sta_reader_check_${SLURM_JOB_ID}"

started=$SECONDS
python scripts/extract_sta_features.py \
  --raw-dir "$VSC_SCRATCH/epic_sta/raw" \
  --split train \
  --output-root "$CHECK_ROOT" \
  --limit 64 \
  --shard-size 64 \
  --event-batch-size 8 \
  --workers 8 \
  --device cuda \
  --dtype float16

printf 'Extraction seconds: %d\n' "$((SECONDS - started))"

python scripts/inspect_sta_features.py \
  --feature-root "$CHECK_ROOT" \
  --split train \
  --check-all

python scripts/verify_sta_extraction.py \
  --feature-root "$CHECK_ROOT" \
  --split train \
  --events 64 \
  --shard-size 64 \
  --limit

echo "Reader check passed."
