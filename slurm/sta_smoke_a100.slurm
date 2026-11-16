#!/bin/bash -l
#SBATCH --job-name=sta-a100-smoke
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96000M
#SBATCH --time=00:15:00
#SBATCH --output=logs/sta-a100-smoke-%j.out

set -euo pipefail

module load cluster/wice/gpu_a100
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

printf 'Job ID: %s\n' "$SLURM_JOB_ID"
printf 'Host: %s\n' "$(hostname)"
printf 'Python: %s\n' "$PYTHON"
printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
nvidia-smi

"$PYTHON" - <<'PY'
import torch
import torchvision

assert torch.cuda.is_available(), "CUDA is unavailable"
name = torch.cuda.get_device_name(0)
assert "A100" in name, f"Expected A100, got {name}"
capability = torch.cuda.get_device_capability(0)
assert capability == (8, 0), capability
memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {name}")
print(f"Compute capability: {capability}")
print(f"GPU memory: {memory_gib:.2f} GiB")
print(f"CUDA tensor check: {(torch.ones(1, device='cuda') + 1).item()}")
PY

# The real-video unit tests use this exact fixture path.
test -s data/raw/P01_03.mp4
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" scripts/evaluate_sta_model_0.py

started=$SECONDS
"$PYTHON" scripts/extract_sta_features.py \
  --raw-dir "$VSC_SCRATCH/epic_sta/raw" \
  --split train \
  --output-root results/vsc_a100_smoke_features \
  --device cuda \
  --limit 64 \
  --event-batch-size 8 \
  --workers 8 \
  --shard-size 64 \
  --dtype float16
printf 'Extraction seconds: %d\n' "$((SECONDS - started))"

"$PYTHON" scripts/inspect_sta_features.py \
  --feature-root results/vsc_a100_smoke_features \
  --split train \
  --check-all

"$PYTHON" scripts/verify_sta_extraction.py \
  --feature-root results/vsc_a100_smoke_features \
  --split train \
  --events 64 \
  --shard-size 64 \
  --limit

printf 'A100 smoke job passed.\n'
