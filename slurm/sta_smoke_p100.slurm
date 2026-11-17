#!/bin/bash -l
#SBATCH --job-name=sta-smoke
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_p100_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --time=00:30:00
#SBATCH --output=logs/sta-smoke-%j.out

set -euo pipefail

module load cluster/wice/gpu_p100

PYTHON="$VSC_DATA/conda_envs/rd_epic_cuda/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    printf 'Missing rd_epic_cuda Python: %s\n' "$PYTHON" >&2
    exit 1
fi

export TORCH_HOME="$VSC_DATA/torch-cache"

cd "$VSC_DATA/R-D_Group2"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $(pwd)"
echo "Python: $PYTHON"

nvidia-smi

"$PYTHON" -c "
import torch
import torchvision

print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())

assert torch.cuda.is_available(), 'CUDA is not available'

print('CUDA version:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print(
    'GPU memory GiB:',
    torch.cuda.get_device_properties(0).total_memory / 1024**3,
)
"

"$PYTHON" -m unittest discover -s tests -v

"$PYTHON" scripts/evaluate_sta_model_0.py

"$PYTHON" scripts/extract_sta_features.py \
  --split train \
  --limit 32 \
  --output-root results/vsc_smoke_features \
  --device cuda \
  --event-batch-size 4 \
  --workers 4 \
  --shard-size 32

"$PYTHON" scripts/inspect_sta_features.py \
  --feature-root results/vsc_smoke_features \
  --split train \
  --check-all

echo "STA smoke job completed successfully"
