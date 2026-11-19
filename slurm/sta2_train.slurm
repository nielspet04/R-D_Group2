#!/bin/bash -l

#SBATCH --job-name=sta2-train
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/sta2-train-%j.out

set -euo pipefail

# Override this too when selecting an H100 partition at submission.
module load "${STA2_CLUSTER_MODULE:-cluster/wice/gpu_a100}"
STA2_CONDA_ROOT="${STA2_CONDA_ROOT:-$VSC_DATA/miniconda3}"
source "$STA2_CONDA_ROOT/etc/profile.d/conda.sh"
if [[ -n "${STA2_CONDA_ENV:-}" ]]; then
    conda activate "$STA2_CONDA_ENV"
elif [[ -d "$VSC_DATA/conda_envs/rd_sta2" ]]; then
    conda activate "$VSC_DATA/conda_envs/rd_sta2"
else
    conda activate rd_sta2
fi

cd "$VSC_DATA/R-D_Group2"

STA2_GPUS="${STA2_GPUS:-4}"
STA2_TRAIN_CONFIG="${STA2_TRAIN_CONFIG:-configs/sta2/main.yaml}"
export STA2_RAW_DIR="${STA2_RAW_DIR:-$VSC_SCRATCH/epic_sta/raw_fixed}"
export STA2_METADATA_DIR="${STA2_METADATA_DIR:-$VSC_SCRATCH/epic_sta/sta2_timelines}"
export STA2_CHECKPOINT="${STA2_CHECKPOINT:-$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

case "$STA2_GPUS" in
    1|2|4) ;;
    *) printf 'STA2_GPUS must be 1, 2 or 4\n' >&2; exit 2 ;;
esac

for file in scripts/train_sta2.py "$STA2_TRAIN_CONFIG" \
    "$STA2_METADATA_DIR/manifest.json" "$STA2_CHECKPOINT"; do
    if [[ ! -s "$file" ]]; then
        printf 'Missing required file: %s\n' "$file" >&2
        printf 'The training entrypoint/configuration are supplied by Emiel.\n' >&2
        exit 1
    fi
done
test -d "$STA2_RAW_DIR"

export STA2_GPUS
python - <<'PY'
import os
import torch

expected = int(os.environ["STA2_GPUS"])
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == expected, (
    f"Requested {expected} processes, but {torch.cuda.device_count()} GPUs are visible"
)
names = {torch.cuda.get_device_properties(i).name for i in range(expected)}
assert len(names) == 1, "Use homogeneous GPUs"
for i in range(expected):
    with torch.cuda.device(i):
        assert torch.cuda.is_bf16_supported(), f"GPU {i} does not support BF16"
print("GPUs:", expected, names)
print("Required accumulation for global batch 16, microbatch 1:", 16 // expected)
PY

printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
printf 'Python: %s\n' "$(command -v python)"
printf 'Video directory: %s\n' "$STA2_RAW_DIR"
printf 'Metadata directory: %s\n' "$STA2_METADATA_DIR"
printf 'Training configuration: %s\n' "$STA2_TRAIN_CONFIG"
nvidia-smi

# --mem requests memory per node. Remove conflicting inherited alternatives.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
# Slurm may adjust CPUs per task for the allocation; retain that value rather
# than the original cpu count encoded in SLURM_TRES_PER_TASK.
unset SLURM_TRES_PER_TASK

# One launcher, one process per GPU. The trainer owns DDP and accumulation.
srun --ntasks=1 python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$STA2_GPUS" \
    scripts/train_sta2.py --config "$STA2_TRAIN_CONFIG" "$@"
