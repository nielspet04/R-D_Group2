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

set -Eeuo pipefail

# Emit diagnostics even when environment/module setup fails before Python starts.
sta2_batch_exit() {
    local code=$?
    printf 'STA2 batch exit: job=%s code=%s elapsed_seconds=%s utc=%s\n' \
        "${SLURM_JOB_ID:-unknown}" "$code" "$SECONDS" "$(date -u +%FT%TZ)"
    if [[ -n "${RND_MONITOR_API_URL:-}" && -n "${output_dir:-}" && -n "${STA2_PYTHON:-}" ]]; then
        "$STA2_PYTHON" scripts/monitor_lifecycle.py --output-root "$output_dir" --exit-code "$code" || true
    fi
    exit "$code"
}
trap sta2_batch_exit EXIT
trap 'printf "STA2 shell failure: source=%s line=%s code=%s\n" "${BASH_SOURCE[0]:-$0}" "$LINENO" "$?" >&2' ERR
printf 'STA2 batch start: job=%s host=%s partition=%s utc=%s\n' \
    "${SLURM_JOB_ID:-unknown}" "$(hostname)" "${SLURM_JOB_PARTITION:-unknown}" "$(date -u +%FT%TZ)"

# Match the cluster module to the selected partition.
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
export STA2_METADATA_DIR="${STA2_METADATA_DIR:-$VSC_SCRATCH/epic_sta/sta2_timelines_local}"
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

# Optional diagnostic precision override; preserve the source configuration.
if [[ -n "${STA2_TRAIN_PRECISION:-}" ]]; then
    case "$STA2_TRAIN_PRECISION" in
        bf16|fp32) ;;
        *) printf 'STA2_TRAIN_PRECISION must be bf16 or fp32\n' >&2; exit 2 ;;
    esac
    STA2_TRAIN_CONFIG=$(python - "$STA2_TRAIN_CONFIG" "$STA2_TRAIN_PRECISION" "$SLURM_JOB_ID" <<'PY'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
config["training"]["bf16"] = sys.argv[2] == "bf16"
output = Path("results/sta2") / f"launcher-config-{sys.argv[3]}.yaml"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(yaml.safe_dump(config))
print(output)
PY
    )
fi

export STA2_GPUS
export STA2_TRAIN_CONFIG
python - <<'PY'
import os
from pathlib import Path
import torch
import yaml

expected = int(os.environ["STA2_GPUS"])
config = yaml.safe_load(Path(os.environ["STA2_TRAIN_CONFIG"]).read_text())
bf16 = config["training"].get("bf16", True)
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == expected, (
    f"Requested {expected} processes, but {torch.cuda.device_count()} GPUs are visible"
)
names = {torch.cuda.get_device_properties(i).name for i in range(expected)}
assert len(names) == 1, "Use homogeneous GPUs"
if bf16:
    for i in range(expected):
        with torch.cuda.device(i):
            assert torch.cuda.is_bf16_supported(), f"GPU {i} does not support BF16"
print("GPUs:", expected, names)
print("Training precision:", "bf16" if bf16 else "fp32")
print("Required accumulation for global batch 16, microbatch 1:", 16 // expected)
PY

printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
STA2_PYTHON="$(command -v python)"
printf 'Python: %s\n' "$STA2_PYTHON"
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
# Resolve Python before srun: the step's PATH may differ from the batch shell.
srun --unbuffered --ntasks=1 --export=ALL "$STA2_PYTHON" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$STA2_GPUS" \
    scripts/train_sta2.py --config "$STA2_TRAIN_CONFIG" "$@"
