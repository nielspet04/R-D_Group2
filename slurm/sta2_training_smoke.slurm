#!/bin/bash -l

#SBATCH --job-name=sta2-train-smoke
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/sta2-train-smoke-%j.out

set -euo pipefail

cd "$VSC_DATA/R-D_Group2"

# Use the separately prepared 32-event train / 16-event validation fixture.
# This wrapper does not modify videos, metadata, or the full-run configuration.
STA2_SMOKE_INPUTS="${STA2_SMOKE_INPUTS:-results/sta2/training-smoke-inputs}"
export STA2_GPUS=1
export STA2_TRAIN_CONFIG="$STA2_SMOKE_INPUTS/config.yaml"
# P100 requires FP32. The common launcher writes a job-specific configuration
# rather than changing the prepared fixture or the full-training defaults.
export STA2_CLUSTER_MODULE=cluster/wice/gpu_p100
export STA2_TRAIN_PRECISION=fp32
export STA2_RAW_DIR="${STA2_RAW_DIR:-$VSC_SCRATCH/epic_sta/raw_fixed}"
# This completed audit is separate from the older sta2_timelines directory.
export STA2_METADATA_DIR="$VSC_SCRATCH/epic_sta/sta2_timelines_local"
export STA2_CHECKPOINT="${STA2_CHECKPOINT:-$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt}"

for file in config.yaml annotations.csv train.csv validation.csv; do
    if [[ ! -s "$STA2_SMOKE_INPUTS/$file" ]]; then
        printf 'Missing prepared training-smoke input: %s/%s\n' "$STA2_SMOKE_INPUTS" "$file" >&2
        exit 1
    fi
done

# Source the common launcher; only this wrapper's SBATCH directives apply.
source slurm/sta2_train.slurm \
    --backbone-config configs/sta2/data_backbone.yaml \
    --backbone-checkpoint "$STA2_CHECKPOINT" \
    --raw-dir "$STA2_RAW_DIR" \
    --metadata "$STA2_METADATA_DIR/manifest.json" \
    --annotations "$STA2_SMOKE_INPUTS/annotations.csv" \
    --train-split "$STA2_SMOKE_INPUTS/train.csv" \
    --validation-split "$STA2_SMOKE_INPUTS/validation.csv" \
    --output-dir "results/sta2/training-smoke-$SLURM_JOB_ID" \
    "$@"
