#!/bin/bash -l

#SBATCH --job-name=sta2-compact-a100
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=40:00:00
#SBATCH --output=logs/sta2-compact-a100-%j.out

set -euo pipefail
cd "$VSC_DATA/R-D_Group2"

# Independent two-block-head experiment; retain the development split.
export STA2_GPUS=2
export STA2_CLUSTER_MODULE=cluster/wice/gpu_a100
export STA2_TRAIN_CONFIG=configs/sta2/compact.yaml
unset STA2_TRAIN_PRECISION
export STA2_RAW_DIR="$VSC_SCRATCH/epic_sta/raw_fixed"
export STA2_METADATA_DIR="${STA2_COMPACT_METADATA_DIR:-$VSC_SCRATCH/epic_sta/sta2_timelines_local}"
export STA2_CHECKPOINT="$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt"
export STA2_CONDA_ROOT="$VSC_DATA/miniconda3"
export STA2_CONDA_ENV="$VSC_DATA/miniconda3/envs/rd_sta2"

# Fresh jobs get unique directories. Resume explicitly into the original one.
if (( $# )); then
    printf 'Use STA2_COMPACT_RESUME_RUN for resume; positional overrides are disabled.\n' >&2
    exit 2
fi
resume_args=()
debug_args=()
if [[ -n "${STA2_COMPACT_STOP_AFTER_EPOCH:-}" ]]; then
    if [[ "$STA2_COMPACT_STOP_AFTER_EPOCH" != 1 || -n "${STA2_COMPACT_RESUME_RUN:-}" ]]; then
        printf 'The compact timing run requires a fresh run and stop epoch 1.\n' >&2
        exit 2
    fi
    debug_args=(--stop-after-epoch 1)
fi
if [[ -n "${STA2_COMPACT_RESUME_RUN:-}" ]]; then
    output_dir="$STA2_COMPACT_RESUME_RUN"
    test -s "$output_dir/checkpoints/last.pt"
    test -s "$output_dir/history.json"
    resume_args=(--resume "$output_dir/checkpoints/last.pt")
else
    output_dir="results/sta2/compact-a100-$SLURM_JOB_ID"
    if [[ -e "$output_dir" ]]; then
        printf 'Output already exists; use explicit resume: %s\n' "$output_dir" >&2
        exit 2
    fi
fi

# Only this wrapper's SBATCH directives apply when sourcing the common launcher.
source slurm/sta2_train.slurm \
    --backbone-config configs/sta2/data_backbone.yaml \
    --backbone-checkpoint "$STA2_CHECKPOINT" \
    --raw-dir "$STA2_RAW_DIR" \
    --metadata "$STA2_METADATA_DIR/manifest.json" \
    --annotations data/raw/EPIC_100_train.csv \
    --train-split data/processed/EPIC_100_train_participant_split.csv \
    --validation-split data/processed/EPIC_100_validation_participant_split.csv \
    --output-dir "$output_dir" \
    --require-complete-metadata \
    ${debug_args[@]+"${debug_args[@]}"} \
    ${resume_args[@]+"${resume_args[@]}"}
