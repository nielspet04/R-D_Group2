#!/bin/bash -l

#SBATCH --job-name=sta2-final-h100
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=72:00:00
#SBATCH --output=logs/sta2-final-h100-%j.out

set -euo pipefail
cd "$VSC_DATA/R-D_Group2"
if (( $# )); then
    printf 'Use STA2_SELECTION_CHECKPOINT or STA2_FINAL_RESUME_RUN; positional overrides are disabled.\n' >&2
    exit 2
fi
export STA2_GPUS=2
export STA2_CLUSTER_MODULE=cluster/wice/gpu_h100
unset STA2_TRAIN_PRECISION
export STA2_RAW_DIR="$VSC_SCRATCH/epic_sta/raw_fixed"
export STA2_METADATA_DIR="$VSC_SCRATCH/epic_sta/sta2_timelines_local"
export STA2_CHECKPOINT="$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt"
export STA2_CONDA_ROOT="$VSC_DATA/miniconda3"
export STA2_CONDA_ENV="$VSC_DATA/conda_envs/rd_sta2"
module load "$STA2_CLUSTER_MODULE"
source "$STA2_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$STA2_CONDA_ENV"

resume_args=()
if [[ -n "${STA2_FINAL_RESUME_RUN:-}" ]]; then
    output_dir="$STA2_FINAL_RESUME_RUN"
    test -s "$output_dir/checkpoints/last.pt"
    resume_args=(--resume "$output_dir/checkpoints/last.pt")
else
    : "${STA2_SELECTION_CHECKPOINT:?Set this to the development run checkpoints/best.pt}"
    output_dir="results/sta2/final-h100-$SLURM_JOB_ID"
    if [[ -e "$output_dir" ]]; then
        printf 'Output already exists: %s\n' "$output_dir" >&2
        exit 2
    fi
    python scripts/prepare_sta2_final.py \
        --selection-checkpoint "$STA2_SELECTION_CHECKPOINT" \
        --backbone-checkpoint "$STA2_CHECKPOINT" \
        --metadata "$STA2_METADATA_DIR/manifest.json" \
        --output-dir "$output_dir/inputs"
fi
export STA2_TRAIN_CONFIG="$output_dir/inputs/config.yaml"
final_epochs=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_epochs"])' "$output_dir/inputs/selection.json")

source slurm/sta2_train.slurm \
    --backbone-config "$output_dir/inputs/backbone.yaml" \
    --source-provenance "$output_dir/inputs/source_provenance.json" \
    --backbone-checkpoint "$STA2_CHECKPOINT" \
    --raw-dir "$STA2_RAW_DIR" \
    --metadata "$STA2_METADATA_DIR/manifest.json" \
    --annotations data/raw/EPIC_100_train.csv \
    --train-split "$output_dir/inputs/train.csv" \
    --validation-split "$output_dir/inputs/validation.csv" \
    --output-dir "$output_dir" \
    --final-epochs "$final_epochs" \
    --require-complete-metadata \
    "${resume_args[@]}"
