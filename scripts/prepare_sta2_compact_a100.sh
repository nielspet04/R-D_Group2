#!/usr/bin/env bash
# Read-only VSC preflight and credit quote; never submits a training job.
set -euo pipefail
launcher=slurm/sta2_compact_a100.slurm
walltime=40:00:00
if [[ "${1:-}" == --timing && $# == 1 ]]; then
    launcher=slurm/sta2_compact_a100_timing.slurm
    walltime=12:00:00
elif (( $# )); then
    printf 'Usage: bash scripts/prepare_sta2_compact_a100.sh [--timing]\n' >&2
    exit 2
fi

: "${VSC_DATA:?Run this on your VSC Bash login node}"
: "${VSC_SCRATCH:?VSC_SCRATCH must be set}"
cd "$VSC_DATA/R-D_Group2"

for command in squeue sacct sam-balance sam-quote; do
    command -v "$command" >/dev/null
done
git status --short --branch
git rev-parse HEAD
sam-balance

# A failed scheduler query must not be interpreted as an empty queue.
queue=$(squeue -M wice -u "$USER" -h -o '%i|%j|%P|%T|%M|%l|%R')
printf '\nCurrent account queue (job|name|partition|state|elapsed|limit|reason):\n%s\n' "$queue"
sacct -M wice -u "$USER" -S 2026-11-19 -X \
    --format=JobID,JobName,Partition,State,Elapsed,AllocTRES
printf '\nExisting compact checkpoints (inspect history before starting again):\n'
if [[ -d results/sta2 ]]; then
    find results/sta2 -path '*/compact-*/checkpoints/best.pt' -print
fi
if [[ "$queue" == *'|sta2-compact-'* ]]; then
    printf '\nA compact job already exists on this account. Follow it before preparing another.\n' >&2
    exit 1
fi
if [[ -n "${STA2_COMPACT_RESUME_RUN:-}" ]]; then
    printf 'This preflight is for a fresh run. Use the runbook resume procedure for %s\n' "$STA2_COMPACT_RESUME_RUN" >&2
    exit 1
fi

for file in \
    "$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt" \
    "$VSC_SCRATCH/epic_sta/sta2_timelines_local/manifest.json" \
    "$VSC_DATA/miniconda3/etc/profile.d/conda.sh" \
    data/raw/EPIC_100_train.csv \
    data/processed/EPIC_100_train_participant_split.csv \
    data/processed/EPIC_100_validation_participant_split.csv; do
    if [[ ! -s "$file" ]]; then
        printf 'Missing required input: %s\n' "$file" >&2
        exit 1
    fi
done
test -d "$VSC_SCRATCH/epic_sta/raw_fixed"
sta2_python="$VSC_DATA/miniconda3/envs/rd_sta2/bin/python"
if [[ ! -x "$sta2_python" ]]; then
    printf 'Missing STA2 environment Python: %s\n' "$sta2_python" >&2
    exit 1
fi
# Test the exact interpreter used by the compact wrapper, independent of the
# login shell's active environment. CUDA hardware is checked inside the job.
"$sta2_python" - <<'PYENV'
import sys
import torch
import torchvision
import numpy
import pandas
import yaml
import cv2
import av
import timm
import einops
import tqdm
print("STA2 environment Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("STA2 dependency imports passed; GPU availability is checked in the allocation.")
PYENV
bash -n slurm/sta2_compact_a100.slurm
bash -n slurm/sta2_train.slurm
bash -n "$launcher"

if [[ -n "${RND_MONITOR_API_URL:-}" && -n "${RND_MONITOR_UPLOAD_KEY_FILE:-}" ]]; then
    "$VSC_DATA/miniconda3/envs/rd_sta2/bin/python" scripts/monitor_upload.py --probe
else
    printf '\nHosted uploads are not configured in this shell. Source your private dashboard env before submission if wanted.\n'
fi

printf '\nQuote for two A100s, 16 CPUs, 128000 MiB, %s:\n' "$walltime"
sam-quote sbatch --time="$walltime" "$launcher"
printf '\nPreflight passed; no job submitted.\n'
printf 'Check the other account for a duplicate and record spent/pending credits.\n'
printf 'Submit only if the quote fits the remaining 1.1M development allowance and 3M campaign cap:\n'
printf 'mkdir -p logs\nunset STA2_COMPACT_RESUME_RUN STA2_COMPACT_STOP_AFTER_EPOCH\nsbatch --export=ALL --time=%s %s\n' "$walltime" "$launcher"
