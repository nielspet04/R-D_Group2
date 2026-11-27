#!/bin/bash -l
# One FULL epoch on the production GPU partition; not the 30-minute debug queue.
#SBATCH --job-name=sta2-compact-a100-timing
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/sta2-compact-a100-timing-%j.out
#SBATCH --open-mode=append

set -euo pipefail
cd "$VSC_DATA/R-D_Group2"
export STA2_COMPACT_STOP_AFTER_EPOCH=1
# Preserve the normal config, schedule, full splits and output layout.
# The shared wrapper rejects accidental resume in this timing mode.
source slurm/sta2_compact_a100.slurm
