#!/bin/bash -l

#SBATCH --job-name=sta2-p100-smoke
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/sta2-p100-smoke-%j.out

set -euo pipefail

cd "$VSC_DATA/R-D_Group2"

# P100 has no native BF16 support. This is an explicit FP32 feasibility
# diagnostic using the same checkpoint, resolution and sampling protocol.
# Full ViT-G inference may exceed the P100's 16 GiB GPU memory.
export STA2_CLUSTER_MODULE=cluster/wice/gpu_p100
export STA2_PRECISION=fp32

source slurm/sta2_smoke.slurm "$@"
