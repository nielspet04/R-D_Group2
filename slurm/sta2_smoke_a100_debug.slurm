#!/bin/bash -l

#SBATCH --job-name=sta2-a100-debug
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000M
#SBATCH --time=00:30:00
#SBATCH --output=logs/sta2-a100-debug-%j.out

set -euo pipefail

cd "$VSC_DATA/R-D_Group2"

# Reuse the verified environment, input checks, and profiler invocation.
# Sourcing this file does not apply its #SBATCH resource directives.
source slurm/sta2_smoke.slurm "$@"
