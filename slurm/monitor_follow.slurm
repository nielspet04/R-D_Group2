#!/bin/bash -l
#SBATCH --job-name=dashboard-upload
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/dashboard-upload-%j.out

set -euo pipefail
module load cluster/wice/batch
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda activate "$VSC_DATA/conda_envs/rd_sta2"
cd "${STA2_PROJECT_DIR:-$VSC_DATA/R-D_Group2}"
if [[ -f "$HOME/.config/rd-dashboard/env" ]]; then
    source "$HOME/.config/rd-dashboard/env"
fi
export PYTHONUNBUFFERED=1
exec python scripts/monitor_follow.py "$@"
