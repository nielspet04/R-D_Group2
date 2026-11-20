#!/bin/bash -l

#SBATCH --job-name=sta2-metadata
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/sta2-metadata-%j.out

set -euo pipefail

module load cluster/wice/batch

# Reuse the verified private environment; do not load Miniforge3.
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda activate "$VSC_DATA/conda_envs/rd_sta2"

cd "${STA2_PROJECT_DIR:-$VSC_DATA/R-D_Group2}"

STA2_RAW_DIR="${STA2_RAW_DIR:-$VSC_SCRATCH/epic_sta/raw_fixed}"
STA2_METADATA_DIR="${STA2_METADATA_DIR:-$VSC_SCRATCH/epic_sta/sta2_timelines}"
STA2_ANNOTATIONS="${STA2_ANNOTATIONS:-data/raw/EPIC_100_train.csv}"
STA2_VIDEO_INFO="${STA2_VIDEO_INFO:-data/raw/EPIC_100_video_info.csv}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

test -d "$STA2_RAW_DIR"
test -s "$STA2_ANNOTATIONS"
test -s "$STA2_VIDEO_INFO"
if ! grep -q '^def reusable_timeline(' scripts/build_sta2_metadata.py; then
    printf 'Sync the updated metadata script with resume support before running.\n' >&2
    exit 1
fi

mkdir -p "$STA2_METADATA_DIR"
# Prevent two copies of this batch job from writing the same manifest.
# Do not run a separate manual metadata scan against this directory.
exec 9>"$STA2_METADATA_DIR/.metadata-job.lock"
if ! flock -n 9; then
    printf 'Another metadata batch job is using: %s\n' "$STA2_METADATA_DIR" >&2
    exit 1
fi

printf 'Job ID: %s\n' "$SLURM_JOB_ID"
printf 'Host: %s\n' "$(hostname)"
printf 'Python: %s\n' "$(command -v python)"
printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
printf 'Video directory: %s\n' "$STA2_RAW_DIR"
printf 'Resuming metadata directory: %s\n' "$STA2_METADATA_DIR"

python scripts/build_sta2_metadata.py \
    --annotations "$STA2_ANNOTATIONS" \
    --raw-dir "$STA2_RAW_DIR" \
    --video-info "$STA2_VIDEO_INFO" \
    --video-id-column video_id \
    --fps-column fps \
    --output-dir "$STA2_METADATA_DIR"

python - "$STA2_METADATA_DIR/manifest.json" "$STA2_ANNOTATIONS" <<'PY'
from collections import Counter
import csv
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
with Path(sys.argv[2]).open() as stream:
    required = {row["video_id"].upper() for row in csv.DictReader(stream)}
assert set(manifest["videos"]) == required, "Incomplete video coverage"
assert manifest.get("complete") is True, "Metadata scan did not finish"
counts = Counter(item["status"] for item in manifest["videos"].values())
print("Completed video coverage:", len(required))
print("Status counts:", dict(counts))
print("Video errors remain recorded; review them before the GPU pilot.")
PY
