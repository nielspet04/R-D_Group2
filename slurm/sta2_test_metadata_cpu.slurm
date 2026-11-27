#!/bin/bash -l
#SBATCH --job-name=sta2-test-metadata
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/sta2-test-metadata-%j.out

set -euo pipefail
cd /data/leuven/397/vsc39706/R-D_Group2
module load cluster/wice/batch
STA2_PYTHON=/data/leuven/397/vsc39706/conda_envs/rd_sta2/bin/python
COURSE_TEST="$PWD/data/raw/EPIC_100_test.csv"
VIDEO_INFO="$PWD/data/raw/EPIC_100_video_info.csv"
RAW_DIR=/scratch/leuven/397/vsc39706/epic_sta/raw_fixed
TEST_METADATA=/scratch/leuven/397/vsc39706/epic_sta/sta2_test_timelines_62771307
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1

printf 'CPU test-audit job=%s host=%s started=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)"
test -x "$STA2_PYTHON"
test -s "$COURSE_TEST"
test -s "$VIDEO_INFO"
test -d "$RAW_DIR"
"$STA2_PYTHON" -c 'import av, numpy, pandas; print("CPU audit dependencies: OK", flush=True)'
mkdir -p "$TEST_METADATA"
exec 9>"$TEST_METADATA/.metadata-job.lock"
flock -n 9 || { printf 'Another test-audit writer is running.\n' >&2; exit 1; }
git rev-parse HEAD
sha256sum "$COURSE_TEST" "$VIDEO_INFO" scripts/build_sta2_metadata.py \
  > "$TEST_METADATA/inputs-${SLURM_JOB_ID}.sha256"
"$STA2_PYTHON" -u scripts/build_sta2_metadata.py \
  --annotations "$COURSE_TEST" --video-info "$VIDEO_INFO" \
  --raw-dir "$RAW_DIR" --output-dir "$TEST_METADATA"

"$STA2_PYTHON" - "$TEST_METADATA/manifest.json" "$COURSE_TEST" <<'PY'
import csv, json, sys
from collections import Counter
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
with Path(sys.argv[2]).open(newline="", encoding="utf-8-sig") as handle:
    required = {row["video_id"].upper() for row in csv.DictReader(handle)}
assert manifest.get("complete") is True, "Test audit did not complete"
assert set(manifest["videos"]) == required, "Incomplete test-video coverage"
print("Completed test-video coverage:", len(required), flush=True)
print("Status counts:", dict(Counter(item["status"] for item in manifest["videos"].values())), flush=True)
print("Test manifest:", sys.argv[1], flush=True)
PY
