#!/bin/bash -l
#SBATCH --job-name=sta2-test-62771307
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=04:00:00
#SBATCH --output=logs/sta2-test-62771307-%j.out
set -euo pipefail
cd /data/leuven/397/vsc39706/R-D_Group2
module load cluster/wice/gpu_a100
source /data/leuven/397/vsc39706/miniconda3/etc/profile.d/conda.sh
conda activate /data/leuven/397/vsc39706/conda_envs/rd_sta2
: "${COURSE_TEST:?Set the actual course CSV path}"
: "${PREP:?Set the generated-report directory}"
: "${STA_CKPT:?}" "${BACKBONE:?}" "${BACKBONE_CONFIG:?}"
: "${RAW_DIR:?}" "${METADATA:?}" "${ANNOTATIONS:?}" "${TRAIN_SPLIT:?}" "${VAL_SPLIT:?}"
python - "$PREP/preflight.json" "$COURSE_TEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
assert report.get("preflight_passed") is True, "CPU preflight must pass before inference"
assert not report["reusable_predictions"], "Reuse the verified artifact instead of spending GPU credits"
assert hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest() == report["course_test"]["sha256"]
PY
# Recheck inputs on the node. This performs no GPU inference.
python scripts/sta2_submission_preflight.py
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK
STA_OUTPUT="results/sta2/competition-final-${SLURM_JOB_ID}/selected"
test ! -e "$STA_OUTPUT"
mkdir -p "$STA_OUTPUT"
srun --ntasks=1 --export=ALL "$(command -v python)" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 scripts/evaluate_sta2.py \
  --split test --events "$COURSE_TEST" --checkpoint "$STA_CKPT" \
  --backbone-checkpoint "$BACKBONE" --backbone-config "$BACKBONE_CONFIG" \
  --raw-dir "$RAW_DIR" --metadata "$METADATA" --annotations "$ANNOTATIONS" \
  --train-split "$TRAIN_SPLIT" --validation-split "$VAL_SPLIT" \
  --output-dir "$STA_OUTPUT/verified"
printf 'Inference completed. CPU export source: %s/verified/predictions.pt\n' "$STA_OUTPUT"
