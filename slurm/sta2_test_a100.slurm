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
STA2_PYTHON=/data/leuven/397/vsc39706/conda_envs/rd_sta2/bin/python
export PREP="$PWD/results/sta2/codabench-first-62771307/preparation"
export STA_CKPT="$PWD/results/sta2/full-h100-62771307/checkpoints/best.pt"
export BACKBONE=/data/leuven/397/vsc39706/sta2/checkpoints/vjepa2_1_vitG_384.pt
export BACKBONE_CONFIG="$PWD/configs/sta2/data_backbone.yaml"
export ANNOTATIONS="$PWD/data/raw/EPIC_100_train.csv"
export TRAIN_SPLIT="$PWD/data/processed/EPIC_100_train_participant_split.csv"
export VAL_SPLIT="$PWD/data/processed/EPIC_100_validation_participant_split.csv"
export RAW_DIR=/scratch/leuven/397/vsc39706/epic_sta/raw_fixed
export METADATA=/scratch/leuven/397/vsc39706/epic_sta/sta2_test_timelines_62771307/manifest.json
export COURSE_TEST="$PWD/data/raw/EPIC_100_test.csv"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
printf 'STA test inference job=%s host=%s started=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)"
test -x "$STA2_PYTHON"
"$STA2_PYTHON" - "$PREP/preflight.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

report = json.loads(Path(sys.argv[1]).read_text())
assert report.get("preflight_passed") is True, "CPU preflight must pass before inference"
assert not report["reusable_predictions"], "Reuse the verified artifact instead of spending GPU credits"
expected_checkpoint = "9890d287eb4e9a7b6da61c470adcff955859b5e1d372e71821ca026e11a12610"
expected_course = "b88a87c285d8c004e36c55a2eea021324a9649823ef80c9463896ca7c0325412"
assert report["checkpoint_sha256"] == expected_checkpoint
assert report["course_test"]["sha256"] == expected_course
assert report["course_test"]["events"] == 9668
for key, supplied in report["paths"].items():
    assert Path(os.environ[key]).resolve() == Path(supplied).resolve(), f"Input path changed: {key}"
assert Path(os.environ["COURSE_TEST"]).resolve() == Path(report["course_test"]["path"]).resolve()
print("Checking selected checkpoint and test input identities...", flush=True)
checks = {
    "STA_CKPT": expected_checkpoint, "COURSE_TEST": expected_course,
    "BACKBONE_CONFIG": report["backbone_config_sha256"], "METADATA": report["metadata_sha256"],
    "ANNOTATIONS": report["verified"]["split_hashes"]["annotations"],
    "TRAIN_SPLIT": report["verified"]["split_hashes"]["train_split"],
    "VAL_SPLIT": report["verified"]["split_hashes"]["validation_split"],
}
for key, expected in checks.items():
    assert sha256(os.environ[key]) == expected, f"Input contents changed: {key}"
print("PASS frozen checkpoint, course CSV, splits, config, and test audit; evaluator will verify backbone weights", flush=True)
PY
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK
STA_OUTPUT="results/sta2/competition-final-${SLURM_JOB_ID}/selected"
test ! -e "$STA_OUTPUT"
mkdir -p "$STA_OUTPUT"
cp "$PREP/preflight.json" "$STA_OUTPUT/preflight.json"
git rev-parse HEAD > "$STA_OUTPUT/inference-source-commit.txt"
sha256sum slurm/sta2_test_a100.slurm scripts/evaluate_sta2.py \
  "$BACKBONE_CONFIG" "$METADATA" "$COURSE_TEST" > "$STA_OUTPUT/inference-inputs.sha256"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
srun --ntasks=1 --export=ALL "$STA2_PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 scripts/evaluate_sta2.py \
  --split test --events "$COURSE_TEST" --checkpoint "$STA_CKPT" \
  --backbone-checkpoint "$BACKBONE" --backbone-config "$BACKBONE_CONFIG" \
  --raw-dir "$RAW_DIR" --metadata "$METADATA" --annotations "$ANNOTATIONS" \
  --train-split "$TRAIN_SPLIT" --validation-split "$VAL_SPLIT" \
  --output-dir "$STA_OUTPUT/verified"
printf 'Inference completed. CPU export source: %s/verified/predictions.pt\n' "$STA_OUTPUT"
