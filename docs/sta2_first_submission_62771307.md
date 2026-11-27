# Emiel: first basic Codabench submission, job 62771307

Prepared against local `STA2_dev` commit `d41f90f53e78f3dd83c67b0bb7ce97d176059a78` on 27 November 2026.
The preparation files are not predictions and are not a Codabench submission.
No training, GPU allocation, public label retrieval, or upload was performed.

The workflow is: verify the existing epoch-5 checkpoint and its exact inputs on CPU;
identify the course test CSV; reuse matching verified test predictions if available;
otherwise quote and run two-pass STA inference; export and strictly validate STA on CPU;
obtain independently generated AR predictions with their provenance; validate both against
the same complete event table; create and hash the combined ZIP. Stop before uploading.

## 1. Start here in your existing VSC Bash terminal

You do not need to configure SSH. Run this read-only block in the VSC terminal you
used for training. It checks only the selected checkpoint, its run files, the repository's
course-data folder, named STA test output folders, and documented AR candidate locations.

```bash
cd /data/leuven/397/vsc39706/R-D_Group2
git status --short --branch
git rev-parse HEAD
printf '%s  %s\n' \
  9890d287eb4e9a7b6da61c470adcff955859b5e1d372e71821ca026e11a12610 \
  results/sta2/full-h100-62771307/checkpoints/best.pt | sha256sum -c -
ls -lh results/sta2/full-h100-62771307/history.json \
  results/sta2/full-h100-62771307/run-manifest-*.json
find data -type f -name EPIC_100_test.csv -print
find results/sta2/full-h100-62771307 -type f \
  \( -name predictions.pt -o -name metrics.json -o -name submission_sta.pt \) -print
for directory in results/sta2/competition-final-*; do
  if [[ -d "$directory" ]]; then
    find "$directory" -type f \
      \( -name predictions.pt -o -name metrics.json -o -name submission_sta.pt \) -print
  fi
done
for directory in \
  /scratch/leuven/394/vsc39410/runs/ar_videomae_full \
  /scratch/leuven/394/vsc39410/runs/videomae_b_f16_seed1; do
  if [[ -r "$directory" && -x "$directory" ]]; then
    find "$directory" -maxdepth 2 -type f \
      \( -name best.pt -o -name submission.pt -o -name config.json \
      -o -name best_metrics.json -o -name '*provenance*.json' \) -print
  else
    printf 'AR candidate missing or inaccessible: %s\n' "$directory"
  fi
done
```

Expected checkpoint result: `results/sta2/full-h100-62771307/checkpoints/best.pt: OK`.
The other lines print actual paths or no matches; no course test path is known yet.
No output from `find data ...` means the course CSV has not been located under that folder.
Obtain the course-supplied CSV through the course/team source; do not download official public labels.
Do not substitute the validation CSV, filter rows, or infer missing IDs from available videos.

## 2. Synchronize the source files through Git and set exact inputs

The reusable files live in `scripts/sta2_submission_preflight.py`,
`scripts/validate_codabench_bundle.py`, `slurm/sta2_test_a100.slurm`, and this guide.
Commit and push these files through the project's normal Git workflow. Then update
the VSC checkout on the same branch. A clean VSC checkout can use `git pull --ff-only`;
inspect `git status --short --branch` first and preserve existing local changes.
No manual upload of a generated results directory is needed for the source files.

The `PREP` directory below contains generated reports and logs and remains ignored.
Local `ar_reference/` snapshots are evidence from the stored AR branch, not a selected
model or runnable handoff; they do not need transferring to run the checks.

```bash
cd /data/leuven/397/vsc39706/R-D_Group2
export PREP="$PWD/results/sta2/codabench-first-62771307/preparation"
mkdir -p "$PREP"
test -s scripts/sta2_submission_preflight.py
test -s scripts/validate_codabench_bundle.py
test -s slurm/sta2_test_a100.slurm
module load cluster/wice/gpu_a100
source /data/leuven/397/vsc39706/miniconda3/etc/profile.d/conda.sh
conda activate /data/leuven/397/vsc39706/conda_envs/rd_sta2
python -c 'import sys, torch; print(sys.executable); print(torch.__version__)'

export STA_CKPT="$PWD/results/sta2/full-h100-62771307/checkpoints/best.pt"
export ANNOTATIONS="$PWD/data/raw/EPIC_100_train.csv"
export TRAIN_SPLIT="$PWD/data/processed/EPIC_100_train_participant_split.csv"
export VAL_SPLIT="$PWD/data/processed/EPIC_100_validation_participant_split.csv"
export BACKBONE=/data/leuven/397/vsc39706/sta2/checkpoints/vjepa2_1_vitG_384.pt
export BACKBONE_CONFIG="$PWD/configs/sta2/data_backbone.yaml"
export RAW_DIR="$VSC_SCRATCH/epic_sta/raw_fixed"
export METADATA="$VSC_SCRATCH/epic_sta/sta2_timelines_local/manifest.json"
# Set only after identifying the real supplied file in step 1:
# export COURSE_TEST=/absolute/path/from/step/1/EPIC_100_test.csv

python scripts/sta2_submission_preflight.py
```

The environment path above comes from `slurm/sta2_full_h100.slurm`. It differs from
the runbook example's `miniconda3/envs/rd_sta2`. Do not create an environment or
change model inputs merely to make a missing path pass. The original run manifest
and checkpoint configuration record the actual inputs.

The preflight first checks the checkpoint's SHA-256, epoch, metric, ten history rows,
full development split, backbone weights, action vocabulary, and backbone YAML.
It writes `preflight.json` even if the course file is still missing, preserving
completed independent checks. With no `COURSE_TEST`, expect `BLOCKED: FileNotFoundError: ...`;
set the actual file and rerun. Hash mismatches stop the workflow; inspect the recorded
training paths and recover the matching original files instead of overwriting them.

Expected passing lines include:

```text
PASS checkpoint: epoch 5, 10 completed epochs, Action MT5R 8.774957568731417%
PASS backbone, annotation, split hashes, action vocabulary, and backbone configuration
PASS course table: <actual count> unique events; SHA-256 <actual hash>
PASS timing audit, media identities, timeline hashes, and legal observation sampling: <counts>
```

No hidden-test count or hash is assumed. Test loading currently requires `narration_id`,
`video_id`, and `sta_observation_end_seconds`. When `start_timestamp` is present,
the loader verifies the cutoff equals start minus one second. If the supplied table
has start timestamps but no cutoff column, preserve the original and report its
header before preparing a derived table; no automatic schema conversion is included.

Every test video must have a timing-audit entry, including events with no usable
context. A missing audit entry is an input error, not permission to drop an event.
Use `scripts/build_sta2_metadata.py --help` if an audit needs extending; the builder
also needs the actual course video-info/FPS file. Preserve the training audit and
use a separate test audit directory if extension is needed. Record its identity.
The preflight checks timelines without decoding video. Actual decode failures are
counted later by inference and receive verified training-only priors.

### Build a separate course test audit when needed

VSC inspection confirmed all 138 course test videos have raw files and supplied FPS
rows in `data/raw/EPIC_100_video_info.csv`. None is present in the development audit.
Use `slurm/sta2_test_metadata_cpu.slurm` to create a separate test-only audit.
It uses the confirmed `/data/leuven/397/vsc39706/conda_envs/rd_sta2/bin/python`
directly and requests four CPU cores, 8 GB memory, no GPU, and a 12-hour walltime cap.
The cap is not a runtime prediction. This fully decodes the test videos to inspect
timestamps; no model is run and no visual features or predictions are produced.

The quote tool must receive a real script file. Run from the VSC repository root:

```bash
mkdir -p logs
set -o pipefail
bash -n slurm/sta2_test_metadata_cpu.slurm
sam-quote sbatch slurm/sta2_test_metadata_cpu.slurm \
  | tee logs/sta2-test-metadata-quote.txt
```

After the quote succeeds and fits the available allowance, submit the same script:

```bash
sbatch slurm/sta2_test_metadata_cpu.slurm
```

Record the actual job ID. The log is `logs/sta2-test-metadata-<JOB_ID>.out`.
Require a successful job exit and `Completed test-video coverage: 138`; inspect
the reported status counts and diagnose video errors. Quote credits are estimates;
record the actual settled charge separately. The new manifest is
`/scratch/leuven/397/vsc39706/epic_sta/sta2_test_timelines_62771307/manifest.json`.
The script locks this output directory and records source hashes for each job.
VSC job `62942164` completed this audit with all 138 test videos marked `ok`.
Rerun the CPU preflight using the completed test manifest:

```bash
bash scripts/preflight_sta2_codabench.sh
```

This runner sets every required path, uses the confirmed STA2 Python directly,
and prints progress. No Conda activation or previous terminal exports are needed.
It writes `results/sta2/codabench-first-62771307/preparation/preflight.json`.
Require `preflight_passed: true` before fresh GPU inference. The original development
audit remains intact. The runner's exports apply to its own process; the explicit
environment setup above remains available for subsequent manual commands.

## 3. Reuse first; quote GPU inference only if needed

`REUSE /.../predictions.pt` means the artifact and metrics match the selected
checkpoint, backbone, data/splits, course table, configuration, and scoring identity.
The artifact must have fresh-reload verification and correct training priors.
Go directly to CPU export in step 4. To check another exact artifact supplied by
the team, set `export PREDICTIONS=/actual/path/predictions.pt` and rerun preflight.
It also needs its sibling `metrics.json`; do not use a validation artifact as test output.

The E4 artifact does not independently record the actual inference `--metadata`
and `--raw-dir` arguments. Keep its original command/job log when reusing it.
For a new run, retain this preflight report and the submitted script alongside it.

Only when no matching artifact exists and `preflight_passed` is true:

```bash
set -o pipefail
mkdir -p logs
bash -n slurm/sta2_test_a100.slurm
sam-balance
export INFERENCE_WALLTIME=04:00:00
sam-quote sbatch --time="$INFERENCE_WALLTIME" slurm/sta2_test_a100.slurm \
  | tee "$PREP/inference-quote.txt"
```

This quotes **two A100s, 16 CPU cores, 128000 MB, four hours**. Four hours is a
provisional allocation limit, not a measured test-runtime prediction. The evaluator
makes **two complete inference passes**. Use the actual course event count and any
existing evaluator timing to adjust walltime, then re-quote the identical request.
Read the live quote against the remaining campaign allowance before submitting.
No credit figure has been measured here. A quote is an estimate, not an actual charge.

After the live quote fits the available allowance, submit that same request:

```bash
sbatch --time="$INFERENCE_WALLTIME" slurm/sta2_test_a100.slurm
# Set to the actual ID printed by sbatch:
export EVAL_JOB=REPLACE_WITH_ACTUAL_JOB_ID
squeue -M wice -j "$EVAL_JOB" -o '%i %j %P %T %M %l %R'
tail -n 40 "logs/sta2-test-62771307-${EVAL_JOB}.out"
sacct -M wice -j "$EVAL_JOB" -X \
  --format=JobID,State,Elapsed,Timelimit,AllocTRES,ExitCode
```

Expected submission response: `Submitted batch job <ID>`. Require `COMPLETED` and
`ExitCode 0:0`, the evaluator's `Split: test`, `Checkpoint epoch: 5`, actual event and
fallback counts, and both saved output paths. Do not retrain or run the old H100
training launcher. GPU resources release automatically when this batch script exits.
The accounting command records actual runtime/resources; obtain actual credit
charges from VSC accounting after settlement and record them separately from the quote.

New artifact location:
`results/sta2/competition-final-<EVAL_JOB>/selected/verified/{predictions.pt,metrics.json}`.

## 4. Export and strictly validate STA on CPU

Run in the same configured environment on the login/CPU side after inference ends.
For reuse, set `PREDICTIONS` to the verified path; otherwise use the completed job:

```bash
# For new inference:
export PREDICTIONS="$PWD/results/sta2/competition-final-${EVAL_JOB}/selected/verified/predictions.pt"
# For reuse, instead export PREDICTIONS=/actual/verified/predictions.pt
export BUNDLE="$PWD/results/sta2/codabench-first-62771307"
export STA_FILE="$BUNDLE/submission_sta.pt"
test ! -e "$STA_FILE"
python scripts/export_sta2.py --checkpoint "$STA_CKPT" --predictions "$PREDICTIONS" \
  --backbone-checkpoint "$BACKBONE" --annotations "$ANNOTATIONS" \
  --train-split "$TRAIN_SPLIT" --validation-split "$VAL_SPLIT" \
  --events "$COURSE_TEST" --output "$STA_FILE"
python scripts/validate_sta_submission.py "$STA_FILE" --events "$COURSE_TEST"
sha256sum "$COURSE_TEST" "$STA_CKPT" "$PREDICTIONS" \
  "$STA_FILE" "$STA_FILE.provenance.json" | tee "$PREP/sta-sha256.txt"
```

Expected: `Exported and validated <N> hidden-test events`, then
`Valid STA submission: <N> events`. Both counts must equal the complete course table.
The strict validator requires exactly three keys, original CSV row order, CPU FP32
probabilities with 97/300 entries, finite values and sums of one, and no missing,
duplicated, or extra IDs. Export rechecks all input identities and fallback priors.
It writes `submission_sta.pt.provenance.json` with source and submission hashes.
If the output already exists, validate its provenance and reuse it; do not overwrite
it blindly. Preserve the exporter metadata and the preflight/log files outside the ZIP.

## 5. AR: what the repository actually establishes

The stored `origin/AR_dev` reference is commit
`398bc60c6f86c4b74b4fd04c8fed90b76667d274` (18 November 2026). Its tracked baseline
config records training revision `ca3871951b7415466b46613e4948a25f2739e1d9`, VideoMAE
`MCG-NJU/videomae-base-finetuned-kinetics`, 16 frames, crop 224, seed 1, 97/300 heads,
and output `/scratch/leuven/394/vsc39410/runs/ar_videomae_full`.
Its `best_metrics.json` reports epoch 7, 7,450 validation events, AR Action Top-1
28.44295302013423%. This is a historical candidate, not confirmation of team selection,
checkpoint availability, hidden-test inference, or checkpoint hash.

The actual AR workflow uses trimmed `[start_timestamp, stop_timestamp]` segments,
deterministic centre-of-bin sampling for evaluation, and VideoMAE verb/noun heads.
`--eval_only --export` writes `preds_val.csv` containing top-five class IDs, not all
97/300 score vectors. `--export_json` retains full marginals in a different JSON schema.
The supplied eval wrapper uses `--skip_missing`, which drops events; it is unsuitable
for final complete-coverage inference. Decode failures currently substitute black
frames and log a warning. The decoder selects the next frame at/after each sample;
a final AR adapter needs to enforce trimmed-clip boundaries, especially near action end.
Do not simply run that wrapper as the Codabench solution. Do not reconstruct full scores
from the top-five CSV. No AR code was changed for an unconfirmed checkpoint.

For either readable AR candidate directory printed in step 1, inspect only these files:

```bash
# Set this to a directory actually found in step 1:
export AR_RUN=/scratch/leuven/394/vsc39410/runs/ar_videomae_full
ls -lh "$AR_RUN/best.pt" "$AR_RUN/config.json" "$AR_RUN/best_metrics.json"
sha256sum "$AR_RUN/best.pt" "$AR_RUN/config.json" "$AR_RUN/best_metrics.json"
cat "$AR_RUN/config.json" "$AR_RUN/best_metrics.json"
python - "$AR_RUN/best.pt" <<'PY'
import json, sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
print(json.dumps({key: ck.get(key) for key in ('epoch', 'best', 'args')}, indent=2))
PY
```

Require the AR owner/team's selected checkpoint or existing `submission.pt` path,
its SHA-256, source training annotations/internal-split identities, selected recipe,
and original hidden-test inference command/log. The AR checkpoint stores paths and
arguments but no training annotation/split SHA-256; hashing today's files alone
cannot prove their historical training contents. Recover archived input identities
or the owner's run evidence. Verify the test-table hash and trimmed-video input
identity as well as the model and output hashes before calling a reused AR file verified.
Do not contact the owner automatically or browse another account beyond these named inputs.

If there is only a selected checkpoint, AR still needs a proper unlabeled-test adapter
and basic `.pt` export of actual full model scores, preserving every event. Inspect
the supplied course timing/clip schema first. Quote that separate GPU inference job
before allocation. No AR inference time or cost can be quoted accurately until the
selected workflow, input coverage, and resources are established.

If an existing verified basic `submission.pt` is supplied, keep it in its original
location and set `AR_FILE` to that absolute path. Its coverage and vector schema
will be checked with the same complete CSV in step 6. Keep the provenance evidence
beside the final bundle. Never use STA predictions as AR or invent an AR output.

## 6. Combined validation, ZIP and hashes (CPU only)

Proceed only with the verified AR file and the passing STA export. This runs the
combined validator copied from runbook section 8. The preparation uses basic outputs;
the extra check below rejects optional fields in either file for this submission.

```bash
set -o pipefail
export AR_FILE=/absolute/path/to/verified/AR/submission.pt
export ZIP_FILE="$BUNDLE/group2-first-62771307.zip"
test -s "$COURSE_TEST" && test -s "$AR_FILE" && test -s "$STA_FILE"
test ! -e "$ZIP_FILE"
python scripts/validate_sta_submission.py "$STA_FILE" --events "$COURSE_TEST"
python - "$AR_FILE" "$STA_FILE" <<'PY'
import sys, torch
for path in sys.argv[1:]:
    rows = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(rows, list) or not all(isinstance(row, dict) and set(row) ==
        {'narration_id', 'verb_output', 'noun_output'} for row in rows):
        raise ValueError(f'{path}: this submission uses the basic three-field format')
PY
python scripts/validate_codabench_bundle.py "$COURSE_TEST" "$AR_FILE" "$STA_FILE" "$ZIP_FILE" \
  | tee "$PREP/combined-validation.txt"
sha256sum "$COURSE_TEST" "$AR_FILE" "$STA_FILE" "$ZIP_FILE" \
  | tee "$PREP/final-sha256.txt"
python - "$ZIP_FILE" <<'PY'
import sys
from zipfile import ZipFile
with ZipFile(sys.argv[1]) as archive:
    print(archive.namelist())
    print('Integrity:', archive.testzip())
PY
```

The blocks set `pipefail` so a failing validator cannot be hidden by `tee`.
Inspect all validation output before proceeding to upload.

Expected combined output:

```text
<AR path>: <N> events, 0 with explicit action scores
<STA path>: <N> events, 0 with explicit action scores
Validated <ZIP path>
['submission.pt', 'submission_sta.pt']
Integrity: None
```

Both files must cover every CSV ID exactly once. The combined course validator
allows AR finite floating scores/logits and any row order; the strict STA validator
adds probability and order checks. Do not use the strict STA probability validator
to reject an otherwise valid AR logit file. No local hidden-test metric is computed.

## 7. Current evidence and remaining upload steps

Local checks passed: 67,217 annotations; 59,767 training and 7,450 validation events;
disjoint participants and event IDs; complete ID partition; matching source metadata;
all 488 unseen-pair validation events retained across 251 pairs. Exact local hashes
and AR reference evidence are in local
`results/sta2/codabench-first-62771307/preparation/local_inspection.json`. Local Python lacks PyTorch;
preparation Python scripts passed syntax checks, not VSC runtime/inference validation.
The Bash launcher also needs the VSC environment and live quote before use.

The VSC checkpoint hash in this handoff is the value supplied by Emiel; it has not
been independently rehashed from this PC. VSC backbone/split provenance, the actual
course test path/hash/count and usable timing audit, verified STA test artifacts,
and selected AR artifact/provenance remain to be checked. Consequently no genuine
`submission.pt`, `submission_sta.pt`, or combined upload ZIP exists in this preparation.

Once both validators pass, the planned upload artifact is:
`/data/leuven/397/vsc39706/R-D_Group2/results/sta2/codabench-first-62771307/group2-first-62771307.zip`.
Keep `final-sha256.txt`, both task provenance records, the course CSV hash, inference
job log, quote, actual runtime and settled credit charge. Stop here with files ready.

For the later manual upload, follow the course runbook: use the course Codabench
competition, student email and the assigned team organization (confirm whether its
course name is `Group 2`); select both AR and STA and upload only the combined ZIP.
The runbook records limits of five submissions/day and 50 total. Confirm the live
competition's limits before uploading, then record its separate AR and STA scores
and the exact ZIP hash. This task performs no upload.
