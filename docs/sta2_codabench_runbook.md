# EPIC-KITCHENS-100 competition runbook

**Run the two-block STA2 head on A100, compare it with the existing four-block model, and select the submission recipe on held-out validation.** Prepare the combined Codabench ZIP after model selection. The team targets **3 December 2026** for model selection and **9 December** for submission, ahead of the **11 December** course deadline.

The course's advanced `action_output` option makes a JFAA-style action classifier a valid candidate for the STA score. The existing STA2 export supports the basic format. Evaluate direct and combined action scoring on the course split before choosing the advanced format.

## 1. Owners, dates, and budget

| Owner | Work | Done when |
| --- | --- | --- |
| Niels | Check for the compact job, then monitor or launch it. | We have a stable `best.pt`, its validation history, and measured credits per epoch. |
| Emiel | Recover the original checkpoint and run the like-for-like STA comparison. | The original, compact, and blend are scored on the same 7,450 events. |
| AR owner | Record the selected AR recipe and locate its checkpoint and hidden-test inference command. | The selected AR model can produce a valid `submission.pt` from trimmed clips when the team freezes its submission. |
| Designated submitter | Check organization membership and submit the selected AR/STA bundle. | Both tasks complete scoring and appear under the team's organization. |

Treat **3.0 million credits** as the cap for this STA campaign. Record actual charges plus the quoted cost of pending work before authorizing another allocation.

| STA work | Credit allowance |
| --- | ---: |
| Development training, including compact | 1,100,000 |
| One all-label retrain, if selected | 1,400,000 |
| Validation and test inference | 200,000 |
| Retries and queue contingency | 300,000 |
| Total | 3,000,000 |

Two A100s with 16 CPU cores are approximately **19.4k credits/hour**, or **778k** for 40 hours. Two H100s with the same core count are approximately **71.7k credits/hour**. The stock 72-hour H100 request is approximately **5.16M**, above the campaign cap. Quotes assume the entire requested allocation runs. Use the live quote and observed job cost for decisions. Prefer A100. Consider a shorter H100 allocation only if its earlier expected finish protects the submission date and its quote fits the cap. The previous H100 run took around five hours per epoch. Measure the compact run before extrapolating. [VSC credit rates and quoting](https://docs.vscentrum.be/leuven/credits.html) · [VSC accounting](https://docs.vscentrum.be/leuven/slurm_accounting.html)

## 2. Course inputs and metrics

| Task | Allowed video input | Primary metric | Additional metrics |
| --- | --- | --- | --- |
| Action Recognition (AR) | The event's trimmed action clip | Action Top-1 accuracy | Top-5, tail-class, and unseen-participant results |
| Short-Term Action Anticipation (STA) | Observed video up to one second before the event starts | Action class-mean Top-5 recall (MT5R) | Verb, noun, tail-class, and unseen-participant MT5R |

- Both tasks are video-only. Develop and tune on the supplied `EPIC_100_train.csv` and the documented internal split. Do not retrieve or use public official EK100 labels for training, tuning, or local evaluation. The course's `EPIC_100_test.csv` is the unlabeled event table. Use it for final inference and coverage checks after selection.
- The STA boundary applies during training, validation, test inference, and feature extraction. For an action starting at 20 s, the cutoff is 19 s. This run uses 32 samples at 8 fps over approximately four seconds of preceding video. The timestamp sampler requires every selected frame to precede the cutoff. The predictor generates latent features from that same legal input. It receives no video from the one-second gap or target action.
- Retain every hidden event in both outputs. For STA events starting before one second, or any event with no legal observation, use the training prior without decoding future frames. Preserve that rule during calibration, ensembling, and optional direct action scoring. AR still makes a prediction for the trimmed event.
- MT5R computes recall@5 separately for each class present in the evaluated subset, then averages those recalls equally. Report AR and STA separately. The course defines no combined metric.

The STA development split has **59,767 training events from 26 participants** and **7,450 validation events** from **P08, P20, P21, P28, P29, and P37**. The full provided labels contain **67,217 events from 32 participants**. Preserve the entire validation set, including the **488 events across 251 action pairs** absent from the development training split. Only use the full 67,217 events for the final retrain after selection is frozen.

## 3. Check and launch compact training

Niels’s focused A100 instructions are in [compact training](sta2_compact_training.md).
Start with its one-complete-epoch timing job (`bash scripts/prepare_sta2_compact_a100.sh --timing`),
record timing and actual/estimated credits, then resume the saved checkpoint if selected.
This requests 12 hours on the normal A100 partition, not the 30-minute debug queue.
The A100 wrapper defaults to 40 hours. `bash scripts/prepare_sta2_compact_a100.sh`
checks local-account prerequisites and prints the live quote without submitting.
Source your existing private dashboard environment first if uploads are wanted;
direct `sbatch` uses the trainer’s integrated reporting. Check both accounts and
the remaining credit allowance before submission.

Run the shell commands on a **VSC Bash** login node. The compact launcher fixes the repository, dataset, checkpoint, and environment locations shown below. Check them from the account that will run the job. On that account, update the checkout to the agreed `STA2_dev` revision if needed.

```bash
cd "$VSC_DATA/R-D_Group2"
git status --short --branch
git rev-parse HEAD
sam-balance
squeue -M wice -u "$USER" -o '%i %j %P %T %M %l %R'
sacct -M wice -u "$USER" -S 2026-11-19 -X --format=JobID,JobName,Partition,State,Elapsed,AllocTRES
find results/sta2 -path '*/compact-*/checkpoints/best.pt' -print
```

Check with both accounts before a fresh launch. If compact is pending or running, record its job ID, partition, walltime, and quote, and follow that job. If it completed, use its history and checkpoint. The hosted dashboard may lag VSC. Use `squeue`, `sacct`, and the native history to determine state.

```bash
set -e
cd "$VSC_DATA/R-D_Group2"
test -s "$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt"
test -s "${STA2_COMPACT_METADATA_DIR:-$VSC_SCRATCH/epic_sta/sta2_timelines_local}/manifest.json"
test -d "$VSC_SCRATCH/epic_sta/raw_fixed"
test -s "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
test -x "$VSC_DATA/miniconda3/envs/rd_sta2/bin/python"
test -s data/raw/EPIC_100_train.csv
test -s data/processed/EPIC_100_train_participant_split.csv
test -s data/processed/EPIC_100_validation_participant_split.csv
mkdir -p logs
bash -n slurm/sta2_compact_a100.slurm
bash -n slurm/sta2_train.slurm
sam-quote sbatch --time=40:00:00 slurm/sta2_compact_a100.slurm
```

When a fresh run is needed and the quote fits the remaining development allowance:

```bash
unset STA2_COMPACT_RESUME_RUN
sbatch --time=40:00:00 slurm/sta2_compact_a100.slurm
```

Set `COMPACT_JOB` to the job ID printed by `sbatch`. Keep the 40-hour override in the submission command. The YAML retains the original schedule horizon and patience, so the shorter Slurm request does not change the learning-rate schedule.

```bash
COMPACT_JOB=REPLACE_WITH_JOB_ID
COMPACT_RUN="results/sta2/compact-a100-$COMPACT_JOB"
squeue -M wice -j "$COMPACT_JOB" --start
tail -f "logs/sta2-compact-a100-$COMPACT_JOB.out"
# After the job has started, run these in a separate terminal:
sacct -M wice -j "$COMPACT_JOB" -X --format=JobID,State,Elapsed,Timelimit,AllocTRES,ExitCode
ls -lh "$COMPACT_RUN"/run-manifest-*.json "$COMPACT_RUN"/history.json "$COMPACT_RUN"/checkpoints/*.pt
```

The manifest should show depth **2**, `observed_only: false`, seed **42**, `learning_rate: 0.0001`, `weight_decay: 0.01`, effective batch **16**, **BF16**, and the expected split counts and participants. Compare the annotation and split hashes and the frozen checkpoint SHA-256 against the reference manifest. The smaller head has approximately **70M** trainable parameters, down from **136M**. Validate that actual training throughput improved and check the usable/fallback counts as well as the score. A read or decode error should be diagnosed against the video audit and timestamps before its run is used for selection.

After the first complete epoch, estimate the hours per epoch from `train_seconds + validation_seconds` in `history.json`, plus checkpoint overhead. Record that number alongside the credits/hour estimate. Budget for at least three completed epochs if feasible. Skim `validation_loss` for overfitting and select checkpoints using **Action MT5R**. Hold off spending on extra epochs if the score is flat or falling once several epochs are available. Check the current result before resuming. The trainer saves only at epoch boundaries and uses a walltime heuristic to stop early. The last incomplete epoch can be lost if the job is killed.

Resume only a walltime-limited run with a saved history and `last.pt` and with the same wrapper, world size, inputs, and settings. For example, quote and submit a 12-hour A100 continuation only if there is room for an additional epoch and its quote fits the remaining allowance:

```bash
export STA2_COMPACT_RESUME_RUN="$COMPACT_RUN"
test -s "$STA2_COMPACT_RESUME_RUN/history.json"
test -s "$STA2_COMPACT_RESUME_RUN/checkpoints/last.pt"
sam-quote sbatch --time=12:00:00 slurm/sta2_compact_a100.slurm
# After checking the quote and the new estimated finish:
sbatch --time=12:00:00 slurm/sta2_compact_a100.slurm
unset STA2_COMPACT_RESUME_RUN
```

## 4. Compare the reference, compact, and blend

Recover the best completed reference checkpoint from VSC run `62771307`. Its recorded result was **8.77% Action MT5R at epoch 5**. Verify that with `results/sta2/full-h100-62771307/history.json` and the checkpoint metadata. Use a stable `best.pt` from each run. If a job can still replace its best checkpoint, copy a consistent checkpoint after an epoch save and name the copy by job and epoch.

Run inference on a GPU allocation. One example on wICE uses two A100s, one task, and two PyTorch workers launched by that task. Set the allocation duration from your deadline and budget. The evaluator performs **two complete inference passes** per checkpoint to verify reload consistency. If the output of a verified run already exists, reuse its `predictions.pt` and `metrics.json` for the same checkpoint and split.

```bash
# Request this only once a GPU validation session is needed.
srun --account=lp_edu_rdlab --clusters=wice --partition=gpu_a100 \
    --nodes=1 --ntasks=1 --gpus-per-node=2 --cpus-per-task=16 \
    --mem=128000M --time=04:00:00 --pty bash -l
```

In the allocated shell, set `COMPACT_CKPT` to the real path. Use the H100 cluster module instead only if the allocation uses H100.

```bash
set -e
cd "$VSC_DATA/R-D_Group2"
module load cluster/wice/gpu_a100
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda activate "$VSC_DATA/miniconda3/envs/rd_sta2"

export ANNOTATIONS=data/raw/EPIC_100_train.csv
export TRAIN_SPLIT=data/processed/EPIC_100_train_participant_split.csv
export VAL_SPLIT=data/processed/EPIC_100_validation_participant_split.csv
export RAW_DIR="$VSC_SCRATCH/epic_sta/raw_fixed"
export METADATA="$VSC_SCRATCH/epic_sta/sta2_timelines_local/manifest.json"
export BACKBONE="$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt"
export BACKBONE_CONFIG=configs/sta2/data_backbone.yaml
export REFERENCE_CKPT=results/sta2/full-h100-62771307/checkpoints/best.pt
export COMPACT_CKPT=results/sta2/compact-a100-REPLACE_WITH_JOB_ID/checkpoints/best.pt
export COMPARE_DIR="results/sta2/competition-validation-${SLURM_JOB_ID}"

test -s "$REFERENCE_CKPT"
test -s "$COMPACT_CKPT"
evaluate_validation() {
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
        scripts/evaluate_sta2.py --split validation --checkpoint "$1" \
        --backbone-checkpoint "$BACKBONE" --backbone-config "$BACKBONE_CONFIG" \
        --raw-dir "$RAW_DIR" --metadata "$METADATA" --annotations "$ANNOTATIONS" \
        --train-split "$TRAIN_SPLIT" --validation-split "$VAL_SPLIT" --output-dir "$2"
}
evaluate_validation "$REFERENCE_CKPT" "$COMPARE_DIR/reference"
evaluate_validation "$COMPACT_CKPT" "$COMPARE_DIR/compact"
```

Release the GPU allocation once inference is done. The comparison below is CPU-only. Run it from the repo root with the same Python environment, setting `COMPARE_DIR` to the directory printed or used above. It verifies the shared scoring setup, checks equivalent fallback predictions, and uses the repository's MT5R implementation.

```bash
COMPARE_DIR=results/sta2/competition-validation-REPLACE_WITH_EVAL_JOB_ID
python - "$COMPARE_DIR" <<'PY'
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.sta2.artifacts import comparison_identity, load_prediction_artifact
from src.sta2.data import load_events
from src.sta2.evaluation import evaluation_action_pairs, score_prediction_records

root = Path(sys.argv[1])
annotations_path = Path("data/raw/EPIC_100_train.csv")
train_split = Path("data/processed/EPIC_100_train_participant_split.csv")
val_split = Path("data/processed/EPIC_100_validation_participant_split.csv")
events = load_events(
    annotations_path, split="validation", train_split=train_split, validation_split=val_split
)
pairs = evaluation_action_pairs(pd.read_csv(annotations_path), num_verbs=97, num_nouns=300)
identity = comparison_identity(annotations_path, train_split, val_split, pairs)
ref_meta, reference = load_prediction_artifact(root / "reference/predictions.pt", events, identity)
cmp_meta, compact = load_prediction_artifact(root / "compact/predictions.pt", events, identity)
if ref_meta["backbone_sha256"] != cmp_meta["backbone_sha256"]:
    raise ValueError("The reference and compact runs used different backbone weights")
if len(reference) != len(compact) or len(reference) != 7450:
    raise ValueError("The comparison requires both complete validation prediction sets")
train = pd.read_csv(train_split)
train_pairs = tuple(sorted(set(zip(train.verb_class.astype(int), train.noun_class.astype(int)))))

blend = []
for left, right in zip(reference, compact):
    keys = ("narration_id", "verb_label", "noun_label", "status", "used_frequency_prior")
    if any(left[key] != right[key] for key in keys):
        raise ValueError(f"Event, label, or context mismatch for {left['narration_id']}")
    row = copy.copy(left)
    for field in ("verb_probabilities", "noun_probabilities"):
        if left["used_frequency_prior"]:
            if not np.array_equal(left[field], right[field]):
                raise ValueError(f"Prior mismatch for {left['narration_id']}")
        else:
            row[field] = (left[field] + right[field]) * np.float32(0.5)
    blend.append(row)

report = {}
for name, records in (("reference", reference), ("compact", compact), ("blend", blend)):
    report[name], _ = score_prediction_records(
        records, action_pairs=pairs, auxiliary_action_vocabulary=train_pairs
    )
    score = 100 * report[name]["action_mt5r"]
    print(f"{name}: Action MT5R {score:.3f}%")

destination = root / "comparison.json"
with destination.open("x") as output:
    json.dump(report, output, indent=2)
    output.write("\n")
PY
```

Record the best epoch, Action/Verb/Noun MT5R, runtime, and credits for each trained model. Also inspect performance across held-out participants and seen/unseen action pairs, especially before switching to a direct action scorer. Count fallbacks separately. A dashboard percentage rounded to two decimals is for display. Use the unrounded metrics when applying the selection rule.

- **Single model:** take the one with the higher validation Action MT5R. Use the reference for an exact tie.
- **Blend:** select the fixed 50/50 blend if it beats the stronger single by at least **0.5 percentage points** (`0.005` on a 0–1 scale). Save the rule and source checkpoints. A blend can use two development checkpoints directly.
- **One more training experiment:** after the main candidate is ready and only if time and credits allow, test JFAA-style **sigmoid focal loss** on the same compact head against its cross-entropy run. Hold the split, observation rule, seed, and training settings fixed. Preserve the 97/300 marginal outputs. Treat the focal run as a separate experiment with its own checkpoint and recipe.

## 5. Advanced action scoring experiment

The course guide's example uses **`action_output`** as the record key. Use that exact key. The optional field is a dictionary of **exactly 100** distinct pairs with finite float scores, for example `"0,1": 0.082`. Each key is a comma-separated `<verb_class>,<noun_class>` using verb indices **0–96** and noun indices **0–299**. Keep both verb and noun vectors. When the optional field is supplied, the scorer uses its pair ranking for the action metric. For a consistent recipe, emit the field for every row in a file or omit it for the entire file.

This changes the earlier assumption that our action classifier could only be auxiliary. The trained STA2 checkpoints already have an action classifier. The current evaluator computes its logits but drops them from the prediction artifact, and the current exporter and strict STA validator expect exactly three keys. Passing an existing artifact to those scripts produces the basic output. The combined ZIP check in section 8 accepts the advanced field once an advanced exporter has produced it.

If time allows, test the existing checkpoints' action heads **before spending GPU credits on a focal-loss retrain**:

1. Add an opt-in inference artifact that retains CPU FP32 action logits for usable events and records the exact ordered `action_vocabulary` from the checkpoint. Bind it to the checkpoint hash, test/validation event table, and split hashes. Include the logits in the independent reload comparison. Keep the default three-field export usable for existing checkpoints.
2. Evaluate three variants over the same 7,450 rows. Use the current `P(verb) × P(noun)` as the baseline. Test the dedicated head alone as a diagnostic. Test a blend of pair scores by normalizing the factored scores over the allowed action-pair list and scattering the softmax of head logits into that same list by pair ID. Sweep direct-head weights **0.25** and **0.50** on validation. The factored component gives pairs missing from the head's training vocabulary a way to rank.
3. Match export behavior during validation. Rank pairs deterministically by descending score then numeric pair ID, write the top 100, and compute action hits using the first five. Keep all examples and classes. Use the training-prior verb/noun product over valid pairs for no-context action outputs. Check overall Action MT5R, seen/unseen pair MT5R, and scores by participant. Promote the advanced recipe only if it beats the best basic recipe by **at least 0.5 percentage points** and the chosen weight still improves pooled performance when selected on five participants and applied to the sixth in turn.
4. Verify any advanced exporter on representative output before test inference. Cover correct label remapping, a pair absent from the training head, a fallback event, exactly 100 finite unique scores, reload reproducibility, and the same top-five ranking before and after serialization. Save the chosen weight and validation evidence. If advanced scoring is selected, use that same logic for the final candidate and run the ZIP check in section 8.

The public JFAA score of **27.95%** motivates this experiment. Its released inference config anchors to **action end minus one second**, while the course permits **action start minus one second**. Keep our sampler and course data when borrowing any JFAA loss or scoring ideas. [JFAA report](https://arxiv.org/html/2605.20904v1) · [JFAA inference config](https://raw.githubusercontent.com/CorrineQiu/JFAA/main/configs/jfaa_vjepa2_vitG_infer.yaml)

## 6. Freeze the choice and optionally retrain on all supplied labels

By **3 December**, record the selected scoring method, checkpoint(s), comparison metrics, and expected finish time. Keep the validation partition unchanged until this decision.

- If a single recipe wins and a full run fits, retrain that configuration from a fresh head on all **67,217 supplied labels**, for the epoch count of its selected development `best.pt`. The development checkpoint supplies the settings and epoch count. The final run has **zero validation events** and preserves the learning-rate schedule horizon.
- If the development blend wins, use the two selected development checkpoints. If an all-label retrain cannot finish in time, use the selected development model for the final inference. A partially completed all-label checkpoint cannot be exported by the current evaluator.
- Estimate final training time from the selected configuration's measured training time per epoch × `67217 / 59767` × selected epochs. Add head checkpoint overhead and queue time, then reserve a GPU window for test inference. Use a walltime that covers that forecast and fits the remaining quote allowance.

The checked-in final launcher targets H100. This snippet creates its A100 counterpart on VSC, preserving its final-data and resume logic. It is needed only if the team selects an all-label A100 run.

```bash
cd "$VSC_DATA/R-D_Group2"
python - <<'PY'
from pathlib import Path

source = Path("slurm/sta2_final_h100.slurm")
destination = Path("slurm/sta2_final_a100.slurm")
original = source.read_text()
for expected in ("final-h100", "gpu_h100"):
    if expected not in original:
        raise ValueError(f"{source} does not contain {expected}")
converted = original.replace("final-h100", "final-a100").replace("gpu_h100", "gpu_a100")
if destination.exists():
    if destination.read_text() != converted:
        raise FileExistsError(f"{destination} exists with different contents")
else:
    with destination.open("x") as output:
        output.write(converted)
PY
bash -n slurm/sta2_final_a100.slurm
mkdir -p logs
unset STA2_FINAL_RESUME_RUN
export STA2_SELECTION_CHECKPOINT=REPLACE_WITH_SELECTED_BEST_PT
test -s "$STA2_SELECTION_CHECKPOINT"
FINAL_WALLTIME=REPLACE_WITH_QUOTED_HH_MM_SS
sam-quote sbatch --time="$FINAL_WALLTIME" slurm/sta2_final_a100.slurm
# After checking the quote, queue, and ETA:
sbatch --time="$FINAL_WALLTIME" slurm/sta2_final_a100.slurm
```

Use the VSC checkout that can read the selected checkpoint and the exact timing metadata it was trained with. The final manifest must show `train_events: 67217`, `validation_events: 0`, and the selected epoch count. Verify the final run completed that count in `history.json` before using `checkpoints/last.pt`. If the allocation ran out of time, resume only if the remaining runtime fits. Use the same final wrapper and set `STA2_FINAL_RESUME_RUN` to the final run directory for that continuation. See [final-training guide](sta2_final_training.md) for the resume and provenance checks.

## 7. Generate the selected test predictions

Start after model selection. Use the **course** `EPIC_100_test.csv`. Set the actual test CSV path, the selected checkpoint, and its corresponding split files. At this point the AR owner runs the selected AR inference on **trimmed clips** and supplies `submission.pt` for the same event IDs. Record the source recipe and checkpoint for both tasks.

For the basic STA path, use the same two-GPU shell setup as section 4. Reapply the environment variables from section 4 in this new allocation. In the snippet below, use either the all-label variables **or** the two development-split paths from section 4.

```bash
export COURSE_TEST=REPLACE_WITH_COURSE_EPIC_100_TEST_CSV
test -s "$COURSE_TEST"
export STA_OUTPUT="results/sta2/competition-final-${SLURM_JOB_ID}"
mkdir -p "$STA_OUTPUT"

# Example for a completed all-label run. For a development winner, instead
# use its best.pt and the development TRAIN_SPLIT, VAL_SPLIT, BACKBONE_CONFIG.
FINAL_RUN=results/sta2/final-a100-REPLACE_WITH_FINAL_JOB_ID
STA_CKPT="$FINAL_RUN/checkpoints/last.pt"
TRAIN_SPLIT="$FINAL_RUN/inputs/train.csv"
VAL_SPLIT="$FINAL_RUN/inputs/validation.csv"
BACKBONE_CONFIG="$FINAL_RUN/inputs/backbone.yaml"

evaluate_test() {
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
        scripts/evaluate_sta2.py --split test --events "$COURSE_TEST" --checkpoint "$1" \
        --backbone-checkpoint "$BACKBONE" --backbone-config "$BACKBONE_CONFIG" \
        --raw-dir "$RAW_DIR" --metadata "$METADATA" --annotations "$ANNOTATIONS" \
        --train-split "$TRAIN_SPLIT" --validation-split "$VAL_SPLIT" --output-dir "$2/verified"
}
export_test() {
    python scripts/export_sta2.py --checkpoint "$1" --predictions "$2/verified/predictions.pt" \
        --backbone-checkpoint "$BACKBONE" --annotations "$ANNOTATIONS" \
        --train-split "$TRAIN_SPLIT" --validation-split "$VAL_SPLIT" \
        --events "$COURSE_TEST" --output "$2/submission_sta.pt"
}
evaluate_test "$STA_CKPT" "$STA_OUTPUT/selected"
export_test "$STA_CKPT" "$STA_OUTPUT/selected"
```

For a basic blend, use the **development** split files for both source checkpoints and execute `evaluate_test` and `export_test` once per checkpoint into separate `reference` and `compact` directories instead of the single-model commands above. Record their separate hashes and drop the GPU allocation when inference finishes. Then blend the two verified basic exports on CPU. The source exports use identical event order and the same priors for unavailable context. Their validators already checked each source prior against the training split.

```bash
# Set STA_OUTPUT to the directory from the completed GPU job if needed.
python - "$STA_OUTPUT" <<'PY'
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
left = torch.load(root / "reference/submission_sta.pt", map_location="cpu", weights_only=True)
right = torch.load(root / "compact/submission_sta.pt", map_location="cpu", weights_only=True)
if len(left) != len(right):
    raise ValueError("The source submissions have different event counts")
blended = []
for a, b in zip(left, right):
    if a["narration_id"] != b["narration_id"]:
        raise ValueError("The verified source submission orders differ")
    record = {"narration_id": a["narration_id"]}
    for field in ("verb_output", "noun_output"):
        record[field] = (a[field] + b[field]) * 0.5
    blended.append(record)
with (root / "submission_sta.pt").open("xb") as output:
    torch.save(blended, output)
PY
python scripts/validate_sta_submission.py "$STA_OUTPUT/submission_sta.pt" --events "$COURSE_TEST"
```

For an advanced STA selection, run the verified advanced exporter from section 5 using the frozen recipe, the selected checkpoint(s), and the same course test CSV. Include the chosen `action_output` on every record. Run the combined validation below. The basic `validate_sta_submission.py` command deliberately expects three keys and applies stricter CPU FP32 probability and row-order checks than the course.

## 8. Validate and submit the combined ZIP

Each prediction file is a `torch.save` list with one dictionary per event:

```python
{
    "narration_id": "P01_101_1",
    "verb_output": verb_scores,  # 97 finite floats
    "noun_output": noun_scores,  # 300 finite floats
}
```

The optional field is `"action_output": {"<verb_class>,<noun_class>": float_score, ...}` with exactly 100 distinct entries. Finite logits and probabilities are accepted. Save dense CPU floating-point tensors for the vectors for reproducible loading. IDs are matched by value, so the course accepts any row order. Exact coverage is mandatory in **both** files. The required ZIP root entries are `submission.pt` (AR) and `submission_sta.pt` (STA).

Set the four paths below before running the CPU validation. Set `STA_FILE` to the actual selected export. The script validates both files and writes the final archive only after both pass. Use a fresh ZIP path in an existing directory.

```bash
COURSE_TEST=REPLACE_WITH_COURSE_EPIC_100_TEST_CSV
AR_FILE=REPLACE_WITH_AR_SUBMISSION_PT
STA_FILE=REPLACE_WITH_STA_SUBMISSION_PT
ZIP_FILE=REPLACE_WITH_FINAL_ZIP_PATH
python - "$COURSE_TEST" "$AR_FILE" "$STA_FILE" "$ZIP_FILE" <<'PY'
import csv
import math
import re
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch

events_path, ar_path, sta_path, zip_path = map(Path, sys.argv[1:])
with events_path.open(newline="", encoding="utf-8-sig") as handle:
    event_ids = [row["narration_id"] for row in csv.DictReader(handle)]
expected_ids = set(event_ids)
if not event_ids or "" in expected_ids or len(expected_ids) != len(event_ids):
    raise ValueError("The course CSV must contain nonempty, unique narration IDs")

required = {"narration_id", "verb_output", "noun_output"}
optional = {"action_output"}
pair_pattern = re.compile(r"([0-9]+),([0-9]+)")

def validate(path):
    rows = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError(f"{path}: expected a list of {len(expected_ids)} predictions")
    seen = set()
    advanced_rows = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) not in (required, required | optional):
            raise ValueError(f"{path}: invalid prediction keys")
        identity = row["narration_id"]
        if not isinstance(identity, str) or identity not in expected_ids or identity in seen:
            raise ValueError(f"{path}: duplicate, unexpected, or invalid ID {identity!r}")
        seen.add(identity)
        for field, size in (("verb_output", 97), ("noun_output", 300)):
            scores = torch.as_tensor(row[field])
            if scores.shape != (size,) or not scores.is_floating_point():
                raise ValueError(f"{path}: {identity} needs {size} float scores for {field}")
            if not torch.isfinite(scores).all().item():
                raise ValueError(f"{path}: {identity} has a nonfinite {field} score")
        if "action_output" in row:
            advanced_rows += 1
            scores = row["action_output"]
            if not isinstance(scores, dict) or len(scores) != 100:
                raise ValueError(f"{path}: {identity} needs exactly 100 action scores")
            numeric_pairs = set()
            for key, score in scores.items():
                match = pair_pattern.fullmatch(key) if isinstance(key, str) else None
                if match is None:
                    raise ValueError(f"{path}: {identity} has invalid action key {key!r}")
                verb, noun = map(int, match.groups())
                if not (0 <= verb < 97 and 0 <= noun < 300) or (verb, noun) in numeric_pairs:
                    raise ValueError(f"{path}: {identity} has invalid or repeated action pair {key!r}")
                numeric_pairs.add((verb, noun))
                if type(score) is not float or not math.isfinite(score):
                    raise ValueError(f"{path}: {identity} needs a finite Python float for {key!r}")
    if seen != expected_ids or advanced_rows not in (0, len(rows)):
        raise ValueError(f"{path}: incomplete coverage or mixed action scoring modes")
    print(f"{path}: {len(rows)} events, {advanced_rows} with explicit action scores")

for path in (ar_path, sta_path):
    validate(path)

entries = ("submission.pt", "submission_sta.pt")
with ZipFile(zip_path, "x", compression=ZIP_DEFLATED) as archive:
    for path, entry in zip((ar_path, sta_path), entries):
        archive.write(path, arcname=entry)
with ZipFile(zip_path) as archive:
    if archive.namelist() != list(entries) or archive.testzip() is not None:
        raise ValueError(f"{zip_path}: archive root entries or integrity check failed")
print(f"Validated {zip_path}")
PY
sha256sum "$COURSE_TEST" "$AR_FILE" "$STA_FILE" "$ZIP_FILE"
```

Use the **course** Codabench competition. Register with your **student email** and join the team organization **`Group X`**, replacing `X` with the number assigned by teaching staff. In the competition, select **both AR and STA** for the combined ZIP. Upload only after checking the root entries and confirming the selected team. The course allows **five submissions per day and 50 total**. Confirm both tasks score successfully, and record their separate results and the ZIP hash. Use the course validator's feedback to diagnose format failures. Keep model and recipe selection on held-out validation.
