# Independent STA2 experiment on Niels's VSC account

Emiel runs the default `configs/sta2/main.yaml`: four-block head, learning rate
1e-4, weight decay 0.01, seed 42, observed-plus-future, and the original 26/6
participant split. Niels runs `configs/sta2/compact.yaml`, changing only head
depth from four to two. This is development training, not final full-data
retraining. It does not need Emiel's checkpoint or access to his account.

## Hypothesis and comparison

The supplied first five epochs show train loss decreasing from 11.837 to 6.085,
while validation loss reaches its minimum at epoch 2 and then rises. This is
consistent with possible overfitting, but validation loss also contains an
auxiliary action loss and is not the selection metric. Best Action MT5R still
improves at epoch 5, so these logs do not establish convergence or a cause.

A two-block head tests whether less trainable capacity generalizes better to
the held-out participants. It contains one self-attention block and the final
cross-attention block, versus three self-attention blocks plus cross-attention
in the default. At the checkpoint's 1664-dimensional embeddings and the real
3,317 training action classes, trainable head parameters are 69,915,010 versus
136,411,778. Backbone cost is unchanged; do not assume training time halves.

Keep all other hyperparameters and the complete validation set unchanged.
The training split remains 59,767 actions from 26 participants. Validation is
7,450 actions from P08, P20, P21, P28, P29 and P37. Both heads retain the
97-verb, 300-noun and auxiliary action outputs. Compare each run's best full
validation Action MT5R under the same 20-epoch cap/patience-five rule. Emiel's
8.77% is his best so far, not necessarily his eventual best. There is no
guarantee the compact variant wins. One seed is an initial comparison, not
proof of a robust gain.

## Pull and launch on VSC

The experiment is on branch `STA2_dev`. Use Git to transfer the configuration,
launcher and this guide to your account's existing checkout:

```bash
cd "$VSC_DATA/R-D_Group2"
git status --short
git switch STA2_dev
git pull --ff-only origin STA2_dev
mkdir -p logs
bash -n slurm/sta2_compact_h100.slurm
unset STA2_COMPACT_RESUME_RUN
sbatch slurm/sta2_compact_h100.slurm
```

Run the prerequisite checks below before submitting if the data, environment
or checkpoint has not already been verified on your own account. Preserve
local VSC changes if Git reports a conflict; do not reset or clean the checkout.

## Alternative: create the experiment directly on VSC

The two new files already exist in the local checkout. The following commands
also generate their exact contents from the default files on VSC, without an
SSH alias, Git push or file transfer. Run on YOUR account's login shell.
They preserve the original files and refuse to replace differing experiment
files that already exist.

```bash
cd "$VSC_DATA/R-D_Group2"
python3 - <<'PY'
from pathlib import Path

def save_new(name, text):
    path = Path(name)
    if path.exists() and path.read_text() != text:
        raise SystemExit(f"Existing experiment differs; preserve it before continuing: {name}")
    path.write_text(text)

config = Path("configs/sta2/main.yaml").read_text()
assert config.count("  depth: 4") == 1, "Expected Emiel's default four-block config"
save_new("configs/sta2/compact.yaml",
    "# Independent development experiment: reduce only prediction-head depth.\n"
    + config.replace("  depth: 4", "  depth: 2"))

script = Path("slurm/sta2_full_h100.slurm").read_text()
for old, new in (
    ("sta2-full-h100", "sta2-compact-h100"),
    ("configs/sta2/main.yaml", "configs/sta2/compact.yaml"),
    ("STA2_RESUME_RUN", "STA2_COMPACT_RESUME_RUN"),
    ("results/sta2/full-h100-", "results/sta2/compact-h100-"),
    ("# Fixed production inputs: never inherit the smoke wrapper's configuration.",
     "# Independent two-block-head experiment; retain the development split."),
):
    assert old in script, f"Unexpected baseline launcher; missing {old}"
    script = script.replace(old, new)
save_new("slurm/sta2_compact_h100.slurm", script)
PY
```

This assumes your VSC checkout contains the existing working full-training
launcher. It creates only a configuration and wrapper; no new trainer code or
final-retraining changes are required for this experiment.

## Check your account's prerequisites and launch

Emiel's account having the data does not establish that your account has it.
These are the existing Niels paths used by the new wrapper:

```bash
test -s "$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt"
test -s "$VSC_SCRATCH/epic_sta/sta2_timelines_local/manifest.json"
test -d "$VSC_SCRATCH/epic_sta/raw_fixed"
test -d "$VSC_DATA/conda_envs/rd_sta2"
test -s data/processed/EPIC_100_train_participant_split.csv
test -s data/processed/EPIC_100_validation_participant_split.csv
mkdir -p logs
bash -n slurm/sta2_compact_h100.slurm
unset STA2_COMPACT_RESUME_RUN
sbatch slurm/sta2_compact_h100.slurm
```

Run the checks before sbatch and resolve any failure. The trainer independently
checks exact source/split coverage, disjoint participants, current video audit
identities and complete video availability. The wrapper requests two H100s,
16 CPU cores, 128000 MiB memory and up to 72 hours under `lp_edu_rdlab`.
It starts a fresh model head, with no resume or dependency on Emiel's job.

## A100 alternative during H100 maintenance

`slurm/sta2_compact_a100.slurm` runs the same compact-head experiment on two
A100 GPUs. It changes the partition, matching cluster module, job/log names
and new-run directory to A100; BF16 and all model/data/training settings stay
the same. Its 16 CPUs and 128000 MiB memory for two GPUs are within the
[documented A100 resource limits](https://docs.vscentrum.be/leuven/slurm_specifics.html).

Check live availability; a different partition is not a guarantee of an
earlier start:

```bash
sinfo -M wice -p gpu_a100,gpu_h100,gpu_v100 -o "%P %a %l %D %t %G"
```

After pulling the new launcher, use
`sbatch slurm/sta2_compact_a100.slurm`. If switching away from an H100 job,
cancel only the pending H100 job after successful replacement submission:
`scancel --state=PENDING OLD_H100_JOBID`. Do not leave two equivalent training
jobs queued unintentionally. A100 logs are
`logs/sta2-compact-a100-JOBID.out`, with outputs under
`results/sta2/compact-a100-JOBID`.

Keep `STA2_COMPACT_RESUME_RUN` unset for a fresh run. If resuming, use the A100
launcher only with an existing A100 run; the resolved configuration includes
output/input paths, and cross-launcher resume has not been validated.

V100 is listed among the [wICE GPU partitions](https://docs.vscentrum.be/leuven/wice_quick_start.html)
but does not provide native BF16. The current trainer supports BF16 or FP32,
not an FP16/GradScaler path. A V100 attempt therefore needs an explicitly
configured FP32 run and memory/throughput profiling before committing to full
training. No V100 performance or fit claim has been established.

## Monitor and resume

For the original H100 launcher, replace JOBID with your new job ID:

```bash
squeue -u "$USER"
tail -f logs/sta2-compact-h100-JOBID.out
```

Outputs are `results/sta2/compact-h100-JOBID`. The run manifest must show
`head.depth: 2`, `train_events: 59767`, `validation_events: 7450`,
the expected participant IDs and `observed_only: false`. Check best/last
checkpoints and `history.json` in this new directory.

If the allocation stops at an epoch boundary before development finishes:

```bash
export STA2_COMPACT_RESUME_RUN="results/sta2/compact-h100-JOBID"
sbatch slurm/sta2_compact_h100.slurm
unset STA2_COMPACT_RESUME_RUN
```

Compare the new best Action MT5R against Emiel's eventual best and check that
both run manifests have identical split hashes and backbone checkpoint SHA256.
Different account paths are fine. If this run wins, select it for later
full-data retraining and Codabench export. Do not use the full-data final job
to evaluate whether this architecture improves held-out validation.

## Local verification

The candidate YAML differs from the baseline only in head depth. The real
training annotations supplied the event, participant and action-class counts.
Meta-device construction verified full-size head parameter counts. A small
CPU model using the candidate's head settings passed finite forward/backward,
one AdamW update and evaluation with 97/300 outputs and activation
checkpointing enabled. Bash syntax and Git whitespace checks passed.
These checks do not establish VSC resource availability, full-size GPU
throughput or an accuracy improvement. No remote job has been launched here.
