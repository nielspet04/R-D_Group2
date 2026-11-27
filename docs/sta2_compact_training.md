# Niels: compact STA2 head on A100

Run the two-block head on **two A100 GPUs** and select its best checkpoint by
held-out validation **Action MT5R**. This is the compact development experiment
in the [competition runbook](sta2_codabench_runbook.md), which is the source of
truth for budget, comparison, model selection and submission. Target model
selection: **3 December 2026**; team submission: **9 December**.

## Fixed experiment

| Setting | Compact run |
| --- | --- |
| Configuration | `configs/sta2/compact.yaml` |
| Head | Two blocks, approximately 69.9M trainable parameters |
| Reference | Four blocks, approximately 136.4M trainable parameters |
| Training / validation | 59,767 / 7,450 events; original 26/6 participant split |
| Validation participants | P08, P20, P21, P28, P29, P37 |
| Seed / learning rate / weight decay | 42 / 0.0001 / 0.01 |
| Effective batch / microbatch per GPU | 16 / 1 (eight accumulation steps on two GPUs) |
| Precision | BF16 |
| Model input | Observed plus predicted future features (`observed_only: false`) |
| Schedule | Original 20-epoch horizon, patience five |
| Allocation | Two A100s, 16 CPUs, 128000 MiB, **40 hours** |
| Output | `results/sta2/compact-a100-JOBID` |

The YAML differs from the reference only in head depth. The frozen backbone is
unchanged, so halving head depth does not imply halving epoch time. All decoded
frames must precede the one-second anticipation cutoff. Predicted future
features come from that legal observation. Keep all 7,450 validation events,
including unseen training action pairs and prior-fallback events.

## Conda path correction — 27 November 2026

Niels's login-node output confirmed that `$VSC_DATA/conda_envs/rd_sta2` does not
exist. The existing environment is `$VSC_DATA/miniconda3/envs/rd_sta2`.
Its interpreter successfully imported Torch, NumPy, pandas and YAML and reported
PyTorch `2.5.1+cu118`, CUDA build `11.8`. The activation command using the old
path failed; the subsequent Python command succeeded in the already-active
environment. This verifies those imports, not the failed activation or GPU/DDP.

The compact A100 wrapper (also used by the timing job), preflight and competition
runbook now use the confirmed path. The preflight runs dependency imports with
that exact interpreter before quoting. Run on VSC after transferring the changes:

```bash
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda activate "$VSC_DATA/miniconda3/envs/rd_sta2" && \
    python -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda)'
```

The `&&` prevents a failed activation from being mistaken for a successful
environment check. CUDA hardware/BF16 and distributed execution remain to be
verified inside the A100 timing allocation. Other launchers have their own
environment settings; this correction covers the compact A100 workflow.

## Prepare on your VSC Bash login node

Preserve any local VSC edits before updating. Do not reset or clean the checkout.
These local changes must first be transferred to VSC through your normal Git
or file-transfer workflow; pulling fetches only changes already on the remote.

```bash
cd "$VSC_DATA/R-D_Group2"
git status --short --branch
git switch STA2_dev
git pull --ff-only origin STA2_dev

# If already configured: exports only the endpoint and private key-file path.
if [[ -f "$HOME/.config/rd-dashboard/env" ]]; then
    source "$HOME/.config/rd-dashboard/env"
fi
bash scripts/prepare_sta2_compact_a100.sh
```

The preflight checks your account's queue and recent accounting, lists existing
compact best checkpoints, checks required files and dependency imports, validates Bash syntax and
quotes 40 hours. It stops if your account already has a queued compact job or
if a required input is missing. It does not submit, decode videos or validate
the complete dataset. The trainer performs its full split/audit checks at start.
Check the other account for an existing compact run as required by the runbook;
your own queue alone cannot rule out a duplicate.

If upload variables are configured, the preflight probes the authenticated
upload endpoint. This proves login-node connectivity only. See
[hosted dashboard setup](hosted_dashboard_setup.md) for private credential setup
and compute-node connectivity checks. Never put the key in Git or chat.

Record actual spending plus quotes for pending work. The campaign cap is
**3M credits**, including **1.1M for development training**. The runbook estimates
roughly **778k credits** for this 40-hour request; the live quote decides whether
it fits. Existing spending reduces the remaining allowance.

## Missing timeline manifest — diagnose before allocating GPUs

On 27 November, Niels's preflight stopped because
`$VSC_SCRATCH/epic_sta/sta2_timelines_local/manifest.json` was missing. No GPU
job was submitted. This is independent of the Conda path correction; the
preflight had not reached its dependency-import check yet. The project balance
is shared and does not establish the remaining STA campaign allowance.

First look for metadata already on your account and confirm the raw videos:

```bash
for base in "$VSC_SCRATCH/epic_sta" "$VSC_DATA/sta2"; do
    if [[ -d "$base" ]]; then
        find "$base" -maxdepth 4 -type f -name manifest.json -print
    fi
done
ls -ld "$VSC_SCRATCH/epic_sta/raw_fixed"
ls -lh data/raw/EPIC_100_train.csv data/raw/EPIC_100_video_info.csv
```

An existing manifest is not automatically usable. Confirm its complete flag,
annotation identity, source-video paths/stat identities, status counts and
referenced timeline files. `sta2_timelines` was recorded as stale in the older
full-training guide; do not rename it to bypass the missing-file check. Do not
copy another account's manifest without validating source identities. If raw
videos are absent, restore/stage the supplied videos first.

Niels subsequently inspected `sta2_timelines/manifest.json`: `complete: true`,
495 `ok` videos, both input hashes matching, exact annotation video coverage,
no recorded errors and no missing timeline files. This supersedes the assumption
that this account's directory was necessarily stale. Source-video identities
and timeline contents still need the full check below.

After transferring the updated checker, preflight and compact wrapper to VSC:

```bash
export STA2_COMPACT_METADATA_DIR="$VSC_SCRATCH/epic_sta/sta2_timelines"
python scripts/check_sta2_metadata.py \
    --metadata-dir "$STA2_COMPACT_METADATA_DIR" \
    --raw-dir "$VSC_SCRATCH/epic_sta/raw_fixed"
bash scripts/prepare_sta2_compact_a100.sh --timing
```

The checker reads source-video size/mtime, the saved timeline SHA-256, PTS dtype,
frame count, zero origin and strict timestamp order, plus supplied FPS and
positive time base. It does not decode videos or modify metadata. The preflight
also runs it automatically before quoting. Keep `STA2_COMPACT_METADATA_DIR`
exported in the submission shell; both the timing and normal compact A100
wrappers use it. Preserve that same export when resuming, and use the same
manifest for the later validation comparison. Do not rename or symlink the
metadata directory to make a hardcoded path pass.

If no usable manifest exists and the raw videos and both input CSVs are
available, build it with the CPU job. The metadata launcher now uses the
confirmed Conda path and defaults to the same `sta2_timelines_local` directory
as compact training. Transfer those changes first; then quote and inspect the
CPU allocation before submitting:

```bash
mkdir -p logs
export STA2_RAW_DIR="$VSC_SCRATCH/epic_sta/raw_fixed"
export STA2_METADATA_DIR="$VSC_SCRATCH/epic_sta/sta2_timelines_local"
sam-quote sbatch slurm/sta2_metadata_cpu.slurm
# Submit after checking the CPU quote and that no metadata scan is active:
sbatch --export=ALL slurm/sta2_metadata_cpu.slurm
```

This is a four-CPU, 8 GB, up-to-12-hour scan with a directory lock and resumable
verified timeline files. It does not train or request GPUs. Inspect
`logs/sta2-metadata-JOBID.out` and accounting after completion; the final coverage
check must pass, and recorded video errors still require review. Rerun the
timing preflight only after the manifest is available. A GPU timing measurement
cannot be made until these data prerequisites pass.

## First run: one complete epoch for timing and cost

Use `slurm/sta2_compact_a100_timing.slurm` before committing to 40 hours. It
requests **12 hours on the regular `gpu_a100` partition**, two A100s, 16 CPUs
and 128000 MiB. The existing `gpu_a100_debug` profiler has a 30-minute limit;
it cannot establish the time for a full epoch. Twelve hours is a provisional
ceiling, not a measured duration or a guarantee that the first epoch fits.

The timing job uses the full training and validation splits and stops after
saving epoch 1, `last.pt`, `best.pt` and history. `--stop-after-epoch 1` leaves
the original 20-epoch scheduler unchanged, making the saved run resumable.
Its stop request is recorded separately in the run manifest. Do not set
`max_epochs: 1`, which would change the learning-rate schedule.

After transferring these changes to VSC, run:

```bash
cd "$VSC_DATA/R-D_Group2"
# Source the private dashboard env here if already configured.
bash scripts/prepare_sta2_compact_a100.sh --timing
```

Save the live quote. At the runbook's approximate rate, 12 hours is about
233k credits for the allocation ceiling. Actual accounting and the live quote
take precedence. The timing job consumes the development allowance; resuming
its checkpoint avoids paying for epoch 1 twice. Check both accounts for a
duplicate and check remaining credits before submitting:

```bash
mkdir -p logs
unset STA2_COMPACT_RESUME_RUN STA2_COMPACT_STOP_AFTER_EPOCH
sbatch --export=ALL --time=12:00:00 slurm/sta2_compact_a100_timing.slurm
```

Record the returned ID and inspect the log:

```bash
TIMING_JOB=REPLACE_WITH_JOB_ID
COMPACT_RUN="results/sta2/compact-a100-$TIMING_JOB"
tail -f "logs/sta2-compact-a100-timing-$TIMING_JOB.out"
# In another shell, or after leaving tail with Ctrl-C:
sacct -M wice -j "$TIMING_JOB" -X \
    --format=JobID,State,ElapsedRaw,Timelimit,AllocTRES,ExitCode
```

The log includes batch start (host, partition and UTC time), setup/manifest,
training progress at the first optimizer update and every 100 updates,
validation progress every 1,000 rank-zero batches with elapsed seconds,
`epoch_complete`, `requested_epoch_boundary` and batch exit code/time.
Output is unbuffered through Python and `srun`. Environment/setup failures
now print source/line and batch-exit diagnostics too. Python/DDP tracebacks
remain in the same Slurm output file. A missing log can mean the job is still
pending or Slurm could not open the output path; create `logs` before `sbatch`.
The timing job's log prefix differs from the standard launcher's prefix.

Once epoch 1 completes, save a cost report using the **total credits quoted for
the 12-hour request**, replacing both placeholders below. For `ElapsedRaw`, use
the completed top-level job row, not a `.batch` or `.extern` step:

```bash
python scripts/summarize_sta2_timing.py \
    --run-dir "$COMPACT_RUN" \
    --quoted-credits REPLACE_WITH_TOTAL_QUOTE --quoted-hours 12 \
    --allocation-seconds REPLACE_WITH_ELAPSED_RAW
```

The report is saved as `$COMPACT_RUN/timing-report.json`. It records measured
train, validation, checkpoint and total epoch seconds, quote-derived credits
per epoch and estimated allocation credits. The allocation includes model
loading, audits, shutdown and other overhead. Add `--actual-credits NUMBER`
when VSC reports the real charge; estimates are never labelled actual cost.
If you changed the requested walltime, pass its hours and matching quote.
First-epoch caches/warmup may differ from later epochs. No completed history
means no measured full-epoch estimate: diagnose the log and accounting first.

After checking metrics, legal-input/fallback counts and the remaining budget,
continue this same run with the normal A100 wrapper and no stop flag:

```bash
export STA2_COMPACT_RESUME_RUN="$COMPACT_RUN"
unset STA2_COMPACT_STOP_AFTER_EPOCH
test -s "$COMPACT_RUN/checkpoints/last.pt"
# Choose a walltime that fits the measured epoch time plus headroom and budget.
sam-quote sbatch --time=12:00:00 slurm/sta2_compact_a100.slurm
# Submit only after checking that quote:
sbatch --export=ALL --time=12:00:00 slurm/sta2_compact_a100.slurm
unset STA2_COMPACT_RESUME_RUN
```

Epoch 2 continues the same schedule and output directory. Do not start a fresh
40-hour job if you intend to continue the measured checkpoint. The commands
in the next section are for an intentionally fresh run.

## Submit once the checks and quote pass

```bash
mkdir -p logs
unset STA2_COMPACT_RESUME_RUN
sbatch --export=ALL --time=40:00:00 slurm/sta2_compact_a100.slurm
```

The wrapper now defaults to 40 hours too. The explicit override keeps the
allocation clear. Direct `sbatch` uses the trainer's integrated monitoring;
`scripts/submit_monitored.py` currently has no compact A100 job type.
Do not use its full-H100 job type for this experiment.

Set the real job ID returned by `sbatch`:

```bash
COMPACT_JOB=REPLACE_WITH_JOB_ID
COMPACT_RUN="results/sta2/compact-a100-$COMPACT_JOB"
squeue -M wice -j "$COMPACT_JOB" --start
sacct -M wice -j "$COMPACT_JOB" -X \
    --format=JobID,State,Elapsed,Timelimit,AllocTRES,ExitCode
tail -f "logs/sta2-compact-a100-$COMPACT_JOB.out"
```

After the first saved epoch, inspect `run-manifest-*.json`, `history.json`,
`checkpoints/best.pt` and `checkpoints/last.pt` in `$COMPACT_RUN`. Confirm the
settings above, split hashes and backbone SHA-256 against the reference.
Measure `train_seconds + validation_seconds` plus checkpoint overhead to
estimate credits per epoch. Aim for at least three completed epochs if feasible;
use Action MT5R and usable/fallback counts to decide whether more training is
worth the cost. Diagnose decode errors before using the run for selection.

The [hosted dashboard](https://training-observatory.pages.dev/) requires sign-in
and may lag. Native history and Slurm accounting determine actual state.
Closing the laptop does not stop a submitted batch job.

## Resume only after checking the saved result

For a walltime-limited A100 run with `history.json` and `checkpoints/last.pt`,
quote a continuation only if another epoch fits its time and credit allowance:

```bash
export STA2_COMPACT_RESUME_RUN="$COMPACT_RUN"
test -s "$STA2_COMPACT_RESUME_RUN/history.json"
test -s "$STA2_COMPACT_RESUME_RUN/checkpoints/last.pt"
sam-quote sbatch --time=12:00:00 slurm/sta2_compact_a100.slurm
# Submit after checking the quote, remaining credits and expected finish:
sbatch --export=ALL --time=12:00:00 slurm/sta2_compact_a100.slurm
unset STA2_COMPACT_RESUME_RUN
```

Keep the same wrapper, world size, inputs and settings. Checkpoints save at
epoch boundaries; an interrupted incomplete epoch can be lost. Do not switch
an H100 run into this A100 wrapper without validating cross-launcher resume.

## Handoff for comparison

Give Emiel the job ID, Git revision, stable best checkpoint path, best epoch,
Action/Verb/Noun MT5R, split/backbone identities, fallback counts, measured epoch
time and actual credits. The recorded reference is run `62771307`, **8.77% Action
MT5R at epoch 5**; verify its native history/checkpoint before comparison.
Follow runbook section 4 to compare both models and the fixed 50/50 blend on
the same complete validation split. Full-data retraining follows model selection.

## Verification record — 27 November 2026

Passed locally: Python syntax, Bash syntax, Git whitespace, quote-derived cost
arithmetic and invalid-quote rejection; the real `fit` control flow with stubbed
model math/storage (stop at epoch 1, preserve the 20-epoch horizon, resume epoch
2); wrapper arguments and accidental-resume rejection; environment-failure
logging with preserved nonzero status. These checks use no GPU or live scheduler.

A regression test for the real model's epoch-stop/resume behavior was added to
`tests/test_sta2/test_sta2_training.py`. It could not run locally because the
available Python environments lack NumPy/PyTorch. Before submitting, run it in
the existing VSC training environment:

```bash
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda activate "$VSC_DATA/miniconda3/envs/rd_sta2"
python -m unittest discover -s tests/test_sta2 -p test_sta2_training.py
```

The shared training launcher's log changes also affect other STA2 wrappers.
No live GPU/DDP execution, queue availability, data validity, credit balance,
throughput or accuracy has been verified here. No timing result or actual charge
exists yet. A real job is launched only by the explicit `sbatch` command on VSC.
