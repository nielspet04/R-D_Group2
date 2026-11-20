# STA2 full training on wICE

Prepared 19 November 2026 for vsc39706, branch `STA2_dev`, Slurm account
`lp_edu_rdlab`. Run cluster commands manually. No data download, metadata rebuild,
or repeat of the completed 32/16-event pipeline test is required.

## Evidence and resource choice

The cluster artifacts are not in this local checkout. The following observations
are from the supplied handoff, not independently re-read remote files:

- Metadata audit 62748900: all 495 videos passed. Use
  `$VSC_SCRATCH/epic_sta/sta2_timelines_local`; `sta2_timelines` is stale.
- A100 pilot 62757701 (`results/sta2/pilot-future-62757701`): 64 unique events,
  100 usable measured steps, BF16 observed-plus-future, ~8.5 GiB reserved and
  ~1.84 usable events/s. This excludes production head training. It did not
  verify backbone state hashes after an update.
- P100 pipeline test 62768804 (`results/sta2/training-smoke-62768804`): 32 usable
  training events, 16 usable validation events, two updates, finite losses
  13.41895 / 17.53294, saved best/last checkpoints (~1.5 GB each). The final
  summary failed after completion; the missing import is already fixed.

Use **two H100 GPUs on one node, BF16, 16 CPU cores, 128000 MiB RAM, 72 hours**.
The official [wICE guide](https://docs.vscentrum.be/leuven/wice_quick_start.html)
lists `gpu_h100` and its 72-hour limit. The request is below the documented
[per-GPU CPU/RAM limits](https://docs.vscentrum.be/leuven/slurm_specifics.html).
This avoids the reported long A100 queue and keeps the existing BF16 setting.
Queue availability, H100 environment compatibility, production head memory and
two-rank NCCL/DDP are **unverified**. Two GPUs is a starting allocation, not a
measured optimum. Do not change GPU count on resume.

P100 FP32 is a fallback for pipeline debugging, not the selected full run.
The aggregate smoke-job rate is only 48 / 1080 = 0.044 events/s, including
startup, validation, saves and failure overhead. Naively scaling that gives
62.5 hours per 10,000 events per epoch on one P100; it is not a steady-state
measurement. Conversely, the A100 backbone rate alone is ~1.51 hours per
10,000 usable events on one GPU, before head/backward and other costs. Neither
supports a promise that 20 epochs fit one allocation. Do not assume linear
two-GPU scaling or apply A100 throughput to H100.

## Fixed experiment and inputs

`slurm/sta2_full_h100.slurm` fixes these paths and rejects positional overrides:

- Config `configs/sta2/main.yaml`; backbone `configs/sta2/data_backbone.yaml`.
- Annotations `data/raw/EPIC_100_train.csv`.
- Original participant splits `data/processed/EPIC_100_train_participant_split.csv`
  and `data/processed/EPIC_100_validation_participant_split.csv`.
- Videos `$VSC_SCRATCH/epic_sta/raw_fixed` and audited local timelines above.
- Checkpoint `$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt`.
- Conda initialization `$VSC_DATA/miniconda3/etc/profile.d/conda.sh`;
  environment `$VSC_DATA/conda_envs/rd_sta2`.

The model, 384-pixel preprocessing, temporal sampling, observed-plus-future
protocol, LR 1e-4, seed 42, maximum 20 epochs, warmup one epoch and early-stop
patience five remain configured as before. Microbatch one with two ranks and
eight accumulation steps preserves effective global batch 16. This is the
existing main experiment; it does not substitute smoke data or implement the
other observed-only / learning-rate experiments discussed in the older plan.

Existing split validation checks exact annotation coverage, disjoint narration
IDs/participants and matching source labels. Full-run mode also requires every
referenced video to be present with `ok` metadata. Dataset startup checks media
size/mtime against the audit. Early/no-context events still follow the existing
mask/fallback protocol. No new decoding audit is requested.

The entrypoint now records counts, participant IDs, training/evaluation action
vocabulary sizes, validation vocabulary coverage, split hashes, metadata hash,
backbone configuration, checkpoint SHA256, resolved configuration and Git state
in `run-manifest-JOBID.json`, before constructing the model. Exact full-split
counts and vocabulary coverage cannot be calculated locally because the CSVs
are absent. Inspect this manifest and the `training_setup` log on VSC.

`validation_action_loss_events=1` in the smoke test means only one usable
validation event had a verb/noun pair in the 32-event training vocabulary.
Unseen pairs are excluded only from auxiliary action-classifier CE; verb/noun
CE still covers usable events. Action accuracy/MT5R is computed from joint
verb/noun probabilities over action pairs from the complete annotation table,
including unseen training pairs, with priors for no-context events. It does
not rank the auxiliary action logits. Zero smoke accuracy is not a quality
assessment. Full training will report its actual vocabulary coverage.

Two correctness fixes accompany the setup: workers restart at each epoch so
epoch-dependent augmentation reaches worker copies, and validation losses/counts
are deduplicated along with predictions when DistributedSampler pads the split.
GPU-mapped RNG tensors are also moved back to CPU before generator restoration,
fixing a CUDA resume failure; a local CUDA regression test covers this case.
Training still uses the existing sampler's padding at the end of an epoch;
reported training throughput counts processed events including that padding.

## Pull and submit, preserving VSC changes

From a wICE login shell, with the VSC environment variables available:

```bash
cd "$VSC_DATA/R-D_Group2"
git status --short
git switch STA2_dev
git pull --ff-only origin STA2_dev
git status --short
mkdir -p logs
bash -n slurm/sta2_train.slurm
bash -n slurm/sta2_full_h100.slurm
unset STA2_RESUME_RUN
sbatch slurm/sta2_full_h100.slurm
```

The patch does not edit `slurm/sta2_metadata_cpu.slurm`, `slurm/sta2_smoke.slurm`
or the VSC-only `slurm/sta2_train_smoke.slurm`. An ordinary fast-forward preserves
those staged/untracked changes; do not reset, clean or discard them. If Git
reports another conflicting local path, retain it and reconcile that path
before pulling. `slurm/sta2_training_smoke.slurm` is a different tracked file.

The common launcher retains the absolute Conda Python executable,
`srun --export=ALL`, and cleanup of `SLURM_MEM_PER_CPU`, `SLURM_MEM_PER_GPU` and
`SLURM_TRES_PER_TASK`. The full wrapper fixes module and GPU count together;
do not override only the SBATCH partition/GPU count.

## Monitor, budget and resume

Fresh output is `results/sta2/full-h100-JOBID`; log is
`logs/sta2-full-h100-JOBID.out`. Replace JOBID with the submission's numeric ID:

```bash
squeue -u vsc39706
tail -f logs/sta2-full-h100-JOBID.out
sacct -M wice -j JOBID --format=JobID,State,Elapsed,AllocTRES,MaxRSS
```

The first optimizer update and every 100 updates log elapsed time and rank-zero
processed events. Validation logs every 1,000 rank-zero batches. Each completed
epoch logs train/validation seconds, global event rates, rank-zero peak reserved
GPU bytes, losses, fallback counts, action loss coverage and Action MT5R; these
also appear in `history.json`. GPU memory is per rank, not a sum or cross-rank
maximum. Setup and checkpoint I/O are outside the reported phase timings.

After an epoch, estimate remaining compute as
`(20 - completed_epoch) * (train_seconds + validation_seconds)` plus checkpoint
overhead, allowing for early stopping. For all 20 epochs to fit 72 hours, mean
epoch time would need to be below 3.6 hours including saves. Use actual full-run
counts/rates; if that budget fails, use further allocations with resume or
reassess resources explicitly. Do not reduce sampling/model/epoch settings to
hide the shortfall. Queue delays are additional.

`last.pt` is replaced atomically after each completed train+validation epoch;
`best.pt` updates when Action MT5R improves. Checkpoints contain head, AdamW,
scheduler, per-rank RNG, epoch, early-stop state and provenance, not the frozen
backbone. Allow space for best, last and a temporary save (at least ~5 GB based
on the smoke files; full vocabulary may increase this). Keep the source
checkpoint accessible. A heuristic stops after a saved epoch if the remaining
Slurm time is less than 1.2 times the preceding epoch plus ten minutes.
There is no mid-epoch save or automatic resubmission; the first epoch must fit,
and unexpected slowdowns/timeouts can lose the current epoch.

Resume only the full-run directory, keeping configuration, paths, metadata,
checkpoint, participant splits and two-GPU world size identical:

```bash
cd "$VSC_DATA/R-D_Group2"
export STA2_RESUME_RUN="results/sta2/full-h100-JOBID"
sbatch slurm/sta2_full_h100.slurm
unset STA2_RESUME_RUN
```

The new job keeps the original output directory and gets a separate Slurm log
and run manifest. Resume loads `checkpoints/last.pt` and starts at the next
epoch; it does not extend the 20-epoch schedule. Earlier smoke checkpoints lack
the new resolved provenance and are not suitable for this full experiment.
An interrupted save can leave history one epoch behind the atomic checkpoint;
the checkpoint determines the restart epoch. If training already reached early
stopping or 20 epochs, resume does no further training.

Remaining cluster acceptance work: two-GPU execution (especially an invalid-only
rank), real full-vocabulary head memory/backward, full-run throughput, H100 CUDA
kernel compatibility, GPU checkpoint resume and real backbone state-hash
verification after a head update. Local synthetic tests do not establish these.

Local verification uses the installed `rd_sta2` environment (PyTorch 2.5.1+cu118):
the full STA2 suite and targeted training/CUDA RNG regression tests pass.
The installed Windows Bash parser timed out, including outside the sandbox;
shell syntax has been reviewed manually, with `bash -n` included above for VSC.

## BF16 integration failure in job 62770181

The first H100 submission failed because the adapter returned observed and
future tokens with different dtypes. The original synthetic tests exercised
FP32 outputs for both branches and missed this AMP integration case. A new
local CUDA test with small instances of the actual vendored encoder/predictor
reproduces FP32 encoder output and BF16 predictor output. The adapter now
promotes both to their common dtype before returning them to the model. This
preserves the FP32 values rather than downcasting them, while retaining BF16
autocast for the backbone and head operations. No model, sampling, batch or
training schedule settings change.

The regression tests exercise head forward/backward and an optimizer update,
including activation checkpointing; the CUDA integration test also verifies
the frozen backbone state hash is unchanged. These are small local modules,
not a full ViT-G checkpoint or two-H100 validation. Pull the fix and submit a
fresh job using the commands above (`STA2_RESUME_RUN` unset); retain the failed
job's output for diagnosis.
