# STA Models 1 and 2 Full Implementation and VSC Training Plan

## Summary

Model 0 and both video baselines are complete and evaluated. Model 0 predicts
training-label frequencies without reading video and obtains an Action MT5R of
`0.0027`. Models 1 and 2 use exactly the same split and evaluator:

| Model | Input | Temporal processing | Purpose |
|---|---|---|---|
| Model 0 | Training-label frequencies | None | Minimum baseline |
| Model 1 | Frozen R3D-18 features | Masked mean pooling | Test whether video helps |
| Model 2 | Same frozen R3D-18 features | Masked GRU | Test whether temporal order helps |

Both video models use cached features. R3D-18 processed the raw videos once on
VSC; subsequent model training was fast and did not repeatedly decode the videos.

The pipeline will retain every event:

- Full context: use all five one-second segments.
- Partial context: zero-pad unavailable early segments and store a validity mask.
- No legal context or unreadable video: use Model 0 probabilities as fallback.
- Hidden-test events are never removed.

The current split contains:

| Split | Total | Full five seconds | Partial context | No context |
|---|---:|---:|---:|---:|
| Train | 59,767 | 59,041 | 650 | 76 |
| Validation | 7,450 | 7,333 | 93 | 24 |

The full-mask counts follow stored segment validity. A segment clipped at the
start of a video remains valid when it still contains usable frames; the older
`58,895` and `7,307` values counted complete five-second windows instead.

## Current progress — 17 November 2026

### Cached features and models

- The version 1 feature-shard agreement is documented in
  [`docs/sta_feature_shard_contract.md`](sta_feature_shard_contract.md), with
  shared validation and atomic shard save/load helpers in `src/sta_features.py`.
- Niels's `STAFeatureDataset` is now implemented in `src/sta_features.py`. It
  reads manifest-backed validated shards lazily, retains metadata and labels, and
  converts stored `float16` features to training-time `float32`. The trainer uses
  this shared dataset directly; no temporary competing shard adapter is needed.
- `scripts/create_sta_example_shard.py` still provides the deterministic
  eight-event contract shard covering full, partial, no-context and video-error
  events. The feature-shard smoke notebook has passed against this contract.
- Model 1 is implemented as `MaskedMeanPoolSTA` in `src/sta_model_1.py`, with
  seven permanent unit tests covering masking, padding isolation, all-false
  events, gradients and input validation.
- Model 2 is implemented as `MaskedGRUSTA` in `src/sta_model_2.py`, with 14
  permanent unit tests covering masked recurrent updates, chronological order,
  padding isolation, gradients and input validation.
- Notebook 5 remains intentionally skipped. Model 2 is tested directly through
  its permanent unit tests.

### Shared training and validation core

- `src/sta_training.py` now contains one shared configuration and training core
  for `mean_pool` and `gru`. The fixed baseline defaults are seed `42`, AdamW,
  learning rate `0.001`, weight decay `0.0001`, train batch size `256`, validation
  batch size `512`, at most 30 epochs, early-stopping patience `5`, dropout `0.3`
  and gradient clipping at `1.0`.
- The training boundary explicitly converts features to `float32`. Full and
  partial events contribute combined verb and noun cross-entropy, while all-false
  events remain counted but are excluded from both losses. A batch with no usable
  context skips the forward pass and optimizer update entirely, preventing AdamW
  weight decay or optimizer state from changing the model without evidence.
- Shared validation runs under inference mode, applies separate verb and noun
  softmax operations for usable events and substitutes Model 0 frequency-prior
  probabilities for all-false events. Every validation row remains represented,
  including `no_context` and `video_error` events.
- Validation reuses `src/evaluation.py` to report verb, noun and valid joint-action
  Top-1, Top-5 and MT5R. Joint-action scores use `P(verb) * P(noun)`, and validation
  Action MT5R is the checkpoint-selection metric.
- `tests/test_sta_training.py` contains six focused tests. They prove that
  no-context labels cannot affect loss, empty-context batches do not update either
  architecture, usable batches update both architectures, `float16` inputs reach
  the models as `float32`, validation retains fallback events, and checkpoint
  save/reload reproduces predictions and optimizer state.

### Checkpoints and experiment orchestration

- Checkpoints are saved atomically and contain a schema version, model state,
  AdamW state, epoch, seed, full configuration, validation metrics, Git commit
  hash and an accurate dirty-working-tree flag. Loading can inspect metadata or
  restore both model and optimizer for resumed training.
- Git provenance is collected with read-only Git commands and includes untracked
  files in the dirty-tree check. This prevents an uncommitted experiment from
  being presented as fully reproducible from its commit hash alone.
- `src/sta_trainer.py` now owns experiment-level orchestration so that
  `src/sta_training.py` remains focused on reusable single-epoch operations. It
  runs training and validation, selects strictly improving Action MT5R, applies
  patience-based early stopping, writes history after every completed epoch and
  restores the selected best checkpoint before returning.
- The orchestrator writes the planned artifacts under the supplied result root:
  `checkpoints/<model>/best.pt`, `logs/<model>/history.csv`,
  `tables/<model>_validation.json` and `configs/<model>.json`.
- `tests/test_sta_trainer.py` supplies a controlled MT5R sequence and proves that
  the best epoch is retained, worse epochs do not replace it, patience stops at
  the expected point, all four artifacts are written and the returned model is
  restored to the best rather than final epoch.

### Training entry point

- `scripts/train_sta.py` trains either `mean_pool` or `gru` directly from the
  shared `STAFeatureDataset`. It retains the fixed `TrainingConfig` defaults,
  fits the Model 0 fallback from training-shard labels only, and builds valid
  action pairs from the complete supplied annotation CSV in the same way as the
  Model 0 evaluator.
- Training uses a deterministic shard-aware batch sampler. Each epoch randomizes
  shard order and event order within each shard with seed 42, but no batch spans
  two shards. Validation remains sequential. This preserves event randomness
  without thrashing the dataset's one-shard cache.
- The CLI accepts the model, feature root, annotation path, output root and
  device. `--smoke-test-epochs` is the only baseline override and is explicitly
  limited to small correctness checks.
- `tests/test_train_sta.py` creates independent contract-valid train and
  validation shard trees. The validation fixture covers full, partial and
  all-false masks, and the end-to-end test runs both architectures for one smoke
  epoch, checks fallback accounting and all four output artifacts, and reloads
  both checkpoints.

### Verification and remaining work

- Full feature verification passes for all `59,767` training and `7,450`
  validation events. It reports `76` and `24` no-context fallbacks respectively,
  zero video errors, valid shard sizes and finite `float32` dataset reads.
- Reported full mean-pooling and GRU runs have completed and their best
  checkpoints reload under the independent evaluator.
- The GRU result was reproduced exactly by a second deterministic seed-42 run.
  This confirms repeatability of the configured experiment, not variation across
  independent seeds.
- The VSC jobs now call
  `$VSC_DATA/conda_envs/rd_epic_cuda/bin/python` directly. They do not load
  Miniforge, bootstrap Miniconda or depend on `conda activate`. Commit `08cdf39`
  records this correction and also resolves the `/data/...` versus
  `/vsc-hard-mounts/...` project-path aliases in the monitored submit wrapper.
- The monitoring tests pass after the environment correction. Full training,
  evaluation and monitoring are complete; final all-label retraining and the
  hidden-test submission remain future work.

## 1. Build and validate the shared STA feature pipeline

### Temporal window definition

For an action starting at time `t`, create five fixed relative segments:

```text
segment 0: t-6 to t-5
segment 1: t-5 to t-4
segment 2: t-4 to t-3
segment 3: t-3 to t-2
segment 4: t-2 to t-1
                           action starts at t
```

The latest permitted frame is strictly before `t-1`.

For every segment:

1. Convert its start and stop times to video-frame indices using the video FPS.
2. Use `ceil(start × FPS)` as the first frame.
3. Use `ceil(stop × FPS)` as the exclusive upper boundary.
4. Uniformly select 16 frame indices within that interval.
5. Confirm every selected frame satisfies `frame_index / FPS < t-1`.
6. Repeat frames when a short partial interval contains fewer than 16 unique frames.
7. Produce a zero clip and `valid_mask=False` when the entire segment occurs before video time zero.

Examples:

```text
Action at 20 s:
mask = [True, True, True, True, True]

Action at 4 s:
relative segments = [-2,-1], [-1,0], [0,1], [1,2], [2,3]
mask = [False, False, True, True, True]

Action at 0.8 s:
no observation exists before the one-second gap
mask = [False, False, False, False, False]
```

### R3D-18 feature extraction

Use `torchvision.models.video.r3d_18` with `R3D_18_Weights.DEFAULT`.

Implementation rules:

- Replace the final classification layer with `nn.Identity()` so the network returns 512 features.
- Set `encoder.eval()`.
- Set every parameter’s `requires_grad=False`.
- Run extraction under `torch.inference_mode()`.
- Use the preprocessing supplied by `R3D_18_Weights.DEFAULT.transforms()`.
- Feed 16 RGB frames per valid one-second segment.
- Flatten the event and segment dimensions before encoding, then restore them afterward.
- Never pass invalid padded segments through R3D-18.
- Save features as `float16`; convert them back to `float32` during model training.

Each event will produce:

```python
features.shape == [5, 512]
valid_mask.shape == [5]
```

### Resumable feature storage

Do not create one file per event. Save fixed shards of 2,048 events:

```text
data/processed/sta_features/
├── train/
│   ├── shard_00000.pt
│   ├── shard_00001.pt
│   └── manifest.csv
├── validation/
│   └── ...
└── test/
    └── ...
```

Each shard contains:

```python
{
    "schema_version": 1,
    "split": str,                           # train, validation or test
    "features": float16 tensor [N, 5, 512],
    "valid_mask": bool tensor [N, 5],
    "verb_labels": int64 tensor [N],       # absent for hidden test
    "noun_labels": int64 tensor [N],       # absent for hidden test
    "narration_ids": list[str],
    "video_ids": list[str],
    "action_start_seconds": float32 tensor [N],
    "observation_end_seconds": float32 tensor [N],
    "status": list[str],  # full, partial, no_context, video_error
}
```

The exact executable handoff agreement, including dtypes, status/mask rules,
ordering and the manifest columns, is documented in
[`docs/sta_feature_shard_contract.md`](sta_feature_shard_contract.md) and validated
by `src/sta_features.py`.

Fixed shard boundaries make extraction resumable. With `--resume`, a completed shard is validated and skipped. A temporary file is written first and renamed only after validation, preventing interrupted jobs from leaving apparently complete shards.

Expected feature size is approximately 300 MiB for training and 40 MiB for validation.

### Public interfaces to add

Implement these reusable interfaces:

```python
sta_segment_boundaries(action_start, context_seconds=5, gap_seconds=1, segments=5)
sample_segment_frames(video_path, segment_start, segment_stop, frame_count=16)
STAFeatureDataset(feature_root, split)
extract_sta_features(...)
```

Provide command-line entrypoints:

```bash
python scripts/audit_sta_data.py
python scripts/extract_sta_features.py --split train
python scripts/inspect_sta_features.py --split train
```

## 2. Implement Models 1 and 2 and their shared trainer

### Model 1: masked mean pooling

Input:

```python
features: [batch, 5, 512]
valid_mask: [batch, 5]
```

Behavior:

1. Multiply features by the validity mask.
2. Sum across the five temporal segments.
3. Divide by the number of valid segments, clamped to at least one.
4. Apply dropout `0.3`.
5. Produce separate verb and noun logits.

```text
features → masked mean → dropout
                           ├→ Linear(512, 97)
                           └→ Linear(512, 300)
```

Model 1 uses video information but discards temporal order.

### Model 2: masked GRU

Use one `GRUCell` with:

```text
input size: 512
hidden size: 512
layers: 1
dropout after final representation: 0.3
```

Process the five features in chronological order. At each step:

- Calculate the candidate hidden state.
- Update the hidden state only when `valid_mask=True`.
- Preserve the previous hidden state for padded segments.

This avoids allowing leading zero-padding to change the GRU’s internal memory.

```text
feature 0 → hidden 0
feature 1 → hidden 1
...
feature 4 → final hidden
                  ├→ Linear(512, 97)
                  └→ Linear(512, 300)
```

Model 2 uses the same visual features as Model 1, so any performance difference tests the value of temporal order.

### Shared training configuration

Use the same fixed configuration for both models:

```text
Random seed:             42
Loss:                    verb cross-entropy + noun cross-entropy
Optimizer:               AdamW
Learning rate:           0.001
Weight decay:            0.0001
Training batch size:     256
Validation batch size:   512
Maximum epochs:          30
Early-stopping patience: 5 epochs
Gradient clipping:       1.0
Primary selection metric: validation Action MT5R
```

Do not use class weighting in the initial baseline. Class-balanced loss becomes a later controlled experiment.

Exclude all-false events from gradient updates. During validation, replace their model scores with Model 0 prior probabilities so all 7,450 validation events remain represented.

For neural logits:

1. Apply softmax separately to verbs and nouns.
2. Calculate valid joint-action scores with `P(verb) × P(noun)`.
3. Report verb, noun and action Top-1, Top-5 and MT5R.
4. Select the checkpoint with the best validation Action MT5R.

Each run must save:

```text
results/checkpoints/<model>/best.pt
results/logs/<model>/history.csv
results/tables/<model>_validation.json
results/configs/<model>.json
```

The checkpoint contains model state, optimizer state, epoch, seed, configuration, validation metrics and Git commit hash.

Training commands:

```bash
python scripts/train_sta.py --model mean_pool
python scripts/train_sta.py --model gru
```

Evaluation commands:

```bash
python scripts/evaluate_sta_model.py \
  --checkpoint results/checkpoints/mean_pool/best.pt

python scripts/evaluate_sta_model.py \
  --checkpoint results/checkpoints/gru/best.pt
```

## 3. Local development and staged testing

Only four videos are available locally, so local work is for correctness, not full training.

### Local test sequence

Activate the current environment:

```bash
conda activate rd_lab
cd "/Users/nielspeters/Library/CloudStorage/OneDrive-KULeuven/MASTER 2/R&D project"
```

Install local CPU/MPS PyTorch and TorchVision without changing the VSC CUDA environment:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m unittest discover -s tests -v
python scripts/evaluate_sta_model_0.py
```

Extract a tiny local feature set from readable videos:

```bash
python scripts/extract_sta_features.py \
  --split train \
  --limit 8 \
  --output-root results/local_sta_smoke \
  --device cpu \
  --shard-size 8
```

Inspect it:

```bash
python scripts/inspect_sta_features.py \
  --feature-root results/local_sta_smoke \
  --split train
```

Expected checks:

- Shape is `[8, 5, 512]`.
- Feature values are finite.
- Every mask contains the expected valid segments.
- No selected timestamp reaches the one-second gap.
- Narration IDs and labels match the source CSV.
- Running extraction twice produces the same features.

### Tiny overfit test

Use 128 VSC examples after the GPU smoke extraction:

```bash
python scripts/train_sta.py \
  --model mean_pool \
  --feature-root results/vsc_smoke_features \
  --max-events 128 \
  --epochs 50 \
  --early-stopping-patience 50

python scripts/train_sta.py \
  --model gru \
  --feature-root results/vsc_smoke_features \
  --max-events 128 \
  --epochs 50 \
  --early-stopping-patience 50
```

This is not a real result. It verifies that:

- gradients flow;
- loss decreases substantially;
- checkpoints can be reloaded;
- predictions change from their initialization;
- the training loop can learn from fixed features.

## 4. VSC environment, real-data extraction and full training

### Create a clean VSC environment

Add a committed `environment-vsc.yml` defining:

```text
Environment name: rd_epic_cuda
Python:           3.11
PyTorch:          2.5.1
TorchVision:      0.20.1
CUDA runtime:     11.8
NumPy:            1.26
Pandas:           2.2
OpenCV headless:  4.10
Matplotlib:       3.9
TQDM:             4.66
```

The existing VSC environment uses CUDA 11.8 for compatibility with the P100
GPUs. Its interpreter is used directly so jobs do not depend on a particular
Conda distribution or shell activation.

On the VSC login node:

```bash
cd "$VSC_DATA/R-D_Group2"
git pull --ff-only

PYTHON="$VSC_DATA/conda_envs/rd_epic_cuda/bin/python"
test -x "$PYTHON"
"$PYTHON" -c "import torch; print(torch.__version__)"

export TORCH_HOME="$VSC_DATA/torch-cache"
mkdir -p "$TORCH_HOME"
```

Download and verify the encoder weights once:

```bash
"$PYTHON" -c "
import torch
from torchvision.models.video import r3d_18, R3D_18_Weights
r3d_18(weights=R3D_18_Weights.DEFAULT)
print('torch:', torch.__version__)
print('cuda visible on login node:', torch.cuda.is_available())
"
```

CUDA may be false on the login node; the actual GPU check happens inside a Slurm job.

### Audit the real VSC data

Run this lightweight audit before requesting a GPU:

```bash
cd "$VSC_DATA/R-D_Group2"
PYTHON="$VSC_DATA/conda_envs/rd_epic_cuda/bin/python"

"$PYTHON" scripts/audit_sta_data.py
```

Acceptance criteria:

- 67,217 provided training annotations.
- 495 unique training video IDs.
- Every training video ID resolves to exactly one MP4.
- No duplicate narration IDs.
- Train and validation participants remain disjoint.
- Counts of full, partial and no-context events match the documented split.
- No video is written or modified.

Also check storage:

```bash
du -sh "$VSC_DATA/R-D_Group2/data/raw"
du -sh "$VSC_DATA/R-D_Group2/data/processed"
myquota
sam-balance
```

### Slurm resources

Use the course account and wICE P100 partition:

```text
Account:   lp_edu_rdlab
Cluster:   wice
GPU:       one P100, 16 GiB
```

The official wICE documentation lists `gpu_p100`, `gpu_p100_long` and `gpu_p100_debug`; P100 jobs may request up to nine CPU cores and 45,000 MiB CPU memory per GPU. [wICE quick-start guide](https://docs.vscentrum.be/leuven/wice_quick_start.html), [KU Leuven Slurm specifics](https://docs.vscentrum.be/leuven/slurm_specifics.html)

Every job must:

```bash
module load cluster/wice/gpu_p100
PYTHON="$VSC_DATA/conda_envs/rd_epic_cuda/bin/python"
test -x "$PYTHON"
export TORCH_HOME="$VSC_DATA/torch-cache"
cd "$VSC_DATA/R-D_Group2"
```

Do not train or extract features directly on a login node.

### GPU smoke job

Create `slurm/sta_smoke_p100.slurm` with:

```text
partition: gpu_p100_debug
time:      00:30:00
GPUs:      1
CPUs:      4
```

The job performs:

1. `nvidia-smi`.
2. Prints PyTorch, TorchVision and CUDA versions.
3. Confirms the device name.
4. Runs unit tests.
5. Evaluates Model 0.
6. Extracts 32 real training examples.
7. Validates feature shapes and leakage rules.
8. Runs a forward pass through both models.

Submit and monitor:

```bash
mkdir -p logs
sbatch slurm/sta_smoke_p100.slurm
squeue -M wice -u "$USER"
```

After completion:

```bash
sacct -M wice -j JOB_ID \
  --format=JobID,State,Elapsed,MaxRSS,ExitCode

less logs/sta_smoke-JOB_ID.out
```

Do not proceed if the state is not `COMPLETED` or if CUDA is unavailable.

### Full feature extraction

Use `slurm/sta_extract_a100.slurm` for the full extraction jobs:

```text
partition:      gpu_a100
time:           12:00:00
GPUs:           1
CPU cores:      12
event batch:    8
decoder workers: 8
shard size:     2048
```

Submit training and validation separately:

```bash
sbatch slurm/sta_extract_a100.slurm train
sbatch slurm/sta_extract_a100.slurm validation
```

The extraction command inside the job is:

```bash
python scripts/extract_sta_features.py \
  --raw-dir "$VSC_SCRATCH/epic_sta/raw" \
  --split "$1" \
  --output-root data/processed/sta_features \
  --device cuda \
  --event-batch-size 8 \
  --workers 8 \
  --shard-size 2048 \
  --dtype float16 \
  --resume
```

If a job times out, submit the same command again. Completed shards are skipped.

After both jobs finish:

```bash
python scripts/inspect_sta_features.py --split train --check-all
python scripts/inspect_sta_features.py --split validation --check-all
```

Acceptance criteria:

- Exactly 59,767 train and 7,450 validation rows are represented.
- Every feature tensor has shape `[5, 512]`.
- All stored values are finite.
- All 76 train and 24 validation no-context events have all-false masks.
- Partial events have at least one valid segment.
- No valid frame reaches or crosses `action_start - 1`.
- Any `video_error` is listed explicitly and uses prior fallback.
- Re-running with `--resume` performs no duplicate work.

### Full Model 1 training

Before a reported run, execute a one-epoch correctness smoke test on the A100
debug partition. It reads the complete cached feature sets so it exercises all
shards, but writes to `results/smoke_a100` and must not be reported as a model
result:

```bash
mkdir -p logs
sbatch slurm/sta_train_smoke_a100.slurm mean_pool
```

The `gpu_a100_debug` partition accepts jobs up to one hour. Only one A100 debug
job may be queued at a time, so smoke the mean-pooling model first and submit a
GRU smoke separately only if needed.

Create `slurm/sta_train_p100.slurm` with:

```text
partition: gpu_p100
time:      04:00:00
GPUs:      1
CPUs:      4
```

Submit the reported run through the isolated monitoring wrapper:

```bash
PYTHON="$VSC_DATA/conda_envs/rd_epic_cuda/bin/python"
"$PYTHON" scripts/submit_monitored.py --model mean_pool
```

The wrapper evaluates the selected checkpoint automatically. A manual repeat can
still be run with:

```bash
"$PYTHON" scripts/evaluate_sta_model.py \
  --checkpoint results/runs/RUN_ID/checkpoints/mean_pool/best.pt
```

Acceptance criteria:

- Training and validation loss are finite.
- The best checkpoint reloads successfully.
- Evaluation covers all 7,450 validation events.
- Action MT5R exceeds Model 0’s `0.002742`.
- Results and configuration are saved.

### Full Model 2 training

Submit:

```bash
PYTHON="$VSC_DATA/conda_envs/rd_epic_cuda/bin/python"
"$PYTHON" scripts/submit_monitored.py --model gru
```

The wrapper evaluates automatically. A manual repeat is:

```bash
"$PYTHON" scripts/evaluate_sta_model.py \
  --checkpoint results/runs/RUN_ID/checkpoints/gru/best.pt
```

Model 2 is successful if it trains correctly and produces a reproducible result. It is a research comparison, so it is not required to outperform Model 1; a negative result must be reported honestly.

### Executed full-baseline results — 15-16 November 2026

The reported runs used the fixed shared configuration: seed `42`, AdamW,
learning rate `0.001`, weight decay `0.0001`, dropout `0.3`, batches of `256`
for training and `512` for validation, a 30-epoch maximum and early-stopping
patience `5`.

| Run | Model | Best epoch | Verb MT5R | Noun MT5R | Action MT5R |
|---|---|---:|---:|---:|---:|
| `20261115-152555-mean_pool-0cc225f3` | Mean pool | 6 | 0.1058 | 0.0767 | **0.0150** |
| `20261116-141102-gru-7e415c5d` | GRU reproduction | 9 | 0.1075 | 0.0563 | 0.0122 |
| `20261116-141325-gru-cf070b53` | GRU reported run | 9 | 0.1075 | 0.0563 | 0.0122 |

The reported GRU job was Slurm job `62715163`. It completed 14 epochs: epoch 9
was best, followed by five non-improving epochs that triggered early stopping.
Training took 69 seconds on one P100. Independent evaluation covered all `7,450`
validation events, used neural predictions for `7,426`, used the Model 0 fallback
for the expected `24`, found `3,568` valid action classes and passed all finite-
metric and coverage checks. Its additional metrics were:

| Metric | Verb | Noun | Action |
|---|---:|---:|---:|
| Top-1 accuracy | 0.2329 | 0.0752 | 0.0118 |
| Top-5 accuracy | 0.6638 | 0.2226 | 0.0549 |
| Class-mean Top-5 recall | 0.1075 | 0.0563 | 0.0122 |

The two GRU directories contain identical metrics because the experiment is
deterministic and both used seed `42`; they are repeated executions of the same
configuration rather than two independent-seed estimates.

The mean-pooling model is selected by validation Action MT5R. The GRU improves
Verb MT5R slightly (`0.1058` to `0.1075`) but reduces Noun MT5R (`0.0767` to
`0.0563`), producing an Action MT5R about `18.7%` lower than mean pooling. This
is a valid negative result: temporal ordering in the five frozen R3D-18 feature
vectors did not improve the primary metric, and recurrent compression likely
discarded static object evidence retained by mean pooling.

### VSC environment incident and correction

GRU job `62707239` failed after 19 seconds before training because the batch job
resolved the central Miniforge base interpreter, which did not contain PyTorch.
Its log reported `ModuleNotFoundError: No module named 'torch'`. The successful
mean-pooling job showed the actual environment interpreter to be
`$VSC_DATA/conda_envs/rd_epic_cuda/bin/python` with PyTorch `2.5.1+cu118`.

Commit `689ad45` first removed the central Miniforge module. After verifying the
real VSC paths, commit `08cdf39` made the final robust correction: every Slurm
entry point invokes the `rd_epic_cuda` interpreter explicitly, checks that it is
executable and never relies on a Conda distribution or shell activation. The
monitored submit wrapper now resolves both sides of its project-root comparison,
so the logical `/data/leuven/...` path and its `/vsc-hard-mounts/...` target are
treated as the same checkout.

### Monitoring and troubleshooting

Use:

```bash
squeue -M wice -u "$USER"
scontrol show job JOB_ID
tail -f logs/JOB_NAME-JOB_ID.out
```

Typical responses:

- `OUT_OF_MEMORY`: reduce event batch size from 4 to 2.
- `TIMEOUT`: resubmit extraction with `--resume`.
- `CUDA unavailable`: check the loaded cluster module and Conda CUDA installation.
- DataLoader worker crash: retry with `--workers 0`; if successful, investigate OpenCV multiprocessing.
- Missing video: run the data audit and repair the dataset before continuing.
- Disk quota error: inspect feature and duplicate video directories with `du`; do not delete anything until exact duplicates are confirmed.

### Dashboard verification

The Training Observatory UI has been rendered successfully, but the inspected
screenshot is the synthetic demonstration, not a live VSC snapshot. This is
confirmed by `DEMO — synthetic data`, job `84231`, run `demo-gru`, node
`demo-node`, output path `/DEMO/results/runs/demo-gru` and Git commit
`synthetic-demo`. The four summary cards and curves are correct for the demo
fixture, but they must not be interpreted as experiment results.

Keep demo and real records in separate SQLite databases. Start the real local
dashboard with a working SSH alias or `user@login-host` and the verified VSC
paths:

```powershell
python scripts/dashboard.py `
  --host YOUR_VSC_SSH_ALIAS `
  --remote-root /data/leuven/397/vsc39706/R-D_Group2 `
  --remote-python /data/leuven/397/vsc39706/conda_envs/rd_epic_cuda/bin/python `
  --db results/monitoring/vsc.sqlite3 `
  --interval 120
```

The real dashboard is confirmed only when the header reports cluster `wice` and
the job list contains actual Slurm IDs such as `62706675` and `62715163`, rather
than `842xx` demo IDs. If non-interactive SSH is unavailable, generate a snapshot
on VSC with `scripts/monitor_snapshot.py`, copy the JSON to the laptop and open it
with `--snapshot` while retaining the separate `vsc.sqlite3` database.

## 5. Model selection, final training and hidden-test submission

### Compare the three models

Generated comparison table:

| Model | Video | Temporal order | Verb MT5R | Noun MT5R | Action MT5R |
|---|---:|---:|---:|---:|---:|
| Frequency prior | No | No | 0.0694 | 0.0287 | 0.0027 |
| R3D-18 mean pooling | Yes | No | 0.1058 | 0.0767 | **0.0150** |
| R3D-18 GRU | Yes | Yes | 0.1075 | 0.0563 | 0.0122 |

Select the final model using validation Action MT5R only.

Do not select using hidden-test feedback.

The selected architecture is Model 1, masked mean pooling. It improves Action
MT5R over Model 0 by approximately `5.5x` and exceeds the GRU by approximately
`23%` relative to the GRU score (`0.0150 / 0.0122 - 1`).

### Retrain on all provided annotations

After choosing Model 1 or Model 2:

1. Combine the cached train and validation features logically; do not rewrite them.
2. Fit the frequency fallback on all 67,217 provided training events.
3. Train the selected architecture on all labelled features.
4. Use the selected validation run’s best epoch count as the fixed final epoch count.
5. Keep the same seed, optimizer and hyperparameters.
6. Save the final checkpoint and its Git commit hash.

### Hidden-test extraction

Run the same extractor with the provided hidden-test CSV:

```bash
sbatch slurm/sta_extract_a100.slurm test
```

Use `sta_observation_end_seconds` as the authoritative cutoff. Never infer or retrieve hidden labels.

All hidden events remain in output order. Events without context or with read errors receive frequency-prior scores.

### Submission generation

Generate:

```bash
python scripts/create_sta_submission.py \
  --checkpoint results/checkpoints/final_sta.pt \
  --output submission_sta.pt
```

Each entry must be:

```python
{
    "narration_id": str,
    "verb_output": float tensor [97],
    "noun_output": float tensor [300],
}
```

Validation before packaging:

- IDs exactly match the hidden-test CSV.
- No missing, duplicate or unexpected IDs.
- Every score is finite.
- Every verb vector has length 97.
- Every noun vector has length 300.
- Original hidden-test order is preserved.
- The file reloads with `torch.load`.

Package it together with the AR team’s file:

```text
submission.zip
├── submission.pt
└── submission_sta.pt
```

## Test Plan and acceptance criteria

### Unit tests

Add tests for:

- Action at 20 seconds produces segments 14-19 and all-true mask.
- Action at 4 seconds produces `[False, False, True, True, True]`.
- Action before one second produces an all-false mask.
- Every sampled frame is strictly before the cutoff.
- Feature extractor returns `[5,512]`.
- Invalid segments remain zero.
- Masked mean ignores padded segments.
- Masked GRU does not update its state on padded segments.
- Both models return `[B,97]` and `[B,300]`.
- No-context evaluation uses Model 0 probabilities.
- Checkpoint save/reload gives identical predictions.
- Submission validation rejects wrong shapes, IDs and non-finite values.

### Integration gates

1. [x] Existing Model 0 tests and metrics remain unchanged.
2. [x] Local eight-event extraction passes.
3. [x] VSC GPU smoke and reader checks complete.
4. [x] Both architectures pass training-loop and decreasing-loss checks.
5. [x] Full feature extraction covers all `59,767` train and `7,450` validation rows.
6. [x] Model 1 beats Model 0 on Action MT5R (`0.0150` versus `0.002742`).
7. [x] Model 2 produces a valid, independently evaluated and reproducible result.
8. [ ] Final all-label retraining and submission validation remain to be completed before Codabench upload.

## Team division and working order

The work is split at the cached-feature boundary. Niels owns the upstream path from
annotations and videos to validated feature shards. Emiel owns the downstream path
from those feature shards to checkpoints, metrics and the final submission. This
keeps the main handoff small and explicit:

```python
features: float tensor [batch, 5, 512]
valid_mask: bool tensor [batch, 5]
verb_labels: int tensor [batch]
noun_labels: int tensor [batch]
```

Both people should first agree on one eight-event example shard and verify that it
loads correctly before either side is considered stable.

### Niels — data and feature pipeline

Primary deliverable: reproducible, leak-free feature shards for train, validation
and hidden test.

- Implement `sta_segment_boundaries` and `sample_segment_frames`.
- Implement the video reader, partial-context padding and validity masks.
- Implement frozen R3D-18 preprocessing and feature extraction.
- Implement resumable shard writing, manifest generation and shard validation.
- Implement `STAFeatureDataset` and the feature-inspection command.
- Implement `scripts/audit_sta_data.py` and its split/video checks.
- Add unit tests for boundaries, frame selection, the one-second gap, padding,
  deterministic extraction and feature shapes.
- Create `environment-vsc.yml`, `slurm/sta_smoke_p100.slurm` and
  `slurm/sta_extract_a100.slurm`.
- Run the local eight-event extraction, VSC data audit, 32-event GPU smoke
  extraction and full train/validation/test extraction.
- Record extraction errors and make sure they are represented as `video_error`
  rather than silently dropping events.

Niels's handoff is complete when the inspection script validates every shard and
the documented counts, masks, IDs and leakage constraints all pass.

### Emiel — models, training, evaluation and submission

Primary deliverable: reproducible Model 1 and Model 2 results plus a validated
hidden-test submission.

- Implement masked mean pooling and the masked `GRUCell` model.
- Implement the shared trainer, optimizer, gradient clipping, early stopping and
  deterministic seeding.
- Implement checkpoint save/reload, history logging, configuration capture and
  Git commit recording.
- Implement verb, noun and joint-action Top-1, Top-5 and MT5R evaluation.
- Implement Model 0 fallback for all-false masks and `video_error` events while
  retaining every validation/test row.
- Add unit tests for masking, output shapes, fallback behavior, metric coverage,
  checkpoint round trips and submission validation.
- Create `slurm/sta_train_p100.slurm`.
- Run the 128-example overfit tests and full Model 1 and Model 2 training jobs.
- Generate the comparison table, select the model using validation Action MT5R,
  retrain it on all labelled features and create `submission_sta.pt`.
- Validate IDs, order, shapes and finite outputs, then help package the final ZIP
  with the AR submission.

Emiel's handoff is complete when both validation result files are reproducible,
the chosen final checkpoint is saved and the submission validator passes.

### Shared checkpoints and review

- Before coding: freeze the shard schema, label mapping and Model 0 probability
  interface together.
- After the local smoke test: Emiel reviews one shard and Niels reviews the model
  input/masking code.
- After the VSC smoke job: jointly confirm CUDA use, feature shapes, cutoff rules
  and forward passes through both models.
- Before full training: Niels signs off on feature coverage; Emiel signs off on a
  successful tiny overfit and checkpoint reload.
- After full training: inspect both learning curves, compare all three models and
  document failures or negative results honestly.
- Before submission: independently run the submission validator and verify the
  final archive contents.

### Recommended implementation order

1. Agree on interfaces and write the unit-test skeletons.
2. Work in parallel: Niels builds extraction using a synthetic/local tiny set;
   Emiel builds models and training using synthetic feature tensors.
3. Integrate on the eight-event local shard and fix interface mismatches.
4. Run the joint 32-event VSC smoke job.
5. Niels launches full feature extraction. While it runs, Emiel finishes the
   trainer, evaluator, logging and submission validator.
6. Emiel runs the tiny overfit tests, then full Model 1 and Model 2 training.
7. Jointly select, retrain and package the final model.

Use separate feature branches, review through pull requests and merge only after
tests pass. Suggested branches are `codex/niels-sta-features` and
`codex/emiel-sta-models`. Push code through Git and pull it on VSC; never copy raw
videos, feature shards or checkpoints into Git.

## Assumptions and fixed defaults

- VSC project path is `$VSC_DATA/R-D_Group2`.
- VSC account is `lp_edu_rdlab`.
- Compute uses one wICE P100 GPU.
- A new Conda environment named `rd_epic` is used.
- R3D-18 remains frozen for Models 1 and 2.
- Five one-second segments and 16 frames per segment are used.
- Cached features are deterministic and contain no augmentation.
- Model 1 uses masked mean pooling.
- Model 2 uses a one-layer masked GRU with hidden size 512.
- Initial baselines use unweighted cross-entropy.
- Participant split, action mapping and evaluator remain unchanged from Model 0.
- Class-balanced loss, RU-LSTM, temporal transformers and end-to-end encoder fine-tuning are later experiments, not part of Models 1 and 2.
