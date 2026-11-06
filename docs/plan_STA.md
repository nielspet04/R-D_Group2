# STA Models 1 and 2 Full Implementation and VSC Training Plan

## Summary

Model 0 is complete and committed. It predicts training-label frequencies without reading video and obtains an Action MT5R of `0.0027`. The next objective is to build two video-based models using exactly the same split and evaluator:

| Model | Input | Temporal processing | Purpose |
|---|---|---|---|
| Model 0 | Training-label frequencies | None | Minimum baseline |
| Model 1 | Frozen R3D-18 features | Masked mean pooling | Test whether video helps |
| Model 2 | Same frozen R3D-18 features | Masked GRU | Test whether temporal order helps |

Both new models will use cached features. R3D-18 processes the raw videos once on VSC; subsequent model training will be fast and will not repeatedly decode the videos.

The pipeline will retain every event:

- Full context: use all five one-second segments.
- Partial context: zero-pad unavailable early segments and store a validity mask.
- No legal context or unreadable video: use Model 0 probabilities as fallback.
- Hidden-test events are never removed.

The current split contains:

| Split | Total | Full five seconds | Partial context | No context |
|---|---:|---:|---:|---:|
| Train | 59,767 | 58,895 | 796 | 76 |
| Validation | 7,450 | 7,307 | 119 | 24 |

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
python scripts/evaluate_sta_prior.py
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
Environment name: rd_epic
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

Use the `pytorch`, `nvidia` and `conda-forge` channels. CUDA 11.8 is chosen for compatibility with the P100 GPUs.

On the VSC login node:

```bash
cd "$VSC_DATA/R-D_Group2"
git pull --ff-only

source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda env create -f environment-vsc.yml
conda activate rd_epic

export TORCH_HOME="$VSC_DATA/torch-cache"
mkdir -p "$TORCH_HOME"
```

Download and verify the encoder weights once:

```bash
python -c "
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
conda activate rd_epic

python scripts/audit_sta_data.py
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
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda activate rd_epic
export TORCH_HOME="$VSC_DATA/torch-cache"
cd "$VSC_DATA/R-D_Group2"
```

Do not train or extract features directly on a login node.

### GPU smoke job

Create `slurm/sta_smoke.slurm` with:

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
sbatch slurm/sta_smoke.slurm
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

Create `slurm/sta_extract.slurm` with:

```text
partition:      gpu_p100
time:           48:00:00
GPUs:           1
CPU cores:      8
event batch:    4
decoder workers: 4
shard size:     2048
```

Submit training and validation separately:

```bash
sbatch slurm/sta_extract.slurm train
sbatch slurm/sta_extract.slurm validation
```

The extraction command inside the job is:

```bash
python scripts/extract_sta_features.py \
  --split "$1" \
  --device cuda \
  --event-batch-size 4 \
  --workers 4 \
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

Create `slurm/sta_train.slurm` with:

```text
partition: gpu_p100
time:      04:00:00
GPUs:      1
CPUs:      4
```

Submit:

```bash
sbatch slurm/sta_train.slurm mean_pool
```

After completion:

```bash
python scripts/evaluate_sta_model.py \
  --checkpoint results/checkpoints/mean_pool/best.pt
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
sbatch slurm/sta_train.slurm gru
```

Evaluate identically:

```bash
python scripts/evaluate_sta_model.py \
  --checkpoint results/checkpoints/gru/best.pt
```

Model 2 is successful if it trains correctly and produces a reproducible result. It is a research comparison, so it is not required to outperform Model 1; a negative result must be reported honestly.

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

## 5. Model selection, final training and hidden-test submission

### Compare the three models

Create one generated comparison table:

| Model | Video | Temporal order | Verb MT5R | Noun MT5R | Action MT5R |
|---|---:|---:|---:|---:|---:|
| Frequency prior | No | No | 0.0694 | 0.0287 | 0.0027 |
| R3D-18 mean pooling | Yes | No | result | result | result |
| R3D-18 GRU | Yes | Yes | result | result | result |

Select the final model using validation Action MT5R only.

Do not select using hidden-test feedback.

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
sbatch slurm/sta_extract.slurm test
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

1. Existing Model 0 tests and metrics remain unchanged.
2. Local eight-event extraction passes.
3. VSC 32-event GPU smoke job completes.
4. Both models overfit a tiny feature subset sufficiently to show decreasing loss.
5. Full feature extraction covers every split row.
6. Model 1 beats Model 0 on Action MT5R.
7. Model 2 produces a valid comparable result.
8. Final submission passes the local validator before Codabench upload.

## Team division and working order

### Person 1

- STA temporal-window and frame-sampling logic.
- Data audit and feature extraction.
- Leakage tests.
- Evaluation, fallback behavior and submission validation.
- VSC feature-extraction jobs.

### Person 2

- Mean-pooling and GRU models.
- Shared training loop.
- Checkpointing and logging.
- VSC training jobs.

### Shared

- Review tensor interfaces before implementation.
- Run the 32-event smoke job together.
- Inspect Model 1 and Model 2 learning curves.
- Compare results and qualitative examples.
- Document failures and final model choice.

Use separate feature branches, review through pull requests and merge only after tests pass. Push code through Git and pull it on VSC; never copy raw videos, feature shards or checkpoints into Git.

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
