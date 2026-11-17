# Modern STA implementation and division of work

**Owners:** Niels and Emiel  
**Implementation window:** 17 November–1 December 2026  
**Course deadline:** 11 December 2026  
**Status:** Implementation plan; GPU feasibility and experimental results remain to be verified.

## 1. Objective and fixed decisions

Implement a **V-JEPA 2.1–based short-term action anticipation model**, using JFAA’s encoder–predictor approach as the architectural reference.

The system will:

1. Read legal pre-action video.
2. Extract representations with a frozen pretrained video encoder.
3. Predict future latent representations using its pretrained predictor.
4. Train an attention-based head to predict verbs and nouns, with an auxiliary action-pair objective.
5. Evaluate and export predictions using the course protocol.

The main model is **V-JEPA 2.1 ViT-G/384**, the approximately 2B-parameter variant. The official repository supplies pretrained weights. Use the self-supervised backbone checkpoint, **not a probe already trained on EPIC-KITCHENS labels**. [Official models](https://github.com/facebookresearch/vjepa2#models)

This is a **JFAA-inspired implementation adapted to the course protocol**. Do not describe it as an exact reproduction: the released JFAA configuration and data filtering differ from the required evaluation. [Released configuration](https://raw.githubusercontent.com/CorrineQiu/JFAA/main/configs/jfaa_vjepa2_vitG_train.yaml), [released loader](https://raw.githubusercontent.com/CorrineQiu/JFAA/main/evals/action_anticipation_frozen/epickitchens.py)

### Fixed scope

- STA only; AR remains outside this implementation.
- Video only.
- Develop on the provided training annotations using the existing participant split.
- Retain every validation and hidden-test event.
- Keep existing R3D-18 implementations and results as baselines.
- Run the modern backbone online initially; do not build a full dense-token cache.
- Freeze both encoder and predictor throughout these experiments.
- Start with one prediction head, rather than JFAA’s large collection of probe candidates.
- No backbone fine-tuning, VANE integration, language model, audio branch or ensembling in this two-week milestone.

### What counts as success

A complete implementation must produce:

- A reproducible GPU pilot with measured memory and throughput.
- A trained modern model and an observed-only ablation.
- Full-coverage validation metrics under the course protocol.
- Reloadable checkpoints and reproducible inference.
- A validated `submission_sta.pt`.
- A comparison against the existing baselines.

Beating the baseline is the experimental objective, not a guaranteed acceptance condition. Report negative results honestly.

## 2. Ownership and the shared interface

### Division of responsibility

| Area | Niels | Emiel |
|---|---|---|
| Event metadata, split integrity and video availability | Owner | Review |
| Legal frame sampling and video decoding | Owner | Review |
| Video preprocessing and augmentation | Owner | Consumer |
| Pretrained encoder and predictor integration | Owner | Consumer |
| GPU environment, backbone profiling and SLURM allocation | Owner | Training requirements |
| Shared tensor contract and fixtures | Owner | Co-review |
| Attention head and auxiliary action head | Review | Owner |
| Loss, optimizer and training loop | Support | Owner |
| Distributed training and checkpoint recovery | Infrastructure support | Owner |
| Evaluation, model selection and ablations | Data coverage review | Owner |
| Submission export and validation | Independent verification | Owner |

**Niels delivers video batches and frozen backbone representations. Emiel delivers the trainable model and everything from optimization to submission.**

Unlike the previous plan, Emiel does not wait for full-dataset feature extraction.

### Interface A: video dataset

Niels implements `STAClipDataset` and its collate function.

Each batch contains:

```python
{
    "video": FloatTensor[B, 3, 32, 384, 384],
    "has_context": BoolTensor[B],
    "verb_label": LongTensor[B],       # omitted for hidden test
    "noun_label": LongTensor[B],       # omitted for hidden test
    "row_index": LongTensor[B],
    "narration_id": list[str],
    "status": list[str],
    "cutoff_seconds": FloatTensor[B],
    "frame_indices": LongTensor[B, 32],
}
```

`status` is one of `full`, `partial`, `no_context`, or `video_error`.

`row_index` refers to the original event-table order and must survive every loader, distributed gather and export operation.

### Interface B: frozen backbone

Niels implements:

```python
class FrozenAnticipationBackbone(nn.Module):
    embed_dim: int

    def forward(
        self,
        video: Tensor,                 # [B, 3, 32, 384, 384]
        anticipation_seconds: Tensor,  # [B], always 1.0 here
    ) -> dict[str, Tensor]:
        ...
```

Output:

```python
{
    "observed_tokens": Tensor[B, N_observed, D],
    "future_tokens": Tensor[B, N_future, D],
}
```

Requirements:

- Both outputs use the same final embedding dimension `D`.
- Expose `embed_dim`; Emiel must not hardcode R3D-18’s 512 dimensions.
- Preserve the official predictor’s positional indexing and hierarchical feature handling.
- Predict one future step with two output frames, following the reference wrapper.
- Never load actual future video to produce `future_tokens`.
- In observed-only mode, bypass the predictor and return an empty future-token sequence.
- Document actual token counts and dimensions in the pilot report.

The upstream wrapper contains the relevant predictor and hierarchy handling. [Reference implementation](https://raw.githubusercontent.com/CorrineQiu/JFAA/main/evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar.py)

### Interface C: trainable head

Emiel implements:

```python
class AnticipationHead(nn.Module):
    def forward(
        self,
        observed_tokens: Tensor,
        future_tokens: Tensor,
    ) -> dict[str, Tensor]:
        ...
```

Output:

```python
{
    "verb_logits": Tensor[B, 97],
    "noun_logits": Tensor[B, 300],
    "action_logits": Tensor[B, A],
}
```

`A` is the number of action pairs observed in the **internal training split**, stored as a sorted, checkpointed mapping.

The primary evaluation and submission use the verb and noun outputs. The action output is an auxiliary training objective.

### Shared fixture

Before real GPU integration, Niels supplies an eight-event synthetic fixture covering:

- Full context.
- Partial context.
- No legal context.
- Video read failure.

It contains both a dataset-batch example and small synthetic backbone outputs. Emiel must be able to develop without downloading the pretrained checkpoint.

Synthetic fixture dimensions are configurable; the production dimension comes from the backbone.

## 3. Niels — exact implementation tasks

### N1. Isolate the modern environment and upstream dependencies

Add:

- `environment-sta2.yml`
- A source-provenance manifest.
- Namespaced upstream model dependencies.

Tasks:

1. Create a separate environment so the existing R3D-18 environment remains usable.
2. Pin the upstream V-JEPA and JFAA revisions.
3. Download the self-supervised ViT-G/384 checkpoint to VSC storage.
4. Record its source URL, SHA-256 checksum and checkpoint keys.
5. Save exact installed package versions after the successful GPU pilot.
6. Preserve upstream license notices.

**Import collision:** both this project and upstream V-JEPA use a top-level `src` package. Integrate the required upstream dependency closure under a dedicated package namespace and rewrite its internal imports accordingly. Do not resolve this through competing `PYTHONPATH` entries or global `sys.path` switching.

Acceptance:

- Existing project imports and upstream model imports work in the same Python process.
- Import smoke tests run from the repository root and from the training entrypoint.

### N2. Implement the legal video dataset

Add `src/sta2/data.py`.

Reuse existing annotation parsing, video indexing and participant-split utilities.

Sampling rules:

- Fixed anticipation gap: **1.0 second**, during both training and evaluation.
- **32 samples at 8 fps**, covering a nominal four-second observation window.
- For cutoff `c`, desired timestamps are:

```python
c - 4.0 + k / 8.0  # k = 0, ..., 31
```

- Convert each nonnegative timestamp to the nearest preceding source frame.
- Check the actual selected frame timestamp against the cutoff.
- Never select a frame at or after the cutoff.
- For hidden test, `sta_observation_end_seconds` is authoritative.
- Reject inconsistent metadata rather than silently changing the cutoff.
- Preserve source-frame timing from the provided video metadata; do not assume all videos have the same frame rate.

Early events:

- If some desired timestamps are negative but legal context exists, repeat the first legal frame for those positions and mark the event `partial`.
- If no frame exists strictly before the cutoff, return `no_context` without reading video.
- For unreadable video or failed requested-frame reads, return `video_error`.
- Invalid events return a zero placeholder video and `has_context=False`.
- Do not silently filter any event.

Use the provided 640 × 360 videos. Record that upscaling to model input size does not restore lost image detail.

### N3. Implement preprocessing

Training:

- One spatial transformation shared across all frames in a clip.
- Random resized crop to 384 × 384.
- Crop scale `[0.5, 1.0]`, aspect ratio `[3/4, 4/3]` (use the exact ratios).
- Horizontal flip probability `0.5`.
- No temporal jitter into the anticipation gap.

Validation and test:

- Resize the shorter side to `int(384 * 256 / 224) = 438`, using bilinear interpolation, then center crop as below. This follows the official single-view evaluation transform's resize-to-crop ratio. It retains a tighter spatial view than resizing the shorter side to 384; any accuracy benefit on the supplied downscaled videos remains to be measured.
- Center crop to 384 × 384.
- No random augmentation.

For all splits:

- Convert BGR decoding output to RGB.
- Convert pixels to floating point in `[0, 1]`.
- Use the official checkpoint preprocessing normalization: RGB mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`, applied after conversion to `[0, 1]`. Record these constants, resize interpolation, antialias behavior and the resolved validation resize size in the resolved configuration.
- Seed training transformations by seed, epoch and event index for reproducible resume behavior.

These augmentation choices are project defaults, not a claim of exact JFAA reproduction.

**Preprocessing decision, 17 November 2026:** Niels's implementation uses the upstream single-view validation resize convention (438 before a 384 crop), replacing the earlier shorter-side size of 384. The reason is alignment with the pretrained model's released evaluation preprocessing, not an observed validation gain. Keep training crop scale `[0.5, 1.0]` and flip probability `0.5` unchanged; do not import JFAA's temporal jitter or annotation filtering. Source: [official evaluation transforms](https://github.com/facebookresearch/vjepa2/blob/main/evals/video_classification_frozen/utils.py). Pin this source revision with the model dependencies.

### N4. Integrate and validate the backbone

Add `src/sta2/backbone.py`.

Tasks:

- Load encoder and predictor with the matching official architecture configuration.
- Verify checkpoint tensor names and shapes.
- Permit only documented namespace-prefix removal.
- Fail on unexplained missing weights or shape mismatches.
- Do not copy the reference loader’s behavior of silently retaining randomly initialized tensors when checkpoint shapes differ.
- Keep all backbone parameters frozen and the backbone in evaluation mode.
- Use BF16 autocast on supported GPUs.
- Run the frozen forward under `torch.no_grad()` so the head can subsequently train on its outputs.
- Keep the backbone in evaluation mode even when the enclosing system enters training mode.

Acceptance:

- Finite observed and future tokens.
- No backbone gradients.
- Backbone weights remain unchanged after a head optimizer step.
- A changed prediction horizon changes predictor positions without reading extra video.

### N5. Implement GPU profiling and launch infrastructure

Add:

- `scripts/profile_sta2.py`
- `slurm/sta2_smoke.slurm`
- `slurm/sta2_train.slurm`

Default distributed target: one node with four homogeneous GPUs. Make GPU count configurable so the same code supports one or two GPUs.

Each GPU stores one frozen backbone replica. **DDP does not split the backbone’s memory across GPUs.**

Profile:

- GPU model and memory.
- Peak allocated and reserved memory.
- Decode time.
- Backbone forward time.
- Head forward/backward time.
- End-to-end events per second.
- Observed-only versus observed-plus-future cost.
- Actual token shapes.

Use 64 representative events, including early and failed events, followed by at least 100 measured training steps after warm-up.

Before full jobs, write an estimated schedule for tuning, both main experiments and final retraining. If the model cannot fit on one GPU at microbatch size one, report the blocker; do not silently substitute a smaller backbone.

**Niels’s handoff is complete when the dataset, backbone and profile report pass the agreed checks.**

## 4. Emiel — exact implementation tasks

### E1. Implement the attention head

Add `src/sta2/head.py`.

Use the pinned reference attentive pooler:

- Three learned queries: verb, noun and action.
- Four pooler blocks.
- Sixteen attention heads.
- MLP ratio four.
- Separate linear classifiers for the three outputs.
- Activation checkpointing for trainable attention blocks.
- Memory-efficient attention where supported.

Concatenate observed and future tokens for the main model. Use only observed tokens for the ablation.

Preserve the upstream block structure rather than replacing it with a different generic Transformer. [Reference pooler](https://raw.githubusercontent.com/facebookresearch/vjepa2/main/src/models/attentive_pooler.py)

### E2. Implement the model wrapper and losses

Add `src/sta2/model.py`.

Tasks:

- Connect Niels’s backbone with the trainable head.
- Exclude unusable videos from backbone inference.
- Supply finite placeholder tokens to the head where needed for synchronized distributed execution.
- Mask invalid events from all losses.
- Skip the optimizer update when the entire distributed accumulation window has no usable events.

Initial loss:

```python
loss = verb_cross_entropy + noun_cross_entropy + action_cross_entropy
```

Use equal weights of `1.0`.

This deliberate project default simplifies the first integration and keeps the primary branches aligned with categorical prediction. Focal loss is outside the first two-week implementation.

For an action pair absent from the training action map:

- Exclude that event from the auxiliary action loss only.
- Keep its verb and noun supervision.
- Retain it in validation and test accounting.

### E3. Implement distributed training and checkpoints

Add:

- `src/sta2/training.py`
- `scripts/train_sta2.py`
- `configs/sta2/main.yaml`

Defaults:

| Setting | Value |
|---|---|
| Optimizer | AdamW, head parameters only |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Betas | `(0.9, 0.999)` |
| Maximum epochs | 20 |
| Warm-up | One epoch |
| Schedule | Cosine decay to zero |
| Gradient clipping | Norm `1.0` |
| Seed | 42 |
| Early stopping | Five epochs without improvement |
| Selection metric | Course-compatible validation Action MT5R |
| Precision | BF16 autocast; FP32 loss calculation |
| Per-GPU microbatch | One event |
| Effective global batch | 16 events |

Use gradient accumulation to retain global batch 16 with one, two or four GPUs.

Distributed requirements:

- `torchrun` and NCCL on one node.
- Wrap the trainable head in DDP; keep the frozen backbone outside DDP.
- Set the training sampler epoch.
- Ensure every rank takes the same number of backward and optimizer steps.
- Normalize losses by the global usable-example counts, including accumulation-window remainders.
- Handle ranks containing only invalid events without deadlocks.
- Gather validation predictions by `row_index`; remove sampler-padding duplicates before scoring.
- Log metrics and save checkpoints on rank zero.

Checkpoint contents:

- Head state.
- Optimizer and scheduler state.
- Epoch and optimizer-step count.
- RNG and sampler state needed for recovery.
- Resolved configuration.
- Backbone checksum and upstream revisions.
- Split hashes and action-label mapping.
- Best metric and early-stopping state.
- Project Git commit and dirty-state flag.

Save `last.pt` and `best.pt` atomically. Do not duplicate the 2B backbone in every training checkpoint.

Resume at completed epoch boundaries. A preempted partial epoch is rerun from the last completed checkpoint.

### E4. Implement course-compatible evaluation

Add `scripts/evaluate_sta2.py`.

Reuse the metric functions in the existing evaluation module.

Primary action scores must be:

```python
P(verb=v) * P(noun=n)
```

Use softmax probabilities from the two primary heads. Use the same valid-action vocabulary and ordering as the existing baseline evaluator, derived from the provided labelled annotation table. Store its identity in the run metadata.

Do not use the auxiliary action-head ranking for primary model selection.

Report:

- Verb, noun and action Top-1.
- Verb, noun and action Top-5.
- Verb, noun and action MT5R.
- Event counts by status.
- Prior-fallback count.
- Unsupported auxiliary action-label count.

For missing context or video failures, use training-only frequency priors.

Every event must contribute to the primary metrics, including pairs not observed in the internal training split.

Additional requirements:

- No filtering based on labels, class frequency or video availability.
- A fresh checkpoint reload must reproduce deterministic validation predictions.
- Save per-event predictions and their original IDs.
- Report MT5R on the `[0, 1]` scale; multiply by 100 only in presentation tables.
- Keep any auxiliary action-head diagnostic clearly separate.

### E5. Implement experiment comparison and final export

Add:

- `scripts/compare_sta2.py`
- `scripts/export_sta2.py`
- `scripts/validate_sta_submission.py`

Required comparison:

| Model | Role |
|---|---|
| Frequency prior | Nonvisual baseline |
| Existing R3D-18 mean pooling | Basic visual baseline |
| Existing R3D-18 GRU | Existing temporal baseline |
| V-JEPA 2.1 observed-only | Modern representation ablation |
| V-JEPA 2.1 observed-plus-future | Main model |

Evaluate every row on the same internal validation split and scoring protocol. Missing baseline results must be generated or explicitly marked unavailable.

Export the course format:

```python
[
    {
        "narration_id": str,
        "verb_output": FloatTensor[97],
        "noun_output": FloatTensor[300],
    },
    ...
]
```

Use CPU FP32 softmax probabilities and `torch.save`.

Do not export the auxiliary action vector or JFAA’s public-challenge JSON format.

Validator requirements:

- Exact event coverage.
- No duplicate or unexpected IDs.
- Original event order.
- Correct vector lengths.
- Finite floating-point values.
- Nonnegative probabilities summing approximately to one.
- Successful file reload.

**Emiel’s handoff is complete when training, independent evaluation and export are reproducible and the submission validator passes.**

## 5. Integration, experiments and acceptance tests

### Implementation sequence

| Period | Niels | Emiel | Shared gate |
|---|---|---|---|
| Days 1–2 | Environment, dataset and sampling tests | Synthetic head, losses and evaluator | Approve interfaces and fixture |
| Days 3–4 | Backbone and GPU profiling | Training loop and checkpoint recovery | 64-event GPU integration |
| Days 5–6 | Decode performance and distributed support | Tiny overfit and DDP verification | Training correctness |
| Days 7–10 | Monitor data coverage and GPU throughput | Tuning, main model and ablation | Validation comparison |
| Days 11–12 | Independent inference audit | Final retraining and export | Submission validation |
| Days 13–14 | Reproduction documentation | Tables and learning curves | Report-ready handoff |

Treat 1 December as the target for a working, evaluated STA system. Preserve the remaining time before 11 December for the report, poster and defense.

### GPU integration acceptance

The initial joint run must verify:

1. Correct checkpoint loading.
2. Legal selected-frame timestamps.
3. Finite backbone tokens.
4. Finite head outputs and loss.
5. A successful head update.
6. Unchanged backbone weights.
7. Correct fallback predictions.
8. Identical event coverage before and after distributed gathering.
9. Checkpoint reload and resumed training.
10. Measured resource use.

Do not launch full training until these checks pass.

### Tiny overfit test

Use 32 usable training events and deterministic preprocessing.

For this diagnostic only:

- Disable augmentation.
- Use FP32 for the head.
- Run at most 500 optimizer updates.
- Require a substantial loss decrease and at least 90% verb and noun training accuracy.

Failure blocks full training until investigated. This is a wiring diagnostic, not evidence of generalization.

### Experiment budget

1. On a fixed, seeded pilot subset, compare learning rates `1e-4` and `3e-4` for three epochs.
2. Keep weight decay and all other settings fixed.
3. Select using pilot validation Action MT5R; break ties in favor of `1e-4`.
4. Use the selected learning rate for both full modern experiments.
5. Train observed-only and observed-plus-future with the same epoch cap, seed and evaluation.
6. Select the final model by full internal validation Action MT5R.
7. Break exact ties in favor of observed-only because inference is cheaper.
8. Retrain the selected configuration from a fresh head on all provided labelled events for its selected best-epoch count.
9. Refit fallback priors and the auxiliary action map on that final training set.
10. Generate hidden-test predictions without using hidden-test labels or leaderboard feedback for selection.

Before scheduling the full runs, their measured combined ETA—including final retraining—must fit within the remaining experiment window with two days reserved for export and debugging. If not, report the measured shortfall before committing to full runs; do not silently change the backbone or evaluation protocol.

### Required automated tests

**Niels:**

- Exact cutoff and frame-rate rounding boundaries.
- Fractional frame rates.
- Partial-context padding.
- No-context handling without video reads.
- Decode failure retention.
- RGB ordering and clip-consistent transforms.
- Strict checkpoint-key and shape checking.
- Frozen-backbone behavior.
- Upstream/project import coexistence.

**Emiel:**

- Output dimensions and action-map ordering.
- Loss masking and empty-context behavior.
- Gradient updates confined to the head.
- Prior fallback and full event coverage.
- Course-compatible product-based action ranking.
- Auxiliary-unknown pairs retained in primary evaluation.
- Checkpoint round trip and epoch-boundary resume.
- Two-GPU training with an invalid-only rank.
- Distributed validation deduplication.
- Submission rejection for missing IDs, wrong shapes and nonfinite values.

Run the existing baseline regression suite alongside the new tests.

## 6. Artifacts and documentation

Each experiment gets its own directory under `results/sta2/<run_id>/`, containing:

- `resolved_config.yaml`
- `provenance.json`
- `profile.json`
- `history.csv`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `validation_metrics.json`
- Per-event validation predictions.
- Coverage and error reports.

Do not commit videos, checkpoints, token tensors or prediction artifacts to Git.

Niels documents:

- Sampling and preprocessing.
- Backbone checkpoint provenance.
- Environment reproduction.
- GPU allocation and measured costs.
- Input-resolution limitations.

Emiel documents:

- Architecture and losses.
- Distributed training and recovery.
- Hyperparameter selection.
- Evaluation and ablation results.
- Final model selection and submission validation.

Both independently verify the final STA export. Packaging it with the existing AR submission is a separate final integration step; uploading to Codabench is not part of these implementation scripts.

### Sources and course constraints

- [Official V-JEPA 2.1 code and pretrained models](https://github.com/facebookresearch/vjepa2)
- [JFAA paper](https://arxiv.org/abs/2605.20904)
- [JFAA implementation](https://github.com/CorrineQiu/JFAA)
- [Supplied course brief](reference/rd_project_2026_2027.pdf)

The supplied course brief defines the allowed training data, video-only restriction, anticipation boundary, submission format and 11 December deadline.
