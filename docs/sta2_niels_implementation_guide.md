# Niels's manual STA2 implementation walkthrough

Written 17 November 2026 against `docs/plan2.md` and project commit `99e330e`.

**Implementation update (17 November):** The modules below now exist. The actual
configuration lives at `configs/sta2/data_backbone.yaml` and controls the clip
transform settings. The backbone uses memory-mapped checkpoint loading. Metadata
generation automatically resumes matching manifests: verified successes are
reused, failed/changed/corrupt entries are retried, and different annotation or
video-info inputs require a separate output directory. Do not run two writers
against the same metadata directory. The A100 smoke script uses private Miniconda
and scratch data paths. `slurm/sta2_train.slurm` supports one/two/four GPUs but
requires Emiel's training entrypoint and configuration. Set
`STA2_PROFILE_ADAPTER=module:factory` when submitting the smoke job to profile
Emiel's real head; this also verifies backbone state hashes before/after updates.
Without an adapter, production head costs and state-update verification remain
explicitly pending. The later sections preserve the original implementation
walkthrough; use the current scripts' `--help` for supported options.

This is a guide for implementing the work yourself. The new Python modules, environment, fixture generator and SLURM scripts described below have **not** been implemented by writing this document. Commands invoking those new files are acceptance commands to run **after** you create them. Existing utilities and upstream sources were inspected; the checkpoint has not been downloaded and no STA2 GPU pilot has run.

Your deliverable is **event table → legal RGB clips → frozen observed/future tokens → measured GPU infrastructure**. Emiel implements the attention head, losses, training loop, evaluation and export. You supply launch infrastructure and independently audit coverage and the final export. A temporary diagnostic linear layer in a test is fine; it is not the production head.

## 1. Understand the changes to the plan

Use the updated preprocessing in `plan2.md`:

- Training: random resized crop to 384, scale `(0.5, 1.0)`, ratio `(3/4, 4/3)`, horizontal flip probability 0.5. One random crop and flip decision per clip.
- Validation/test: bilinear resize of the shorter side to **438**, then center crop 384. The official single-view transform computes `int(crop_size * 256 / 224)`. This is alignment with upstream preprocessing, not proof of improved accuracy on our data.
- RGB mean `(0.485, 0.456, 0.406)`; standard deviation `(0.229, 0.224, 0.225)` after dividing pixels by 255.
- Record interpolation, antialias setting, library versions and resize size. For a tensor implementation, choose `InterpolationMode.BILINEAR` and `antialias=True` explicitly. This is a resolved project implementation choice, not a claim of pixel equality with the upstream PIL/video implementation.

Keep the four-second observation window, one-second gap, 32 samples at 8 Hz, backbone size, split and full event coverage fixed. The downscaled input remains 640 × 360; interpolation cannot recover missing detail.

## 2. Start an isolated branch and capture the baseline

On your Mac, from the project root:

```bash
git status --short
git switch -c codex/sta2-niels
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Preserve any existing changes before switching branches. The branch command is for you to execute; this guide has not created it. Record pre-existing test failures so that new failures can be distinguished. Run the existing suite again at the end in the original environment as well as the modern environment where compatible.

Create this layout incrementally:

```text
environment-sta2.yml
configs/sta2/data_backbone.yaml
docs/sta2_source_provenance.json
docs/sta2_niels_handoff.md
src/sta2/__init__.py
src/sta2/data.py
src/sta2/transforms.py
src/sta2/backbone.py
src/sta2/fixtures.py
src/sta2/vendor/__init__.py
src/sta2/vendor/vjepa21/...
scripts/create_sta2_fixture.py
scripts/audit_sta2_data.py
scripts/profile_sta2.py
slurm/sta2_smoke.slurm
slurm/sta2_train.slurm
tests/test_sta2_data.py
tests/test_sta2_transforms.py
tests/test_sta2_backbone.py
tests/test_sta2_imports.py
```

The extra transform, fixture and audit files are small organizational choices within your ownership. Keep CPU imports lightweight: importing `src.sta2.data` must not construct a 2B-parameter model or require a downloaded checkpoint.

## 3. Audit the data before changing sampling

Read these existing functions:

| Location | Reuse or learn from |
|---|---|
| `src/data_processing.py` | `timestamp_to_seconds`, `index_videos`, `participant_split`, annotation discovery |
| `scripts/audit_sta_data.py` | `validate_split_integrity` |
| `scripts/extract_sta_features.py` | Separate labelled/test schemas and authoritative test cutoff |
| `src/sta_feature_extraction.py` | `prepare_event_frames`: one capture, one seek, read forward, reconstruct repeats, release decoder |
| `tests/test_sta_sampling_regressions.py` | Boundary tests at fractional frame rates |

Do not use `match_available_annotations`: it drops events. Do not use `sta_segment_boundaries`, `sample_segment_indices` or the R3D transforms as the STA2 sampling/preprocessing implementation. They describe the older five-segment pipeline. The old audit's context counts also use that older window; reuse split integrity checks, then implement STA2 context counting separately.

The existing split files are:

```text
data/processed/EPIC_100_train_participant_split.csv
data/processed/EPIC_100_validation_participant_split.csv
```

Load the original labelled table and both split tables. Verify unique narration IDs, disjoint train/validation IDs, disjoint participants, and exact union equal to the original table. Preserve existing membership. Historical scripts expect 59,767 training and 7,450 validation events; derive counts from the actual files and investigate differences instead of forcing those constants.

Assign `row_index = 0..N-1` to the **original source table before split selection**. Join split membership by narration ID. Avoid resetting this identifier after shuffling or taking a subset. For hidden test, index the hidden-test source table from zero independently. Save table identity and original CSV checksum with the index convention. Emiel must know whether an ID belongs to the labelled or hidden table.

Use Python's `hashlib.sha256(path.read_bytes()).hexdigest()` for CSV checksums. Hash the ordered narration IDs of each split as well, with a documented separator/encoding. Store the hashes in the run provenance.

Locate `EPIC_100_video_info.csv` in the supplied data on VSC. It is not present among this checkout's local raw files inspected for this guide. Inspect its header before writing a parser; do not invent column names. Build a per-video metadata record with frame rate, frame count when supplied, frame-index origin and timing source. Cross-check against the downscaled MP4. If source metadata describes different frame counts/rates from the provided videos, investigate and document the mapping before sampling.

Create `scripts/audit_sta2_data.py` to accept `--annotations`, `--train-split`, `--validation-split`, `--video-info`, `--raw-dir`, and `--output`. Produce counts, split hashes, missing video IDs and metadata inconsistencies. Missing videos remain events. Invalid annotation/timing metadata is a hard audit error; a corrupt or unreadable file is an event-level `video_error`.

## 4. Create a separate VSC environment

Do model loading on an allocated GPU node. Keep the baseline `rd_epic_cuda` environment.

Create `environment-sta2.yml` with environment name `rd_sta2`, conda-forge, Python 3.11 and pip. As a **provisional compatibility starting point**, retain the project's existing `torch==2.5.1+cu118`, `torchvision==0.20.1+cu118`, `numpy==1.26.4`, `pandas==2.2.3`, `opencv-python-headless==4.10.0.84` and `tqdm==4.66.6`. Add `PyYAML`, `timm`, and `einops`; resolve and pin their installed versions before the GPU pilot. Use the same PyTorch extra index as `environment-vsc.yml`. This environment is not yet validated for the new backbone.

The required model dependency closure imports torch, timm and einops; avoid installing the upstream training stack wholesale. If a newer PyTorch/CUDA pair is needed, use a matched torch/torchvision pair supported by the allocated node's driver, update only the modern environment, and record the reason.

On VSC, using the module convention in the existing SLURM files:

```bash
module load Miniforge3/25.3.0-3
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$VSC_DATA/R-D_Group2"
conda env create -f environment-sta2.yml
conda activate rd_sta2
python -m pip check
```

Treat that module name and repository path as the project's current templates. Check availability on VSC before submitting; this guide did not query the cluster. On an allocated GPU, print torch/torchvision versions, `torch.version.cuda`, CUDA availability, GPU model, total memory and `torch.cuda.is_bf16_supported()`. Fail a requested BF16 pilot if unsupported rather than silently changing precision.

After the successful pilot:

```bash
mkdir -p results/sta2/pilot
python -m pip freeze > results/sta2/pilot/pip-freeze.txt
conda env export > results/sta2/pilot/environment-resolved.yml
```

Keep the reproducible, portable pins in the committed environment file; machine-specific export paths belong in the run artifact.

## 5. Pin upstream source revisions

Clone sources outside the Python import path on VSC:

```bash
mkdir -p "$VSC_DATA/sta2-upstream"
git clone https://github.com/facebookresearch/vjepa2.git "$VSC_DATA/sta2-upstream/vjepa2"
git clone https://github.com/CorrineQiu/JFAA.git "$VSC_DATA/sta2-upstream/JFAA"
git -C "$VSC_DATA/sta2-upstream/vjepa2" rev-parse HEAD
git -C "$VSC_DATA/sta2-upstream/JFAA" rev-parse HEAD
```

Record both full 40-character SHAs immediately in `docs/sta2_source_provenance.json`. Use those exact revisions for later reproduction. Do not record `main` as a pin. Capture URLs, the reviewed reference configuration path, vendored source-file hashes, import rewrites and license-file locations. Future `git checkout --detach <recorded-sha>` commands must use the recorded real values, not placeholder text.

Review these at the pinned revisions:

```text
V-JEPA: app/vjepa_2_1/models/vision_transformer.py
V-JEPA: app/vjepa_2_1/models/predictor.py
V-JEPA: src/hub/backbones.py
V-JEPA: evals/video_classification_frozen/utils.py
JFAA: configs/jfaa_vjepa2_vitG_train.yaml
JFAA: evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar.py
```

Upstream branches can change between reading this guide and implementation. If imports/configuration differ at your pin, update the closure and document the difference.

## 6. Vendor the model dependency closure without `src` collisions

Use this mapping under `src/sta2/vendor/vjepa21/`:

```text
app/vjepa_2_1/models/vision_transformer.py → models/vision_transformer.py
app/vjepa_2_1/models/predictor.py          → models/predictor.py
app/vjepa_2_1/models/utils/modules.py     → models/utils/modules.py
app/vjepa_2_1/models/utils/patch_embed.py → models/utils/patch_embed.py
src/masks/utils.py                       → masks/utils.py
src/utils/tensors.py                     → utils/tensors.py
```

These six files form the observed model import closure in the sources checked for this guide. Add `__init__.py` files to each package directory. Copy the files with their copyright headers. Retain the applicable root license notices, including `LICENSE` and `APACHE-LICENSE` where present, in the vendor directory. Preserve JFAA notices for any adapted wrapper code too.

Rewrite only internal imports in the vendored model files:

```text
src.masks.utils                    → src.sta2.vendor.vjepa21.masks.utils
src.utils.tensors                  → src.sta2.vendor.vjepa21.utils.tensors
app.vjepa_2_1.models.utils.modules  → src.sta2.vendor.vjepa21.models.utils.modules
app.vjepa_2_1.models.utils.patch_embed
                                  → src.sta2.vendor.vjepa21.models.utils.patch_embed
```

Check remaining imports with `rg -n '^(from|import) ' src/sta2/vendor`. There must be no dependency on a competing upstream top-level `src` or `app` package. Do not use `torch.hub.load` as an import workaround, or add the upstream repositories to `PYTHONPATH`.

Run an import-only smoke test that imports `src.data_processing`, `src.sta_feature_extraction`, and both namespaced upstream model modules in the same process. Print `__file__` for each. Repeat in a subprocess through `scripts/profile_sta2.py --import-smoke` after implementing that mode, and through Emiel's training entrypoint once available. The entrypoint may follow the existing convention of adding **this repository root** to `sys.path`; that is distinct from switching between competing upstream roots.

## 7. Download and inspect the correct checkpoint

The official model table distinguishes uppercase **ViT-G (2B)** from lowercase **ViT-g (1B)**. Download the self-supervised uppercase model:

```bash
mkdir -p "$VSC_DATA/sta2/checkpoints"
curl --fail --location --retry 3 \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt \
  --output "$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt.part"
sha256sum "$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt.part"
mv "$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt.part" \
  "$VSC_DATA/sta2/checkpoints/vjepa2_1_vitG_384.pt"
```

Run sequentially and only rename after a successful download. A locally computed checksum records identity; it does not establish an independently published checksum match. Never put the checkpoint in Git.

In a sufficiently sized CPU/GPU allocation, load on CPU and record the top-level keys, then tensor names and shapes under `target_encoder` and `predictor`. Use `torch.load(..., map_location="cpu", weights_only=True)` first. If the official artifact contains non-tensor serialization requiring another loading policy, inspect and document that compatibility issue explicitly. Do not print tensor contents. Discard other training-state entries before moving modules to GPU to reduce peak host memory.

Add the source URL, actual SHA-256, byte count, actual checkpoint keys and source revisions to the provenance manifest. Do not invent these values before download.

## 8. Implement pure timestamp sampling first

In `src/sta2/data.py`, separate these responsibilities:

```python
resolve_cutoff(row, split) -> float
sample_frame_indices(cutoff, frame_timestamps) -> sampling_result
decode_requested_frames(video_path, indices) -> RGB_uint8_array
```

`sampling_result` should hold exactly 32 zero-based decoder indices, selected timestamps, and status. Keep full-precision timing internally; create the contract's float32 cutoff tensor only when assembling the returned sample.

For labelled data, cutoff is `timestamp_to_seconds(start_timestamp) - 1.0`. For test, use `sta_observation_end_seconds` directly. If both start and cutoff metadata are present and inconsistent with the one-second protocol, raise a descriptive error. Negative cutoffs are legitimate early events and lead to `no_context`; NaN/infinite timing is malformed metadata.

Construct desired times as:

```python
desired = cutoff - 4.0 + np.arange(32, dtype=np.float64) / 8.0
```

The last desired sample is `cutoff - 0.125`, not the cutoff. Do not use a linspace that includes the endpoint.

Build or obtain an authoritative monotonic frame timestamp sequence. For verified constant-rate video starting at time zero, it is `np.arange(frame_count, dtype=np.float64) / fps`, with that video's precise rate. Preserve rational rates such as `60000/1001` when available. For variable-rate/nonzero-origin media, use actual presentation timestamps and the documented timeline mapping; average FPS alone is insufficient. A per-video timestamp index is metadata, not a dense token cache. If unavailable, stop the audit until timing is resolved.

The algorithm is:

1. Find source frames with timestamp **strictly below** cutoff. If none exist, return `no_context`; do not open a decoder.
2. For each nonnegative desired time, use the rightmost source timestamp `<= desired`. `np.searchsorted(timestamps, desired, side="right") - 1` expresses this when timestamps are sorted.
3. For desired times below zero, repeat the earliest legal frame and set `partial`.
4. Reject unexplained gaps before the first frame or targets beyond the available video interval rather than silently extending video with its last frame. An incomplete/truncated file becomes `video_error`; inconsistent timing metadata is a hard audit error.
5. Assert every selected source timestamp is below cutoff, every index is in range, and indices are nondecreasing.
6. For normal zero-origin media, use `full` when all desired times are nonnegative and all reads succeed; otherwise `partial` for the prescribed left padding.

Avoid multiplying by a rounded integer FPS. Avoid adding an arbitrary epsilon to permit a frame across the cutoff. For constant-rate arithmetic, test neighboring candidate indices against reconstructed source timestamps to correct floating-point boundary errors while preserving the preceding-frame rule.

Examples at 8 FPS with first frame time zero:

| Cutoff | Desired times | Status |
|---|---|---|
| 6.0 s | 2.0 through 5.875 | full |
| 0.5 s | -3.5 through 0.375 | partial; initial positions repeat frame 0 |
| 0.01 s | -3.99 through -0.115 | partial; all positions can repeat legal frame 0 |
| 0.0 s | all negative | no_context; frame 0 is not strictly before cutoff |
| -0.2 s | all negative | no_context |

## 9. Implement decoding with explicit failure retention

Use the existing reader's one-seek/read-forward structure, but feed it the new indices:

1. Deduplicate the requested indices for reading.
2. Open one decoder per event/worker; never share a mutable `VideoCapture` across processes.
3. Seek to the first requested frame and verify the backend's positioning semantics. If OpenCV cannot establish correct positions for the media, use a timestamp-aware backend for that media.
4. Read forward to the final requested frame, retaining only requested frames.
5. Convert each retained BGR frame to RGB exactly once.
6. Reconstruct the original 32 positions, including repeated padding indices.
7. Release resources in `finally` on every exit path.

Check source metadata/frame positions against the actual downscaled media in the pilot. Keyframe seeking alone does not prove that the first decoded frame has the requested identity.

Missing file, failed open/seek/read, or truncated requested context yields `video_error`, zero video and `has_context=False`. Preserve the event's labels, row index and narration ID. Keep an error-detail record outside the fixed batch contract. Catch expected media I/O failures near the decoder; do not turn programming errors or invalid annotation schemas into video errors.

## 10. Implement clip-consistent preprocessing

Use `src/sta2/transforms.py`. Input is `[32,H,W,3]` RGB uint8; output is contiguous float32 `[3,32,384,384]`.

For training, sample crop area/aspect once, derive one `(top,left,height,width)` rectangle with the standard random-resized-crop fallback, and apply it to the entire `[T,C,H,W]` tensor. Sample one flip decision. Use torchvision functional `resized_crop`/`hflip` with explicit bilinear interpolation and antialiasing. Do not call a random transform independently for each frame.

For validation/test, resize the shorter side to 438 while preserving aspect ratio; then center crop 384. With a 640 × 360 source, a torchvision tensor resize with integer size 438 gives approximately 778 × 438 before cropping. Verify the actual rounded dimensions in your implementation. Make the crop deterministic.

Convert to float in `[0,1]`, normalize along the RGB channel, and permute to `[C,T,H,W]`. Normalized output is not constrained to `[0,1]`. For unusable events, bypass augmentation/normalization and return an **exact zero placeholder in final model-input space**.

Seed training augmentation from `(base_seed, epoch, original_row_index)` using a stable numeric scheme such as `np.random.SeedSequence`; do not use Python's process-randomized `hash()`. Use a local generator for crop and flip decisions. Verify identical results after resume and with different DataLoader worker counts.

Implement `dataset.set_epoch(epoch)`. Initially use `persistent_workers=False` so new epoch state reaches worker copies. If enabling persistent workers later, use shared epoch state or another explicit mechanism; mutating the parent dataset alone does not update existing worker copies. Tell Emiel to call both sampler and dataset epoch setters.

## 11. Assemble `STAClipDataset` and the collate function

Recommended constructor:

```python
STAClipDataset(
    events, videos_by_id, video_metadata,
    *, split, training=False, seed=42,
)
```

Validate metadata in initialization, preserve every input row, and make `__len__` exactly the selected event count. Make `__getitem__` follow this order: cutoff → timing/sampling → early no-context return → decoding → transform → assemble sample. Keep GPU operations out of DataLoader workers.

Implement `collate_sta_clips(samples)` to produce:

```text
video             float32 [B,3,32,384,384]
has_context       bool    [B]
verb_label        int64   [B]    labelled splits only
noun_label        int64   [B]    labelled splits only
row_index         int64   [B]
narration_id      list[str]
status            list[str]
cutoff_seconds    float32 [B]
frame_indices     int64   [B,32]
```

Omit labels entirely for test. Reject mixed labelled/unlabelled samples in the same batch. Assert the same ordering across every field. Define `frame_indices=-1` when no indices could be selected; retain attempted legal indices after a subsequent decode failure, and document that convention. `has_context` must be true exactly for `full` and `partial`.

One production float32 video is 54 MiB before intermediate allocations. An eight-event fixture batch is 432 MiB; keep CI fixtures small by default. Excessive worker prefetch can consume substantial host memory. Start with two workers and `prefetch_factor=1`, measure, then tune.

## 12. Deliver the eight-event fixture early

Implement `src/sta2/fixtures.py` and a generator script, with configurable height/width, embedding dimension and token counts. Generate exactly two events of each status, fixed IDs, nontrivial original row indices, deterministic labels, cutoffs, and indices.

Provide two modes:

- Small development tensors, for example spatial size 32, D=64, 16 observed tokens and four future tokens. Clearly label these as synthetic dimensions.
- Production-shaped dataset batch with 32 frames at 384, generated only on request. Small synthetic token tensors can remain separate; they must not pretend to be real ViT-G output.

Return a dictionary containing `batch`, `backbone_outputs` and explicit fixture metadata. Invalid video placeholders are zero and masks/statuses agree. Include an observed-only fixture with `[B,0,D]` future tokens. Add labelled and unlabelled variants.

Design this proposed command after implementing the CLI:

```bash
python scripts/create_sta2_fixture.py \
  --output results/sta2/fixtures/eight_events.pt \
  --height 32 --width 32 --embed-dim 64 \
  --observed-tokens 16 --future-tokens 4
```

Give Emiel the generator and contract immediately. Generated `.pt` fixtures stay outside Git. Unit tests should generate them in temporary directories and require no pretrained download.

## 13. Load encoder and predictor strictly

Create `src/sta2/backbone.py` with separate architecture construction, checkpoint validation and forward logic. Start from the pinned JFAA architecture settings and cross-check the official V-JEPA 2.1 ViT-G factory. Strip configuration-only fields such as `checkpoint_key` before passing kwargs into constructors.

Use the **V-JEPA 2.1** namespaced modules, not the older V-JEPA 2 modules. Resolve all constructor settings into `configs/sta2/data_backbone.yaml`, including image size, input frames, patch/tubelet size, RoPE, predictor depth/width, hierarchy settings and mask-token count. The reference uses encoder key `target_encoder` and predictor key `predictor`; verify those against the actual artifact.

Implement a strict loader:

1. Extract the selected state dictionary.
2. Remove only documented leading namespace prefixes, such as `module.` and `backbone.`, anchored at the start of keys.
3. Reject key collisions introduced by removal.
4. Compare the complete expected and supplied key sets.
5. Compare every tensor shape.
6. Report all missing, unexpected and mismatched keys clearly and fail.
7. Call `load_state_dict(..., strict=True)` after the comparison.

Do not inherit the reference wrapper's replacement of mismatched tensors with newly initialized model values. If an architectural buffer genuinely needs a special documented rule, make the exact allowlist and justification explicit; never add a broad `strict=False` escape hatch.

Record the actual `encoder.embed_dim` and expose it as `backbone.embed_dim`. The inspected uppercase ViT-G factory defines D=1664, but the runtime value is authoritative and Emiel must read the attribute.

## 14. Implement frozen observed/future forward behavior

The public API is the one in `plan2.md`. Add an `observed_only` constructor option. Validate video shape and finite nonnegative horizon vector of shape `[B]`. Production callers always pass 1.0 seconds. Freeze every parameter with `requires_grad_(False)` and call `eval()`.

Override `train` so a surrounding `model.train()` cannot enable backbone dropout or training behavior:

```python
def train(self, mode=True):
    super().train(False)
    return self
```

Use `torch.no_grad()` around the forward, with BF16 autocast on supported CUDA GPUs. Avoid `inference_mode()` here: its tensors have restrictions when a downstream trainable layer saves inputs for backward. The trainable head's forward and backward must occur **outside** your no-grad block.

Preserve hierarchical information for the predictor. Expose final-layer-width tokens to the head in **both** modes; the upstream observed-only shortcut must not accidentally return a wider hierarchical vector. Treat tuple predictor output as future prediction plus optional predicted context; return the future component through the contract.

Keep positional indexing identical to the pinned reference. As an arithmetic sanity check, with 384 input, patch 16 and tubelet 2:

```text
spatial grid = 24 × 24 = 576 tokens per tubelet
observed sequence = (32 / 2) × 576 = 9,216 patch tokens
two output frames = one tubelet = 576 predicted tokens
one-second gap = (1.0 × 8 / 2) = four tubelet positions
first future position ID = 9,216 + 4 × 576 = 11,520
```

These are inferred expected patch-token counts, **not measured outputs**. Verify them against the pinned wrapper and real forward; do not remove unexpected tokens just to match the arithmetic. Keep the predictor's configured temporal capacity and its bounds checks.

Return separate `observed_tokens` and `future_tokens`, both `[B,N,D]`. In observed-only mode do not call the predictor and return `observed.new_empty(B,0,D)`. Future tokens come only from prediction over observed representations and future position IDs; the backbone API should have no video path or decoder access.

## 15. Write your automated tests before a large GPU run

Use small deterministic fake encoders/predictors for most backbone tests, not randomly initialized full-size ViT-G. Keep real-checkpoint tests explicit and GPU-gated.

| Test file | Required checks |
|---|---|
| `test_sta2_data.py` | Exact boundaries; 29.97, 30000/1001 and 60000/1001 rates; desired-time preceding frame; strict cutoff; partial repeats; c≤0 and tiny positive c; hidden-test authority; conflicting metadata rejection; no decoder call for no-context; missing/open/seek/read failures retain rows; collate order/dtypes; original IDs after subset/shuffle |
| `test_sta2_transforms.py` | Known red/blue BGR conversion; normalization numeric values; identical input frames remain identical after one shared crop/flip; same seed/epoch/row repeats exactly; validation deterministic; final shape; invalid final zeros |
| `test_sta2_backbone.py` | Strict missing/unexpected/shape failures; prefix collision rejection; equal output D; finite tokens; observed-only never calls predictor; hierarchy reaches predictor; horizon changes target IDs; train() remains eval; all parameter gradients absent |
| `test_sta2_imports.py` | Project and upstream modules coexist from root and profiling entrypoint |

For freezing behavior, attach a tiny diagnostic linear layer to detached backbone output, compute loss outside no-grad and take one optimizer step using **only layer parameters**. Require changed layer weights, no backbone gradients, and unchanged backbone parameters/buffers. A complete before/after state hash avoids allocating a second full GPU backbone; hash tensors in chunks on CPU if needed. This is a correctness check, not a proxy for production-head memory.

Run:

```bash
python -m unittest discover -s tests -p 'test_sta2_*.py' -v
python -m unittest discover -s tests -v
```

## 16. Implement a deterministic 64-event GPU pilot

Give `scripts/profile_sta2.py` the following proposed flags:

```text
--annotations --train-split --validation-split --video-info --raw-dir
--checkpoint --config --output-dir --seed
--events 64 --warmup-steps 10 --measured-steps 100
--microbatch 1 --workers 2 --mode observed-only|observed-plus-future
--head-checkpoint (optional, if integrating Emiel's head)
--import-smoke
```

Freeze the selected 64 IDs in a pilot manifest. Select across participants, videos and action times, deliberately including partial and no-context events and any real known decode failures. If real failures are absent, add a separately labelled fault-injection check; do not count fabricated failures as measured real coverage. Keep enough usable events to exercise the backbone. Cycle deterministically over the fixed subset to reach at least 100 measured steps after warm-up.

Measure both modes separately, ideally in fresh processes. Include:

- GPU name, total device memory, BF16 support, package versions and driver/runtime information.
- Peak allocated and reserved GPU memory, including backbone weight residency.
- Decode/preprocess time and host-to-device time.
- Encoder/predictor forward time and actual tensor shapes, including internal hierarchy width.
- Head forward/backward and optimizer time using Emiel's real head when available.
- End-to-end events/second, usable events/second, per-status counts and sample count.

Synchronize CUDA when using wall clocks for GPU sections or use CUDA events. Reset peak statistics after warm-up and record the baseline resident allocation. Do not count enqueued asynchronous operations as completed work. Timing `next(loader)` measures loader waiting under prefetch, not the full independent decode cost; add a worker-free decoder microbenchmark or instrumentation and label each measurement correctly.

Before Emiel's head is ready, report backbone throughput and explicitly mark production head time/memory and training ETA as pending. A tiny test layer establishes gradient wiring only. Complete the integrated measurement once his real head exists; do not declare the handoff fully complete from a cheap substitute head.

Save `profile.json`, `provenance.json`, `resolved_config.yaml`, coverage/error report, pilot IDs and a readable report under `results/sta2/<run_id>/`. If the full backbone cannot fit at microbatch one on a single allocated GPU, report that blocker with measured memory; four DDP replicas will not solve per-GPU capacity.

## 17. Add the smoke SLURM job

Start from `slurm/sta_extract_h100.slurm`. Preserve its account/module conventions, but verify current partition/account availability on the cluster. Choose a short enough wall time based on your first load/forward timing; 100 modern-model steps may require more than a debug partition allows.

Set one node, one task, one GPU, suitable host RAM (the existing H100 extraction template requests 96,000 MB), and a log path under `logs/`. Activate `rd_sta2`, print Git revision and `nvidia-smi`, then call the new profiler. A proposed command after the CLI exists:

```bash
python scripts/profile_sta2.py \
  --annotations data/raw/EPIC_100_train.csv \
  --train-split data/processed/EPIC_100_train_participant_split.csv \
  --validation-split data/processed/EPIC_100_validation_participant_split.csv \
  --video-info "$STA2_VIDEO_INFO" \
  --raw-dir "$STA2_RAW_DIR" \
  --checkpoint "$STA2_CHECKPOINT" \
  --config configs/sta2/data_backbone.yaml \
  --output-dir "results/sta2/pilot-${SLURM_JOB_ID}" \
  --seed 42 --events 64 --warmup-steps 10 --measured-steps 100 \
  --microbatch 1 --workers 2 --mode observed-plus-future
```

Set `STA2_RAW_DIR` to the verified complete/repaired video root, `STA2_VIDEO_INFO` to the supplied metadata table and `STA2_CHECKPOINT` to the downloaded model. Validate those variables at script start using `: "${STA2_CHECKPOINT:?set STA2_CHECKPOINT}"` and corresponding checks for the others. Do not use a raw directory just because an older script names it.

Submit only after files exist and CPU checks pass:

```bash
mkdir -p logs
sbatch slurm/sta2_smoke.slurm
squeue -u "$USER"
```

Read the log and profile artifacts before progressing. Repeat observed-only with the same IDs and configuration. Do not download checkpoints independently inside each GPU rank.

## 18. Add configurable single-node launch infrastructure

`slurm/sta2_train.slurm` is your launcher; `scripts/train_sta2.py` and training logic belong to Emiel. Agree on his CLI and configuration filename before wiring the final command. Until then, a guarded launcher should report the missing entrypoint/config explicitly.

Use one SLURM task running one `torchrun` launcher, with a default four GPUs on one node. Permit one/two GPUs by overriding the GPU allocation and `STA2_GPUS`, and check the requested process count against visible devices. Conceptual command after Emiel's entrypoint exists:

```bash
srun --ntasks=1 torchrun --standalone --nnodes=1 \
  --nproc_per_node="$STA2_GPUS" \
  scripts/train_sta2.py --config configs/sta2/main.yaml
```

Use the final `--config` convention agreed with Emiel. A one-task allocation should not also spawn four independent `torchrun` launchers. Each process selects `LOCAL_RANK`; Emiel wraps only the trainable head in DDP. Each rank has a complete frozen backbone replica.

The effective global batch requirement is 16 at microbatch one:

| GPUs | Accumulation steps |
|---|---|
| 1 | 16 |
| 2 | 8 |
| 4 | 4 |

Your responsibilities are allocation, correct rank/device launch, CPU worker budget, storage paths, environment and profiling. Emiel owns synchronized backward steps, global valid-example normalization, checkpoint resume and distributed deduplication. Support his two-GPU invalid-only-rank test before the full four-GPU job.

## 19. Calculate the experiment schedule from measurements

Use throughput measured with the real head and intended GPU count. For end-to-end throughput `r` over all events, one epoch is approximately `N_train / r_train + N_validation / r_validation`, plus checkpoint/logging overhead. Include fallback events consistently. Do not multiply single-GPU throughput by four without measuring scaling.

Calculate separate costs for observed-only and observed-plus-future. Include the two three-epoch learning-rate pilots, both full experiments at the maximum 20 epochs as an upper bound, and final retraining on all labelled events. Estimate final retraining with all-labelled event count and the selected best-epoch count once available. Include queue delays, restarts and two days for export/debugging.

Write the resulting hours and planned GPU allocations to the pilot report before full submissions. If the measured budget does not fit the implementation window through 1 December, report the shortfall to Emiel and agree on a revised plan; do not change backbone or evaluation silently.

## 20. Complete the handoff and independent audit

In `docs/sta2_niels_handoff.md`, record the API, code revision, environment, source revisions, checkpoint hash/path, metadata/split identities, preprocessing constants, frame-index origin and sentinel convention, actual token shapes, precision, GPU memory, throughput and exact successful launch command. Link to the generated results directory without committing large artifacts.

Your handoff checklist:

- Every event survives dataset iteration, with a traceable original row index.
- Timing is verified for actual downscaled media; selected frames are strictly legal.
- Partial/no-context/video-error cases match the contract.
- Synthetic fixture runs on CPU without the checkpoint.
- Project/upstream imports coexist in the real entrypoint.
- Encoder and predictor load strictly with known provenance.
- Finite real observed/future outputs have the same final D.
- Predictor is bypassed in observed-only mode.
- Backbone remains frozen/eval after a head update.
- Both modes have measured GPU cost; production head cost is recorded when available.
- One/two/four-GPU infrastructure agrees with Emiel's training API.
- Existing baseline regression tests still pass.

Later, independently verify Emiel's hidden-test export against the original hidden table: identical ID set, no duplicates, original order, CPU FP32 vectors of lengths 97/300, finite nonnegative normalized probabilities and successful reload. Review fallback count against your data coverage report. You are checking his export, not implementing or uploading his submission pipeline.

Suggested commits, scoped to your files: environment/provenance and vendor closure; legal dataset and sampling tests; clip transforms and fixture; frozen backbone and strict loading tests; profiling/SLURM/handoff. Stage explicit paths rather than `git add .` in shared work.

## Source references

- [Project implementation plan](plan2.md).
- [Official model table and checkpoint links](https://github.com/facebookresearch/vjepa2#models).
- [Official V-JEPA 2.1 encoder](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/vision_transformer.py).
- [Official V-JEPA 2.1 predictor](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/predictor.py).
- [Official model factories](https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py).
- [Official evaluation preprocessing](https://github.com/facebookresearch/vjepa2/blob/main/evals/video_classification_frozen/utils.py).
- [JFAA reference wrapper](https://github.com/CorrineQiu/JFAA/blob/main/evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar.py).
- [JFAA reference architecture configuration](https://github.com/CorrineQiu/JFAA/blob/main/configs/jfaa_vjepa2_vitG_train.yaml).

These web links were inspected while preparing the guide. During implementation, replace moving-branch references in provenance with the exact pinned commit URLs. Architecture compatibility, checkpoint contents, GPU fit and model quality remain acceptance work, not established results.
