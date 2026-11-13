# STA extraction overhead: findings and proposed fix

Date: 13 November 2026

This note records the original code investigation, a small local benchmark, and the implementation plan. Priority 1 is now implemented in both extraction paths; see [implementation validation and VSC rollout](sta_event_reader_rollout.md). Descriptions of the current reader below refer to the code before that change. VSC profiling and later optimisations remain follow-up work.

## Main finding

The current reader repeatedly opens a video and seeks to individual frames. We can reduce this work while selecting the same frames and keeping the same model inputs.

**First fix: open the video once per event, seek once to the first required frame, and read forward through the observation window.**

This is a pipeline optimisation. It does not require changing R3D-18, the feature format, the sampling rule, or the participant split.

## Where the overhead comes from

In [sta_feature_extraction.py](../src/sta_feature_extraction.py):

- `sample_segment_frames()` creates and releases a `cv2.VideoCapture` for each segment, around lines 117–139.
- Inside that function, every selected frame is read using a separate `capture.set(CAP_PROP_POS_FRAMES, ...)` followed by `capture.read()`, around lines 130–136.
- Both `extract_event_features()` and the batch path's `prepare_event()` call this reader separately for each of the five segments.

For a full-context event, this means:

| Operation | Current reader | Proposed event reader |
|---|---:|---:|
| Open the video | 5 times | 1 time |
| Explicitly seek | 80 times | 1 time |
| Frames selected for the encoder | 80 positions | The same 80 positions |

Seeking can reposition the decoder and cause repeated decoding of compressed video. A sequential reader still decodes intermediate frames, but avoids repeatedly jumping through the file. Only selected frames need RGB conversion and storage.

At 59,767 training events, the upper bounds are 298,835 video opens and 4,781,360 explicit seeks. Partial-context events reduce these counts. Overlapping events can also request the same video content repeatedly.

## What was measured

A local reader-only test used `data/raw/P01_03.mp4` with an example action starting at 20 seconds:

- Legal observation window: `[14, 19)` seconds.
- Five one-second segments, with 16 frame positions selected per segment.
- The existing `sample_segment_indices()` selected the indices for both methods.
- No encoder inference, GPU processing, shard writing, or cluster storage was included.
- Execution order was current reader, sequential reader, sequential reader, current reader.

| Method | First measurement | Second measurement |
|---|---:|---:|
| Current per-frame-seek reader | 1.1243 s | 1.0818 s |
| One-open, one-seek sequential reader | 0.0674 s | 0.0632 s |

The selected RGB frame arrays were **exactly equal**, with a maximum pixel difference of zero. The reading step was approximately **17 times faster on this one example**.

This is a small local benchmark, not a production throughput estimate. It does not establish a 17-fold speedup for the complete extraction or show which stage dominates the reported 15-hour training-feature extraction. Longer videos, different seek positions, codecs, storage, and concurrent readers need testing.

## Priority 1: implement one reader for the whole event

Add a shared event-frame preparation helper and use it in **both** `extract_event_features()` and `extract_event_batch()`. Updating only the single-event path would leave the production batch path unchanged.

Algorithm:

1. Call `sta_segment_boundaries(action_start)` as before.
2. If no segment is valid, return the existing `no_context` representation without opening a video.
3. Open one `VideoCapture`; read FPS and frame count once.
4. For every valid segment, call the existing `sample_segment_indices()` with the same arguments. Keep each segment's complete ordered list of indices.
5. Combine the requested indices into a sorted set of unique frame indices.
6. Seek once to the earliest requested index. Read sequentially through the latest requested index. Store RGB frames only at requested indices.
7. Reconstruct each segment's 16-frame array from its original index list. Preserve duplicate indices when short segments require repeated frames.
8. Release the capture in a `finally` block. Pass the prepared segments to the existing preprocessing and encoder code.

Conceptually:

```text
Current:
  for each segment:
      open video
      for each sampled frame:
          seek to frame
          read frame
      close video

Proposed:
  calculate all required frame indices for this event
  open video
  seek to the earliest required frame
  read forward, retaining the requested frames
  close video
  reconstruct the five segments in their original order
```

Store only the selected frames, not the whole decoded video. Keep decoder handles private to the event or worker; do not share one mutable `VideoCapture` across concurrent threads.

## Behaviour that must be preserved

- Same frame indices, chronological segment order, RGB conversion, and pretrained transforms.
- Every selected timestamp satisfies `frame_index / FPS < action_start - 1.0`.
- Same `[5, 512]` features and `[5]` validity mask per event.
- Missing segments remain zero-padded and invalid; repeated frame indices remain repeated.
- No-context events and unreadable-video events retain their existing status and handling.
- A read failure must not silently substitute another frame or drop an event.
- Event IDs, labels, CSV order, manifests, and shard boundaries remain consistent with [the shard contract](sta_feature_shard_contract.md).
- The encoder remains frozen and in evaluation mode for this first optimisation.

## Priority 2: profile the remaining stages on VSC

Measure a representative sample before deciding the next optimisation:

| Stage | Measure separately |
|---|---|
| Video preparation | Opening, seeking, decoding, and RGB conversion |
| Preprocessing | Tensor conversion, resizing, cropping, and normalisation |
| Transfer | Moving the prepared batch to the GPU |
| Encoder | R3D-18 forward pass |
| Output | Copying features back, converting storage dtype, and writing shards |

Use CUDA events or synchronise CUDA appropriately when timing GPU work. Timing only the Python call can miss asynchronous GPU execution. Record GPU utilisation, event throughput, batch size, worker count, hardware, and code revision.

Also compare encoder-only throughput using already-prepared batches with complete pipeline throughput. This helps identify whether the GPU is waiting for data.

## Later optimisations, guided by profiling

### Overlap CPU preparation with GPU inference

Currently, `extract_event_batch()` waits for the whole batch to be prepared before encoding it. CPU decoding of the next batch is not prefetched during the current batch's GPU inference.

Use a persistent worker pool and a bounded prefetch queue so batch `k+1` can be prepared while batch `k` is encoded. Keep output order deterministic and limit queued batches to control memory. Reusing the worker pool also avoids recreating it for every batch.

### Reuse decoded frames across nearby events

After the simpler event-reader change is validated, consider processing nearby events from the same video together. A bounded, worker-local frame cache could avoid decoding overlapping windows repeatedly. Restore the original CSV event order before writing shards.

This is a larger change and needs explicit limits on memory, open handles, and cache lifetime. Do not cache an entire long recording by default.

### Test mixed-precision inference separately

The current `--dtype float16` option controls feature storage. It does not enable mixed-precision encoder inference: the inspected code uses `torch.inference_mode()` without `autocast`.

If profiling shows substantial GPU compute time, benchmark CUDA autocast against the existing precision. Check finite outputs, numerical differences, and downstream validation results. Reduced precision may change features, so treat it as a separate experiment rather than an exact-equivalence reader optimisation.

Likewise, a new encoder, different frames, or a different context length changes the feature definition. Use separate output roots for these experiments; the current resume check does not identify every possible extraction-configuration change.

## Validation and rollout

1. Extend `tests/test_sta_feature_extraction.py` and the sampling regressions to compare old and new readers on full, partial, and no-context events, repeated indices, missing files, and read failures.
2. Compare sampled indices and decoded pixels at early and late positions in several available videos. Include non-integer FPS cases and check the strict anticipation cutoff.
3. Check the actual batched path with workers enabled, including event order and status preservation.
4. Compare encoder outputs under unchanged preprocessing, precision, and batch arrangement. Investigate any differences instead of assuming matching tensor shapes are enough.
5. Run a small VSC comparison into a separate output directory. Measure complete throughput and stage timings, not just decoder time.
6. Inspect and verify the resulting shards against the source annotations. Preserve all 7,450 validation events and all 59,767 training events when running the full splits.
7. Only after these checks, use the optimised reader for subsequent extraction. Retain existing completed shards and their provenance; avoid changing files underneath an active job.

Success means **less measured decoding overhead and better end-to-end throughput, with the same selected inputs, cutoff rules, event coverage, and feature contract**. The full-job speedup must be measured on VSC.

## Short explanation for the team

> “Our reader was reopening videos for each segment and seeking separately to every frame. In one local test, opening once and reading forward gave exactly the same sampled frames and reduced reading time from about 1.1 seconds to 0.065 seconds. We should validate that reader change, then measure the full pipeline on VSC to see how much extraction time it saves.”
