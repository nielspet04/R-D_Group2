# Event-level STA reader: implementation and rollout

Date: 13 November 2026

The first optimisation in [the extraction overhead note](sta_extraction_overhead_and_fix.md)
is implemented in `src/sta_feature_extraction.py`. Both single-event extraction and
the production batch path call `prepare_event_frames()`. Each event opens its own
capture, samples with the existing index function, seeks once, and reads forward.
Only requested frames are converted to RGB and retained. Segment reconstruction
preserves repeated indices. All decoder exit paths release the capture; a failed
seek or read becomes an event error through the existing extraction error policy.
No-context events do not open a video.

The encoder, transforms, precision, event order, masks, statuses and shard format
are unchanged. The original `sample_segment_frames()` remains available as the
per-segment reader and equivalence reference. Prefetching, caching and mixed
precision are separate follow-ups, after profiling on VSC.

## Local validation

`tests/test_sta_feature_extraction.py` and the sampling regressions passed all 25
tests, including:

- One capture and one seek, full/partial/no context, repeated indices, and exact
  cutoff checks at 10, 29.97 and 50 FPS.
- Failed open, seek and decode, capture cleanup, and a synthetic MJPEG video at
  29.97 FPS.
- Worker-enabled batches preserving event order, status, zero padding and features.
- Exact pretrained R3D-18 feature equality against the old reader with identical
  batch arrangement and preprocessing; encoder input tensors also match exactly.

All 11 downstream shard-contract, dataset and inspection tests also passed. The
inspection subprocess required running outside the local sandbox because OpenMP
could not allocate shared memory inside it.

Additional reader-only checks on the two readable local EPIC videos gave identical
RGB pixels for every selected frame:

| Video | Action start (s) | Old reader (s) | Event reader (s) |
|---|---:|---:|---:|
| P01_03 | 4.50 | 0.4172 | 0.0574 |
| P01_03 | 20.00 | 1.8822 | 0.1010 |
| P01_03 | 95.08 | 0.9827 | 0.0989 |
| P01_04 | 4.50 | 1.4809 | 0.0622 |
| P01_04 | 20.00 | 1.3194 | 0.0976 |
| P01_04 | 84.19 | 1.4668 | 0.1083 |

Both videos report 59.940060 FPS. These are single measurements per case, old
reader first, on local storage; they are not an end-to-end or VSC speed estimate.
Local P01_02 and P01_06 files could not be opened (missing MP4 moov atom) and were
excluded from pixel comparisons.

## Switch the VSC job

The existing extraction SLURM scripts automatically use the new reader once the
updated source is deployed. Keep event batch size 8, workers 8, shard size 2048
and the same annotation CSV when resuming.

1. Stop the old extraction job before deploying updated source or starting another
   writer against its output directory. A running Python process will not adopt
   the new code merely because its source file changes.
2. Deploy the changed source to `$VSC_DATA/R-D_Group2`, recording its revision.
3. On an allocated GPU node with the existing CUDA environment active, run a short
   extraction into a fresh, separate directory. For the A100 scratch layout:

```bash
cd "$VSC_DATA/R-D_Group2"
time python scripts/extract_sta_features.py \
  --raw-dir "$VSC_SCRATCH/epic_sta/raw" \
  --split train \
  --output-root data/processed/sta_reader_check_20261113 \
  --limit 64 --shard-size 64 \
  --device cuda --event-batch-size 8 --workers 8 --dtype float16

python scripts/inspect_sta_features.py \
  --feature-root data/processed/sta_reader_check_20261113 --split train --check-all

python scripts/verify_sta_extraction.py \
  --feature-root data/processed/sta_reader_check_20261113 \
  --split train --events 64 --shard-size 64 --limit
```

Use the actual raw-video directory on H100. Compare elapsed time with the old
reader on the same annotations and hardware using a different output root. Stage
profiling and full VSC throughput measurement remain to be done on the cluster.

4. After the check passes, submit the existing script for the intended GPU:

```bash
sbatch slurm/sta_extract_a100.slurm train
# Or, for an H100 allocation:
# sbatch slurm/sta_extract_h100.slurm train
```

Both scripts already use `--resume`. Completed valid shards are retained; an
interrupted shard is recomputed. Shards containing `video_error` are retried by
the existing resume logic. Keep the old job logs to retain provenance for the
shards that it completed. The scripts inspect and verify the full split after
extraction (59,767 training events or 7,450 validation events).
