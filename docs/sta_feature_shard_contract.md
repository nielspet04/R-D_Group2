# STA feature-shard contract (version 1)

This file is the handoff agreement between Niels's feature-extraction pipeline and
Emiel's model-training pipeline. The executable source of truth is
`src/sta_features.py`.

## Directory and ordering

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

Shards are read in numeric filename order and rows are read in tensor order. This
order must match the source split CSV. The last shard may contain fewer than 2,048
events. A narration ID may occur only once in a complete split.

`manifest.csv` has one row per shard:

```text
schema_version,split,shard_index,shard_file,start_index,end_index_exclusive,event_count
1,train,0,shard_00000.pt,0,2048,2048
1,train,1,shard_00001.pt,2048,4096,2048
```

`start_index` and `end_index_exclusive` refer to positions in the source split CSV.
This makes ordering and resume checks unambiguous.

## Contents of each `.pt` file

Each shard is a dictionary saved with `torch.save`. All tensors are stored on CPU.

| Key | Type and shape | Rule |
|---|---|---|
| `schema_version` | Python `int` | Must be `1`. |
| `split` | Python `str` | `train`, `validation` or `test`. |
| `features` | `float16 [N,5,512]` | Finite R3D-18 features; invalid segments are exactly zero. |
| `valid_mask` | `bool [N,5]` | `True` only for segments containing legal video context. |
| `verb_labels` | `int64 [N]` | Required for train/validation; absent for test. Range `0..96`. |
| `noun_labels` | `int64 [N]` | Required for train/validation; absent for test. Range `0..299`. |
| `narration_ids` | `list[str]` length `N` | Non-empty and unique within the shard. |
| `video_ids` | `list[str]` length `N` | Source video ID for each event. |
| `action_start_seconds` | `float32 [N]` | Action start from the annotation. |
| `observation_end_seconds` | `float32 [N]` | Authoritative legal cutoff; never later than action start minus one second. |
| `status` | `list[str]` length `N` | `full`, `partial`, `no_context` or `video_error`. |

Status and mask must agree:

- `full`: all five mask values are true.
- `partial`: between one and four mask values are true.
- `no_context`: all mask values are false because no legal context exists.
- `video_error`: all mask values are false; evaluation uses the Model 0 fallback.

Training converts `features` to `float32` after loading. It must not change the
stored shard. All-false events remain in the dataset but are excluded from gradient
updates and use Model 0 probabilities during evaluation or submission generation.

## First integration check

After installing PyTorch, Niels or Emiel creates the agreed deterministic example:

```bash
python scripts/create_sta_example_shard.py
```

This produces:

```text
results/local_sta_contract/train/shard_00000.pt
```

It contains eight artificial events covering full context, partial context, no
context and a video error. It is only an interface test, not extracted video data.

Run the tests with:

```bash
python -m unittest tests.test_sta_features -v
```

Once this passes in both environments, Niels can replace the artificial values
with extracted R3D-18 features while Emiel can safely build the trainer against the
same loader and validator.
