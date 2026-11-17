"""Profile legal STA2 inputs and the frozen backbone on one GPU."""

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sta2.backbone import FrozenAnticipationBackbone, state_sha256
from src.sta2.data import (
    STAClipDataset,
    collate_sta_clips,
    load_events,
    resolve_cutoff,
    sha256_file,
)


def git_output(*arguments):
    return subprocess.check_output(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def choose_pilot(dataset, count, seed):
    rng = np.random.default_rng(seed)

    groups = {
        "no_context": [],
        "partial": [],
        "video_error": [],
        "full": [],
    }

    for index, row in enumerate(dataset.events):
        cutoff = resolve_cutoff(row, dataset.split)
        metadata = dataset.metadata[row["video_id"]]

        if cutoff <= 0:
            group = "no_context"
        elif metadata["status"] != "ok":
            group = "video_error"
        elif cutoff < 4:
            group = "partial"
        else:
            group = "full"

        groups[group].append(index)

    selected = []

    # Include early/error groups when they exist.
    for values in groups.values():
        shuffled = rng.permutation(values)
        selected.extend(
            int(value)
            for value in shuffled[:min(8, len(values))]
        )

    remaining = np.array(
        sorted(set(range(len(dataset))) - set(selected)),
        dtype=np.int64,
    )

    selected.extend(
        int(value)
        for value in rng.permutation(remaining)[
            :max(0, count - len(selected))
        ]
    )

    return selected[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--annotations", required=True)
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--validation-split", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument("--events", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--observed-only",
        action="store_true",
    )

    parser.add_argument(
        "--adapter",
        help=(
            "Optional Emiel-owned module:factory. "
            "factory(embed_dim, device) returns "
            "(head, optimizer, loss_function). "
            "loss_function(head_output, usable_batch) "
            "returns the real scalar training loss."
        ),
    )

    args = parser.parse_args()

    if args.events < 1 or args.measured_steps < 100:
        raise ValueError(
            "Use a positive pilot size and at least 100 measured steps"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This pilot requires BF16 support")

    config = yaml.safe_load(
        Path(args.config).read_text()
    )
    args.seed = config.get("seed", 42) if args.seed is None else args.seed
    config["seed"] = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    events = load_events(
        args.annotations,
        split="train",
        train_split=args.train_split,
        validation_split=args.validation_split,
    )

    dataset = STAClipDataset(
        events,
        args.raw_dir,
        args.metadata,
        split="train",
        training=True,
        seed=args.seed,
        preprocessing=config.get("preprocessing"),
    )
    config["preprocessing"] = dataset.transform.preprocessing

    indices = choose_pilot(
        dataset,
        args.events,
        args.seed,
    )

    if len(indices) != args.events:
        raise ValueError("Not enough events for the pilot")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cpu_decode = []
    cpu_transform = []
    unique_statuses = Counter()
    usable_count = 0

    # Worker-free pass separates decode and spatial transform cost.
    for index in indices:
        sample = dataset[index]

        unique_statuses[sample["status"]] += 1

        if sample["has_context"].item():
            usable_count += 1
            cpu_decode.append(
                dataset.last_decode_seconds
            )
            cpu_transform.append(
                dataset.last_transform_seconds
            )

        del sample

    if usable_count == 0:
        raise RuntimeError(
            "The pilot contains no usable video"
        )

    subset = Subset(dataset, indices)

    loader_options = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": False,
        "collate_fn": collate_sta_clips,
    }

    if args.workers:
        loader_options["prefetch_factor"] = 1

    loader = DataLoader(
        subset,
        **loader_options,
    )

    device = torch.device("cuda:0")

    print("Loading frozen backbone with memory-mapped checkpoint...", flush=True)
    backbone = (
        FrozenAnticipationBackbone.from_checkpoint(
            args.checkpoint,
            config,
            observed_only=args.observed_only,
            bf16=True,
        )
        .to(device)
    )

    head = optimizer = loss_function = None

    if args.adapter:
        module_name, factory_name = (
            args.adapter.split(":", 1)
        )

        factory = getattr(
            importlib.import_module(module_name),
            factory_name,
        )

        head, optimizer, loss_function = factory(
            backbone.embed_dim,
            device,
        )

        head.train()

        backbone_ids = {
            id(parameter)
            for parameter in backbone.parameters()
        }

        optimized_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

        if backbone_ids & optimized_ids:
            raise ValueError(
                "Optimizer contains backbone parameters"
            )

    def infinite_batches():
        while True:
            yield from loader

    batches = iter(infinite_batches())

    # Outside timing: a full streaming digest checks values, including buffers.
    if head is not None:
        print("Hashing backbone state before head updates (outside timing)...", flush=True)
    state_before = state_sha256(backbone) if head is not None else None

    def perform_step(batch):
        valid = batch["has_context"]

        if not valid.any().item():
            return None, 0.0, 0.0

        video = batch["video"][valid].to(
            device,
            non_blocking=True,
        )

        horizons = torch.ones(
            len(video),
            device=device,
        )

        torch.cuda.synchronize()
        started = time.perf_counter()

        tokens = backbone(video, horizons)

        torch.cuda.synchronize()
        backbone_seconds = (
            time.perf_counter() - started
        )

        head_seconds = 0.0

        if head is not None:
            usable_batch = {
                key: value[valid].to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
                and key != "video"
            }

            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            started = time.perf_counter()

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                outputs = head(
                    tokens["observed_tokens"],
                    tokens["future_tokens"],
                )

            # The adapter owns real loss construction and FP32 policy.
            loss = loss_function(
                outputs,
                usable_batch,
            )

            if loss.ndim != 0 or not torch.isfinite(loss):
                raise RuntimeError(
                    "Adapter returned an invalid scalar loss"
                )

            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            head_seconds = (
                time.perf_counter() - started
            )

        shapes = {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in tokens.items()
        }

        return shapes, backbone_seconds, head_seconds

    warmup_completed = 0

    while warmup_completed < args.warmup_steps:
        result, _, _ = perform_step(next(batches))

        if result is not None:
            warmup_completed += 1

    torch.cuda.synchronize()

    resident_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    backbone_times = []
    head_times = []

    attempted_events = 0
    usable_steps = 0
    measured_statuses = Counter()
    shapes = None

    started = time.perf_counter()

    # Count usable forward/training steps, not invalid-only iterations.
    while usable_steps < args.measured_steps:
        batch = next(batches)

        attempted_events += len(batch["status"])
        measured_statuses.update(batch["status"])

        result, backbone_seconds, head_seconds = (
            perform_step(batch)
        )

        if result is not None:
            shapes = result
            usable_steps += 1
            backbone_times.append(backbone_seconds)

            if head is not None:
                head_times.append(head_seconds)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # Capture memory statistics before verification transfers.
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if head is not None:
        print("Verifying backbone state after head updates (outside timing)...", flush=True)
    state_after = state_sha256(backbone) if head is not None else None
    if state_before is not None and state_before != state_after:
        raise RuntimeError("Backbone state changed during head optimization")

    if any(
        parameter.requires_grad
        or parameter.grad is not None
        for parameter in backbone.parameters()
    ):
        raise RuntimeError(
            "Backbone is not fully frozen"
        )

    properties = torch.cuda.get_device_properties(0)

    profile = {
        "mode": (
            "observed_only"
            if args.observed_only
            else "observed_plus_future"
        ),
        "head_profile": (
            "production_adapter"
            if head is not None
            else "pending"
        ),
        "gpu": properties.name,
        "gpu_total_bytes": properties.total_memory,
        "resident_after_warmup_bytes": resident_bytes,
        "peak_allocated_bytes":
            peak_allocated,
        "peak_reserved_bytes":
            peak_reserved,
        "backbone_state_verified_unchanged": (
            state_before == state_after if state_before is not None else None
        ),
        "backbone_state_sha256": state_after,
        "pilot_unique_events": len(indices),
        "pilot_status_counts": dict(unique_statuses),
        "measured_status_counts": dict(measured_statuses),
        "measured_usable_steps": usable_steps,
        "measured_attempted_events": attempted_events,
        "elapsed_seconds": elapsed,
        "events_per_second": attempted_events / elapsed,
        "usable_events_per_second": usable_steps / elapsed,
        "mean_decode_seconds": float(
            np.mean(cpu_decode)
        ),
        "mean_preprocessing_seconds": float(
            np.mean(cpu_transform)
        ),
        "mean_backbone_seconds": float(
            np.mean(backbone_times)
        ),
        "mean_head_step_seconds": (
            float(np.mean(head_times))
            if head_times
            else None
        ),
        "token_outputs": shapes,
    }

    print("Recording checkpoint checksum; this reads the full checkpoint...", flush=True)
    provenance = {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(
            git_output("status", "--porcelain")
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "checkpoint_sha256":
            sha256_file(args.checkpoint),
        "annotations_sha256":
            sha256_file(args.annotations),
        "train_split_sha256":
            sha256_file(args.train_split),
        "validation_split_sha256":
            sha256_file(args.validation_split),
        "metadata_manifest_sha256":
            sha256_file(args.metadata),
        "arguments": {
            key: str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        },
    }

    source_provenance = Path(
        "configs/sta2/source_provenance.json"
    )

    if source_provenance.exists():
        provenance["upstream"] = json.loads(
            source_provenance.read_text()
        )

    pilot_ids = [
        {
            "narration_id":
                dataset.events[index]["narration_id"],
            "row_index":
                int(dataset.events[index]["row_index"]),
        }
        for index in indices
    ]

    for name, value in (
        ("profile.json", profile),
        ("provenance.json", provenance),
        ("pilot_events.json", pilot_ids),
    ):
        (args.output_dir / name).write_text(
            json.dumps(value, indent=2) + "\n"
        )

    (args.output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )

    installed = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
    )

    (args.output_dir / "pip-freeze.txt").write_text(
        installed
    )

    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
