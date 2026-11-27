"""Collect a stable, local STA2 handoff snapshot on VSC; does not send files."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=45)
        return {"command": args, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.SubprocessError) as error:
        return {"command": args, "returncode": None, "error": str(error)}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--credits-per-hour", type=float,
                        help="Quote-derived allocation rate, not an actual charge")
    parser.add_argument("--actual-credits", type=float,
                        help="Actual charge for this job only, from VSC accounting")
    args = parser.parse_args()
    if not args.job_id.isdigit():
        parser.error("job ID must be numeric")
    import math
    for value in (args.credits_per_hour, args.actual_credits):
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error("credits and rate must be finite and nonnegative")
    run = args.run_dir or Path(f"results/sta2/compact-a100-{args.job_id}")
    best = run / "checkpoints/best.pt"
    if not best.is_file():
        parser.error(f"No saved best checkpoint: {best}; do not claim a handoff exists before an epoch saves")
    # Check dependency before creating an incomplete bundle.
    import torch

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path("results/sta2/handoffs") / f"compact-{args.job_id}-{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    # The trainer replaces best.pt atomically. Copy from one open inode so a
    # concurrent epoch save cannot mix checkpoint versions within the snapshot.
    with best.open("rb") as source, (destination / "best.pt").open("xb") as target:
        shutil.copyfileobj(source, target)
    checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=True)
    epoch = int(checkpoint["epoch"])
    missing = []
    for source in [run / "history.json", *sorted(run.glob("run-manifest-*.json")), run / "timing-report.json"]:
        if source.is_file():
            shutil.copyfile(source, destination / source.name)
        elif source.name != "timing-report.json":
            missing.append(str(source))
    if not list(destination.glob("run-manifest-*.json")):
        missing.append("run-manifest-*.json")
    logs = []
    # Include the requested attempt plus every resumed attempt recorded in a manifest.
    job_ids = {args.job_id}
    for manifest in run.glob("run-manifest-*.json"):
        suffix = manifest.stem.removeprefix("run-manifest-")
        if suffix.isdigit():
            job_ids.add(suffix)
    for job_id in sorted(job_ids):
        logs.extend(sorted(Path("logs").glob(f"sta2-compact-a100*-{job_id}.out")))
    for source in logs:
        shutil.copyfile(source, destination / source.name)
    if not logs:
        missing.append("compact Slurm training log")
    queries = {
        "squeue": command(["squeue", "-M", "wice", "-j", args.job_id, "-h", "-o", "%i|%j|%P|%T|%M|%l|%R"]),
        "sacct": command(["sacct", "-M", "wice", "-j", args.job_id, "-X", "-n", "-P", "--format=JobIDRaw,State,ElapsedRaw,Start,End,ExitCode,AllocTRES"]),
        "revision": command(["git", "rev-parse", "HEAD"]),
        "local_changes": command(["git", "status", "--short", "--untracked-files=normal"]),
        "tracked_patch": command(["git", "diff", "HEAD", "--binary"]),
    }
    for key, value in queries.items():
        (destination / f"{key}.txt").write_text(value.get("stdout", "") + value.get("stderr", "") + value.get("error", ""))
    state = "UNKNOWN"
    elapsed = None
    if queries["sacct"]["returncode"] == 0:
        for line in queries["sacct"]["stdout"].splitlines():
            fields = line.split("|")
            if len(fields) >= 3 and fields[0] == args.job_id:
                state = fields[1]
                elapsed = int(fields[2]) if fields[2].isdigit() else None
    history_path = destination / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    epoch_rows = []
    for row in history:
        seconds = row.get("epoch_seconds")
        phases = float(row["train_seconds"]) + float(row["validation_seconds"])
        hours = (seconds if seconds is not None else phases) / 3600
        epoch_rows.append({"epoch": row["epoch"], "hours": hours,
            "includes_checkpoint_saves": seconds is not None,
            "estimated_credits": hours * args.credits_per_hour if args.credits_per_hour is not None else None})
    metadata = {key: checkpoint.get(key) for key in (
        "epoch", "best_metric", "git_commit", "git_dirty", "split_hashes",
        "configuration", "backbone_provenance",
    )}
    (destination / "checkpoint-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    digest = sha256(destination / "best.pt")
    terminal = state.split()[0].rstrip("+") in {
        "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
    }
    report = {
        "captured_utc": stamp, "job_id": args.job_id, "scheduler_state": state,
        "scheduler_queries": queries | {"tracked_patch": {"saved_to": "tracked_patch.txt"}},
        "run_directory": str(run.resolve()), "snapshot_directory": str(destination.resolve()),
        "checkpoint_epoch": epoch, "checkpoint_sha256": digest,
        "checkpoint_git_revision": checkpoint.get("git_commit"),
        "checkpoint_git_dirty": checkpoint.get("git_dirty"),
        "status_note": (
            "Attempt is terminal according to accounting; this is its stable saved best checkpoint. Experiment may still be resumed; final selection is not confirmed."
            if terminal else "Attempt is active or unconfirmed; this is a stable snapshot, not a final checkpoint claim."
        ),
        "best_epoch_present_in_copied_history": any(int(row["epoch"]) == epoch for row in history),
        "epoch_timing": epoch_rows,
        "quote_derived_credits_per_hour": args.credits_per_hour,
        "allocation_hours": elapsed / 3600 if elapsed is not None else None,
        "estimated_allocation_credits": elapsed / 3600 * args.credits_per_hour if elapsed is not None and args.credits_per_hour is not None else None,
        "actual_job_credits": args.actual_credits,
        "cost_note": "Epoch costs are quote-derived estimates. Actual job charges require VSC accounting; they include startup/shutdown and any incomplete work. Multi-attempt run costs are not inferred from one job.",
        "snapshot_note": "Checkpoint is stable; history/logs may be newer or incomplete while training is active. Current git patch may differ from code at checkpoint save; untracked file contents are not included.",
        "missing_files": missing,
    }
    (destination / "handoff.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in sorted(destination.iterdir()) if path.is_file()
    ))
    print(f"Job: {args.job_id}; accounting state: {state}\nRun: {run.resolve()}\nStable best epoch: {epoch}\nSHA-256: {digest}\nBundle: {destination.resolve()}")
    print(report["status_note"])
    print("Missing files:", missing)
    print("Actual credits:", args.actual_credits if args.actual_credits is not None else "NOT PROVIDED")
    for row in epoch_rows:
        print(f"Epoch {row['epoch']}: {row['hours']:.4f} hours; estimated credits: {row['estimated_credits']}")


if __name__ == "__main__":
    main()
