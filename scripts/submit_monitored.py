"""Submit the existing P100 training script with isolated output and metadata."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mean_pool", "gru"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected_root = (
        Path(os.environ.get("VSC_DATA", "/missing")) / "R-D_Group2"
    ).resolve()
    if root != expected_root:
        parser.error("The existing P100 script expects this checkout at $VSC_DATA/R-D_Group2")
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + args.model + "-" + uuid.uuid4().hex[:8]
    directory = root / "results/runs" / run_id
    directory.mkdir(parents=True)
    (root / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, RND_OUTPUT_ROOT=str(directory), RND_MONITOR_RUN_DIR=str(directory))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    metadata = {"run_id": run_id, "model": args.model, "submitted": time.time(),
                "cluster": "wice", "output_root": str(directory), "git_commit": commit,
                "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout),
                "state": "SUBMITTING"}
    def save():
        temporary = directory / "run.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.replace(directory / "run.json")
    save()
    result = subprocess.run(["sbatch", "--parsable", "--export=ALL", "slurm/sta_train_p100.slurm", args.model],
                            cwd=root, env=env, capture_output=True, text=True)
    if result.returncode:
        metadata.update(state="SUBMISSION_FAILED", error=result.stderr.strip())
        save()
        raise SystemExit(result.stderr)
    job_id = result.stdout.strip().split(";")[0]
    if not re.fullmatch(r"\d+", job_id):
        raise SystemExit(f"Unexpected sbatch reply; check squeue before retrying: {result.stdout!r}")
    metadata.update(job_id=job_id, state="PENDING", log_path=str(root / f"logs/sta-train-{job_id}.out"))
    save()
    print(f"Job: {job_id}\nRun: {run_id}\nOutputs: {directory}")


if __name__ == "__main__":
    main()
