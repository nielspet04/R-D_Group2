"""Read a bounded, read-only snapshot on VSC; stdout is one JSON object."""
from __future__ import annotations

import argparse
import csv
import getpass
import json
from pathlib import Path
import subprocess
import time


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def tail(path, limit=16000):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def collect(root, cluster="wice", days=7):
    jobs, errors = {}, []
    user = getpass.getuser()
    commands = [
        (["sacct", "-M", cluster, "-u", user, "-X", "-n", "-P", "-S", f"now-{days}days",
          "--format=JobIDRaw,JobName,State,Partition,Elapsed,NodeList,ExitCode"],
         ["job_id", "name", "state", "partition", "elapsed", "node", "exit_code"]),
        (["squeue", "-M", cluster, "-u", user, "-h", "-r", "-o", "%i|%j|%T|%P|%M|%N|%R"],
         ["job_id", "name", "state", "partition", "elapsed", "node", "reason"]),
    ]
    for command, fields in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            if result.returncode:
                errors.append(f"{command[0]}: {result.stderr.strip()}")
                continue
            for line in result.stdout.splitlines():
                values = line.strip().split("|")
                if len(values) < len(fields):
                    continue
                job = dict(zip(fields, values))
                jobs[job["job_id"]] = {**jobs.get(job["job_id"], {}), **job}
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"{command[0]}: {error}")
    runs = []
    paths = sorted((root / "results/runs").glob("*/run.json"), reverse=True)[:200]
    for path in paths:
        run = read_json(path)
        if not run.get("run_id"):
            continue
        directory = path.parent
        run["telemetry"] = read_json(directory / "telemetry.json")
        run["history"] = []
        for history in directory.glob("logs/*/history.csv"):
            try:
                with history.open(encoding="utf-8", newline="") as stream:
                    run["history"] = list(csv.DictReader(stream))[-1000:]
            except (OSError, csv.Error):
                pass
        run["config"] = read_json(directory / "configs" / f"{run.get('model')}.json")
        log = Path(run.get("log_path", str(root / "missing")))
        if log.resolve().is_relative_to(root.resolve()):
            run["log_tail"] = tail(log)
        runs.append(run)
    return {"collected_at": time.time(), "cluster": cluster, "jobs": list(jobs.values()), "runs": runs, "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cluster", default="wice")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.cluster, args.days)))
