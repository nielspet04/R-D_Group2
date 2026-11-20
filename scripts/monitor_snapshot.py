"""Read a bounded, read-only snapshot on VSC; stdout is one JSON object."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import getpass
import json
from pathlib import Path
import subprocess
import time


def bounded_text(path, limit=8*1024*1024):
    with path.open('rb') as stream:
        payload = stream.read(limit+1)
    if len(payload) > limit:
        raise ValueError(f'File exceeds {limit} bytes: {path.name}')
    return payload.decode('utf-8')


def read_json(path):
    try:
        value = json.loads(bounded_text(path))
        return value if isinstance(value, dict) else {}
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
    jobs, errors, sources = {}, [], {}
    user = getpass.getuser()
    commands = [
        (["sacct", "-M", cluster, "-u", user, "-X", "-n", "-P", "-S", f"now-{days}days",
          "--format=JobIDRaw,JobName,State,Partition,Elapsed,NodeList,ExitCode,Submit,Start,End,Timelimit"],
         ["job_id", "name", "state", "partition", "elapsed", "node", "exit_code", "submitted_at", "started_at", "ended_at", "time_limit"]),
        (["squeue", "-M", cluster, "-u", user, "-h", "-r", "-o", "%i|%j|%T|%P|%M|%N|%R"],
         ["job_id", "name", "state", "partition", "elapsed", "node", "reason"]),
    ]
    for command, fields in commands:
        sources[command[0]] = {'status':'ok', 'complete':True, 'observed_at':time.time()}
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            if result.returncode:
                sources[command[0]] = {"status":"error", "complete":False}
                errors.append(f"{command[0]}: {result.stderr.strip()}")
                continue
            for line in result.stdout.splitlines():
                values = line.strip().split("|")
                if len(values) < 7:
                    continue
                job = dict(zip(fields, values))
                job["state"] = job["state"].split()[0].rstrip("+")
                jobs[job["job_id"]] = {**jobs.get(job["job_id"], {}), **job}
        except (OSError, subprocess.SubprocessError) as error:
            sources[command[0]] = {"status":"error", "complete":False}
            errors.append(f"{command[0]}: {error}")
    runs = []
    paths = list((root / 'results/runs').glob('*/run.json'))
    paths += [p.parent.parent / 'history.json' for p in (root / 'results/runs').glob('*/monitoring/experiment.json')]
    if (root / 'results/monitoring/experiment.json').exists():
        paths.append(root / 'results/history.json')
    paths += list((root / 'results/sta2').glob('*/history.json'))
    paths += [p.parent / 'history.json' for p in (root / 'results/sta2').glob('*/run-manifest-*.json')]
    # Instrumented runs can be discovered before their first completed epoch.
    paths += [p.parent.parent / 'history.json' for p in (root / 'results/sta2').glob('*/monitoring/experiment.json')]
    directories = sorted(set(p.parent for p in paths), key=lambda p:p.stat().st_mtime, reverse=True)
    if len(directories) > 200:
        errors.append('Discovery limited to 200 recent runs')
    for directory in directories[:200]:
        try:
            identity = read_json(directory / 'monitoring/experiment.json')
            legacy = read_json(directory / 'run.json')
            if not legacy and identity.get('model') in ('mean_pool','gru'):
                legacy = {'run_id':identity['experiment_id'],'model':identity['model'],'output_root':str(directory)}
            sta2 = not legacy or legacy.get("model") == "sta2"
            if sta2:
                manifests = sorted(directory.glob('run-manifest-*.json'), key=lambda p:p.stat().st_mtime)
                config = read_json(manifests[-1]) if manifests else {}
                run_id = identity.get('experiment_id') or legacy.get('run_id') or 'legacy-sta2-' + hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:16]
                attempts = [{'job_id':p.stem.removeprefix('run-manifest-')} for p in manifests]
                legacy = {**legacy, 'run_id':run_id, 'model':'sta2', 'submitted':directory.stat().st_mtime,
                          'output_root':str(directory), 'attempts':attempts, 'config':config,
                          'job_id': attempts[-1]['job_id'] if attempts else None}
            run = dict(legacy)
            if identity:
                run['run_id'] = identity['experiment_id']
            telemetry_path = directory / 'telemetry.json'
            if telemetry_path.exists():
                run['telemetry'] = json.loads(bounded_text(telemetry_path))
            history_paths = [directory / 'history.json'] if sta2 else list(directory.glob('logs/*/history.csv'))
            for history in history_paths:
                if not history.exists():
                    continue
                try:
                    content = bounded_text(history)
                    run['history'] = json.loads(content) if sta2 else list(csv.DictReader(io.StringIO(content)))
                    if not isinstance(run['history'], list):
                        raise ValueError('History must be a list')
                except (OSError, ValueError, csv.Error) as error:
                    run.pop('history',None)
                    errors.append(f'{directory.name}: history unavailable ({type(error).__name__})')
            if not sta2:
                config = read_json(directory / 'configs' / f"{run.get('model')}.json")
                if config:
                    run['config'] = config
            attempt_paths = sorted((directory / 'monitoring/attempts').glob('*/attempt.json'), key=lambda p:p.stat().st_mtime)
            if attempt_paths:
                run['attempts'] = [read_json(p) for p in attempt_paths]
                latest = run['attempts'][-1]
                run.update(attempt_id=latest.get('attempt_id'), job_id=latest.get('job_id') or run.get('job_id'))
                tracking = read_json(attempt_paths[-1].parent / 'tracking-status.json')
                if tracking:
                    run['wandb'] = tracking
            log = Path(run.get('log_path') or root / 'missing')
            if sta2 and run.get('job_id'):
                candidates = [root / 'logs' / f"{prefix}-{run['job_id']}.out" for prefix in ('sta2-full-h100','sta2-train')]
                log = next((p for p in candidates if p.exists()), log)
            if log.resolve().is_relative_to(root.resolve()) and log.exists():
                run['log_tail'] = tail(log)
            if sum(len(json.dumps(r)) for r in runs) + len(json.dumps(run)) > 30*1024*1024:
                errors.append('Snapshot size budget reached; remaining runs deferred')
                break
            runs.append(run)
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(f'{directory.name}: unavailable ({type(error).__name__})')
    return {'schema_version':2, 'collected_at':time.time(), 'cluster':cluster,
            'jobs':list(jobs.values()), 'runs':runs, 'errors':errors, 'sources':sources}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cluster", default="wice")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.cluster, args.days)))
