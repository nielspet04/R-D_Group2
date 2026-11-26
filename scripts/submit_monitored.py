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
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mean_pool", "gru"])
    parser.add_argument("--job-type", choices=['baseline','sta2-full-h100'], default='baseline')
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--dry-run", action='store_true')
    args = parser.parse_args()
    if args.job_type == 'baseline' and not args.model:
        parser.error('--model is required for baseline')
    if args.job_type != 'baseline' and args.model:
        parser.error('--model only applies to baseline')
    if args.resume_run and args.job_type != 'sta2-full-h100':
        parser.error('--resume-run only applies to STA2')
    script = 'slurm/sta_train_p100.slurm' if args.job_type == 'baseline' else 'slurm/sta2_full_h100.slurm'
    command = ['sbatch','--parsable','--export=ALL',script] + ([args.model] if args.model else [])
    if args.dry_run:
        print(json.dumps({'command':command,'resume_run':str(args.resume_run) if args.resume_run else None}))
        return
    root = Path(__file__).resolve().parents[1]
    expected_root = (
        Path(os.environ.get("VSC_DATA", "/missing")) / "R-D_Group2"
    ).resolve()
    if root != expected_root:
        parser.error("The existing P100 script expects this checkout at $VSC_DATA/R-D_Group2")
    model = args.model or "sta2"
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + model + "-" + uuid.uuid4().hex[:8]
    directory = root / ('results/runs' if args.job_type == 'baseline' else 'results/sta2') / run_id
    if args.resume_run:
        directory = args.resume_run.resolve()
        if not directory.is_relative_to(root) or not (directory / 'checkpoints/last.pt').is_file():
            parser.error('Resume requires an existing checkpoint directory inside this checkout')
    directory.mkdir(parents=True, exist_ok=bool(args.resume_run))
    (root / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, RND_OUTPUT_ROOT=str(directory), RND_MONITOR_RUN_DIR=str(directory))
    if args.resume_run:
        env['STA2_RESUME_RUN'] = str(directory)
    else:
        env.pop('STA2_RESUME_RUN',None)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    metadata = {"run_id": run_id, "model": model, "submitted": time.time(),
                "cluster": "wice", "output_root": str(directory), "git_commit": commit,
                "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout),
                "state": "SUBMITTING"}
    monitoring = directory / 'monitoring'
    monitoring.mkdir(exist_ok=True)
    identity_path = monitoring / 'experiment.json'
    if args.resume_run and identity_path.exists():
        run_id = json.loads(identity_path.read_text(encoding='utf-8'))['experiment_id']
        metadata['run_id'] = run_id
    else:
        identity_path.write_text(json.dumps({'schema_version':2,'experiment_id':run_id,'model':model}),encoding='utf-8')
    attempt_id = uuid.uuid4().hex
    attempt_dir = monitoring / 'attempts' / attempt_id
    attempt_dir.mkdir(parents=True)
    env['RND_MONITOR_ATTEMPT_ID'] = attempt_id
    pending = {'experiment_id':run_id,'attempt_id':attempt_id,'submitted':metadata['submitted'],'job_id':None,'model':model,'state':'SUBMITTING'}
    outbox = None
    if os.environ.get('RND_MONITOR_API_URL') and os.environ.get('RND_MONITOR_UPLOAD_KEY_FILE'):
        try:
            sys.path.insert(0,str(root))
            from src.monitoring.outbox import Outbox
            outbox = Outbox(attempt_dir,run_id,attempt_id)
            outbox.append('experiment_registered',{'model':model,'configuration':{}})
        except Exception:
            pass
    def publish_submission():
        if outbox:
            try:
                kind = 'attempt_submitted' if metadata.get('job_id') else 'submission_unknown'
                outbox.append(kind,{'job_id':metadata['job_id'],'cluster':'wice'} if metadata.get('job_id') else {'reason':metadata['state']})
                subprocess.run([sys.executable,str(root/'scripts/monitor_upload.py'),'--attempt-dir',str(attempt_dir),'--once'],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=4)
            except Exception:
                pass
    def save():
        temporary = directory / "run.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.replace(directory / "run.json")
        pending.update(state=metadata['state'], job_id=metadata.get('job_id'))
        attempt_temp = attempt_dir / 'attempt.json.tmp'
        attempt_temp.write_text(json.dumps(pending),encoding='utf-8')
        attempt_temp.replace(attempt_dir / 'attempt.json')
    save()
    try:
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        metadata.update(state='SUBMISSION_UNKNOWN', error='Submission interrupted; check squeue before retrying')
        save()
        publish_submission()
        raise SystemExit(metadata['error'])
    if result.returncode:
        metadata.update(state="SUBMISSION_FAILED", error=result.stderr.strip())
        save()
        publish_submission()
        raise SystemExit(result.stderr)
    job_id = result.stdout.strip().split(";")[0]
    if not re.fullmatch(r"\d+", job_id):
        metadata.update(state="SUBMISSION_UNKNOWN", error="Unexpected sbatch reply; inspect queue before retrying")
        save()
        publish_submission()
        raise SystemExit(f"Unexpected sbatch reply; check squeue before retrying: {result.stdout!r}")
    prefix = 'sta-train' if args.job_type == 'baseline' else 'sta2-full-h100'
    metadata.update(job_id=job_id, state="PENDING", log_path=str(root / f"logs/{prefix}-{job_id}.out"))
    save()
    publish_submission()
    print(f"Job: {job_id}\nRun: {run_id}\nOutputs: {directory}")


if __name__ == "__main__":
    main()
