"""Upload saved epochs for an existing job from a finite CPU batch allocation."""
import argparse
import getpass
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.monitoring.outbox import file_lock


def job_active(job_id, cluster):
    """None means scheduler unavailable, never evidence of job completion."""
    try:
        result = subprocess.run(
            ['squeue', '-M', cluster, '-h', '-u', getpass.getuser(), '-o', '%i'],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    return job_id in result.stdout.split()


def upload_history(directory):
    # Reuse the stable import identity and durable retry outbox. A partial
    # history write or network outage is retried on the next poll.
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/monitor_backfill.py'),
             '--run-dir', str(directory), '--upload'], timeout=120,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError) as error:
        print('Upload deferred: ' + type(error).__name__, flush=True)
        return False


def follow(directory, job_id, cluster, interval, max_hours):
    deadline = time.monotonic() + max_hours * 3600
    final_successes = 0
    while time.monotonic() < deadline:
        active = job_active(job_id, cluster)
        uploaded = upload_history(directory)
        print(f'job={job_id} queued={active} upload_ok={uploaded}', flush=True)
        # Two successful final reads handle a history file being replaced near
        # job exit. Never synthesize training success or scheduler events.
        final_successes = final_successes + 1 if active is False and uploaded else 0
        if final_successes >= 2:
            print('Job left queue; saved history uploaded. Helper finished.', flush=True)
            return 0
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
    print('Helper time limit reached; local results retained. Further uploads need a new helper.', flush=True)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--cluster', default='wice')
    parser.add_argument('--interval', default=300, type=int)
    parser.add_argument('--max-hours', default=11.9, type=float)
    args = parser.parse_args()
    if not re.fullmatch(r'\d+', args.job_id):
        parser.error('--job-id must be a numeric Slurm job ID')
    if args.interval < 60 or not 0 < args.max_hours <= 168:
        parser.error('Use an interval >= 60 seconds and 0 < max-hours <= 168')
    directory = args.run_dir.resolve()
    if not directory.is_dir():
        parser.error('Run directory does not exist')
    # Fail visibly if the compute host cannot authenticate/reach the API.
    from src.monitoring.api_transport import post
    post({}, 'probe')
    print('Compute-node API probe succeeded; following completed epochs.', flush=True)
    with file_lock(directory / '.monitor-follow.lock', timeout=0):
        return follow(directory, args.job_id, args.cluster, args.interval, args.max_hours)


if __name__ == '__main__':
    raise SystemExit(main())
