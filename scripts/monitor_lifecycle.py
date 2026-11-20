"""Best-effort batch exit event. No scheduler claims and no training dependencies."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.monitoring.outbox import Outbox


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--exit-code',type=int,required=True)
    args=parser.parse_args()
    if not os.environ.get('RND_MONITOR_API_URL') or not os.environ.get('RND_MONITOR_UPLOAD_KEY_FILE'):
        return
    try:
        active=json.loads((args.output_root/'monitoring/active-attempt.json').read_text())
        directory=Path(active['attempt_dir'])
        if not directory.resolve().is_relative_to(args.output_root.resolve()):
            return
        metadata=json.loads((directory/'attempt.json').read_text())
        if metadata.get('job_id') and os.environ.get('SLURM_JOB_ID') and metadata['job_id'] != os.environ['SLURM_JOB_ID']:
            return
        outbox=Outbox(directory,metadata['experiment_id'],metadata['attempt_id'])
        outbox.append('batch_exit_reported',{'exit_code':args.exit_code})
        # Invoke as a bounded child in the shell; no hanging network call in the EXIT trap.
        import subprocess
        subprocess.run([sys.executable,str(Path(__file__).with_name('monitor_upload.py')),'--attempt-dir',str(directory),'--once'],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=4)
    except Exception:
        pass


if __name__=='__main__':
    main()
