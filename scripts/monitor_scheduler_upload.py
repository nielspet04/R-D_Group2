"""Explicit one-shot scheduler observation upload; never installs a recurring process."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.monitor_snapshot import collect
from src.monitoring.outbox import Outbox
from src.monitoring.api_contracts import identifier
from scripts.monitor_upload import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cluster',default='wice')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.dry_run:
        print('No scheduler query or upload performed.')
    else:
        snapshot=collect(args.root,args.cluster)
        if any(v.get('status')!='ok' for v in snapshot['sources'].values()):
            raise SystemExit('Scheduler collection failed; no partial scheduler state uploaded.')
        outbox=Outbox(args.root/'results/monitoring/scheduler-api',identifier('scheduler:'+args.cluster),identifier('scheduler-producer:'+args.cluster))
        outbox.append('scheduler_snapshot',{'cluster':args.cluster,'jobs':snapshot['jobs'][:200]})
        run(argparse.Namespace(probe=False,attempt_dir=outbox.directory,dry_run=False,once=True))
