"""Upload an attempt's durable outbox, without importing any training packages."""
import argparse
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.monitoring.api_contracts import utc_now
from src.monitoring.api_transport import post, UploadError
from src.monitoring.outbox import Outbox, atomic, file_lock


def upload_once(outbox):
    batch = outbox.batch()
    if not batch['events']:
        return False
    response = post(batch)
    outbox.acknowledge(batch,response)
    if not response.get('accepted') and not response.get('duplicates'):
        raise UploadError('empty_acknowledgement',True)
    atomic(outbox.directory / 'upload-status.json', {'state':'ok','last_success':utc_now(),
           'pending':len(list(outbox.events.glob('*.json')))})
    return True


def run(args):
    if args.probe:
        post({},'probe')
        print('API authentication and database connection succeeded.')
        return
    if not args.attempt_dir:
        raise ValueError('--attempt-dir is required')
    outbox = Outbox(args.attempt_dir)
    if args.dry_run:
        print(f"Queued events: {len(list(outbox.events.glob('*.json')))}; no request sent.")
        return
    failures = 0
    while True:
        try:
            outbox.compact()
            sent = upload_once(outbox)
            failures = 0
            if args.once:
                if sent:
                    continue
                return
            # Drain a backlog quickly, bounded by server-side rate limits.
            wait = 1 if sent and len(list(outbox.events.glob('*.json'))) >= 50 else 60
        except UploadError as error:
            failures += 1
            atomic(outbox.directory/'upload-status.json', {'state':'error','code':error.code,'observed_at':utc_now()})
            if not error.retryable:
                if error.code in ('http_400','http_409','http_413','http_422'):
                    outbox.quarantine(outbox.batch(),error.code)
                raise
            if args.once and failures>=3:
                raise
            wait = max(error.retry_after,min(300,2**min(failures,8)+random.random()*3))
        for _ in range(int(wait)+1):
            if (outbox.directory/'finish.json').exists():
                # One final bounded attempt; retained files are replayable if it fails.
                upload_once(outbox)
                return
            time.sleep(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-dir',type=Path)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--probe',action='store_true')
    try:
        args=parser.parse_args()
        if args.attempt_dir and not args.dry_run:
            with file_lock(args.attempt_dir/'api-upload.lock',timeout=0 if args.once else 15):
                run(args)
        else:
            run(args)
    except (UploadError,OSError,ValueError) as error:
        print('Upload stopped: '+(error.code if isinstance(error,UploadError) else type(error).__name__)+'. Local data retained.',file=sys.stderr)
        raise SystemExit(1)
