"""Prepare or upload existing native history without loading checkpoints or submitting jobs."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.monitoring.api_contracts import identifier, METRICS, public_config, encode
from src.monitoring.outbox import Outbox, atomic
from src.monitoring.store import normalize_run
from scripts.monitor_snapshot import bounded_text


def history_metrics(row):
    """CSV readers return counts as strings as well as floating-point metrics."""
    metrics = {}
    for key, value in row.items():
        if key not in METRICS:
            continue
        if isinstance(value, str):
            try:
                value = float(value)
                if math.isfinite(value) and value.is_integer():
                    value = int(value)
            except ValueError:
                value = None
        if isinstance(value, bool) or isinstance(value, float) and not math.isfinite(value):
            value = None
        metrics[key] = value
    return metrics


def prepare(directory):
    directory=directory.resolve()
    if not directory.is_dir():
        raise ValueError("Run directory does not exist")
    ledger_dir=directory/'monitoring/api-import'
    ledger_dir.mkdir(parents=True,exist_ok=True)
    identity_file=ledger_dir/'identity.json'
    if identity_file.exists():
        identity=json.loads(identity_file.read_text())
    else:
        # Persist the mapping so a subsequent import from a copied directory is stable.
        legacy=directory/'monitoring/experiment.json'
        metadata=json.loads(bounded_text(legacy)) if legacy.exists() else {}
        run=directory/'run.json'
        if run.exists():
            metadata={**json.loads(bounded_text(run)),**metadata}
        experiment=identifier(metadata.get('experiment_id') or metadata.get('run_id') or uuid.uuid4())
        identity={'experiment_id':experiment,'attempt_id':str(uuid.uuid5(uuid.UUID(experiment),'historical-import'))}
        atomic(identity_file,identity)
    history_path=directory/'history.json'
    if history_path.exists():
        history=json.loads(bounded_text(history_path));model='sta2'
        manifests=sorted(directory.glob('run-manifest-*.json'))
        config=json.loads(bounded_text(manifests[-1])) if manifests else {}
        job_id=manifests[-1].stem.removeprefix('run-manifest-') if manifests else None
    else:
        paths=list(directory.glob('logs/*/history.csv'))
        if len(paths)!=1:
            raise ValueError('Select a directory containing exactly one model history')
        history_path=paths[0];model=history_path.parent.name
        with history_path.open(encoding='utf-8',newline='') as stream:
            if history_path.stat().st_size>8*1024*1024:
                raise ValueError('History exceeds import size limit')
            history=list(csv.DictReader(stream))
        configuration=directory/'configs'/f'{model}.json'
        config=json.loads(bounded_text(configuration)) if configuration.exists() else {}
        metadata_path=directory/'run.json'
        job_id=json.loads(bounded_text(metadata_path)).get('job_id') if metadata_path.exists() else None
    history=normalize_run({'history':history})['history']
    outbox=Outbox(ledger_dir,identity['experiment_id'],identity['attempt_id'])
    ledger_file=ledger_dir/'imported.json'
    ledger=json.loads(ledger_file.read_text()) if ledger_file.exists() else []
    observed=datetime.fromtimestamp(history_path.stat().st_mtime,timezone.utc).isoformat().replace('+00:00','Z')
    def append(kind,payload):
        fingerprint=hashlib.sha256(encode([kind,payload])).hexdigest()
        if fingerprint not in ledger:
            outbox.append(kind,payload,observed_at=observed,event_id=str(uuid.uuid5(uuid.UUID(identity['attempt_id']),fingerprint)))
            ledger.append(fingerprint)
            atomic(ledger_file,ledger)
    append('experiment_registered',{'model':model,'configuration':public_config(config)})
    append('attempt_started',{'job_id':str(job_id) if job_id else None,'cluster':'wice'})
    for row in history:
        append('epoch_committed',{'epoch':row['epoch'],'history_revision':identity['attempt_id'],**history_metrics(row)})
    # There is no invented completion event: an old CSV alone doesn't prove job success.
    return ledger_dir,len(history)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--upload',action='store_true',help='Explicitly send the prepared events')
    args=parser.parse_args()
    try:
        if not args.upload:
            # Dry run must not even create mapping/outbox files in the training directory.
            directory=args.run_dir.resolve()
            paths=[directory/'history.json'] if (directory/'history.json').exists() else list(directory.glob('logs/*/history.csv'))
            if len(paths)!=1:
                raise ValueError('Select a run with one native history')
            history=json.loads(bounded_text(paths[0])) if paths[0].suffix=='.json' else list(csv.DictReader(bounded_text(paths[0]).splitlines()))
            rows=normalize_run({'history':history})['history']
            print(f'Dry run: {directory}; {len(rows)} completed epochs. No changes or uploads made.')
        else:
            directory,count=prepare(args.run_dir)
            print(f'Prepared {count} scientific epochs; native files unchanged.')
            from scripts.monitor_upload import run
            run(argparse.Namespace(probe=False,attempt_dir=directory,dry_run=False,once=True))
    except Exception as error:
        print('Import stopped ('+type(error).__name__+'); original history retained.',file=sys.stderr)
        raise SystemExit(1)
