"""Publish committed epoch files from a separate process; --once replays a completed attempt."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.monitoring.tracking import METRICS


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def publish(directory, once=False):
    metadata = json.loads((directory / 'attempt.json').read_text(encoding='utf-8'))
    mode = os.environ.get('RND_WANDB_MODE','disabled')
    project = os.environ.get('RND_WANDB_PROJECT','')
    if mode not in ('online','offline') or '/' not in project:
        raise ValueError('Set RND_WANDB_MODE=online|offline and RND_WANDB_PROJECT=entity/project')
    if mode == 'online' and not os.environ.get('WANDB_API_KEY'):
        raise ValueError('WANDB_API_KEY is missing')
    import wandb
    entity, project_name = project.split('/',1)
    worker_id = metadata['attempt_id']
    settings = wandb.Settings(init_timeout=15, console='off', disable_code=True)
    run = wandb.init(entity=entity, project=project_name, id=worker_id,
                     group=metadata['experiment_id'], resume='allow', mode=mode,
                     dir=str(directory), config={**metadata,'monitoring_schema':2}, settings=settings)
    run.define_metric('epoch')
    for key in METRICS.values():
        run.define_metric(key, step_metric='epoch')
    sent = set()
    if mode == 'online':
        try:
            for row in wandb.Api(timeout=15).run(f'{entity}/{project_name}/{worker_id}').scan_history(keys=['epoch']):
                sent.add(int(row['epoch']))
        except Exception:
            # Local epochs remain replayable; duplicates are deduplicated on ingestion.
            pass
    while True:
        history = directory / 'epochs.json'
        if history.exists():
            for record in json.loads(history.read_text(encoding='utf-8')):
                epoch = int(record['epoch'])
                if epoch in sent:
                    continue
                run.log({'epoch':epoch, **{remote:record[key] for key,remote in METRICS.items() if record.get(key) is not None}})
                sent.add(epoch)
        atomic(directory / 'tracking-status.json', {'state':mode,'url':run.url,'path':f'{entity}/{project_name}/{worker_id}','updated_at':time.time()})
        if once or (directory / 'finish.json').exists():
            break
        for _ in range(15):
            if (directory / 'finish.json').exists():
                break
            time.sleep(1)
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-dir', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    try:
        publish(args.attempt_dir, args.once)
    except Exception as error:
        atomic(args.attempt_dir / 'tracking-status.json', {'state':'error','error':type(error).__name__, 'updated_at':time.time()})
        raise SystemExit(1)
