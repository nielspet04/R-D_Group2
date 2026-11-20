"""Versioned cache. Retain source observations; local files own scientific history."""
from contextlib import contextmanager, closing
import json
import math
from pathlib import Path
import sqlite3
import time


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def dumps(value):
    return json.dumps(clean(value), allow_nan=False)


@contextmanager
def connect(path):
    db = sqlite3.connect(path, timeout=10)
    try:
        version = db.execute('PRAGMA user_version').fetchone()[0]
        if version > 2:
            raise ValueError('Database is newer than this dashboard; use a compatible version')
        if version == 0 and db.execute("SELECT name FROM sqlite_master WHERE name='records'").fetchone():
            backup = Path(str(path) + '.before-v2.sqlite3')
            if not backup.exists():
                with closing(sqlite3.connect(backup)) as destination:
                    db.backup(destination)
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, payload TEXT, PRIMARY KEY(kind,id))')
            db.execute('CREATE TABLE IF NOT EXISTS samples (run_id TEXT, timestamp REAL, payload TEXT, PRIMARY KEY(run_id,timestamp))')
            db.execute('CREATE TABLE IF NOT EXISTS observations (source TEXT, kind TEXT, id TEXT, timestamp REAL, payload TEXT, PRIMARY KEY(source,kind,id))')
            db.execute('PRAGMA user_version=2')
            yield db
    finally:
        db.close()


def status(path, source, value):
    with connect(path) as db:
        previous = db.execute("SELECT payload FROM records WHERE kind='meta' AND id=?", (source,)).fetchone()
        payload = json.loads(previous[0]) if previous else {}
        payload.update(value)
        db.execute("INSERT OR REPLACE INTO records VALUES ('meta',?,?)", (source, dumps(payload)))


def normalize_run(value):
    value = dict(value)
    if 'history' in value:
        rows = value['history']
        if not isinstance(rows,list):
            value.pop('history')
            return value
        normalized = {}
        for row in rows:
            if not isinstance(row,dict):
                continue
            try:
                epoch = float(row.get('epoch'))
                if not math.isfinite(epoch) or epoch < 1 or epoch != int(epoch):
                    continue
            except (TypeError,ValueError):
                continue
            record = dict(row,epoch=int(epoch))
            for key,number in list(record.items()):
                if key.endswith(('_loss','_mt5r','_seconds','_per_second')) or key=='learning_rate':
                    try:
                        record[key] = float(number) if number is not None and not isinstance(number,bool) else None
                    except (TypeError,ValueError):
                        record[key] = None
            for key,metric_value in list(record.items()):
                if key.endswith('_mt5r') and metric_value is not None and not 0 <= metric_value <= 1:
                    record[key] = None
            normalized[int(epoch)] = clean(record)
        value['history'] = [normalized[e] for e in sorted(normalized)]
    return value


def ingest(path, snapshot, source='vsc'):
    if snapshot.get('schema_version', 1) not in (1, 2):
        raise ValueError('Unsupported snapshot version')
    if not isinstance(snapshot.get('jobs'), list) or not isinstance(snapshot.get('runs'), list):
        raise ValueError('Snapshot must contain jobs and runs lists')
    stamp = float(snapshot['collected_at'])
    if not math.isfinite(stamp):
        raise ValueError('Invalid snapshot timestamp')
    with connect(path) as db:
        previous_sync = db.execute("SELECT payload FROM records WHERE kind='meta' AND id=?", (source,)).fetchone()
        previous_sync = json.loads(previous_sync[0]) if previous_sync else {}
        identity = snapshot.get('project') if source == 'wandb' else snapshot.get('cluster')
        if previous_sync.get('identity') and identity and previous_sync['identity'] != identity:
            raise ValueError('Use a separate database for a different cluster or W&B project')
        if stamp < previous_sync.get('observed_at', 0):
            return
        if source == 'vsc' and (not snapshot.get('errors') or all(snapshot.get('sources',{}).get(name,{}).get('complete') for name in ('squeue','sacct'))):
            seen = {str(j['job_id']) for j in snapshot['jobs']}
            for key, payload in db.execute("SELECT id,payload FROM records WHERE kind='jobs'").fetchall():
                job = json.loads(payload)
                if key not in seen and job.get('state') in {'RUNNING','PENDING','CONFIGURING','COMPLETING'}:
                    job['state'] = 'UNKNOWN'
                    db.execute("UPDATE records SET payload=? WHERE kind='jobs' AND id=?", (dumps(job),key))
        for kind, field in [('jobs','job_id'), ('runs','run_id')]:
            for incoming in snapshot[kind]:
                if not isinstance(incoming, dict) or not incoming.get(field):
                    continue
                value = clean(normalize_run(incoming) if kind == 'runs' else dict(incoming))
                key = str(value[field])
                value[field] = key
                old = db.execute('SELECT payload FROM records WHERE kind=? AND id=?', (kind,key)).fetchone()
                old = json.loads(old[0]) if old else {}
                db.execute('INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?)', (source,kind,key,stamp,dumps(value)))
                if kind == 'runs':
                    # File failures omit fields; empty successful histories are meaningful (resume).
                    if source == 'wandb' and old.get('metric_source') == 'vsc':
                        local_history = old.get('history',[])
                        cloud_by_epoch = {row['epoch']:row for row in value.get('history',[])}
                        merged_history = [{**cloud_by_epoch.get(row['epoch'],{}), **row} for row in local_history]
                        value = {**value, **old, 'history':merged_history, 'wandb':value.get('wandb', old.get('wandb'))}
                    else:
                        value = {**old, **value, 'metric_source':source}
                    value['observed_at'] = stamp
                    telemetry = value.get('telemetry', {})
                    if telemetry.get('heartbeat'):
                        db.execute('INSERT OR IGNORE INTO samples VALUES (?,?,?)', (key,telemetry['heartbeat'],dumps(telemetry)))
                else:
                    value = {**old, **value, 'observed_at':stamp}
                db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?)', (kind,key,dumps(value)))
        meta = {**previous_sync, 'identity':identity, 'observed_at':stamp, 'last_attempt':time.time(), 'errors':snapshot.get('errors',[]),
                'sources':snapshot.get('sources',{}), 'cluster':snapshot.get('cluster'),
                'status':'partial' if snapshot.get('errors') else 'ok'}
        if not snapshot.get('errors'):
            meta['last_success'] = stamp
        db.execute("INSERT OR REPLACE INTO records VALUES ('meta',?,?)", (source,dumps(meta)))
        if source == 'vsc':
            db.execute("INSERT OR REPLACE INTO records VALUES ('meta','sync',?)", (dumps(meta),))
        # Keep a bounded high-resolution window; epoch histories are never pruned.
        db.execute('DELETE FROM samples WHERE timestamp < ?', (stamp-14*86400,))


def read_data(path):
    data = {'jobs':[], 'runs':[], 'meta':{}}
    with connect(path) as db:
        for kind,key,payload in db.execute('SELECT kind,id,payload FROM records'):
            if kind == 'meta':
                data['meta'][key] = json.loads(payload)
            elif kind in ('jobs','runs'):
                data[kind].append(json.loads(payload))
        for run in data['runs']:
            run['samples'] = [json.loads(r[0]) for r in db.execute('SELECT payload FROM samples WHERE run_id=? ORDER BY timestamp DESC LIMIT 300',(run['run_id'],))][::-1]
    data['jobs'].sort(key=lambda j:j['job_id'], reverse=True)
    data['runs'].sort(key=lambda r:r.get('submitted',0), reverse=True)
    return data
