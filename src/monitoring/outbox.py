"""Crash-safe immutable event files. Producer and uploader own different files."""
import json
import os
from pathlib import Path
import threading
import uuid
import time
from contextlib import contextmanager
from .api_contracts import encode, new_event, MAX_EVENTS, MAX_BATCH_BYTES


def atomic(path, value):
    path = Path(path)
    temporary = path.parent / ('.' + uuid.uuid4().hex[:16] + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def file_lock(path, timeout=2):
    """Kernel-owned locks are released after a crash; no stale PID lockfiles."""
    with Path(path).open('a+b') as stream:
        stream.seek(0,2)
        if stream.tell()==0:
            stream.write(b'0')
            stream.flush()
        deadline=time.monotonic()+timeout
        while True:
            try:
                stream.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic()>=deadline:
                    raise TimeoutError('monitor_lock_busy') from None
                time.sleep(0.02)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name=='nt':
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


class Outbox:
    def __init__(self, directory, experiment=None, attempt=None):
        self.directory = Path(directory)
        self.events = self.directory / 'api-outbox'
        self.events.mkdir(parents=True, exist_ok=True)
        self.experiment, self.attempt = experiment, attempt
        self.lock = threading.Lock()
        state = self.directory / 'api-producer.json'
        self.sequence = json.loads(state.read_text())['sequence'] if state.exists() else 0
        # Recover a crash after event rename but before saving producer state.
        self.sequence = max([self.sequence]+[int(p.stem.split('-')[0]) for p in self.events.glob('*.json')])

    def append(self, kind, payload, **kwargs):
        with self.lock, file_lock(self.directory/'api-producer.lock'):
            state = self.directory/'api-producer.json'
            saved = json.loads(state.read_text())['sequence'] if state.exists() else 0
            self.sequence=max([self.sequence,saved]+[int(p.stem.split('-')[0]) for p in self.events.glob('*.json')])
            self.sequence += 1
            event = new_event(self.experiment,self.attempt,self.sequence,kind,payload,**kwargs)
            atomic(self.events / f"{self.sequence:010d}.json", event)
            atomic(self.directory / 'api-producer.json', {'sequence':self.sequence})
            return event

    def batch(self):
        rows = []
        for path in sorted(self.events.glob('*.json')):
            event = json.loads(path.read_text(encoding='utf-8'))
            candidate = {'schema_version':1, 'batch_id':str(uuid.uuid4()), 'events':rows+[event]}
            if len(encode(candidate)) > MAX_BATCH_BYTES or len(rows) >= MAX_EVENTS:
                break
            rows.append(event)
        return {'schema_version':1, 'batch_id':str(uuid.uuid4()), 'events':rows}

    def acknowledge(self, batch, response):
        accepted = set(response.get('accepted',[])) | set(response.get('duplicates',[]))
        requested = {event['event_id'] for event in batch['events']}
        if not accepted <= requested:
            raise ValueError('Server acknowledged unknown events')
        # Only the explicitly acknowledged immutable files can be removed.
        for event in batch['events']:
            if event['event_id'] in accepted:
                path = self.events / f"{event['sequence']:010d}.json"
                path.unlink(missing_ok=True)
        atomic(self.directory / 'api-acks.json', {'last_batch':batch['batch_id'], 'acknowledged':sorted(accepted)})

    def compact(self, budget=50*1024*1024):
        paths = sorted(self.events.glob('*.json'))
        total = sum(p.stat().st_size for p in paths)
        if total <= budget:
            return
        # Preserve lifecycle and epochs; remove only superseded queued telemetry.
        for path in paths[:-1]:
            event = json.loads(path.read_text(encoding='utf-8'))
            if event['kind'] in ('heartbeat','progress'):
                size = path.stat().st_size
                path.unlink()
                total -= size
                if total <= budget:
                    return

    def quarantine(self, batch, reason):
        directory = self.directory / 'api-quarantine'
        directory.mkdir(exist_ok=True)
        for event in batch['events']:
            path = self.events / f"{event['sequence']:010d}.json"
            if path.exists():
                path.replace(directory / path.name)
        atomic(directory / 'last-error.json', {'code':reason})
