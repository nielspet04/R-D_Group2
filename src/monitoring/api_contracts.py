"""Portable, bounded monitoring events. This module imports no ML/cloud packages."""
from datetime import datetime, timezone
import json
import math
import re
import uuid

MAX_BATCH_BYTES = 256 * 1024
MAX_EVENTS = 50
METRICS = {
    'train_loss', 'validation_loss', 'action_mt5r', 'verb_mt5r', 'noun_mt5r',
    'learning_rate', 'train_seconds', 'validation_seconds', 'best_action_mt5r',
    'train_events_per_second', 'validation_events_per_second',
    'train_events', 'validation_events', 'train_usable_events',
    'validation_usable_events', 'train_excluded_events', 'validation_fallback_events',
    'rank_peak_reserved_gpu_bytes', 'best_epoch',
}
FIELDS = {
    'experiment_registered': {'model', 'configuration', 'comparison_identity'},
    'attempt_submitted': {'job_id', 'cluster'},
    'submission_unknown': {'reason'},
    'attempt_started': {'job_id', 'cluster'},
    'resume_confirmed': {'parent_attempt_id', 'cutoff'},
    'heartbeat': {'phase', 'epoch', 'completed_epoch', 'progress_at', 'gpus'},
    'progress': {'phase', 'epoch', 'completed_epoch', 'progress_at'},
    'epoch_committed': {'epoch', 'history_revision'} | METRICS,
    'training_finished': {'completion_reason'},
    'training_failed': {'error_code'},
    'batch_exit_reported': {'exit_code'},
    'scheduler_snapshot': {'jobs', 'cluster'},
}


def identifier(value):
    """Stable mapping for legacy IDs; never infer identity from model names."""
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, 'rd-monitor:' + str(value)))


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def encode(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':'), sort_keys=True).encode('utf-8')


def public_config(value, depth=0):
    """Do not project credentials, host paths or unbounded arbitrary objects."""
    if depth > 5:
        return None
    if isinstance(value, dict):
        return {str(k): public_config(v, depth+1) for k,v in list(value.items())[:80]
                if not re.search(r'key|token|secret|password|path|directory|root|resume', str(k), re.I)}
    if isinstance(value, list):
        return [public_config(v, depth+1) for v in value[:100]]
    if isinstance(value, str):
        return '[local path]' if value.startswith(('/', '\\')) or re.match(r'^[A-Za-z]:[\\/]', value) else value[:500]
    if value is None or isinstance(value, (bool,int)):
        return value
    if isinstance(value,float):
        return value if math.isfinite(value) else None
    return None


def validate_event(event):
    required = {'event_id','experiment_id','attempt_id','sequence','kind','observed_at','payload'}
    if not isinstance(event,dict) or set(event) != required:
        raise ValueError('Invalid event envelope')
    for key in ('event_id','experiment_id','attempt_id'):
        uuid.UUID(event[key])
    if type(event['sequence']) is not int or not 0 < event['sequence'] < 2**53:
        raise ValueError('Invalid sequence')
    stamp = datetime.fromisoformat(event['observed_at'].replace('Z','+00:00'))
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError('Timestamp must be UTC')
    payload = event['payload']
    kind = event['kind']
    if kind not in FIELDS or not isinstance(payload,dict) or set(payload)-FIELDS[kind]:
        raise ValueError('Invalid event kind or fields')
    if kind == 'experiment_registered' and (not isinstance(payload.get('model'),str) or len(payload['model'])>100):
        raise ValueError('Model required')
    if kind == 'epoch_committed':
        if type(payload.get('epoch')) is not int or payload['epoch'] < 1:
            raise ValueError('Invalid epoch')
        if payload.get('history_revision') != event['attempt_id']:
            raise ValueError('Revision must identify this attempt')
    if kind == 'resume_confirmed':
        uuid.UUID(payload['parent_attempt_id'])
        if type(payload.get('cutoff')) is not int or payload['cutoff'] < 0:
            raise ValueError('Invalid resume cutoff')
    for key in METRICS & payload.keys():
        number = payload[key]
        if number is not None and (type(number) not in (int,float) or not math.isfinite(number)):
            raise ValueError('Invalid metric')
        if key.endswith('mt5r') and number is not None and not 0 <= number <= 1:
            raise ValueError('Recall must be a fraction')
    if kind == 'batch_exit_reported' and (type(payload.get('exit_code')) is not int or not 0 <= payload['exit_code'] <= 255):
        raise ValueError('Invalid exit code')
    if len(encode(event)) > MAX_BATCH_BYTES-1024:
        raise ValueError('Event too large')
    return event


def new_event(experiment, attempt, sequence, kind, payload, observed_at=None, event_id=None):
    return validate_event({'event_id':event_id or str(uuid.uuid4()), 'experiment_id':identifier(experiment),
        'attempt_id':identifier(attempt), 'sequence':sequence, 'kind':kind,
        'observed_at':observed_at or utc_now(), 'payload':payload})
