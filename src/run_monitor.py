"""Optional, failure-isolated file telemetry. No network or extra packages."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import warnings
import sys
import uuid


class RunMonitor:
    def __init__(self):
        self.path = os.environ.get("RND_MONITOR_RUN_DIR")
        self.state = {"phase": "initializing", "host": socket.gethostname()}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.publisher = None
        self.attempt_dir = None
        self.epochs = []
        self.api_publisher = None
        self.outbox = None
        self.parent_attempt = None

    def configure(self, output_dir, model, resume=None):
        """Establish one writer per execution. Monitoring errors never alter training."""
        if int(os.environ.get('RANK','0')) != 0:
            self.path = None
            return
        self.publisher = None
        self.api_publisher = None
        self.outbox = None
        self.attempt_dir = None
        self.parent_attempt = None
        try:
            output = Path(output_dir)
            self.path = str(output)
            directory = output / 'monitoring'
            directory.mkdir(parents=True, exist_ok=True)
            identity_path = directory / 'experiment.json'
            if identity_path.exists() and (resume or os.environ.get('RND_MONITOR_ATTEMPT_ID')):
                identity = json.loads(identity_path.read_text(encoding='utf-8'))
            else:
                legacy = output / 'run.json'
                previous = json.loads(legacy.read_text(encoding='utf-8')) if legacy.exists() else {}
                identity = {'schema_version':2,'experiment_id':previous.get('run_id') or uuid.uuid4().hex,'model':model}
                self.atomic(identity_path, identity)
            attempt_id = os.environ.get('RND_MONITOR_ATTEMPT_ID') or uuid.uuid4().hex
            if int(os.environ.get('SLURM_RESTART_COUNT','0')):
                attempt_id = uuid.uuid4().hex
            previous_attempts = sorted((directory / 'attempts').glob('*/attempt.json'), key=lambda p:p.stat().st_mtime)
            if resume:
                previous_attempts = [p for p in previous_attempts if p.parent.name != attempt_id]
                if previous_attempts:
                    self.parent_attempt = json.loads(previous_attempts[-1].read_text(encoding='utf-8')).get('attempt_id')
            self.attempt_dir = directory / 'attempts' / attempt_id
            self.attempt_dir.mkdir(parents=True, exist_ok=True)
            self.metadata = {**identity,'attempt_id':attempt_id,'job_id':os.environ.get('SLURM_JOB_ID'),
                             'submitted':time.time(),'resumed_from_epoch':None,'configuration':{}}
            self.atomic(self.attempt_dir / 'attempt.json',self.metadata)
            self.state = {'phase':'initializing','host':socket.gethostname(),'attempt_id':attempt_id}
            self.epochs = []
            self.atomic(directory / 'active-attempt.json', {'attempt_dir':str(self.attempt_dir.resolve())})
            if os.environ.get('RND_MONITOR_API_URL') and os.environ.get('RND_MONITOR_UPLOAD_KEY_FILE'):
                from src.monitoring.outbox import Outbox
                self.outbox = Outbox(self.attempt_dir,identity['experiment_id'],attempt_id)
                self.outbox.append('experiment_registered',{'model':model,'configuration':{}})
                self.outbox.append('attempt_started',{'job_id':os.environ.get('SLURM_JOB_ID'),'cluster':os.environ.get('SLURM_CLUSTER_NAME','wice')})
                self.api_publisher = subprocess.Popen([sys.executable,str(Path(__file__).resolve().parents[1] / 'scripts/monitor_upload.py'),'--attempt-dir',str(self.attempt_dir)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except Exception as error:
            warnings.warn(f'Monitoring setup unavailable ({type(error).__name__}); training continues')

    @staticmethod
    def atomic(path, value):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
        temporary.replace(path)

    def epoch(self, record, configuration=None, resumed_from_epoch=None):
        if not self.attempt_dir:
            return
        try:
            self.epochs.append(record)
            self.atomic(self.attempt_dir / 'epochs.json',self.epochs)
            self.metadata['configuration'] = configuration or self.metadata['configuration']
            self.metadata['resumed_from_epoch'] = resumed_from_epoch
            self.atomic(self.attempt_dir / 'attempt.json',self.metadata)
            if self.outbox:
                from src.monitoring.api_contracts import METRICS, identifier, public_config
                if len(self.epochs)==1:
                    self.outbox.append('experiment_registered',{'model':self.metadata['model'],'configuration':public_config(configuration or {})})
                metrics = {key:value for key,value in record.items() if key in METRICS}
                self.outbox.append('epoch_committed',{'epoch':int(record['epoch']),'history_revision':identifier(self.metadata['attempt_id']),**metrics})
            if self.publisher is None and os.environ.get('RND_WANDB_MODE','disabled') in ('online','offline'):
                self.publisher = subprocess.Popen([sys.executable,str(Path(__file__).resolve().parents[1] / 'scripts/wandb_publish.py'), '--attempt-dir',str(self.attempt_dir)], stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except Exception as error:
            warnings.warn(f'Monitoring epoch unavailable ({type(error).__name__}); training continues')

    def confirm_resume(self, cutoff, configuration=None):
        if not self.outbox:
            return
        try:
            from src.monitoring.api_contracts import identifier, public_config
            self.outbox.append('experiment_registered',{'model':self.metadata['model'],'configuration':public_config(configuration or {})})
            if self.parent_attempt:
                self.outbox.append('resume_confirmed',{'parent_attempt_id':identifier(self.parent_attempt),'cutoff':int(cutoff)})
        except Exception as error:
            warnings.warn(f'API resume event unavailable ({type(error).__name__}); training continues')

    def update(self, **values):
        with self.lock:
            if any(self.state.get(key) != value for key,value in values.items()):
                self.state["progress_at"] = time.time()
            self.state.update(values)

    def write(self):
        if not self.path:
            return
        try:
            with self.lock:
                payload = {**self.state, "heartbeat": time.time()}
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                )
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                payload["gpus"] = []
                for line in result.stdout.splitlines():
                    index, uuid, utilization, used, total = [v.strip() for v in line.split(",")]
                    if visible != [""] and index not in visible and not any(uuid.startswith(v) for v in visible):
                        continue
                    payload["gpus"].append({"index": index, "uuid": uuid,
                        "utilization": float(utilization), "memory_used": float(used), "memory_total": float(total)})
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            directory = Path(self.path)
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / "telemetry.json.tmp"
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(directory / "telemetry.json")
            if self.outbox:
                values = {key:payload[key] for key in ('phase','epoch','completed_epoch','progress_at','gpus') if key in payload}
                self.outbox.append('heartbeat',values)
        except Exception as error:
            warnings.warn(f"Monitoring write failed; training continues: {error}")

    def _loop(self):
        while not self.stop.is_set():
            self.write()
            self.stop.wait(30)

    def __enter__(self):
        self.stop.clear()
        if self.path:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.update(phase="training_failed" if error else "training_finished", error=str(error) if error else None)
        self.write()
        if self.attempt_dir:
            try:
                if self.outbox:
                    self.outbox.append('training_failed' if error else 'training_finished',
                        {'error_code':type(error).__name__} if error else {'completion_reason':self.state.get('completion_reason','finished')})
                self.atomic(self.attempt_dir / 'finish.json', {'failed':bool(error),'finished_at':time.time()})
                for child in (self.publisher,self.api_publisher):
                    if child is None:
                        continue
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:
                            child.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            child.kill()
            except Exception:
                pass


monitor = RunMonitor()
