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


class RunMonitor:
    def __init__(self):
        self.path = os.environ.get("RND_MONITOR_RUN_DIR")
        self.state = {"phase": "initializing", "host": socket.gethostname()}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def update(self, **values):
        with self.lock:
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
        except Exception as error:
            warnings.warn(f"Monitoring write failed; training continues: {error}")

    def _loop(self):
        while not self.stop.is_set():
            self.write()
            self.stop.wait(30)

    def __enter__(self):
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


monitor = RunMonitor()
