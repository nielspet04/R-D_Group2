"""Local SQLite dashboard. Python standard library only; VSC access uses SSH."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def connect(path):
    db = sqlite3.connect(path, timeout=10)
    try:
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, payload TEXT, PRIMARY KEY(kind,id))")
            db.execute("CREATE TABLE IF NOT EXISTS samples (run_id TEXT, timestamp REAL, payload TEXT, PRIMARY KEY(run_id,timestamp))")
            yield db
    finally:
        db.close()


def ingest(path, snapshot):
    with connect(path) as db:
        # Preserve historical jobs; absence from a successful snapshot means unknown,
        # never an inferred success. sacct supplies authoritative terminal states.
        if not snapshot.get("errors"):
            seen = {j["job_id"] for j in snapshot["jobs"]}
            for key, payload in db.execute("SELECT id,payload FROM records WHERE kind='jobs'").fetchall():
                job = json.loads(payload)
                if key not in seen and job.get("state") in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}:
                    job["state"] = "UNKNOWN"
                    db.execute("UPDATE records SET payload=? WHERE kind='jobs' AND id=?", (json.dumps(job), key))
        for kind, key in [("jobs", "job_id"), ("runs", "run_id")]:
            for value in snapshot[kind]:
                db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)", (kind, value[key], json.dumps(value)))
                telemetry = value.get("telemetry", {})
                if kind == "runs" and telemetry.get("heartbeat"):
                    db.execute("INSERT OR IGNORE INTO samples VALUES (?,?,?)",
                               (value[key], telemetry["heartbeat"], json.dumps(telemetry)))
        db.execute("INSERT OR REPLACE INTO records VALUES ('meta','sync',?)", (json.dumps({
            "last_success": snapshot["collected_at"], "errors": snapshot.get("errors", []), "cluster": snapshot.get("cluster")}),))


def read_data(path):
    data = {"jobs": [], "runs": [], "meta": {}}
    with connect(path) as db:
        for kind, key, payload in db.execute("SELECT kind,id,payload FROM records"):
            if kind == "meta":
                data["meta"][key] = json.loads(payload)
            else:
                data[kind].append(json.loads(payload))
        for run in data["runs"]:
            run["samples"] = [json.loads(row[0]) for row in db.execute(
                "SELECT payload FROM samples WHERE run_id=? ORDER BY timestamp DESC LIMIT 300", (run["run_id"],))][::-1]
    data["jobs"].sort(key=lambda j: int(j["job_id"].split("_")[0]) if j["job_id"].split("_")[0].isdigit() else 0, reverse=True)
    data["runs"].sort(key=lambda r: r.get("submitted", 0), reverse=True)
    return data


class Collector:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.last_attempt = 0

    def refresh(self):
        if not self.lock.acquire(blocking=False):
            return
        try:
            if time.time() - self.last_attempt < 60:
                return
            self.last_attempt = time.time()
            if self.args.snapshot:
                snapshot = json.loads(self.args.snapshot.read_text(encoding="utf-8"))
            else:
                command = shlex.join([self.args.remote_python, "-", "--root", self.args.remote_root, "--cluster", self.args.cluster])
                result = subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    self.args.host, "bash -lc " + shlex.quote(command)],
                    input=(ROOT / "scripts/monitor_snapshot.py").read_text(encoding="utf-8"),
                    text=True, encoding="utf-8", capture_output=True, timeout=65)
                if result.returncode:
                    raise RuntimeError(result.stderr.strip() or f"SSH exit {result.returncode}")
                snapshot = json.loads(result.stdout)
            ingest(self.args.db, snapshot)
            with connect(self.args.db) as db:
                db.execute("DELETE FROM records WHERE kind='meta' AND id='connection_error'")
        except Exception as error:
            with connect(self.args.db) as db:
                db.execute("INSERT OR REPLACE INTO records VALUES ('meta','connection_error',?)",
                           (json.dumps({"message": str(error), "at": time.time()}),))
        finally:
            self.lock.release()

    def loop(self):
        while True:
            self.refresh()
            time.sleep(self.args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--host", help="Existing SSH alias or user@login-host")
    source.add_argument("--snapshot", type=Path, help="Import a JSON snapshot (offline/demo)")
    parser.add_argument("--remote-root", help="Absolute project path on VSC")
    parser.add_argument("--remote-python", default="python3", help="Python >=3.9 on VSC; absolute path allowed")
    parser.add_argument("--cluster", default="wice")
    parser.add_argument("--interval", type=int, default=0, help="0: manual refresh, otherwise >=60 seconds")
    parser.add_argument("--db", type=Path, default=ROOT / "results/monitoring/dashboard.sqlite3")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    if args.host and (not args.remote_root or not args.remote_root.startswith("/") or args.host.startswith("-")):
        parser.error("--host requires --remote-root with an absolute Linux path")
    if args.interval != 0 and args.interval < 60:
        parser.error("--interval must be 0 or at least 60 seconds")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    with connect(args.db):
        pass
    collector = Collector(args)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/data":
                payload = json.dumps(read_data(args.db)).encode()
                content_type = "application/json"
            elif self.path in {"/", "/index.html"}:
                payload = (ROOT / "monitoring/index.html").read_bytes()
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if self.path != "/api/refresh":
                self.send_error(404)
                return
            if self.headers.get("X-RND-Dashboard") != "1":
                self.send_error(403)
                return
            threading.Thread(target=collector.refresh, daemon=True).start()
            self.send_response(202)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=collector.loop if args.interval else collector.refresh, daemon=True).start()
    print(f"Dashboard: http://127.0.0.1:{args.port}\nDatabase: {args.db}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
