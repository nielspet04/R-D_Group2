"""SQLite dashboard for SSH, direct VSC, or offline snapshot collection."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse, parse_qs, unquote
import uuid

ROOT = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(ROOT))
from src.monitoring.store import connect, ingest, read_data, status, dumps


class Collector:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.last_attempt = 0
        self.cloud_lock = threading.Lock()
        self.cloud_attempt = 0
        self.state_lock = threading.Lock()
        self.requests = {}

    def refresh(self):
        if not self.lock.acquire(blocking=False):
            return
        try:
            if time.time() - self.last_attempt < 60:
                return
            self.last_attempt = time.time()
            if getattr(self.args, "wandb_only", False):
                return
            if self.args.snapshot:
                from scripts.monitor_snapshot import bounded_text
                snapshot = json.loads(bounded_text(self.args.snapshot, 32*1024*1024))
            elif self.args.local_root:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/monitor_snapshot.py"),
                        "--root",
                        str(self.args.local_root),
                        "--cluster",
                        self.args.cluster,
                    ],
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    timeout=65,
                )
                if result.returncode:
                    raise RuntimeError(
                        result.stderr.strip()
                        or f"Local collector exit {result.returncode}"
                    )
                snapshot = json.loads(result.stdout)
            else:
                command = shlex.join([self.args.remote_python, "-", "--root", self.args.remote_root, "--cluster", self.args.cluster])
                result = subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    self.args.host, "bash -lc " + shlex.quote(command)],
                    input=(ROOT / "scripts/monitor_snapshot.py").read_text(encoding="utf-8"),
                    text=True, encoding="utf-8", capture_output=True, timeout=65)
                if result.returncode:
                    raise RuntimeError(result.stderr.strip() or f"SSH exit {result.returncode}")
                snapshot = json.loads(result.stdout)
            if len(json.dumps(snapshot)) > 32*1024*1024:
                raise ValueError("Snapshot exceeds 32 MiB")
            ingest(self.args.db, snapshot)
            with connect(self.args.db) as db:
                db.execute("DELETE FROM records WHERE kind='meta' AND id='connection_error'")
        except Exception as error:
            status(self.args.db, "vsc", {"status":"error", "last_attempt":time.time(), "error":type(error).__name__})
            with connect(self.args.db) as db:
                db.execute("INSERT OR REPLACE INTO records VALUES ('meta','connection_error',?)",
                           (json.dumps({"message": str(error), "at": time.time()}),))
        finally:
            self.lock.release()

    def refresh_cloud(self):
        project = getattr(self.args, 'wandb_project', None)
        if not project or not self.cloud_lock.acquire(blocking=False):
            return
        try:
            if time.time()-self.cloud_attempt < 60:
                return
            self.cloud_attempt = time.time()
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/wandb_read.py'), project],
                                    capture_output=True, text=True, timeout=55)
            if result.returncode:
                raise RuntimeError('Cloud reader failed; check key, project permissions and optional dependency')
            if len(result.stdout) > 32*1024*1024:
                raise ValueError('Cloud snapshot exceeds limit')
            ingest(self.args.db, json.loads(result.stdout), source='wandb')
        except Exception as error:
            status(self.args.db, 'wandb', {'status':'error', 'last_attempt':time.time(), 'error':type(error).__name__})
        finally:
            self.cloud_lock.release()

    def request(self, sources):
        request_id = uuid.uuid4().hex
        outcome = {}
        now = time.time()
        for source in sources:
            enabled = not getattr(self.args, 'wandb_only', False) if source == 'vsc' else bool(getattr(self.args, 'wandb_project', None))
            lock = self.lock if source == 'vsc' else self.cloud_lock
            last = self.last_attempt if source == 'vsc' else self.cloud_attempt
            outcome[source] = 'disabled' if not enabled else 'coalesced' if lock.locked() else 'cooldown' if now-last < 60 else 'accepted'
        with self.state_lock:
            self.requests[request_id] = {'id':request_id, 'status':'running', 'sources':outcome, 'started_at':now}
            while len(self.requests) > 100:
                del self.requests[next(iter(self.requests))]
        def work():
            workers = []
            for source, state in outcome.items():
                if state == 'accepted':
                    worker = threading.Thread(target=self.refresh if source == 'vsc' else self.refresh_cloud, daemon=True)
                    workers.append(worker)
                    worker.start()
            for worker in workers:
                worker.join()
            for source,state in outcome.items():
                if state == 'coalesced':
                    lock = self.lock if source == 'vsc' else self.cloud_lock
                    with lock:
                        pass
            with self.state_lock:
                self.requests[request_id]['status'] = 'completed'
        threading.Thread(target=work, daemon=True).start()
        return self.requests[request_id]

    def cloud_loop(self):
        while True:
            self.refresh_cloud()
            time.sleep(120)

    def loop(self):
        while True:
            self.refresh()
            time.sleep(self.args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wandb-only", action="store_true", help="Read cloud experiments without SSH")
    source.add_argument("--host", help="Existing SSH alias or user@login-host")
    source.add_argument(
        "--local-root",
        type=Path,
        help="Collect directly on VSC; use with VS Code port forwarding",
    )
    source.add_argument("--snapshot", type=Path, help="Import a JSON snapshot (offline/demo)")
    parser.add_argument("--wandb-project", help="Optional entity/project; key comes from WANDB_API_KEY")
    parser.add_argument("--remote-root", help="Absolute project path on VSC")
    parser.add_argument("--remote-python", default="python3", help="Python >=3.9 on VSC; absolute path allowed")
    parser.add_argument("--cluster", default="wice")
    parser.add_argument("--interval", type=int, default=0, help="0: manual refresh, otherwise >=60 seconds")
    parser.add_argument("--db", type=Path, default=ROOT / "results/monitoring/dashboard.sqlite3")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument('--config', type=Path, help='Non-secret JSON configuration; explicit CLI options override it')
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        config = json.loads(preliminary.config.read_text(encoding='utf-8'))
        allowed = {'wandb_project','cluster','interval','db','port'}
        if not isinstance(config,dict) or set(config)-allowed:
            parser.error('Configuration contains unknown or secret fields')
        if 'db' in config:
            config['db'] = Path(config['db'])
        parser.set_defaults(**config)
    args = parser.parse_args()
    if args.wandb_only and not args.wandb_project:
        parser.error("--wandb-only requires --wandb-project entity/project")
    if args.host and (not args.remote_root or not args.remote_root.startswith("/") or args.host.startswith("-")):
        parser.error("--host requires --remote-root with an absolute Linux path")
    if args.local_root:
        args.local_root = args.local_root.resolve()
        if not args.local_root.is_dir():
            parser.error("--local-root must be an existing project directory")
        if args.remote_root:
            parser.error("--remote-root is only valid with --host")
    if args.interval != 0 and args.interval < 60:
        parser.error("--interval must be 0 or at least 60 seconds")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    with connect(args.db):
        pass
    collector = Collector(args)
    if args.wandb_only:
        status(args.db,'vsc',{'status':'disabled'})
    if not args.wandb_project:
        status(args.db,'wandb',{'status':'disabled'})

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, code=200):
            self.send_payload(dumps(value).encode(), 'application/json', code)

        def send_payload(self, payload, content_type, code=200):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(payload)

        def valid_host(self):
            return urlparse('http://' + self.headers.get('Host','')).hostname in ('localhost','127.0.0.1','::1')

        def do_GET(self):
            if not self.valid_host():
                self.send_error(403)
                return
            parsed = urlparse(self.path)
            route, query = parsed.path, parse_qs(parsed.query)
            if route in ('/', '/index.html', '/dashboard.js', '/dashboard.css'):
                name = 'index.html' if route == '/' else route[1:]
                content_type = {'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','css':'text/css; charset=utf-8'}[name.rsplit('.',1)[-1]]
                self.send_payload((ROOT / 'monitoring' / name).read_bytes(), content_type)
                return
            data = read_data(args.db)
            if route == '/api/data':
                self.send_json(data)
            elif route == '/api/status':
                self.send_json({'version':2, 'sources':data['meta'], 'requests':list(collector.requests.values())})
            elif route.startswith('/api/refresh/'):
                request = collector.requests.get(route.split('/')[-1])
                self.send_json(request or {'error':'Unknown request'}, 200 if request else 404)
            elif route in ('/api/experiments','/api/jobs'):
                try:
                    limit = min(200,max(1,int(query.get('limit',['50'])[0])))
                    offset = max(0,int(query.get('cursor',['0'])[0]))
                except ValueError:
                    self.send_json({'error':'Invalid pagination'},400)
                    return
                items = data['runs'] if route.endswith('experiments') else data['jobs']
                needle = query.get('q',[''])[0].lower()
                items = [r for r in items if needle in dumps(r).lower()]
                summaries = []
                for r in items[offset:offset+limit]:
                    summary = {k:v for k,v in r.items() if k not in ('history','samples','log_tail')}
                    history = r.get('history',[])
                    summary['latest'] = history[-1] if history else {}
                    valid = [h for h in history if h.get('action_mt5r') not in (None,'')]
                    summary['best'] = max(valid, key=lambda h:float(h['action_mt5r'])) if valid else {}
                    summaries.append(summary)
                self.send_json({'items':summaries,'next_cursor':offset+limit if offset+limit < len(items) else None})
            elif route == '/api/compare':
                ids = query.get('ids',[''])[0].split(',')
                if len(ids)>6:
                    self.send_json({'error':'Select at most six experiments'},400)
                    return
                selected = [r for r in data['runs'] if r['run_id'] in ids]
                self.send_json({'runs':selected})
            elif route.startswith('/api/experiments/'):
                run_id = unquote(route.split('/')[3])
                run = next((r for r in data['runs'] if r['run_id']==run_id), None)
                self.send_json(run or {'error':'Unknown experiment'},200 if run else 404)
            else:
                self.send_error(404)

        def do_POST(self):
            if not self.valid_host():
                self.send_error(403)
                return
            if self.path != '/api/refresh':
                self.send_error(404)
                return
            origin = self.headers.get('Origin')
            if self.headers.get('X-RND-Dashboard') != '1' or (origin and urlparse(origin).netloc != self.headers.get('Host')):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get('Content-Length',0))
                if not 0 <= length <= 1024:
                    raise ValueError()
                body = json.loads(self.rfile.read(length)) if length else {}
                sources = body.get('sources',['vsc','wandb'])
                if not isinstance(sources,list) or not sources or any(s not in ('vsc','wandb') for s in sources):
                    raise ValueError()
            except (ValueError, AttributeError):
                self.send_json({'error':'Invalid sources/request'},400)
                return
            request = collector.request(sources)
            states = list(request['sources'].values())
            code = 202 if any(s in ('accepted','coalesced') for s in states) else 429 if 'cooldown' in states else 200
            self.send_json(request,code)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=collector.loop if args.interval else collector.refresh, daemon=True).start()
    if args.wandb_project:
        threading.Thread(target=collector.cloud_loop, daemon=True).start()
    print(f"Dashboard: http://127.0.0.1:{args.port}\nDatabase: {args.db}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
