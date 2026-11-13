"""Loopback-only presentation server with atomic, version-checked text saves."""
from datetime import datetime
from hashlib import sha256
from html import escape
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, unquote
import json
import secrets
import tempfile
import threading
import os
import re

VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

class EditableFields(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.source, self.stack, self.fields = source, [], {}
        self.lines = [0]
        for i, c in enumerate(source):
            if c == '\n': self.lines.append(i + 1)
        self.feed(source)
    def source_offset(self):
        line, col = self.getpos()
        return self.lines[line - 1] + col
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        key = attrs.get('data-edit-id')
        if tag not in VOID:
            self.stack.append((tag, key, self.source_offset() + len(self.get_starttag_text())))
    def handle_endtag(self, tag):
        while self.stack:
            current, key, start = self.stack.pop()
            if key:
                if key in self.fields: raise ValueError('Duplicate editable field')
                self.fields[key] = (start, self.source_offset())
            if current == tag: break

class SafeTextHTML(HTMLParser):
    allowed = {'b', 'strong', 'em', 'i', 'br', 'span', 'div', 'p', 'u', 'sub', 'sup'}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag in self.allowed: self.parts.append('<' + tag + '>')
    def handle_endtag(self, tag):
        if tag in self.allowed and tag != 'br': self.parts.append('</' + tag + '>')
    def handle_data(self, data): self.parts.append(escape(data, quote=False))
    def handle_startendtag(self, tag, attrs): self.handle_starttag(tag, attrs)

def revision(text): return sha256(text.encode('utf-8')).hexdigest()

def make_server(root, port=8765):
    root = Path(root).resolve()
    lock, token = threading.Lock(), secrets.token_urlsafe(32)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs): super().__init__(*args, directory=str(root), **kwargs)
        def allowed_host(self):
            try:
                host = urlsplit('http://' + self.headers.get('Host', '')).hostname
                origin = self.headers.get('Origin')
                return host in {'localhost', '127.0.0.1'} and (not origin or origin == 'http://' + self.headers.get('Host'))
            except ValueError: return False
        def reply(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if not self.allowed_host(): return self.reply(403, {'error': 'Local access only'})
            path = unquote(urlsplit(self.path).path)
            if '.backups' in Path(path).parts: return self.reply(404, {'error': 'Not found'})
            if path == '/api/editor-state':
                text = (root / 'index.html').read_text(encoding='utf-8')
                return self.reply(200, {'revision': revision(text), 'token': token})
            if path in {'/', '/index.html'}:
                text = (root / 'index.html').read_text(encoding='utf-8')
                data = re.sub(r'<body\b[^>]*>', lambda m: re.sub(r'\sdata-save-revision="[^"]*"', '', m[0])[:-1] + ' data-save-revision="' + revision(text) + '">', text, count=1).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers(); self.wfile.write(data)
                return
            return super().do_GET()
        def end_headers(self):
            self.send_header('X-Content-Type-Options', 'nosniff')
            if urlsplit(self.path).path.endswith(('.js', '.css')):
                self.send_header('Cache-Control', 'no-cache')
            super().end_headers()
        def do_POST(self):
            if not self.allowed_host() or self.headers.get('Authorization') != 'Bearer ' + token:
                return self.reply(403, {'error': 'Save authorization failed; reload the page'})
            if urlsplit(self.path).path != '/api/save': return self.reply(404, {'error': 'Not found'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.reply(415, {'error': 'Expected JSON'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 524288: raise ValueError('Invalid save size')
                request = json.loads(self.rfile.read(size))
                changes = request.get('changes')
                if not isinstance(changes, dict) or not changes or not all(isinstance(k, str) and isinstance(v, str) for k, v in changes.items()):
                    raise ValueError('Invalid text changes')
                with lock:
                    path = root / 'index.html'
                    text = path.read_text(encoding='utf-8')
                    if request.get('revision') != revision(text):
                        return self.reply(409, {'error': 'File changed elsewhere. Download your edits before reloading.'})
                    fields = EditableFields(text).fields
                    if set(changes) - set(fields): raise ValueError('Unknown editable field')
                    replacements = []
                    for key, value in changes.items():
                        safe = SafeTextHTML(); safe.feed(value); safe.close()
                        start, end = fields[key]
                        replacements.append((start, end, ''.join(safe.parts)))
                    updated = text
                    for start, end, value in sorted(replacements, reverse=True):
                        updated = updated[:start] + value + updated[end:]
                    if updated != text:
                        backups = root / '.backups'; backups.mkdir(exist_ok=True)
                        name = datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.html'
                        (backups / name).write_text(text, encoding='utf-8')
                        temporary = None
                        try:
                            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root, prefix='.index-', suffix='.tmp', delete=False) as out:
                                temporary = out.name; out.write(updated); out.flush(); os.fsync(out.fileno())
                            os.replace(temporary, path)
                        finally:
                            if temporary and os.path.exists(temporary): os.unlink(temporary)
                    return self.reply(200, {'revision': revision(updated), 'saved': True})
            except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
                return self.reply(400, {'error': 'Invalid save request'})
            except OSError:
                return self.reply(500, {'error': 'Could not write the file; download a backup'})
    return ThreadingHTTPServer(('127.0.0.1', port), Handler)
