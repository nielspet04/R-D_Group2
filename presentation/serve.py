#!/usr/bin/env python3
"""Start the offline presentation. Run: python3 serve.py"""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os, webbrowser, socket
os.chdir(Path(__file__).resolve().parent)
url = 'http://localhost:8765/'
with socket.socket() as probe:
    if probe.connect_ex(('127.0.0.1',8765)) == 0:
        webbrowser.open(url)
        raise SystemExit('A server is already running on port 8765; opened it in your browser.')
print(f'Presentation: {url}\nSpeaker view: press S\nStop server: Ctrl+C', flush=True)
webbrowser.open(url)
ThreadingHTTPServer(('127.0.0.1',8765), SimpleHTTPRequestHandler).serve_forever()
