#!/usr/bin/env python3
"""Run: python3 serve.py. Browser edits autosave to index.html."""
from pathlib import Path
from editor_server import make_server
import argparse
import webbrowser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    url = 'http://localhost:%d/' % args.port
    try:
        server = make_server(Path(__file__).resolve().parent, args.port)
    except OSError as error:
        raise SystemExit('Could not start the editor server. Stop the old server on port %d first. %s' % (args.port, error))
    print('Presentation with autosave: %s\nSpeaker view: S\nStop: Ctrl+C' % url, flush=True)
    if not args.no_browser: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
