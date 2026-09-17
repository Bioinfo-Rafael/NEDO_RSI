"""Serve the built viewer and runs on loopback, without modifying upstream files."""
import argparse
import errno
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        route = unquote(urlsplit(path).path)
        base = ROOT / ('runs' if route.startswith('/runs/') else 'frontend/dist')
        suffix = route[len('/runs/'):] if route.startswith('/runs/') else route.lstrip('/')
        target = (base / suffix).resolve()
        if not target.is_relative_to(base.resolve()):
            return str(ROOT / '.no-such-file')
        return str(target)

    def list_directory(self, path):
        self.send_error(403, 'Directory listing disabled')

    def do_GET(self):
        if self.path == '/':
            self.send_response(302)
            self.send_header('Location', self.server.default_url)
            self.end_headers()
            return
        # A selected run can be viewed before its first append. Serve an empty
        # snapshot without creating got.json or fabricating a research node.
        route = unquote(urlsplit(self.path).path)
        if route == self.server.pending_route and not Path(self.translate_path(self.path)).exists():
            body = json.dumps({"meta": self.server.pending_meta, "nodes": []}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', nargs='?', default='nedo-rsi-test')
    parser.add_argument('session', nargs='?', default='smoke-test')
    parser.add_argument('--port', type=int, default=4500)
    args = parser.parse_args()
    for value in (args.project, args.session):
        if value in ('', '.', '..') or '/' in value or '\\' in value:
            parser.error('project and session must each be a single non-empty path segment')
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        parser.exit(1, f'Port {args.port} is already in use. Stop the existing Viewer with Ctrl+C, '
                    f'or start this Viewer with --port {args.port + 1}.\n')
    server.pending_route = f'/runs/{args.project}/{args.session}/got.json'
    server.pending_meta = {'project_name': args.project, 'session_id': args.session}
    server.default_url = '/?src=/runs/' + quote(args.project, safe='') + '/' + quote(args.session, safe='') + '/got.json'
    print(f'Viewer: http://127.0.0.1:{args.port}{server.default_url}', flush=True)
    if not (ROOT / 'runs' / args.project / args.session / 'got.json').exists():
        print('Waiting for the first append_trace_node for this project/session. No trace has been recorded yet.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
