"""Inspect tools using installed Codex's actual MCP client, without a model turn."""
import asyncio
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'Graph-of-Trace/.cache'


async def main():
    env = dict(os.environ, CODEX_HOME=str(ROOT / '.codex-got'))
    with (CACHE / 'codex-direct-mcp.log').open('w') as log:
        proc = await asyncio.create_subprocess_exec('codex', 'app-server', '--stdio',
            env=env, cwd=str(ROOT), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=log)

        async def rpc(i, method, params):
            proc.stdin.write((json.dumps({'id': i, 'method': method, 'params': params}) + '\n').encode())
            await proc.stdin.drain()
            while True:
                raw = await asyncio.wait_for(proc.stdout.readline(), 40)
                if not raw:
                    raise RuntimeError('Codex exited unexpectedly')
                message = json.loads(raw)
                if message.get('id') == i:
                    assert 'error' not in message, message
                    return message['result']

        try:
            await rpc(1, 'initialize', {'clientInfo': {'name': 'got_direct_check', 'version': '1.0'},
                'capabilities': {'experimentalApi': True}})
            proc.stdin.write(b'{"method":"initialized"}\n')
            await proc.stdin.drain()
            result = await rpc(2, 'mcpServerStatus/list', {})
            server = next(s for s in result['data'] if s['name'] == 'graph_of_trace')
            assert {'build_trace', 'append_trace_node', 'get_trace'} <= set(server['tools'])
            (CACHE / 'codex-direct-mcp-status.json').write_text(json.dumps(result, indent=2))
            (CACHE / 'direct-tool-schemas.json').write_text(json.dumps({
                name: server['tools'][name]['inputSchema'] for name in ('append_trace_node', 'get_trace')}, indent=2))
            print('Codex MCP recognized:', ', '.join(sorted(server['tools'])))
        finally:
            proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except TimeoutError:
                proc.terminate()
                await proc.wait()


asyncio.run(main())
