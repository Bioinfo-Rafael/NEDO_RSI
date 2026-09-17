"""One MCP recording with a stubbed LLM; no provider/network requests."""
import asyncio
import json
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/nedo-rsi-test/smoke-test/got.json'


async def main():
    if OUTPUT.exists():
        raise SystemExit(f'Refusing to overwrite existing smoke trace: {OUTPUT}')
    params = StdioServerParameters(command=sys.executable,
        args=[str(ROOT / 'local-setup/smoke_server.py'), '--transport', 'stdio'], cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = [tool.name for tool in (await session.list_tools()).tools]
            assert 'build_trace' in names
            result = await session.call_tool('build_trace', {'payload': {
                'project': {'name': 'nedo-rsi-test'}, 'session': {'id': 'smoke-test'},
                'subtask': {'title': 'Verified Graph of Trace installation',
                    'description': 'Offline installation smoke test. LLM extraction stubbed; no paid API called.'},
                'artifacts': []}})
            assert not result.isError, result
            data = json.loads(OUTPUT.read_text())
            assert len(data['nodes']) == 2
            assert data['nodes'][1]['title'] == 'Verified Graph of Trace installation'
            print(json.dumps({'tools': names, 'output': str(OUTPUT), 'nodes': len(data['nodes']), 'llm': 'stubbed'}))


asyncio.run(main())
