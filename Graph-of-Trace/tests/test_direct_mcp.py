"""Real stdio MCP calls create Baseline -> A/B -> Compare, without API keys."""
import asyncio
import json
import os
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[1]


async def create_dag(base: Path, project="nedo-rsi-direct", session_id="smoke-test"):
    output = base / project / session_id / "got.json"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing smoke trace: {output}")
    env = {**os.environ, "GOT_OUTPUT_BASE_DIR": str(base),
           "GOT_OUTPUT_PATH_TEMPLATE": "{base_dir}/{project_name}/{session_id}/got.json"}
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "AZURE_OPENAI_API_KEY"):
        env.pop(key, None)
    params = StdioServerParameters(command=sys.executable,
        args=[str(REPO / "tests/direct_stdio_server.py"), "--transport", "stdio"], cwd=str(REPO), env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            names = {tool.name for tool in tools}
            assert {"build_trace", "append_trace_node", "get_trace"} <= names
            query = {"project": {"name": project}, "session": {"id": session_id}}
            for index, (title, parents) in enumerate([
                ("Baseline", []), ("Ablation A", ["N001"]),
                ("Ablation B", ["N001"]), ("Compare A/B", ["N002", "N003"])
            ], 1):
                result = await session.call_tool("append_trace_node", {"payload": {**query, "node": {
                    "title": title, "description": "Installation smoke fixture only; no research experiment or metrics measured.",
                    "status": "completed", "parents": [{"id": pid, "relation": "necessitated_by",
                    "explanation": "Required input to this smoke DAG node"} for pid in parents], "artifacts": []}}})
                assert not result.isError, result
                data = json.loads(next(c.text for c in result.content if c.type == "text"))
                assert data["id"] == f"N{index:03d}"
            before = output.read_bytes()
            invalid = await session.call_tool("append_trace_node", {"payload": {**query, "node": {
                "title": "Invalid", "description": "Must not persist", "parents": [{"id": "N999"}]}}})
            assert invalid.isError
            assert "Unknown parent N999" in str(invalid)
            assert output.read_bytes() == before
            summary = await session.call_tool("get_trace", {"payload": query})
            assert not summary.isError, summary
            nodes = json.loads(next(c.text for c in summary.content if c.type == "text"))["nodes"]
            assert [n["title"] for n in nodes] == ["Baseline", "Ablation A", "Ablation B", "Compare A/B"]
            assert [p["id"] for p in nodes[3]["parents"]] == ["N002", "N003"]
            print(json.dumps({"tools": sorted(names), "nodes": len(nodes), "path": str(output), "network": "forbidden"}))
    return output


def test_direct_stdio_no_network(tmp_path):
    asyncio.run(create_dag(tmp_path))


if __name__ == "__main__":
    asyncio.run(create_dag(REPO / "runs"))
