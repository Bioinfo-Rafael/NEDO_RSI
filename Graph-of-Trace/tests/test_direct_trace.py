"""Direct DAG recording: real writer, no LLM, plus process-level lock tests."""
import asyncio
import json
import os
from pathlib import Path
import socket
import sys

import pytest
from pydantic import ValidationError

from tool import append_trace_node, get_trace


def payload(title="Baseline", parents=()):
    return {"project": {"name": "direct-test"}, "session": {"id": "run"}, "node": {
        "title": title, "description": "Measured accuracy=0.8; artifact results/metrics.json",
        "status": "completed", "parents": [{"id": p, "relation": "necessitated_by",
        "explanation": "Required evidence"} for p in parents],
        "artifacts": [{"path": "results/metrics.json", "type": "json"}]}}


@pytest.fixture
def output(tmp_path, monkeypatch):
    monkeypatch.setenv("GOT_OUTPUT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("GOT_OUTPUT_PATH_TEMPLATE", "{base_dir}/{project_name}/{session_id}/got.json")
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "AZURE_OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path / "direct-test/run/got.json"


def append(title="Baseline", parents=()):
    return asyncio.run(append_trace_node(payload(title, parents)))


def test_root(output):
    result = append()
    assert result["id"] == "N001"
    node = json.loads(output.read_text())["nodes"][0]
    assert node["title"] == "Baseline"
    assert node["parents"] == [{"id": "N001", "relation": "necessitated_by"}]
    assert node["status"] == "completed"


def test_child_siblings_multi_parent_and_schema(output):
    append()
    assert append("A", ["N001"])["id"] == "N002"
    assert append("B", ["N001"])["id"] == "N003"
    assert append("Compare", ["N002", "N003"])["id"] == "N004"
    graph = json.loads(output.read_text())
    assert [p["id"] for p in graph["nodes"][3]["parents"]] == ["N002", "N003"]
    # Field contract in frontend/src/got-to-reactflow.ts (GoTGraph/GoTNode).
    for node in graph["nodes"]:
        assert all(isinstance(node[k], str) for k in ("id", "title", "description", "status"))
        assert all(isinstance(p["id"], str) and isinstance(p["relation"], str) for p in node["parents"])
        assert all(isinstance(a["path"], str) and isinstance(a["type"], str) for a in node["artifacts"])
    query = {k: payload()[k] for k in ("project", "session")}
    summary = asyncio.run(get_trace(query))
    assert len(summary["nodes"]) == 4
    assert set(summary["nodes"][0]) == {"id", "title", "parents", "status"}


@pytest.mark.parametrize("parents", [["N999"], ["N002"], []])
def test_invalid_parent_no_write(output, parents):
    append()
    before = output.read_bytes()
    with pytest.raises(ValueError, match="parent"):
        append("Invalid", parents)
    assert output.read_bytes() == before


def test_root_rejects_supplied_parent(output):
    with pytest.raises(ValueError, match="First node"):
        append(parents=["N001"])
    assert not output.exists()


def test_duplicate_edges(output):
    append()
    result = append("A", ["N001", "N001"])
    assert len(result["parents"]) == 1


def test_client_cannot_assign_id(output):
    p = payload()
    p["node"]["id"] = "N010"
    with pytest.raises(ValidationError, match="Extra inputs"):
        asyncio.run(append_trace_node(p))
    assert not output.exists()


def test_corrupt_file_is_preserved(output):
    output.parent.mkdir(parents=True)
    output.write_text("broken JSON")
    with pytest.raises(ValueError):
        append()
    assert output.read_text() == "broken JSON"


def test_existing_cycle_is_rejected(output):
    append()
    append("A", ["N001"])
    append("B", ["N002"])
    graph = json.loads(output.read_text())
    graph["nodes"][1]["parents"][0]["id"] = "N003"
    output.write_text(json.dumps(graph))
    before = output.read_bytes()
    with pytest.raises(ValueError, match="Cycle"):
        append("C", ["N001"])
    assert output.read_bytes() == before


def test_sanitized_path_collision_rejected(output):
    append()
    p = payload()
    p["project"]["name"] = "../direct-test"
    with pytest.raises(ValueError, match="mismatch"):
        asyncio.run(append_trace_node(p))


def test_get_missing_does_not_write(output):
    query = {k: payload()[k] for k in ("project", "session")}
    assert asyncio.run(get_trace(query))["nodes"] == []
    assert not output.parent.exists()


def test_no_llm_or_network(output, monkeypatch):
    from Monitor import steps_llm
    import httpx

    def forbidden(*args, **kwargs):
        pytest.fail("External API/LLM/network path reached")

    monkeypatch.setattr(steps_llm, "build_nodes", forbidden)
    monkeypatch.setattr(steps_llm, "get_chat_adapter", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket.socket, "sendto", forbidden)
    test_child_siblings_multi_parent_and_schema(output)


def test_concurrent_coroutines(output):
    append()

    async def drive():
        return await asyncio.gather(*(append_trace_node(payload(str(i), ["N001"])) for i in range(20)))

    results = asyncio.run(drive())
    assert len({r["id"] for r in results}) == 20
    assert len(json.loads(output.read_text())["nodes"]) == 21


def test_concurrent_processes(output):
    append()
    code = "import asyncio,json,sys; from tool import append_trace_node; asyncio.run(append_trace_node(json.loads(sys.argv[1])))"

    async def drive():
        procs = [await asyncio.create_subprocess_exec(sys.executable, "-c", code,
            json.dumps(payload(str(i), ["N001"])), cwd=str(Path(__file__).resolve().parents[1]),
            env=dict(os.environ), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            for i in range(8)]
        for proc in procs:
            _, err = await asyncio.wait_for(proc.communicate(), 20)
            assert proc.returncode == 0, err.decode()

    asyncio.run(drive())
    nodes = json.loads(output.read_text())["nodes"]
    assert [n["id"] for n in nodes] == [f"N{i:03d}" for i in range(1, 10)]


def test_legacy_build_trace_still_works(output, monkeypatch):
    from Monitor import steps_llm
    from tool import build_trace

    async def extraction(**kwargs):
        return [{"id": "N002", "title": "Legacy", "parents": [{"id": "N001", "relation": "necessitated_by"}]}]

    monkeypatch.setattr(steps_llm, "build_nodes", extraction)
    p = payload()
    legacy = {"project": p["project"], "session": p["session"],
              "subtask": {"title": "Legacy", "description": "Legacy path"}, "artifacts": []}
    assert asyncio.run(build_trace(legacy))["status"] == "ok"
    assert append("Direct after legacy", ["N002"])["id"] == "N003"
