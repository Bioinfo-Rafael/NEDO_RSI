"""Append caller-authored DAG nodes. No LLM or provider imports/calls."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .got_writer import _atomic_write_json, _resolve_got_path, _session_write_lock

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TraceProject(StrictModel):
    name: Text


class TraceSession(StrictModel):
    id: Text


class TraceQuery(StrictModel):
    project: TraceProject
    session: TraceSession


class TraceParent(StrictModel):
    id: Annotated[str, Field(pattern=r"^N[0-9]{3,}$")]
    relation: Literal["necessitated_by"] = "necessitated_by"
    explanation: str | None = None


class TraceArtifact(StrictModel):
    path: Text
    type: Text


class DirectNode(StrictModel):
    title: Text
    description: Text
    status: Text = "completed"
    parents: list[TraceParent] = Field(default_factory=list)
    artifacts: list[TraceArtifact] = Field(default_factory=list)


class AppendTracePayload(TraceQuery):
    node: DirectNode


def _read_graph(path: Path, project: str, session: str) -> dict:
    """Fail closed on malformed/colliding files; never silently reset a trace."""
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"meta": {"project_name": project, "session_id": session}, "nodes": []}
    if not isinstance(graph, dict) or not isinstance(graph.get("meta"), dict) or not isinstance(graph.get("nodes"), list):
        raise ValueError("Invalid got.json: expected meta object and nodes array")
    if graph["meta"].get("project_name") != project or graph["meta"].get("session_id") != session:
        raise ValueError("Trace project/session mismatch (possible sanitized-path collision)")
    _validate_dag(graph["nodes"])
    return graph


def _validate_dag(nodes: list[dict]) -> None:
    by_id = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Invalid node object")
        nid = node.get("id")
        if not isinstance(nid, str) or not re.fullmatch(r"N[0-9]{3,}", nid) or int(nid[1:]) < 1:
            raise ValueError("Invalid stored node id")
        if nid != f"N{int(nid[1:]):03d}" or nid in by_id:
            raise ValueError("Noncanonical or duplicate stored node id")
        by_id[nid] = node
    if nodes and "N001" not in by_id:
        raise ValueError("Missing N001 root")
    parents_by_id = {}
    children = {nid: [] for nid in by_id}
    for nid, node in by_id.items():
        parents = node.get("parents")
        if not isinstance(parents, list) or not parents:
            raise ValueError(f"Missing parents for {nid}")
        seen = set()
        for parent in parents:
            if not isinstance(parent, dict) or parent.get("relation") != "necessitated_by":
                raise ValueError(f"Invalid parent edge for {nid}")
            pid = parent.get("id")
            if not isinstance(pid, str) or pid not in by_id:
                raise ValueError(f"Unknown parent {pid!r} for {nid}")
            if pid in seen:
                raise ValueError(f"Duplicate stored edge for {nid}")
            seen.add(pid)
        # GoT's root self-edge is a schema sentinel, not a graph dependency.
        if nid == "N001":
            if seen != {"N001"}:
                raise ValueError("Root must reference only itself")
            seen = set()
        elif nid in seen:
            raise ValueError(f"Cycle: self-parent for {nid}")
        parents_by_id[nid] = seen
        for pid in seen:
            children[pid].append(nid)
    # Iterative topological check avoids recursion limits on long trajectories.
    degree = {nid: len(parents) for nid, parents in parents_by_id.items()}
    ready = [nid for nid, count in degree.items() if count == 0]
    visited = 0
    while ready:
        nid = ready.pop()
        visited += 1
        for child in children[nid]:
            degree[child] -= 1
            if degree[child] == 0:
                ready.append(child)
    if visited != len(nodes):
        raise ValueError("Cycle in existing trace; refusing to append")


async def append_node(payload: AppendTracePayload) -> dict:
    project, session = payload.project.name, payload.session.id
    path = _resolve_got_path(project, session)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with _session_write_lock(path.with_name(path.name + ".lock"), str(path)):
        graph = _read_graph(path, project, session)
        nodes = graph["nodes"]
        ids = {node["id"] for node in nodes}
        nid = f"N{max((int(i[1:]) for i in ids), default=0) + 1:03d}"
        node = payload.node.model_dump(exclude_none=True)
        if not nodes:
            if node["parents"]:
                raise ValueError("First node requires parents=[]; root self-edge is server-generated")
            node["parents"] = [{"id": nid, "relation": "necessitated_by"}]
        else:
            if not node["parents"]:
                raise ValueError("Non-root node requires at least one existing parent")
            unique = {}
            for parent in node["parents"]:
                if parent["id"] not in ids:
                    raise ValueError(f"Unknown parent {parent['id']}; use get_trace to select existing IDs")
                unique.setdefault(parent["id"], parent)
            node["parents"] = list(unique.values())
        node["id"] = nid
        # A new ID with only existing parents cannot introduce a cycle.
        nodes.append(node)
        _atomic_write_json(path, graph)
        return {"status": "ok", "id": nid, "parents": node["parents"], "path": str(path)}


async def read_trace(payload: TraceQuery) -> dict:
    path = _resolve_got_path(payload.project.name, payload.session.id)
    # Atomic replacement provides a complete old-or-new snapshot without a write lock.
    graph = _read_graph(path, payload.project.name, payload.session.id)
    return {"meta": graph["meta"], "nodes": [
        {"id": n["id"], "title": n.get("title", n["id"]), "parents": n["parents"], "status": n.get("status")}
        for n in graph["nodes"]
    ]}
