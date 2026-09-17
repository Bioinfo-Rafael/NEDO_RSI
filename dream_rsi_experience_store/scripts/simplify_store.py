#!/usr/bin/env python3
"""Convert the normalized Experience Store into the three-table replay store."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "dream_rsi_experience_store" / "schema.sql"
DEFAULT_DB = ROOT / "experience_store" / "experience.db"

METHOD_PRIORITY = {
    "explicit": 5,
    "filesystem": 4,
    "generation_id": 3,
    "git": 2,
    "temporal_inference": 1,
    "temporal_sequence": 1,
    "inferred": 1,
}


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_json(text: str | None, default: object) -> object:
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def copy_database(source: Path, target: Path) -> None:
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def best_bindings(con: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    selected: dict[str, sqlite3.Row] = {}
    for row in con.execute("SELECT * FROM node_event_bindings ORDER BY event_id, binding_id"):
        old = selected.get(row["event_id"])
        rank = (float(row["confidence"]), METHOD_PRIORITY.get(row["binding_method"], 0))
        if old is None:
            selected[row["event_id"]] = row
            continue
        old_rank = (float(old["confidence"]), METHOD_PRIORITY.get(old["binding_method"], 0))
        if rank > old_rank:
            selected[row["event_id"]] = row
    return selected


def metric_json(con: sqlite3.Connection, node_id: str) -> str:
    grouped: dict[str, list[tuple[float | None, str | None, str | None]]] = defaultdict(list)
    rows = con.execute(
        """SELECT m.name,m.value,m.unit,m.direction
           FROM node_metric_links l JOIN metrics m ON m.metric_id=l.metric_id
           WHERE l.node_id=? ORDER BY m.name,m.metric_id""",
        (node_id,),
    )
    for row in rows:
        item = (row["value"], row["unit"], row["direction"] or "unknown")
        if item not in grouped[row["name"]]:
            grouped[row["name"]].append(item)

    result: dict[str, object] = {}
    for name, items in sorted(grouped.items()):
        encoded = []
        for value, unit, direction in items:
            if unit is None and direction == "unknown":
                encoded.append(value)
            else:
                detail: dict[str, object] = {"value": value}
                if unit is not None:
                    detail["unit"] = unit
                if direction != "unknown":
                    detail["direction"] = direction
                encoded.append(detail)
        result[name] = encoded[0] if len(encoded) == 1 else encoded
    return compact_json(result)


def artifact_json(con: sqlite3.Connection, node_id: str) -> str:
    rows = con.execute(
        """SELECT a.original_path,a.artifact_type,a.content_hash,a.git_commit,
                  a.mime_type,a.size_bytes
           FROM node_artifact_links l JOIN artifacts a ON a.artifact_id=l.artifact_id
           WHERE l.node_id=? ORDER BY a.original_path,a.artifact_id""",
        (node_id,),
    )
    result: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for row in rows:
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        item = {"path": row["original_path"], "type": row["artifact_type"]}
        for source, target in (
            ("content_hash", "sha256"),
            ("git_commit", "git_commit"),
            ("mime_type", "mime_type"),
            ("size_bytes", "size_bytes"),
        ):
            if row[source] is not None:
                item[target] = row[source]
        result.append(item)
    return compact_json(result)


def node_type(node: sqlite3.Row) -> str:
    proposal = (node["proposal_text"] or "").lower()
    metadata = parse_json(node["metadata_json"], {})
    experiment_id = metadata.get("experiment_id") if isinstance(metadata, dict) else None
    if experiment_id == "baseline" or "initial synchronous linear baseline" in proposal:
        return "baseline"
    if node["primary_score"] is not None:
        return "experiment"
    if any(word in proposal for word in ("launcher", "server", "session info", "automatic approval")):
        return "infrastructure"
    if node["git_commit_after"]:
        return "code_improvement"
    return "unknown"


def node_status(node: sqlite3.Row) -> str:
    if node["fail_class"] or node["error_text"] or node["valid"] == 0:
        return "failed"
    if node["completed_at"] or node["evaluated"] == 1 or node["valid"] == 1 or node["git_commit_after"]:
        return "completed"
    return "unknown"


def tree_fields(nodes: dict[str, sqlite3.Row], node_id: str) -> tuple[int, str]:
    path: list[str] = []
    seen: set[str] = set()
    current: str | None = node_id
    while current:
        if current in seen:
            raise ValueError(f"cycle detected while building path for {node_id}")
        seen.add(current)
        path.append(current)
        parent = nodes.get(current)
        if parent is None:
            raise ValueError(f"missing node while building path for {node_id}: {current}")
        current = parent["parent_node_id"]
    path.reverse()
    return len(path) - 1, "/".join(path)


def convert(source: Path, output: Path) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".legacy.db", delete=False) as handle:
        legacy_copy = Path(handle.name)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".new.db", delete=False) as handle:
        new_db = Path(handle.name)
    legacy_copy.unlink(missing_ok=True)
    new_db.unlink(missing_ok=True)

    try:
        copy_database(source, legacy_copy)
        old = sqlite3.connect(f"file:{legacy_copy}?mode=ro", uri=True)
        old.row_factory = sqlite3.Row
        new = sqlite3.connect(new_db)
        new.row_factory = sqlite3.Row
        new.execute("PRAGMA foreign_keys=ON")
        new.executescript(SCHEMA.read_text(encoding="utf-8"))

        kept_runs = {
            row["run_id"]: row
            for row in old.execute(
                """SELECT * FROM discovery_runs r
                   WHERE EXISTS (SELECT 1 FROM discovery_nodes n WHERE n.run_id=r.run_id)
                   ORDER BY r.start_time,r.run_id"""
            )
        }
        nodes = {row["node_id"]: row for row in old.execute("SELECT * FROM discovery_nodes")}
        bindings = best_bindings(old)

        sessions_by_run: dict[str, list[str]] = defaultdict(list)
        threads_by_run: dict[str, list[str]] = defaultdict(list)
        for row in old.execute(
            """SELECT DISTINCT rs.run_id,s.session_id,s.thread_id
               FROM run_sessions rs JOIN sessions s ON s.session_id=rs.session_id
               ORDER BY rs.run_id,s.session_id"""
        ):
            sessions_by_run[row["run_id"]].append(row["session_id"])
            if row["thread_id"] and row["thread_id"] not in threads_by_run[row["run_id"]]:
                threads_by_run[row["run_id"]].append(row["thread_id"])

        for run_id, run in kept_runs.items():
            metadata = parse_json(run["metadata_json"], {})
            source_reference = metadata.get("run_dir") if isinstance(metadata, dict) else None
            source_type = "autoresearch_run" if source_reference else "git_codex_reconstruction"
            if not source_reference:
                source_reference = run["git_branch"]
            merged_metadata = dict(metadata) if isinstance(metadata, dict) else {}
            merged_metadata.update(
                {
                    "legacy_run_type": run["run_type"],
                    "provenance_confidence": run["provenance_confidence"],
                }
            )
            new.execute(
                """INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    run["project_name"],
                    run["task_name"],
                    run["root_node_id"],
                    run["status"],
                    run["start_time"],
                    run["end_time"],
                    int(run["replay_eligible"]),
                    source_type,
                    source_reference,
                    run["git_branch"],
                    compact_json(threads_by_run[run_id]),
                    compact_json(sessions_by_run[run_id]),
                    compact_json(merged_metadata),
                ),
            )

        events_by_node: dict[str, list[sqlite3.Row]] = defaultdict(list)
        source_files_by_node: dict[str, set[str]] = defaultdict(set)
        event_rows = list(
            old.execute(
                """SELECT e.*,f.original_path source_file
                   FROM events e JOIN source_files f ON f.file_id=e.source_file_id
                   ORDER BY e.source_file_id,e.line_no,e.event_id"""
            )
        )
        for event in event_rows:
            binding = bindings.get(event["event_id"])
            if binding:
                events_by_node[binding["node_id"]].append(event)
                source_files_by_node[binding["node_id"]].add(event["source_file"])

        commit_data: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in old.execute(
            """SELECT l.node_id,c.commit_hash,c.message,c.timestamp
               FROM node_commit_links l JOIN git_commits c ON c.commit_hash=l.commit_hash
               ORDER BY l.node_id,c.timestamp,c.commit_hash"""
        ):
            commit_data[row["node_id"]].append(row)

        for node in sorted(nodes.values(), key=lambda n: (n["run_id"], n["sequence_index"], n["node_id"])):
            if node["run_id"] not in kept_runs:
                continue
            depth, display_path = tree_fields(nodes, node["node_id"])
            bound = events_by_node[node["node_id"]]
            timestamps = [e["timestamp"] for e in bound if e["timestamp"]]
            sequences = [e["sequence_no"] for e in bound if e["sequence_no"] is not None]
            commits = commit_data[node["node_id"]]
            commit_hashes = [r["commit_hash"] for r in commits]
            primary_commit = node["git_commit_after"] or (commit_hashes[-1] if commit_hashes else None)
            primary_message = next(
                (r["message"] for r in reversed(commits) if r["commit_hash"] == primary_commit),
                commits[-1]["message"] if commits else None,
            )
            original_metadata = parse_json(node["metadata_json"], {})
            merged_metadata = dict(original_metadata) if isinstance(original_metadata, dict) else {}
            merged_metadata.update(
                {
                    "created_at": node["created_at"],
                    "completed_at": node["completed_at"],
                    "evaluated": node["evaluated"],
                    "valid": node["valid"],
                    "fail_class": node["fail_class"],
                    "error_text": node["error_text"],
                    "parent_relation_type": node["parent_relation_type"],
                    "parent_relation_confidence": node["parent_relation_confidence"],
                    "score_direction_method": node["score_direction_method"],
                    "score_direction_evidence": node["score_direction_evidence"],
                }
            )
            decision = merged_metadata.get("decision")
            result_summary = node["observation_text"] or (f"decision: {decision}" if decision else None)
            source_reference = node["workspace_path"] or node["git_commit_after"]
            source_type = "autoresearch_experiment" if node["workspace_path"] else "git_commit"
            new.execute(
                """INSERT INTO experiences VALUES (
                   ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )""",
                (
                    node["run_id"],
                    node["node_id"],
                    node["parent_node_id"],
                    node["branch_id"],
                    node["attempt_index"],
                    node["sequence_index"],
                    depth,
                    display_path,
                    node_type(node),
                    node["proposal_text"],
                    node["prompt_text"],
                    result_summary,
                    node_status(node),
                    node["primary_score"],
                    node["primary_score_name"],
                    node["primary_score_direction"] or "unknown",
                    metric_json(old, node["node_id"]),
                    artifact_json(old, node["node_id"]),
                    primary_commit,
                    compact_json(commit_hashes),
                    primary_message,
                    node["git_commit_before"],
                    node["git_commit_after"],
                    len(bound),
                    min(timestamps) if timestamps else None,
                    max(timestamps) if timestamps else None,
                    min(sequences) if sequences else None,
                    max(sequences) if sequences else None,
                    source_type,
                    source_reference,
                    compact_json(sorted(source_files_by_node[node["node_id"]])),
                    compact_json(merged_metadata),
                ),
            )

        for event in event_rows:
            binding = bindings.get(event["event_id"])
            node_id = binding["node_id"] if binding else None
            run_id = nodes[node_id]["run_id"] if node_id else event["run_id"]
            if run_id not in kept_runs:
                run_id = None
            new.execute(
                """INSERT INTO raw_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    event["event_id"],
                    run_id,
                    node_id,
                    event["sequence_no"],
                    event["timestamp"],
                    event["event_type"],
                    event["role"],
                    event["tool_name"],
                    event["command"],
                    event["cwd"],
                    event["stdout"],
                    event["stderr"],
                    event["exit_code"],
                    event["message"],
                    event["source_file"],
                    event["line_no"],
                    binding["binding_method"] if binding else None,
                    binding["confidence"] if binding else None,
                    event["raw_json"],
                ),
            )

        violations = list(new.execute("PRAGMA foreign_key_check"))
        if violations:
            raise ValueError(f"foreign key violations: {violations[:5]}")
        new.commit()
        counts = {
            table: new.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("runs", "experiences", "raw_events")
        }
        counts["bound_events"] = new.execute(
            "SELECT count(*) FROM raw_events WHERE node_id IS NOT NULL"
        ).fetchone()[0]
        counts["unbound_events"] = counts["raw_events"] - counts["bound_events"]
        new.close()
        old.close()

        for suffix in ("-wal", "-shm"):
            Path(str(output) + suffix).unlink(missing_ok=True)
        os.replace(new_db, output)
        return counts
    finally:
        legacy_copy.unlink(missing_ok=True)
        new_db.unlink(missing_ok=True)
        for temporary in (legacy_copy, new_db):
            Path(str(temporary) + "-wal").unlink(missing_ok=True)
            Path(str(temporary) + "-shm").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    counts = convert(args.source_db.resolve(), args.output_db.resolve())
    print(compact_json(counts))


if __name__ == "__main__":
    main()
