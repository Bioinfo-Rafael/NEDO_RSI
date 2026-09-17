from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .util import jdump, stable_id


EXPLICIT_METHODS = {"explicit", "filesystem", "generation_id"}


def _role(event: dict, metric_event_ids: set[str]) -> str:
    typ = (event.get("event_type") or "").lower()
    command = (event.get("command") or "").lower()
    tool = (event.get("tool_name") or "").lower()
    if event["event_id"] in metric_event_ids:
        return "metric_source"
    if "proposal" in typ or "hypothesis" in typ:
        return "proposal"
    if "evaluat" in typ or "decision" in typ:
        return "evaluation"
    if "apply_patch" in command or tool in {"apply_patch", "edit", "write_file"}:
        return "code_edit"
    if event.get("command"):
        return "command"
    if event.get("stdout") is not None or event.get("stderr") is not None:
        return "command_output"
    if event.get("role") == "assistant" or "reason" in typ:
        return "reasoning"
    return "context"


def _insert_binding(con, node_id, event, method, confidence, evidence, metric_event_ids):
    bid = stable_id("binding", node_id, event["event_id"])
    con.execute("""INSERT OR IGNORE INTO node_event_bindings
      (binding_id,node_id,event_id,binding_role,binding_method,confidence,evidence,sequence_within_node)
      VALUES(?,?,?,?,?,?,?,NULL)""",
      (bid, node_id, event["event_id"], _role(event, metric_event_ids), method, confidence, evidence))


def bind_node_traces(con):
    """Build evidence-backed node relations. Existing Event.node_id is input evidence only."""
    for table in ("node_event_bindings", "node_trace_ranges", "node_command_summary",
                  "node_metric_links", "node_artifact_links", "run_sessions"):
        con.execute(f"DELETE FROM {table}")

    metric_event_ids = {r[0] for r in con.execute(
        "SELECT DISTINCT source_event FROM metrics WHERE source_event IS NOT NULL")}
    nodes = {r["node_id"]: dict(r) for r in con.execute("SELECT * FROM discovery_nodes")}
    exp_nodes = {}
    commit_nodes = {}
    for node in nodes.values():
        try:
            exp_id = json.loads(node["metadata_json"]).get("experiment_id")
        except (TypeError, json.JSONDecodeError):
            exp_id = None
        if exp_id:
            exp_nodes[(node["run_id"], str(exp_id))] = node["node_id"]
        if node.get("git_commit_after"):
            commit_nodes[node["git_commit_after"]] = node["node_id"]

    # Primary run/session relationship remains explicit but is no longer assumed exclusive.
    con.execute("""INSERT OR IGNORE INTO run_sessions(run_id,session_id,role,confidence,evidence)
      SELECT run_id,session_id,'primary',1.0,'discovery_runs.session_id'
      FROM discovery_runs WHERE session_id IS NOT NULL""")

    events = [dict(r) for r in con.execute("SELECT * FROM events ORDER BY source_file_id,line_no")]
    for event in events:
        node_id = event.get("node_id")
        if node_id:
            node = nodes[node_id]
            if node.get("git_commit_after"):
                # Production trace ranges are rebuilt below from actual commit
                # completion markers. Event.node_id is a legacy hint and may be
                # contaminated by commands that merely print older SHA values.
                continue
            else:
                method, confidence = "filesystem", .95
                source = con.execute("SELECT original_path FROM source_files WHERE file_id=?",
                                     (event["source_file_id"],)).fetchone()[0]
                evidence = f"raw event file is inside node experiment directory: {source}"
            _insert_binding(con, node_id, event, method, confidence, evidence, metric_event_ids)
            if event.get("session_id"):
                con.execute("INSERT OR IGNORE INTO run_sessions VALUES(?,?,?,?,?)",
                            (node["run_id"], event["session_id"], "execution", confidence, evidence))
            continue

        # Harness records carry an exact experiment identifier in their raw payload.
        try:
            raw = json.loads(event["raw_json"])
        except json.JSONDecodeError:
            raw = {}
        exp_id = raw.get("experiment_id")
        target = exp_nodes.get((event.get("run_id"), str(exp_id))) if exp_id is not None else None
        if target:
            evidence = f"raw event experiment_id={exp_id} matches discovery node metadata"
            _insert_binding(con, target, event, "explicit", 1.0, evidence, metric_event_ids)
            if event.get("session_id"):
                con.execute("INSERT OR IGNORE INTO run_sessions VALUES(?,?,?,?,?)",
                            (nodes[target]["run_id"], event["session_id"], "harness", 1.0, evidence))

    # Git reconstructed run: a commit is the result node. Bind the contiguous
    # interval ending at an actual `git commit` completion line. Merely printing
    # a SHA in `git log`, a patch, or a prompt is not a boundary.
    by_file = defaultdict(list)
    production_run_ids = {n["run_id"] for n in nodes.values() if n.get("git_commit_after")}
    for event in events:
        if event.get("run_id") in production_run_ids:
            by_file[event["source_file_id"]].append(event)
    marker_re = re.compile(r"\[[^\]\n]+\s+([0-9a-f]{7,40})\]\s+[^\n]+")
    for file_id, file_events in by_file.items():
        buffer=[]
        for event in sorted(file_events,key=lambda e:(e["line_no"],e["event_id"])):
            buffer.append(event)
            text="\n".join(str(event.get(k) or "") for k in ("stdout","message"))
            matches=marker_re.findall(text)
            target_sha=None
            for short in matches:
                candidates=[sha for sha in commit_nodes if sha.startswith(short)]
                if len(candidates)==1: target_sha=candidates[0]
            if target_sha:
                target=commit_nodes[target_sha]
                evidence=f"contiguous source-file interval ending at git commit completion marker [{target_sha[:7]}]"
                for item in buffer:
                    _insert_binding(con,target,item,"git",.80,evidence,metric_event_ids)
                    if item.get("session_id"):
                        con.execute("INSERT OR IGNORE INTO run_sessions VALUES(?,?,?,?,?)",
                                    (nodes[target]["run_id"],item["session_id"],"execution",.80,evidence))
                buffer=[]

    # Stable ordering within each node follows source file and source line, avoiding
    # false ordering assumptions between duplicate raw copies.
    for node_id, in con.execute("SELECT node_id FROM discovery_nodes"):
        bound = con.execute("""SELECT b.binding_id FROM node_event_bindings b
          JOIN events e ON e.event_id=b.event_id WHERE b.node_id=?
          ORDER BY e.source_file_id,e.line_no,e.event_id""", (node_id,)).fetchall()
        for index, row in enumerate(bound):
            con.execute("UPDATE node_event_bindings SET sequence_within_node=? WHERE binding_id=?",
                        (index, row[0]))

    # A range is scoped by node, session and raw source file. sequence_no resets per
    # file, so mixing files would create artificial overlaps and inversions.
    groups = con.execute("""SELECT b.node_id,e.session_id,e.source_file_id,
      min(e.sequence_no),max(e.sequence_no),min(e.timestamp),max(e.timestamp),count(*)
      FROM node_event_bindings b JOIN events e ON e.event_id=b.event_id
      GROUP BY b.node_id,e.session_id,e.source_file_id""").fetchall()
    for node_id, session_id, file_id, first_seq, last_seq, start, end, count in groups:
        first = con.execute("""SELECT e.event_id FROM node_event_bindings b JOIN events e ON e.event_id=b.event_id
          WHERE b.node_id=? AND e.session_id IS ? AND e.source_file_id=? ORDER BY e.sequence_no,e.line_no LIMIT 1""",
          (node_id, session_id, file_id)).fetchone()[0]
        last = con.execute("""SELECT e.event_id FROM node_event_bindings b JOIN events e ON e.event_id=b.event_id
          WHERE b.node_id=? AND e.session_id IS ? AND e.source_file_id=? ORDER BY e.sequence_no DESC,e.line_no DESC LIMIT 1""",
          (node_id, session_id, file_id)).fetchone()[0]
        candidate = con.execute("""SELECT count(*) FROM events WHERE source_file_id=?
          AND sequence_no BETWEEN ? AND ?""", (file_id, first_seq, last_seq)).fetchone()[0]
        methods = Counter(r[0] for r in con.execute("""SELECT b.binding_method FROM node_event_bindings b
          JOIN events e ON e.event_id=b.event_id WHERE b.node_id=? AND e.session_id IS ? AND e.source_file_id=?""",
          (node_id, session_id, file_id)))
        confidence = con.execute("""SELECT min(b.confidence) FROM node_event_bindings b JOIN events e ON e.event_id=b.event_id
          WHERE b.node_id=? AND e.session_id IS ? AND e.source_file_id=?""",
          (node_id, session_id, file_id)).fetchone()[0]
        if methods.get("explicit"):
            boundary = "explicit_experiment_id"
        elif methods.get("filesystem"):
            boundary = "experiment_directory"
        elif methods.get("git"):
            boundary = "git_commit_interval"
        else:
            boundary = "inferred"
        roles = {r[0] for r in con.execute("""SELECT b.binding_role FROM node_event_bindings b JOIN events e ON e.event_id=b.event_id
          WHERE b.node_id=? AND e.session_id IS ? AND e.source_file_id=?""",
          (node_id, session_id, file_id))}
        complete = int(boundary == "explicit_experiment_id" and "evaluation" in roles)
        coverage = count / candidate if candidate else None
        rid = stable_id("range", node_id, session_id, file_id)
        con.execute("""INSERT INTO node_trace_ranges VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (rid,node_id,session_id,file_id,first,last,first_seq,last_seq,start,end,count,candidate,
           candidate-count,coverage,complete,boundary,confidence,
           f"{count} bound events in source-file sequence interval {first_seq}..{last_seq}; methods={dict(methods)}"))

    # Directly observed relations only. These tables do not create metrics or scores.
    con.execute("""INSERT INTO node_metric_links
      SELECT node_id,metric_id,'explicit',1.0 FROM metrics WHERE node_id IS NOT NULL""")
    con.execute("""INSERT INTO node_artifact_links
      SELECT node_id,artifact_id,'filesystem',.95,original_path FROM artifacts WHERE node_id IS NOT NULL""")
    # Command aggregation uses raw command events. Runtime is NULL unless a numeric
    # duration is explicitly recorded in the event payload.
    for node_id in nodes:
        commands = [dict(r) for r in con.execute("""SELECT e.* FROM node_event_bindings b
          JOIN events e ON e.event_id=b.event_id WHERE b.node_id=? AND e.command IS NOT NULL
          ORDER BY b.sequence_within_node""", (node_id,))]
        success = sum(e.get("exit_code") == 0 for e in commands)
        failed = sum(e.get("exit_code") not in (None, 0) for e in commands)
        runtimes = []
        for e in commands:
            try:
                raw = json.loads(e["raw_json"])
                value = raw.get("seconds")
                if isinstance(value, (int, float)): runtimes.append(float(value))
            except json.JSONDecodeError:
                pass
        con.execute("INSERT INTO node_command_summary VALUES(?,?,?,?,?,?,?)",
                    (node_id,len(commands),success,failed,
                     commands[0]["command"] if commands else None,
                     commands[-1]["command"] if commands else None,
                     sum(runtimes) if runtimes else None))
    con.commit()
