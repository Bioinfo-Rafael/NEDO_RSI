"""Version-tolerant parsing of Codex session JSONL and exec --json streams.

Response usage and cumulative turn/token_count records must never be added
together. Cached input is a subset of input; reasoning output a subset of output.
Opaque reasoning records are not decoded or exported into the analysis tables.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.common import read_json, write_json

TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
              "output_tokens", "reasoning_output_tokens", "total_tokens")


def read_jsonl(path):
    values, invalid = [], 0
    if not Path(path).exists():
        return values, invalid
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except ValueError:
            invalid += 1
    return values, invalid


def seconds_between(start, end):
    if not start or not end:
        return None
    try:
        return max(0., (datetime.fromisoformat(end.replace("Z", "+00:00")) -
                        datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds())
    except (TypeError, ValueError):
        return None


def normalized_usage(usage):
    out = {k: int(usage.get(k) or 0) for k in TOKEN_KEYS}
    if "total_tokens" not in usage:
        out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
    return out


def parse_sessions(records):
    responses, calls, turns, models = {}, {}, [], []
    pending_calls = set()
    session, turn = None, None
    for r in records:
        p = r.get("payload", {})
        kind, ts = r.get("type"), r.get("timestamp")
        if kind == "session_meta":
            session = p.get("id", p.get("session_id"))
            turn = None
            pending_calls.clear()
        elif kind == "turn_context":
            if turn != p.get("turn_id"):
                pending_calls.clear()
            turn = p.get("turn_id")
            models.append({"session_id": session, "turn_id": turn, "model": p.get("model"),
                           "effort": p.get("effort", p.get("reasoning_effort"))})
        elif (kind == "token_usage_record" and p.get("response_id") and
              isinstance(p.get("usage"), dict) and p["usage"]):
            sid = p.get("session_id", session)
            rid = p["response_id"]
            # Current CLI writes completed tool-call items before response usage,
            # then their tool results. This is an ordering inference, not an API
            # foreign key. Never carry an unmatched call past its result/turn.
            if (sid, rid) not in responses:
                for key in list(pending_calls):
                    c = calls[key]
                    if c["session_id"] == sid and c["turn_id"] == p.get("turn_id", turn):
                        c["response_id"] = rid
                        c["response_link_method"] = "inferred_next_usage_before_tool_result"
                        pending_calls.remove(key)
            responses[(sid, rid)] = {"session_id": sid, "turn_id": p.get("turn_id", turn),
                                     "response_id": rid, "timestamp": ts,
                                     "usage": normalized_usage(p.get("usage", {})),
                                     "api_duration_seconds": None}
        elif kind == "event_msg" and p.get("type") == "task_complete":
            turns.append({"session_id": session, "turn_id": p.get("turn_id", turn),
                          "duration_ms": p.get("duration_ms"),
                          "time_to_first_token_ms": p.get("time_to_first_token_ms")})
        elif kind == "response_item":
            typ, cid = p.get("type"), p.get("call_id")
            if typ in ("function_call", "custom_tool_call") and cid:
                calls[(session, cid)] = {"session_id": session, "turn_id": turn, "call_id": cid,
                                         "name": p.get("name"), "started_at": ts,
                                         "input": p.get("arguments", p.get("input")),
                                         "completed_at": None, "duration_seconds": None,
                                         "response_id": p.get("response_id"),
                                         "response_link_method": "explicit" if p.get("response_id") else "unavailable",
                                         "source": "session", "actor": "codex"}
                if not p.get("response_id"):
                    pending_calls.add((session, cid))
            elif typ in ("function_call_output", "custom_tool_call_output") and cid:
                pending_calls.discard((session, cid))
                c = calls.get((session, cid))
                if c:
                    c["completed_at"] = ts
                    c["duration_seconds"] = seconds_between(c["started_at"], ts)
                    c["output"] = p.get("output")
    values = list(responses.values())
    return {"responses": values, "tool_calls": list(calls.values()), "turns": turns,
            "models": models,
            "usage": {k: sum(v["usage"][k] for v in values) if values else None for k in TOKEN_KEYS},
            "usage_granularity": "response" if values else "unavailable"}


def parse_cli(stream):
    threads, calls, usage = [], {}, []
    thread = None
    for row in stream:
        e = row.get("event", row)
        ts = row.get("received_at")
        if e.get("type") == "thread.started":
            thread = e.get("thread_id")
            if thread:
                threads.append(thread)
        if e.get("type") == "turn.completed" and "usage" in e:
            usage.append(normalized_usage(e["usage"]))
        item = e.get("item", {})
        if item.get("type") in ("command_execution", "mcp_tool_call", "file_change", "web_search"):
            key = (thread, item.get("id"))
            c = calls.setdefault(key, {"session_id": thread, "call_id": item.get("id"),
                                       "name": item.get("type"), "started_at": ts,
                                       "completed_at": None, "source": "cli_receipt", "actor": "codex"})
            if e.get("type") == "item.completed":
                c.update(completed_at=ts, status=item.get("status"), exit_code=item.get("exit_code"),
                         duration_seconds=seconds_between(c["started_at"], ts))
            c["input"] = item.get("command", item.get("arguments"))
    return {"thread_ids": list(dict.fromkeys(threads)), "tool_calls": list(calls.values()),
            "usage": {k: sum(v[k] for v in usage) for k in TOKEN_KEYS}, "usage_records": len(usage)}


def collect(agent_dir, experiment_id, sessions_root=None):
    agent_dir = Path(agent_dir)
    stream, invalid = read_jsonl(agent_dir / "stream_events.jsonl")
    cli = parse_cli(stream)
    sessions_root = Path(sessions_root or Path.home() / ".codex/sessions")
    records, saved = [], []
    for tid in cli["thread_ids"]:
        for path in sorted(sessions_root.glob(f"*/*/*/*{tid}*.jsonl")):
            rows, bad = read_jsonl(path)
            invalid += bad
            # Match identity as well as filename; never ingest unrelated chats.
            meta = next((r.get("payload", {}) for r in rows if r.get("type") == "session_meta"), {})
            if meta.get("id", meta.get("session_id")) != tid:
                continue
            dest = agent_dir / "raw_sessions" / path.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(path, dest)
            saved.append(str(dest))
            records.extend(rows)
    result = parse_sessions(records)
    if result["usage_granularity"] == "unavailable" and cli["usage_records"]:
        result["usage"] = cli["usage"]
        result["usage_granularity"] = "turn (exec JSON fallback)"
    elif result["usage_granularity"] == "unavailable":
        result["usage"] = {k: None for k in TOKEN_KEYS}
    if not result["tool_calls"]:
        result["tool_calls"] = cli["tool_calls"]
    result.update(experiment_id=experiment_id, thread_ids=cli["thread_ids"],
                  session_files=saved, invalid_jsonl_lines=invalid,
                  api_duration_available=False)
    for group in ("responses", "tool_calls", "turns", "models"):
        for row in result[group]:
            row["experiment_id"] = experiment_id
    write_json(agent_dir / "telemetry.json", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path, help="one existing rollout JSONL (read only)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows, invalid = read_jsonl(args.session)
    result = parse_sessions(rows)
    result["invalid_jsonl_lines"] = invalid
    write_json(args.out, result)
    print(json.dumps({"responses": len(result["responses"]), "usage": result["usage"]}))


if __name__ == "__main__":
    main()
