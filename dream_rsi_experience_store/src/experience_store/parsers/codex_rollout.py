from __future__ import annotations
import json
from pathlib import Path
from ..secrets import redact


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            raw=line.rstrip("\n")
            try: yield line_no, json.loads(raw), None
            except json.JSONDecodeError as exc: yield line_no, None, f"{exc.msg} at column {exc.colno}"


def thread_id(obj: dict) -> str | None:
    payload=obj.get("payload") or {}
    if obj.get("type")=="session_meta": return payload.get("session_id") or payload.get("id")
    if obj.get("type")=="thread.started": return obj.get("thread_id")
    return payload.get("thread_id")


def project(obj: dict) -> dict:
    """Best-effort normalized fields; raw_json remains authoritative."""
    typ=obj.get("type")
    if not typ:
        typ=".".join(str(obj.get(k) or "unknown") for k in ("actor","phase","state"))
    payload=obj.get("payload") or {}
    role=tool=command=stdout=stderr=message=cwd=None; exit_code=None
    if typ=="session_meta": cwd=payload.get("cwd")
    if typ=="response_item":
        ptype=payload.get("type")
        role=payload.get("role")
        if ptype=="message":
            texts=[]
            for part in payload.get("content") or []:
                if isinstance(part,dict) and isinstance(part.get("text"),str): texts.append(part["text"])
            message="\n".join(texts) or None
        elif ptype in ("function_call","custom_tool_call"):
            tool=payload.get("name"); command=payload.get("arguments") or payload.get("input")
        elif ptype in ("function_call_output","custom_tool_call_output"):
            stdout=payload.get("output")
    elif typ in ("event_msg","item.started","item.completed"):
        item=payload if typ=="event_msg" else obj.get("item") or {}
        subtype=item.get("type")
        if subtype in ("agent_message","user_message"): role=subtype.split("_")[0]; message=item.get("text") or item.get("message")
        if subtype=="command_execution":
            tool="shell";command=item.get("command");stdout=item.get("aggregated_output");exit_code=item.get("exit_code")
        if subtype: typ=f"{typ}.{subtype}"
    fields={"event_type":typ,"role":role,"tool_name":tool,"command":command,"cwd":cwd,
            "stdout":stdout,"stderr":stderr,"exit_code":exit_code,"message":message}
    secret=False
    for key in ("command","stdout","stderr","message"):
        fields[key],found=redact(fields[key] if isinstance(fields[key],str) else (json.dumps(fields[key],ensure_ascii=False) if fields[key] is not None else None));secret|=found
    return fields|{"secret_detected":secret}
