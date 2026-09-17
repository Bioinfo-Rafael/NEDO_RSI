"""Export the simplified replay store as three JSONL files."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


TABLES = ("runs", "experiences", "raw_events")


def export(con: sqlite3.Connection, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.iterdir():
        if stale.is_file():
            stale.unlink()
    for table in TABLES:
        order = {
            "runs": "run_id",
            "experiences": "run_id,sequence_index,node_id",
            "raw_events": "coalesce(run_id,''),coalesce(node_id,''),source_file,source_line,event_id",
        }[table]
        with (out / f"{table}.jsonl").open("w", encoding="utf-8") as stream:
            for row in con.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                stream.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
