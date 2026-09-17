#!/usr/bin/env python3
"""Prepare typed data for the three-sheet artifact-tool workbook."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "experience_store" / "experience.db"
OUT = ROOT / "experience_store" / ".excel_data.json"

QUERIES = {
    "Runs": "SELECT * FROM runs ORDER BY start_time,run_id",
    "Experiences": "SELECT * FROM experiences ORDER BY run_id,sequence_index,node_id",
    "Raw_Events": """SELECT * FROM raw_events
        ORDER BY coalesce(run_id,''),coalesce(node_id,''),source_file,source_line,event_id""",
}


def excel_value(value: object) -> object:
    if isinstance(value, str) and len(value) > 30_000:
        return value[:29_950] + "\n[Excel表示用に省略。完全版はexperience.dbにあります]"
    return value


con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
sheets = {}
for name, sql in QUERIES.items():
    cursor = con.execute(sql)
    columns = [description[0] for description in cursor.description]
    sheets[name] = {
        "columns": columns,
        "rows": [[excel_value(value) for value in row] for row in cursor.fetchall()],
    }
OUT.write_text(json.dumps({"sheets": sheets}, ensure_ascii=False), encoding="utf-8")
con.close()
print(OUT)
