#!/usr/bin/env python3
"""Audit the simplified replay store and write one compact report."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "experience_store" / "experience.db"
MANIFEST = ROOT / "experience_store" / "ingest_manifest.json"
REPORT = ROOT / "experience_store" / "reports" / "simplification_audit.md"


def raw_modified_count() -> tuple[int, int]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    modified = 0
    for record in manifest["files"]:
        path = Path(record["original_path"])
        if not path.exists():
            modified += 1
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        modified += digest.hexdigest() != record["sha256"]
    return len(manifest["files"]), modified


con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
tables = [
    row[0]
    for row in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
]
counts = {table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in tables}
roots = con.execute("SELECT count(*) FROM experiences WHERE parent_id IS NULL").fetchone()[0]
orphans = con.execute(
    """SELECT count(*) FROM experiences child
       LEFT JOIN experiences parent ON parent.node_id=child.parent_id
       WHERE child.parent_id IS NOT NULL AND parent.node_id IS NULL"""
).fetchone()[0]

cycles = 0
for run_id, in con.execute("SELECT run_id FROM runs"):
    parents = dict(
        con.execute("SELECT node_id,parent_id FROM experiences WHERE run_id=?", (run_id,))
    )
    cyclic: set[str] = set()
    for start in parents:
        seen: set[str] = set()
        current: str | None = start
        while current:
            if current in seen:
                cyclic.add(start)
                break
            seen.add(current)
            current = parents.get(current)
    cycles += len(cyclic)

directions = Counter(
    {row[0]: row[1] for row in con.execute("SELECT score_direction,count(*) FROM experiences GROUP BY score_direction")}
)
bound = con.execute("SELECT count(*) FROM raw_events WHERE node_id IS NOT NULL").fetchone()[0]
unbound = con.execute("SELECT count(*) FROM raw_events WHERE node_id IS NULL").fetchone()[0]
fk_violations = list(con.execute("PRAGMA foreign_key_check"))
raw_files, raw_modified = raw_modified_count()

result = {
    "tables": len(tables),
    "runs": counts.get("runs", 0),
    "experiences": counts.get("experiences", 0),
    "raw_events": counts.get("raw_events", 0),
    "roots": roots,
    "orphans": orphans,
    "cycles": cycles,
    "bound_events": bound,
    "unbound_events": unbound,
    "raw_files_checked": raw_files,
    "raw_modified": raw_modified,
    "foreign_key_violations": len(fk_violations),
}

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(
    "\n".join(
        [
            "# Experience Store simplification audit",
            "",
            "| Check | Result |",
            "|---|---:|",
            f"| tables | {result['tables']} |",
            f"| runs | {result['runs']} |",
            f"| experiences | {result['experiences']} |",
            f"| raw events | {result['raw_events']} |",
            f"| root nodes | {result['roots']} |",
            f"| missing-parent orphans | {result['orphans']} |",
            f"| nodes participating in a cycle | {result['cycles']} |",
            f"| node-bound events | {result['bound_events']} |",
            f"| unbound events | {result['unbound_events']} |",
            f"| raw files checked | {result['raw_files_checked']} |",
            f"| raw source modified or missing | {result['raw_modified']} |",
            f"| foreign-key violations | {result['foreign_key_violations']} |",
            "",
            "## Score direction",
            "",
            "| Direction | Experiences |",
            "|---|---:|",
            *[f"| {key} | {value} |" for key, value in sorted(directions.items())],
            "",
            "`raw source modified or missing`は`ingest_manifest.json`に記録されたSHA-256と、",
            "現在のraw fileを照合した件数です。",
            "",
        ]
    ),
    encoding="utf-8",
)
con.close()
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
