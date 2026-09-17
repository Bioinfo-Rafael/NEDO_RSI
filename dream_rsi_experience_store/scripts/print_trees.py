#!/usr/bin/env python3
"""Print Discovery Trees from the simplified Experience Store."""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "experience_store" / "experience.db"


def label(row: sqlite3.Row) -> str:
    proposal = " ".join((row["proposal"] or "").split())
    if len(proposal) > 88:
        proposal = proposal[:85] + "..."
    score = "" if row["score"] is None else f" score={row['score']:.6g}"
    branch = "" if not row["branch_id"] else f" branch={row['branch_id']}"
    return f"{row['node_id']} [{row['node_type']}]{branch}{score} — {proposal}"


def print_run(con: sqlite3.Connection, run: sqlite3.Row) -> None:
    rows = list(
        con.execute(
            "SELECT * FROM experiences WHERE run_id=? ORDER BY sequence_index,node_id",
            (run["run_id"],),
        )
    )
    by_id = {row["node_id"]: row for row in rows}
    children: dict[str | None, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        children[row["parent_id"]].append(row)

    print(f"\n{run['run_id']}  project={run['project_name'] or ''}  task={run['task_name'] or ''}")

    def visit(node_id: str, prefix: str, is_last: bool, root: bool = False) -> None:
        row = by_id[node_id]
        connector = "" if root else ("└── " if is_last else "├── ")
        print(prefix + connector + label(row))
        next_prefix = prefix if root else prefix + ("    " if is_last else "│   ")
        node_children = children.get(node_id, [])
        for index, child in enumerate(node_children):
            visit(child["node_id"], next_prefix, index == len(node_children) - 1)

    roots = children.get(None, [])
    for index, root in enumerate(roots):
        visit(root["node_id"], "", index == len(roots) - 1, root=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experience Storeの探索木を表示します")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    if args.run_id:
        runs = list(con.execute("SELECT * FROM runs WHERE run_id=?", (args.run_id,)))
        if not runs:
            raise SystemExit(f"run not found: {args.run_id}")
    else:
        runs = list(con.execute("SELECT * FROM runs ORDER BY start_time,run_id"))
    for run in runs:
        print_run(con, run)
    con.close()


if __name__ == "__main__":
    main()
