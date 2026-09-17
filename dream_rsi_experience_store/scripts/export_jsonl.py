#!/usr/bin/env python3
"""Export runs, experiences, and raw_events as JSONL."""

from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experience_store.export import export


con = sqlite3.connect(f"file:{ROOT / 'experience_store/experience.db'}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
export(con, ROOT / "experience_store/exports")
con.close()
print(ROOT / "experience_store/exports")
