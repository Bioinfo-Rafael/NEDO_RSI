#!/usr/bin/env python3
from pathlib import Path
import json,sqlite3,sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from experience_store.trace_audit import raw_integrity,trace_audit

con=sqlite3.connect(ROOT/"experience_store/experience.db");con.row_factory=sqlite3.Row
result=trace_audit(con,ROOT/"experience_store/reports")
integrity=raw_integrity(ROOT/"experience_store/ingest_manifest.json",
                        ROOT/"experience_store/reports/raw_integrity_pre.json",
                        ROOT/"experience_store/reports/raw_integrity.md")
print(json.dumps({"trace":result,"raw_integrity":integrity},indent=2))
if integrity["changed"] or integrity["missing"]: raise SystemExit(1)
