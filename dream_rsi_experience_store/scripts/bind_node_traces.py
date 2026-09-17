#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from experience_store.bindings import bind_node_traces
from experience_store.db import connect

db=ROOT/"experience_store/experience.db"
con=connect(db)
bind_node_traces(con)
print(db)
for table in ("node_event_bindings","node_trace_ranges","node_metric_links","node_artifact_links","node_commit_links","run_sessions"):
    print(table,con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
