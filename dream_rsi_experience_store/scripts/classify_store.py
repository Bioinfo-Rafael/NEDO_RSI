#!/usr/bin/env python3
from pathlib import Path
import sqlite3,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from experience_store.classification import audit_replay_eligibility,classify_scores_and_runs
con=sqlite3.connect(ROOT/"experience_store/experience.db");con.row_factory=sqlite3.Row
classify_scores_and_runs(con);audit_replay_eligibility(con,ROOT/"experience_store/reports/replay_eligibility.md")
for row in con.execute("SELECT run_type,count(*),sum(replay_eligible) FROM discovery_runs GROUP BY run_type"):print(*row)
