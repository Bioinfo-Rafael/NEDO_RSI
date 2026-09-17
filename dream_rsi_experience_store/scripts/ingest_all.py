#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from experience_store.ingest import ingest

db=ROOT/"experience_store/experience.db"
con=ingest(ROOT,ROOT/"experience_store/ingest_manifest.json",db)
print(db)
for table in ("sources","source_files","sessions","discovery_runs","discovery_nodes","events","metrics","artifacts","git_commits","warnings"):
 print(table,con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
