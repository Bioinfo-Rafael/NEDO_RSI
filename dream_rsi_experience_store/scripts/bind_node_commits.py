#!/usr/bin/env python3
from pathlib import Path
import sqlite3,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from experience_store.commit_bindings import audit_node_commits,bind_node_commits
con=sqlite3.connect(ROOT/"experience_store/experience.db");con.row_factory=sqlite3.Row
bind_node_commits(con)
suspicious=audit_node_commits(con,ROOT/"experience_store/reports/node_commit_audit.md")
print("node_commit_links",con.execute("SELECT count(*) FROM node_commit_links").fetchone()[0])
print("suspicious",suspicious)
