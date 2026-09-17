#!/usr/bin/env python3
import json,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from experience_store.audit import audit
con=sqlite3.connect(ROOT/"experience_store/experience.db");con.row_factory=sqlite3.Row
result=audit(con,ROOT/"experience_store/reports/audit.md",ROOT/"experience_store/reports/coverage.csv",ROOT/"experience_store/reports/coverage.md")
(ROOT/"experience_store/reports/audit.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
raise SystemExit(0 if result["passed"] else 1)
