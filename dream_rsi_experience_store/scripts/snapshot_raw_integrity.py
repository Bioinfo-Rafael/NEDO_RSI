#!/usr/bin/env python3
"""Capture pre-rebuild SHA-256 values without modifying raw sources."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
manifest=json.loads((ROOT/"experience_store/ingest_manifest.json").read_text())
snapshot={}
for rec in manifest["files"]:
    path=Path(rec["original_path"]);h=hashlib.sha256()
    with path.open("rb") as stream:
        while chunk:=stream.read(1024*1024):h.update(chunk)
    snapshot[rec["file_id"]]={"path":str(path),"sha256":h.hexdigest(),"manifest_sha256":rec["sha256"]}
out=ROOT/"experience_store/reports/raw_integrity_pre.json"
out.write_text(json.dumps(snapshot,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
print(out)
