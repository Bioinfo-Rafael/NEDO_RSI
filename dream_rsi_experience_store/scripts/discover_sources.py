#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from experience_store.discover import discover,render_report

manifest=discover(ROOT,ROOT/"experience_store/ingest_manifest.json")
render_report(manifest,ROOT/"experience_store/reports/source_discovery.md")
print(ROOT/"experience_store/reports/source_discovery.md")
