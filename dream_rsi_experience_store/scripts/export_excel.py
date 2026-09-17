#!/usr/bin/env python3
"""Rebuild experience.xlsx through the Codex artifact-tool runtime."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
deps=Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"
link=HERE/"node_modules"
created=False
if not link.exists():
    if not deps.exists():
        raise SystemExit(f"artifact-tool dependencies not found: {deps}")
    link.symlink_to(deps,target_is_directory=True);created=True
try:
    subprocess.run([sys.executable,str(HERE/"prepare_excel_data.py")],check=True,cwd=ROOT)
    subprocess.run(["node",str(HERE/"export_excel.mjs")],check=True,cwd=ROOT)
finally:
    if created and link.is_symlink(): link.unlink()
    (ROOT/"experience_store/.excel_data.json").unlink(missing_ok=True)
    for temporary in (ROOT/"experience_store").glob(".excel_preview_*.png"):
        temporary.unlink(missing_ok=True)
print(ROOT/"experience_store/experience.xlsx")
