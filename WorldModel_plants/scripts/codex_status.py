#!/usr/bin/env python
"""What has the Codex loop actually done so far? results.tsv rows, the agent's
commits in its own git dir, and the diff of worldmodel.py against the frozen
baseline - the three things that decide whether 'improved' is a fact."""
from __future__ import annotations

import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else None


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, cwd=ROOT).stdout.strip()


print("=== results.tsv ===")
rows = [l for l in (ROOT / "control_research" / "results.tsv").read_text().splitlines() if l.strip()]
for l in rows:
    c = l.split("\t")
    print("  " + "  ".join(f"{x:>10s}" if i in (1, 2, 3) else x for i, x in enumerate(c[:5])) + "  " + (c[5] if len(c) > 5 else ""))
if len(rows) > 2:
    vals = [float(l.split("\t")[1]) for l in rows[1:] if l.split("\t")[1] not in ("0.000000",)]
    kept = [float(l.split("\t")[1]) for l in rows[1:] if l.split("\t")[4] == "keep"]
    if vals and kept:
        print(f"  baseline {vals[0]:.4f} -> best kept {min(kept):.4f}  ({(vals[0]-min(kept))/vals[0]*100:+.1f}% は下がった量)")

gitdirs = sorted((ROOT / "control_research").glob(".*/git"))
if tag:
    gitdirs = [g for g in gitdirs if g.parent.name == f".{tag}"]
for g in gitdirs:
    print(f"\n=== Codex commits in {g.relative_to(ROOT)} ===")
    print(sh("git", f"--git-dir={g}", "log", "--format=  %h %s", "-n", "20") or "  (none)")

print("\n=== worldmodel.py vs frozen baseline (main) ===")
stat = sh("git", "diff", "--stat", "main", "--", "control_research/worldmodel.py")
print("  " + (stat.splitlines()[-1] if stat else "no change"))
