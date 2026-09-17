#!/usr/bin/env python
"""Pull the agent's own commit history and diffs out of the git dirs it made.

Codex could not write to the repository's .git under the sandbox, so it created
its own. That turns out to be useful: the two runs are cleanly separated, and
each experiment is one commit with the agent's own description of what it tried.
"""
from __future__ import annotations

import json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _discover_runs():
    """Every git dir Codex created under control_research/ (it cannot write the
    repository's .git from the sandbox, so each run leaves its own), newest tag
    last, plus the historical prediction-loop run."""
    runs = [{"key": "prediction", "gitdir": "integration/.sep14/git",
             "tsv": "integration/results.tsv", "file": "integration/train.py",
             "metric": "val_nrmse", "label": "第1ループ：予測誤差で採点"}]
    for d in sorted((ROOT / "control_research").glob(".*/git")):
        tag = d.parent.name.lstrip(".")
        runs.append({"key": f"control_{tag}", "gitdir": str(d.relative_to(ROOT)),
                     "tsv": "control_research/results.tsv",
                     "file": "control_research/worldmodel.py", "metric": "control_cost",
                     "label": f"制御性能で採点（{tag}）", "prefix": tag})
    return runs


RUNS = _discover_runs()


def git(gitdir, *args):
    return subprocess.run(["git", f"--git-dir={ROOT/gitdir}", *args],
                          capture_output=True, text=True, cwd=ROOT).stdout


def commits(gitdir, prefix="sep14"):
    """Commits whose subject starts with the run tag."""
    out = git(gitdir, "log", "--format=%H\t%s")
    rows = []
    for ln in out.splitlines():
        h, _, s = ln.partition("\t")
        if s.startswith(prefix):
            rows.append({"sha": h[:7], "subject": s})
    return rows[::-1]


def diff_stat(gitdir, sha, path):
    d = git(gitdir, "show", "--format=", "--unified=0", sha, "--", path)
    add = len([l for l in d.splitlines() if l.startswith("+") and not l.startswith("+++")])
    rem = len([l for l in d.splitlines() if l.startswith("-") and not l.startswith("---")])
    body = [l for l in d.splitlines()
            if (l.startswith("+") and not l.startswith("+++")) and l[1:].strip()]
    return add, rem, [l[1:].rstrip() for l in body[:14]]


def read_tsv(p):
    """Read by header name. The two loops log different column counts, and
    reading by position silently shifted status into the secondary metric."""
    lines = (ROOT / p).read_text().splitlines()
    head = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        if ln.strip():
            rows.append(dict(zip(head, ln.split("\t"))))
    return rows


def main() -> int:
    out = {}
    for r in RUNS:
        cs = commits(r["gitdir"], r.get("prefix", "sep14"))
        tsv = read_tsv(r["tsv"])
        items = []
        for i, t in enumerate(tsv):
            sha = t["commit"]
            add, rem, body = diff_stat(r["gitdir"], sha, r["file"]) if i else (0, 0, [])
            items.append({
                "n": i + 1, "sha": sha,
                "metric": float(t[r["metric"]]),
                "status": t["status"],
                "description": t["description"],
                "added": add, "removed": rem,
                "diff": body,
            })
        out[r["key"]] = {"label": r["label"], "metric": r["metric"], "items": items}
        print(f"{r['label']}: {len(items)} 実験")
        for it in items:
            print(f"  {it['n']}. {it['sha']}  {r['metric']}={it['metric']:.4f} "
                  f"[{it['status']:7s}] +{it['added']}/-{it['removed']}  {it['description'][:52]}")
    (ROOT / "figs" / "codex_history.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n-> figs/codex_history.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
