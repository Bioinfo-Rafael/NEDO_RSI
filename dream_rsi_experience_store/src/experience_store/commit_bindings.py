from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


COMMIT_MARKER = re.compile(r"\[[^\]\n]+\s+([0-9a-f]{7,40})\]\s+[^\n]+")


def bind_node_commits(con):
    """Recompute node/commit relations from node fields and bound commit output only."""
    con.execute("DELETE FROM node_commit_links")
    commits=[r[0] for r in con.execute("SELECT commit_hash FROM git_commits")]

    # A result commit explicitly stored on the node is authoritative.
    for node_id,commit in con.execute("""SELECT node_id,git_commit_after FROM discovery_nodes
      WHERE git_commit_after IS NOT NULL"""):
        if commit in commits:
            con.execute("INSERT INTO node_commit_links VALUES(?,?,?,?,?)",
                        (node_id,commit,"explicit_git",1.0,
                         "discovery_nodes.git_commit_after explicitly identifies the node result commit"))

    # Additional commits require an actual commit completion marker inside the
    # canonical node trace. Same session/run/branch membership alone is ignored.
    for row in con.execute("""SELECT b.node_id,e.event_id,e.stdout,e.message,e.source_file_id,e.line_no
      FROM node_event_bindings b JOIN events e ON e.event_id=b.event_id
      WHERE e.stdout IS NOT NULL OR e.message IS NOT NULL"""):
        text="\n".join(str(x or "") for x in (row[2],row[3]))
        for short in COMMIT_MARKER.findall(text):
            matches=[sha for sha in commits if sha.startswith(short)]
            if len(matches)!=1: continue
            sha=matches[0]
            exists=con.execute("SELECT binding_method FROM node_commit_links WHERE node_id=? AND commit_hash=?",
                               (row[0],sha)).fetchone()
            if exists: continue
            con.execute("INSERT INTO node_commit_links VALUES(?,?,?,?,?)",
                        (row[0],sha,"event_git",.93,
                         f"commit completion marker in bound event {row[1]} at {row[4]}:{row[5]}"))
    con.commit()


def audit_node_commits(con, report:Path):
    per_commit=Counter(r[0] for r in con.execute("SELECT commit_hash FROM node_commit_links"))
    lines=["# Node commit audit","",
      "Bindings are recomputed only from `git_commit_after` or an actual commit-completion marker inside a bound event.",
      "Session, run, branch, and timestamp proximity alone are not binding evidence.","",
      "| node_id | proposal | commits | hashes | methods | confidence | suspicious |","|---|---|---:|---|---|---|---|"]
    suspicious_total=0
    for node in con.execute("SELECT * FROM discovery_nodes ORDER BY run_id,sequence_index"):
        links=list(con.execute("""SELECT l.*,c.timestamp,c.author_name,c.author_email FROM node_commit_links l
          JOIN git_commits c ON c.commit_hash=l.commit_hash WHERE l.node_id=? ORDER BY c.timestamp,l.commit_hash""",(node["node_id"],)))
        reasons=[]
        if len(links)>3: reasons.append("more than 3 commits")
        if any(per_commit[x["commit_hash"]]>3 for x in links): reasons.append("commit bound to more than 3 nodes")
        if node["git_commit_after"] and not any(x["commit_hash"]==node["git_commit_after"] for x in links):
            reasons.append("git_commit_after missing")
        if node["created_at"]:
            if any(x["timestamp"] and x["timestamp"] < node["created_at"] for x in links): reasons.append("commit predates node")
        authors={(x["author_name"],x["author_email"]) for x in links}
        if len(authors)>1: reasons.append("multiple authors")
        suspicious=", ".join(reasons) if reasons else ""
        suspicious_total+=bool(reasons)
        proposal=(node["proposal_text"] or "").replace("|","\\|").replace("\n"," ")[:100]
        hashes=", ".join(x["commit_hash"][:12] for x in links)
        methods=", ".join(sorted({x["binding_method"] for x in links}))
        conf=", ".join(sorted({f"{x['confidence']:.2f}" for x in links}))
        lines.append(f"| `{node['node_id']}` | {proposal} | {len(links)} | `{hashes}` | {methods} | {conf} | {suspicious} |")
    lines += ["",f"Suspicious node count: **{suspicious_total}**",""]
    report.write_text("\n".join(lines),encoding="utf-8")
    return suspicious_total
