from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path


EXPLICIT = ("explicit", "filesystem", "generation_id")


def trace_audit(con: sqlite3.Connection, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    total_nodes=con.execute("SELECT count(*) FROM discovery_nodes").fetchone()[0]
    total_events=con.execute("SELECT count(*) FROM events").fetchone()[0]
    bindings=con.execute("SELECT count(*) FROM node_event_bindings").fetchone()[0]
    bound_events=con.execute("SELECT count(DISTINCT event_id) FROM node_event_bindings").fetchone()[0]
    unbound=total_events-bound_events
    with_trace=con.execute("SELECT count(DISTINCT node_id) FROM node_event_bindings").fetchone()[0]
    without_trace=total_nodes-with_trace
    complete=con.execute("SELECT count(DISTINCT node_id) FROM node_trace_ranges WHERE trace_complete=1").fetchone()[0]
    partial=with_trace-complete
    explicit=con.execute("SELECT count(*) FROM node_event_bindings WHERE binding_method IN (?,?,?)",EXPLICIT).fetchone()[0]
    inferred=bindings-explicit
    methods=dict(con.execute("SELECT binding_method,count(*) FROM node_event_bindings GROUP BY 1"))
    histogram=dict(con.execute("""SELECT printf('%.2f',confidence),count(*) FROM node_event_bindings GROUP BY confidence ORDER BY confidence DESC"""))
    orphan=con.execute("""SELECT count(*) FROM discovery_nodes n LEFT JOIN discovery_nodes p ON p.node_id=n.parent_node_id
      WHERE n.parent_node_id IS NOT NULL AND p.node_id IS NULL""").fetchone()[0]
    ambiguous=con.execute("""SELECT count(DISTINCT node_id) FROM node_trace_ranges WHERE confidence < .9""").fetchone()[0]
    duplicate_bindings=con.execute("""SELECT count(*) FROM (SELECT node_id,event_id,count(*) n FROM node_event_bindings GROUP BY 1,2 HAVING n>1)""").fetchone()[0]
    multi_node_events=con.execute("""SELECT count(*) FROM (SELECT event_id,count(DISTINCT node_id) n FROM node_event_bindings GROUP BY event_id HAVING n>1)""").fetchone()[0]
    overlaps=con.execute("""SELECT count(*) FROM node_trace_ranges a JOIN node_trace_ranges b
      ON a.source_file_id=b.source_file_id AND a.node_id<b.node_id
      AND a.first_sequence_no<=b.last_sequence_no AND b.first_sequence_no<=a.last_sequence_no""").fetchone()[0]
    inversions=con.execute("SELECT count(*) FROM node_trace_ranges WHERE first_sequence_no>last_sequence_no OR start_time>end_time").fetchone()[0]
    source_cov=[dict(r) for r in con.execute("""SELECT s.source_type,count(DISTINCT e.event_id) total,
      count(DISTINCT b.event_id) bound FROM events e JOIN source_files f ON f.file_id=e.source_file_id
      JOIN sources s ON s.source_id=f.source_id LEFT JOIN node_event_bindings b ON b.event_id=e.event_id
      GROUP BY s.source_type ORDER BY s.source_type""")]
    run_cov=[dict(r) for r in con.execute("""SELECT coalesce(e.run_id,'(no run)') run_id,count(DISTINCT e.event_id) total,
      count(DISTINCT b.event_id) bound FROM events e LEFT JOIN node_event_bindings b ON b.event_id=e.event_id
      GROUP BY e.run_id ORDER BY e.run_id""")]
    lines=["# Trace binding audit","","## Summary","", "| item | count |","|---|---:|",
      f"| total nodes | {total_nodes} |",f"| nodes with trace | {with_trace} |",f"| nodes without trace | {without_trace} |",
      f"| nodes with complete trace | {complete} |",f"| nodes with partial trace | {partial} |",
      f"| total events | {total_events} |",f"| distinct bound events | {bound_events} |",f"| unbound events | {unbound} |",
      f"| node-event bindings | {bindings} |",f"| explicit/evidence-direct bindings | {explicit} |",f"| inferred bindings | {inferred} |",
      "","`explicit/evidence-direct`は`explicit`、`filesystem`、`generation_id`です。git区間は根拠を持ちますが、event単体のnode IDではないためinferred側に数えます。",
      "","## Binding methods","","| method | count |","|---|---:|"]
    lines += [f"| {k} | {v} |" for k,v in sorted(methods.items())]
    lines += ["","## Confidence histogram","","| confidence | bindings |","|---|---:|"]
    lines += [f"| {k} | {v} |" for k,v in histogram.items()]
    lines += ["","## Source type coverage","","| source type | total events | bound | coverage |","|---|---:|---:|---:|"]
    lines += [f"| {r['source_type']} | {r['total']} | {r['bound']} | {r['bound']/r['total']:.1%} |" for r in source_cov]
    lines += ["","## Run coverage","","| run | total events | bound | coverage |","|---|---:|---:|---:|"]
    lines += [f"| `{r['run_id']}` | {r['total']} | {r['bound']} | {r['bound']/r['total']:.1%} |" for r in run_cov]
    lines += ["","## Consistency checks","",
      f"- Orphan nodes: `{orphan}`",f"- Ambiguous boundaries (confidence < 0.90): `{ambiguous}`",
      f"- Duplicated node/event bindings: `{duplicate_bindings}`",f"- Events bound to multiple nodes: `{multi_node_events}`",
      f"- Node trace range overlaps within one raw file: `{overlaps}`",f"- Timestamp/sequence inversions: `{inversions}`",
      "","Confidenceはtrace relationの根拠の強さであり、実験性能scoreではありません。新しい実験scoreは生成していません。",""]
    (report_dir/"trace_binding_audit.md").write_text("\n".join(lines),encoding="utf-8")

    # Unbound events remain first-class DB rows; report why no evidence-backed node was selected.
    rows=[dict(r) for r in con.execute("""SELECT e.event_id,e.session_id,e.run_id,e.event_type,e.source_file_id,e.line_no
      FROM events e LEFT JOIN node_event_bindings b ON b.event_id=e.event_id WHERE b.event_id IS NULL""")]
    type_counts=Counter(r["event_type"] or "(null)" for r in rows)
    session_counts=Counter(r["session_id"] or "(no session)" for r in rows)
    reason_counts=Counter()
    for r in rows:
        if not r["run_id"]: reason_counts["event has no reconstructed discovery run"]+=1
        elif not con.execute("SELECT 1 FROM discovery_nodes WHERE run_id=?",(r["run_id"],)).fetchone():
            reason_counts["run has no evidence-backed discovery nodes"]+=1
        else: reason_counts["no explicit identifier, path, generation, workspace, git, or coherent boundary evidence"]+=1
    u=["# Unbound events","",f"Unbound event count: **{len(rows)}**","",
       "Unbound events are retained in `events`; they are not forced into a node.","","## Reasons","","| reason | count |","|---|---:|"]
    u += [f"| {k} | {v} |" for k,v in reason_counts.most_common()]
    u += ["","## Event types","","| event type | count |","|---|---:|"]
    u += [f"| `{k}` | {v} |" for k,v in type_counts.most_common()]
    u += ["","## Sessions","","| session | count |","|---|---:|"]
    u += [f"| `{k}` | {v} |" for k,v in session_counts.most_common()]
    (report_dir/"unbound_events.md").write_text("\n".join(u)+"\n",encoding="utf-8")
    return {"total_nodes":total_nodes,"nodes_with_trace":with_trace,"nodes_without_trace":without_trace,
            "complete":complete,"partial":partial,"total_events":total_events,"bindings":bindings,
            "bound_events":bound_events,"unbound":unbound,"explicit":explicit,"inferred":inferred,
            "methods":methods,"overlaps":overlaps,"multi_node_events":multi_node_events,"inversions":inversions}


def raw_integrity(manifest_path: Path, pre_path: Path, report_path: Path):
    manifest=json.loads(manifest_path.read_text())
    pre=json.loads(pre_path.read_text())
    changed=[];missing=[];checked=0
    for rec in manifest["files"]:
        path=Path(rec["original_path"]);checked+=1
        if not path.exists(): missing.append(str(path));continue
        h=hashlib.sha256()
        with path.open("rb") as f:
            while chunk:=f.read(1024*1024): h.update(chunk)
        before=pre.get(rec["file_id"],{}).get("sha256")
        if h.hexdigest()!=before: changed.append({"path":str(path),"before":before,"after":h.hexdigest()})
    lines=["# Raw source integrity","",f"- Files checked: `{checked}`",f"- Missing files: `{len(missing)}`",
           f"- Modified files: **`{len(changed)}`**","", "Result: **"+("PASS" if not changed and not missing else "FAIL")+"**","",
           "SHA-256 values captured before rebuilding were compared with the same raw paths after all generated outputs were rebuilt.",""]
    if missing: lines += ["## Missing",""]+[f"- `{p}`" for p in missing]
    if changed: lines += ["## Changed",""]+[f"- `{r['path']}`" for r in changed]
    report_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"checked":checked,"missing":len(missing),"changed":len(changed)}
