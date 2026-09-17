from __future__ import annotations
import csv,json,sqlite3
from collections import Counter,defaultdict
from pathlib import Path


COUNT_TABLES=("sources","source_files","sessions","discovery_runs","discovery_nodes","events","metrics","artifacts","git_commits","duplicate_candidates","warnings")


def _cycles(con):
    cycles=[]
    for run in con.execute("SELECT run_id FROM discovery_runs"):
        parents={r[0]:r[1] for r in con.execute("SELECT node_id,parent_node_id FROM discovery_nodes WHERE run_id=?",(run[0],))}
        for start in parents:
            seen=set();cur=start
            while cur:
                if cur in seen:cycles.append((run[0],start));break
                seen.add(cur);cur=parents.get(cur)
    return cycles


def audit(con:sqlite3.Connection,report:Path,coverage_csv:Path,coverage_md:Path):
    counts={t:con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in COUNT_TABLES}
    unique_threads=con.execute("SELECT count(DISTINCT thread_id) FROM sessions WHERE thread_id IS NOT NULL").fetchone()[0]
    duplicate_threads=con.execute("SELECT count(DISTINCT match_value) FROM duplicate_candidates WHERE entity_type='session'").fetchone()[0]
    parse=Counter(dict(con.execute("SELECT parse_status,count(*) FROM source_files GROUP BY parse_status").fetchall()))
    edges=Counter(dict(con.execute("SELECT parent_relation_type,count(*) FROM discovery_nodes GROUP BY parent_relation_type").fetchall()))
    missing_parents=con.execute("""SELECT count(*) FROM discovery_nodes n LEFT JOIN discovery_nodes p ON n.parent_node_id=p.node_id WHERE n.parent_node_id IS NOT NULL AND p.node_id IS NULL""").fetchone()[0]
    roots={r[0] for r in con.execute("SELECT root_node_id FROM discovery_runs WHERE root_node_id IS NOT NULL")}
    orphans=[r[0] for r in con.execute("SELECT node_id FROM discovery_nodes WHERE parent_node_id IS NULL") if r[0] not in roots]
    cycles=_cycles(con)
    timestamp_inversions=con.execute("""SELECT count(*) FROM discovery_nodes n JOIN discovery_nodes p ON n.parent_node_id=p.node_id WHERE n.created_at IS NOT NULL AND p.created_at IS NOT NULL AND n.created_at < p.created_at""").fetchone()[0]
    duplicate_events=con.execute("SELECT count(*) FROM (SELECT source_file_id,line_no,count(*) c FROM events GROUP BY 1,2 HAVING c>1)").fetchone()[0]
    unknown_events=con.execute("SELECT count(*) FROM warnings WHERE category='unknown_event_type'").fetchone()[0]
    score_without=con.execute("""SELECT count(*) FROM discovery_nodes n WHERE n.primary_score IS NOT NULL AND NOT EXISTS (SELECT 1 FROM metrics m WHERE m.node_id=n.node_id AND m.name=n.primary_score_name AND abs(m.value-n.primary_score)<1e-12)""").fetchone()[0]
    prov_missing={}
    for table,key,etype in (("sessions","session_id","session"),("discovery_runs","run_id","discovery_run"),("discovery_nodes","node_id","discovery_node"),("events","event_id","event"),("metrics","metric_id","metric"),("artifacts","artifact_id","artifact"),("git_commits","commit_hash","git_commit")):
        prov_missing[table]=con.execute(f"SELECT count(*) FROM {table} t WHERE NOT EXISTS(SELECT 1 FROM provenance p WHERE p.entity_type=? AND p.entity_id=t.{key})",(etype,)).fetchone()[0]
    coverage=[]
    for row in con.execute("SELECT f.*, EXISTS(SELECT 1 FROM artifacts a JOIN provenance p ON p.entity_type='artifact' AND p.entity_id=a.artifact_id WHERE p.file_id=f.file_id) referenced FROM source_files f ORDER BY source_id,relative_path"):
        d=dict(row);d["discovered"]="yes";d["hashed"]="yes" if d["sha256"] else "no";d["classified"]="yes" if d["file_type"] else "no";d["parsed"]=d["parse_status"];d["referenced_in_db"]="yes" if d.pop("referenced") else "no";coverage.append(d)
    coverage_csv.parent.mkdir(parents=True,exist_ok=True)
    cols=["file_id","source_id","original_path","relative_path","file_type","size_bytes","sha256","discovered","hashed","classified","parsed","referenced_in_db","ignored_reason"]
    with coverage_csv.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r.get(k) for k in cols} for r in coverage)
    unaccounted=[r for r in coverage if r["referenced_in_db"]!="yes" and not r.get("ignored_reason")]
    cov_lines=["# Raw data coverage","",f"All discovered files: **{len(coverage)}**",f"Referenced in DB: **{sum(r['referenced_in_db']=='yes' for r in coverage)}**",f"Unaccounted files: **{len(unaccounted)}**","","| parse status | files |","|---|---:|"]
    for k,v in sorted(parse.items()):cov_lines.append(f"| {k} | {v} |")
    cov_lines += ["","Failed parse files remain represented as artifacts and source_files; `ignored_reason` or a warning records the reason.",""]
    coverage_md.write_text("\n".join(cov_lines))
    lines=["# Experience store audit","", "## Counts","", "| entity | count |","|---|---:|"]
    for k,v in counts.items():lines.append(f"| {k} | {v} |")
    lines += [f"| unique Codex threads | {unique_threads} |",f"| duplicate thread IDs | {duplicate_threads} |","","## Integrity checks","",
              f"- Parsed/referenced/failed source files: `{parse.get('parsed',0)}` / `{parse.get('referenced',0)}` / `{parse.get('failed',0)}`",
              f"- Orphan non-root nodes: `{len(orphans)}`",f"- Missing parent references: `{missing_parents}`",f"- Discovery-tree cycles: `{len(cycles)}`",
              f"- Timestamp inversions: `{timestamp_inversions}`",f"- Duplicate events by source line: `{duplicate_events}`",f"- Unknown event-type warnings: `{unknown_events}`",
              f"- Rows without provenance: `{sum(prov_missing.values())}` (`{json.dumps(prov_missing,sort_keys=True)}`)",
              f"- **Score fields populated without evidence: `{score_without}`**",f"- Raw files without DB reference or ignored reason: `{len(unaccounted)}`","","## Tree edges","", "| relation | nodes |","|---|---:|"]
    for k,v in sorted(edges.items()):lines.append(f"| {k} | {v} |")
    passed=not any((orphans,missing_parents,cycles,timestamp_inversions,duplicate_events,score_without,unaccounted)) and sum(prov_missing.values())==0
    lines += ["",f"## Result: {'PASS' if passed else 'FAIL'}","", "No replay value, policy score, or inferred quality score was generated.",""]
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text("\n".join(lines))
    return {"passed":passed,"counts":counts,"unique_threads":unique_threads,"duplicate_threads":duplicate_threads,"parse":dict(parse),"edges":dict(edges),"score_without_evidence":score_without,"provenance_missing":prov_missing,"unaccounted":len(unaccounted)}
