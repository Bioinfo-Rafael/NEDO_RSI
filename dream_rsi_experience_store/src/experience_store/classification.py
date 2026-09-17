from __future__ import annotations

from pathlib import Path


DIRECTION_EVIDENCE=(
 "WorldModel_plants/research/Prompt.md: Validation selection minimizes "
 "0.5 * forecast_nrmse / baseline_forecast_nrmse + 0.5 * control_cost / baseline_control_cost"
)


def classify_scores_and_runs(con):
    # Unknown is explicit: do not infer direction from a metric name alone.
    con.execute("UPDATE discovery_nodes SET primary_score_direction='unknown',score_direction_method='unknown',score_direction_evidence=NULL")
    con.execute("UPDATE metrics SET direction='unknown'")
    con.execute("""UPDATE discovery_nodes SET primary_score_direction='minimize',
      score_direction_method='task_definition',score_direction_evidence=?
      WHERE primary_score IS NOT NULL AND run_id IN
      (SELECT run_id FROM discovery_runs WHERE task_name='WorldModel autoresearch')""",(DIRECTION_EVIDENCE,))
    # These exact metrics are named in the task definition/report as minimized.
    con.execute("""UPDATE metrics SET direction='minimize' WHERE node_id IN
      (SELECT n.node_id FROM discovery_nodes n JOIN discovery_runs r ON r.run_id=n.run_id
       WHERE r.task_name='WorldModel autoresearch')
      AND name IN ('objective','forecast_nrmse','forecast/forecast_nrmse','control_cost')""")

    con.execute("UPDATE discovery_runs SET run_type='unknown',replay_eligible=0")
    for run in con.execute("SELECT run_id,root_node_id,task_name FROM discovery_runs").fetchall():
        run_id,root,task=run
        nodes=con.execute("SELECT node_id,parent_node_id,valid FROM discovery_nodes WHERE run_id=?",(run_id,)).fetchall()
        if not nodes and task=="Codex session":
            run_type,eligible="raw_session",0
        elif nodes and root and any(n[0]==root for n in nodes):
            parents={n[0]:n[1] for n in nodes};cycle=False
            for start in parents:
                seen=set();cur=start
                while cur:
                    if cur in seen:cycle=True;break
                    seen.add(cur);cur=parents.get(cur)
            valid=any(n[2] in (None,1) for n in nodes)
            run_type,eligible="discovery",int(valid and not cycle)
        elif task and any(word in task.lower() for word in ("setup","launcher","infrastructure")):
            run_type,eligible="infrastructure",0
        else:
            run_type,eligible="unknown",0
        con.execute("UPDATE discovery_runs SET run_type=?,replay_eligible=? WHERE run_id=?",
                    (run_type,eligible,run_id))
    con.commit()


def audit_replay_eligibility(con,report:Path):
    lines=["# Replay eligibility","",
      "Eligibility describes structural replay readiness. It is not an experiment score.","",
      "| run_id | task | project | type | eligible | root | nodes | edges | scored nodes | traced nodes | reason |",
      "|---|---|---|---|---:|---|---:|---:|---:|---:|---|"]
    for r in con.execute("SELECT * FROM discovery_runs ORDER BY start_time,run_id"):
        n=con.execute("SELECT count(*) FROM discovery_nodes WHERE run_id=?",(r["run_id"],)).fetchone()[0]
        edges=con.execute("SELECT count(*) FROM discovery_nodes WHERE run_id=? AND parent_node_id IS NOT NULL",(r["run_id"],)).fetchone()[0]
        scored=con.execute("SELECT count(*) FROM discovery_nodes WHERE run_id=? AND primary_score IS NOT NULL",(r["run_id"],)).fetchone()[0]
        traced=con.execute("""SELECT count(DISTINCT b.node_id) FROM node_event_bindings b JOIN discovery_nodes n ON n.node_id=b.node_id WHERE n.run_id=?""",(r["run_id"],)).fetchone()[0]
        if r["replay_eligible"]: reason="root and valid nodes present; no cycle"
        elif r["run_type"]=="raw_session": reason="Codex trace container with no discovery node"
        elif r["run_type"]=="infrastructure": reason="setup/launcher/bookkeeping only"
        else: reason="insufficient tree evidence"
        lines.append(f"| `{r['run_id']}` | {r['task_name'] or ''} | {r['project_name'] or ''} | {r['run_type']} | {r['replay_eligible']} | `{r['root_node_id'] or ''}` | {n} | {edges} | {scored} | {traced} | {reason} |")
    report.write_text("\n".join(lines)+"\n",encoding="utf-8")
