from __future__ import annotations
import datetime as dt, json, re, subprocess
from collections import defaultdict
from pathlib import Path
from .db import connect
from .parsers.codex_rollout import iter_jsonl, project, thread_id
from .provenance import add as provenance
from .secrets import redact
from .util import file_type, jdump, mime_type, stable_id


KNOWN_CODEX={"session_meta","event_msg","response_item","world_state","turn_context","token_usage_record","thread.started","turn.started","turn.completed","item.started","item.completed"}


def _load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def _session(con, tid, source_id, file_id, meta, kind="codex"):
    sid=stable_id("session",tid)
    prompt=None
    if isinstance(meta.get("base_instructions"),dict):prompt=meta["base_instructions"].get("text")
    prompt,secret=redact(prompt)
    con.execute("""INSERT INTO sessions(session_id,external_session_id,thread_id,session_type,start_time,status,prompt_text,working_directory,model,metadata_json)
      VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(thread_id) DO UPDATE SET
      start_time=COALESCE(sessions.start_time,excluded.start_time),working_directory=COALESCE(sessions.working_directory,excluded.working_directory),model=COALESCE(sessions.model,excluded.model)""",
      (sid,tid,tid,kind,meta.get("timestamp"),meta.get("status"),prompt,meta.get("cwd"),meta.get("model") or meta.get("model_provider"),jdump(meta)))
    actual=con.execute("SELECT session_id FROM sessions WHERE thread_id=?",(tid,)).fetchone()[0]
    con.execute("INSERT OR IGNORE INTO session_sources VALUES(?,?,?,?,?)",(actual,source_id,file_id,"observed",jdump({"secret_detected":secret})))
    provenance(con,"session",actual,source_id,file_id,derivation_method="thread_id",confidence=1.0,evidence=tid)
    return actual


def _insert_event(con, rec, sid, run_id, node_id, line_no, seq, obj):
    fields=project(obj);raw,secret=redact(json.dumps(obj,ensure_ascii=False,separators=(",",":")))
    eid=stable_id("event",rec["file_id"],line_no)
    con.execute("""INSERT OR IGNORE INTO events(event_id,session_id,run_id,node_id,source_file_id,line_no,sequence_no,timestamp,event_type,role,tool_name,command,cwd,stdout,stderr,exit_code,message,raw_json,secret_detected)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
      (eid,sid,run_id,node_id,rec["file_id"],line_no,seq,obj.get("timestamp"),fields["event_type"],fields["role"],fields["tool_name"],fields["command"],fields["cwd"],fields["stdout"],fields["stderr"],fields["exit_code"],fields["message"],raw,int(secret or fields["secret_detected"])))
    provenance(con,"event",eid,rec["source_id"],rec["file_id"],line_no=line_no,event_index=seq,derivation_method="jsonl_line",confidence=1.0)
    return eid,fields


def _warning(con, source_id, category, message, evidence, session_id=None,node_id=None,severity="warning"):
    wid=stable_id("warning",source_id,category,message,evidence)
    con.execute("INSERT OR IGNORE INTO warnings VALUES(?,?,?,?,?,?,?,?)",(wid,source_id,session_id,node_id,severity,category,message,evidence));return wid


def _numeric_leaves(value,pointer=""):
    if isinstance(value,bool):return
    if isinstance(value,(int,float)):yield pointer or "/",float(value),value
    elif isinstance(value,dict):
        for k,v in value.items():yield from _numeric_leaves(v,pointer+"/"+str(k).replace("~","~0").replace("/","~1"))
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from _numeric_leaves(v,pointer+f"/{i}")


def _metric(con,node_id,session_id,name,value,file_id,raw,pointer,event=None):
    mid=stable_id("metric",node_id,session_id,name,file_id,pointer,event)
    con.execute("INSERT OR IGNORE INTO metrics VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (mid,node_id,session_id,name,value,None,None,file_id,event,json.dumps(raw,ensure_ascii=False),pointer))
    source=con.execute("SELECT source_id FROM source_files WHERE file_id=?",(file_id,)).fetchone() if file_id else None
    if source: provenance(con,"metric",mid,source[0],file_id,json_pointer=pointer,derivation_method="explicit_numeric_field" if pointer else "explicit_log_metric",confidence=1.0,evidence=event)
    return mid


def _file_node_map(manifest):
    out={}
    for f in manifest["files"]:
        m=re.search(r"research/runs/([^/]+)/experiments/([^/]+)/",f["original_path"].replace("\\","/"))
        if m:out[f["file_id"]]=(m.group(1),m.group(2))
    return out


def ingest(workspace:Path,manifest_path:Path,db_path:Path):
    manifest=_load(manifest_path); db_path.unlink(missing_ok=True);con=connect(db_path)
    files={f["file_id"]:f for f in manifest["files"]}; path_records=defaultdict(list)
    for s in manifest["sources"]:
        con.execute("INSERT INTO sources VALUES(?,?,?,?,?,?,?,?,?,?)",(s["source_id"],s["source_type"],s["source_root"],s.get("repo_url"),s.get("repo_path"),s.get("git_branch"),s.get("git_commit"),s.get("archive_path"),s["discovered_at"],jdump(s.get("metadata",{}))))
    for f in manifest["files"]:
        path_records[f["original_path"]].append(f);secret=0
        con.execute("INSERT INTO source_files VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
          (f["file_id"],f["source_id"],f["original_path"],f["relative_path"],f["file_type"],f["size_bytes"],f["sha256"],f.get("mtime"),f.get("git_commit"),f.get("archived_copy_path"),"classified",None,None,secret))
    # All git commits reachable from every branch, deduplicated; branch membership separate.
    bare=workspace/"_ingest_raw/git/WorldModel_plants.git";source_id="src_nulla_git"
    fmt="%H%x1f%an%x1f%ae%x1f%aI%x1f%s%x1f%P"
    raw=subprocess.run(["git","-C",str(bare),"log","--all",f"--format={fmt}"],text=True,capture_output=True,check=True).stdout
    for line in raw.splitlines():
        sha,name,email,when,msg,parents=line.split("\x1f")
        changed=subprocess.run(["git","-C",str(bare),"diff-tree","--no-commit-id","--name-status","-r",sha],text=True,capture_output=True,check=True).stdout.splitlines()
        con.execute("INSERT OR IGNORE INTO git_commits VALUES(?,?,?,?,?,?,?,?,?)",(sha,manifest["sources"][0]["repo_url"],name,email,when,msg,parents,jdump(changed),source_id))
        provenance(con,"git_commit",sha,source_id,git_commit=sha,derivation_method="git_object",confidence=1.0)
        branches=subprocess.run(["git","-C",str(bare),"for-each-ref",f"--contains={sha}","--format=%(refname:short)","refs/heads"],text=True,capture_output=True,check=True).stdout.splitlines()
        con.executemany("INSERT OR IGNORE INTO git_commit_branches VALUES(?,?)",[(sha,b) for b in branches])
    # Rafael explicit runs and experiment nodes.
    run_nodes={}; node_by_exp={}; session_by_run={}
    for rec in manifest["files"]:
        if rec["source_id"]!="src_rafael_local" or rec["relative_path"].split("/")[-1]!="manifest.json" or "/runs/" not in rec["original_path"]:continue
        data=_load(rec["original_path"]);run_name=Path(rec["original_path"]).parent.name
        sid=stable_id("session","rafael",run_name);start=data.get("started_at");end=data.get("finished_at")
        con.execute("INSERT OR IGNORE INTO sessions(session_id,external_session_id,session_type,start_time,end_time,status,working_directory,git_repo,initial_commit,model,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
          (sid,run_name,"autoresearch_harness",start,end,data.get("phase"),data.get("config",{}).get("run_dir"),manifest["sources"][0]["repo_url"],data.get("repo_commit"),data.get("config",{}).get("model"),jdump(data)))
        con.execute("INSERT OR IGNORE INTO session_sources VALUES(?,?,?,?,?)",(sid,rec["source_id"],rec["file_id"],"manifest",jdump({})));provenance(con,"session",sid,rec["source_id"],rec["file_id"],json_pointer="/",derivation_method="run_manifest",confidence=1.0)
        rid=stable_id("run","rafael",run_name);session_by_run[run_name]=sid
        con.execute("INSERT INTO discovery_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(rid,sid,0,None,start,end,"WorldModel autoresearch",run_name,None,data.get("phase"),1.0,jdump(data.get("config",{}),),"unknown",0))
        provenance(con,"discovery_run",rid,rec["source_id"],rec["file_id"],json_pointer="/",derivation_method="run_manifest",confidence=1.0);run_nodes[run_name]=rid
        root=Path(rec["original_path"]).parent/"experiments"
        experiments=[]
        for exp_file in sorted(root.glob("*/experiment.json")):
            exp=_load(exp_file);experiments.append((exp_file,exp))
        experiments.sort(key=lambda x:(x[1].get("round",0),x[1].get("arm",x[1].get("id",""))))
        last_by_arm={};baseline=None
        for seq,(exp_file,exp) in enumerate(experiments):
            expid=exp.get("id") or exp_file.parent.name;arm=exp.get("arm") or expid.split("_")[0];round_no=exp.get("round")
            nid=stable_id("node",rid,expid);parent=None;relation="unknown";confidence=None
            if expid=="baseline":baseline=nid
            elif arm in last_by_arm:parent=last_by_arm[arm];relation="inferred_chronological";confidence=.8
            elif baseline:parent=baseline;relation="inferred_chronological";confidence=.5
            proposal=None
            hyp=exp_file.parent/"source/hypothesis.json"
            if hyp.exists():
                h=_load(hyp);proposal=h.get("hypothesis") if isinstance(h,dict) else None
            score=exp.get("objective") if isinstance(exp.get("objective"),(int,float)) else None
            con.execute("INSERT INTO discovery_nodes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (nid,rid,parent,arm,round_no,seq,None,None,proposal,None,None,str(exp_file.parent),None,1 if exp.get("metrics") else 0,1 if exp.get("status")=="ok" else 0,None,exp.get("error"),score,"objective" if score is not None else None,None,None,None,None,None,relation,confidence,jdump({"experiment_id":expid,"decision":exp.get("decision"),"raw":exp}),"unknown","unknown",None))
            node_by_exp[(run_name,expid)]=nid;last_by_arm[arm]=nid
            exp_rec=next((f for f in manifest["files"] if f["original_path"]==str(exp_file.resolve())),rec)
            provenance(con,"discovery_node",nid,exp_rec["source_id"],exp_rec["file_id"],json_pointer="/",derivation_method="experiment_directory",confidence=1.0)
            for ptr,val,rawv in _numeric_leaves(exp.get("metrics",{}),"/metrics"):_metric(con,nid,sid,ptr.removeprefix("/metrics/"),val,exp_rec["file_id"],rawv,ptr)
            if score is not None:_metric(con,nid,sid,"objective",float(score),exp_rec["file_id"],score,"/objective")
        if baseline:con.execute("UPDATE discovery_runs SET root_node_id=? WHERE run_id=?",(baseline,rid))
    # Parse every JSONL losslessly and deduplicate sessions by thread id.
    active_session={};event_fields=[]
    for rec in manifest["files"]:
        if rec["file_type"] not in ("codex_rollout","events_jsonl","jsonl"):continue
        path=Path(rec["original_path"]);sid=None;run_id=None;node_id=None;valid=0;failed=0;seq=0
        m=re.search(r"research/runs/([^/]+)(?:/experiments/([^/]+))?",path.as_posix())
        if m:
            run_id=run_nodes.get(m.group(1));sid=session_by_run.get(m.group(1));node_id=node_by_exp.get((m.group(1),m.group(2))) if m.group(2) else None
        for line_no,obj,error in iter_jsonl(path):
            if error:
                failed+=1;_warning(con,rec["source_id"],"invalid_jsonl_line",error,f"{rec['relative_path']}:{line_no}",sid,node_id);continue
            valid+=1;tid=thread_id(obj)
            if tid:
                meta=obj.get("payload") or obj
                sid=_session(con,str(tid),rec["source_id"],rec["file_id"],meta)
                active_session[rec["file_id"]]=sid
                if not run_id:
                    run_id=stable_id("run","session",sid)
                    con.execute("INSERT OR IGNORE INTO discovery_runs(run_id,session_id,run_index,start_time,task_name,status,provenance_confidence,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
                      (run_id,sid,0,obj.get("timestamp"),"Codex session",None,1.0,jdump({})))
                    provenance(con,"discovery_run",run_id,rec["source_id"],rec["file_id"],line_no=line_no,derivation_method="session_log",confidence=1.0)
            sid=sid or active_session.get(rec["file_id"])
            eid,fields=_insert_event(con,rec,sid,run_id,node_id,line_no,seq,obj);seq+=1;event_fields.append((eid,sid,node_id,rec,fields,obj))
            base_type=(obj.get("type") or "unknown")
            if base_type not in KNOWN_CODEX and rec["file_type"]=="codex_rollout":_warning(con,rec["source_id"],"unknown_event_type",base_type,f"{rec['relative_path']}:{line_no}",sid,node_id,"info")
        status="parsed" if valid else "failed"
        con.execute("UPDATE source_files SET parse_status=?,parser_name=?,ignored_reason=? WHERE file_id=?",(status,"jsonl",None if valid else "no valid JSON lines",rec["file_id"]))
    # Nulla1202's production branch: commits are explicit workspace lineage. Keep
    # all author commits as nodes; attach events after each observed commit marker.
    nulla_commits=sorted(manifest["git"]["nulla_commits"],key=lambda c:(c["timestamp"],c["commit"]))
    prod_tid="01a0a295-6c67-7ad0-9583-6093908fe040"
    prod_sid=stable_id("session",prod_tid)
    if con.execute("SELECT 1 FROM sessions WHERE session_id=?",(prod_sid,)).fetchone() and nulla_commits:
        prod_run=stable_id("run","nulla","control/production-20260915b")
        con.execute("INSERT OR IGNORE INTO discovery_runs(run_id,session_id,run_index,task_name,project_name,git_branch,status,provenance_confidence,metadata_json) VALUES(?,?,?,?,?,?,?,?,?)",
          (prod_run,prod_sid,0,"H200 control autoresearch","WorldModel_plants","control/production-20260915b","interrupted",.9,jdump({"reconstruction":"git lineage plus Codex event log"})))
        provenance(con,"discovery_run",prod_run,"src_nulla_git",git_commit=nulla_commits[0]["commit"],
                   derivation_method="git_branch_and_session_log",confidence=.9,
                   evidence="control/production-20260915b")
        prev=None;commit_nodes={}
        for seq,c in enumerate(nulla_commits):
            nid=stable_id("node",prod_run,c["commit"]);commit_nodes[c["commit"]]=nid
            parent=None;relation="unknown";confidence=None
            parents=con.execute("SELECT parents FROM git_commits WHERE commit_hash=?",(c["commit"],)).fetchone()
            if parents:
                for p in parents[0].split():
                    if p in commit_nodes:parent=commit_nodes[p];relation="inferred_git";confidence=.95;break
            if parent is None and prev is not None:parent=prev;relation="inferred_chronological";confidence=.4
            con.execute("INSERT OR IGNORE INTO discovery_nodes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (nid,prod_run,parent,"control/production-20260915b",seq,seq,c["timestamp"],None,c["message"],None,None,None,None,None,None,None,None,None,None,None,c["commit"],prod_tid,None,None,relation,confidence,jdump({"commit":c["commit"],"branches":c["branches"]}),"unknown","unknown",None))
            provenance(con,"discovery_node",nid,"src_nulla_git",git_commit=c["commit"],derivation_method="git_commit",confidence=1.0)
            con.execute("INSERT OR IGNORE INTO session_commits VALUES(?,?,?,?,?,?)",(prod_sid,nid,c["commit"],"node_result",None,None));prev=nid
        con.execute("UPDATE discovery_runs SET root_node_id=? WHERE run_id=?",(next(iter(commit_nodes.values())),prod_run))
        current=None;updated=[]
        short_map={sha[:7]:nid for sha,nid in commit_nodes.items()}|{sha[:12]:nid for sha,nid in commit_nodes.items()}
        for eid,sid,node_id,rec,fields,obj in event_fields:
            if sid!=prod_sid:updated.append((eid,sid,node_id,rec,fields,obj));continue
            text="\n".join(x for x in (fields.get("stdout"),fields.get("message"),fields.get("command")) if x)
            matches=[]
            for short,nid in short_map.items():
                if re.search(rf"\b{re.escape(short)}\b",text):matches.append((text.rfind(short),nid))
            if matches:current=max(matches)[1]
            if current:
                con.execute("UPDATE events SET run_id=?,node_id=? WHERE event_id=?",(prod_run,current,eid));node_id=current
            updated.append((eid,sid,node_id,rec,fields,obj))
        event_fields=updated
    # Explicit metrics in event output only; no derived values.
    metric_re=re.compile(r"(?m)^([A-Za-z][A-Za-z0-9_.-]*):\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    for eid,sid,node_id,rec,fields,obj in event_fields:
        text="\n".join(x for x in (fields.get("stdout"),fields.get("message")) if x)
        for match in metric_re.finditer(text):_metric(con,node_id,sid,match.group(1),float(match.group(2)),rec["file_id"],match.group(0),None,eid)
        for sha in set(re.findall(r"\b[0-9a-f]{7,40}\b",text)):
            row=con.execute("SELECT commit_hash FROM git_commits WHERE commit_hash LIKE ?",(sha+"%",)).fetchone()
            if row and sid:
                con.execute("INSERT OR IGNORE INTO session_commits VALUES(?,?,?,?,?,?)",(sid,node_id,row[0],"mentioned_in_event",rec["file_id"],None))
    # Artifacts: every discovered raw file is represented, and is attached to an explicit experiment when possible.
    fmap=_file_node_map(manifest)
    for rec in manifest["files"]:
        node_id=None
        if rec["file_id"] in fmap:node_id=node_by_exp.get(fmap[rec["file_id"]])
        aid=stable_id("artifact",rec["source_id"],rec["relative_path"],rec["sha256"])
        con.execute("INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?,?,?,?,?,?)",(aid,node_id,rec["file_type"],rec["original_path"],rec["sha256"],rec.get("git_commit"),mime_type(Path(rec["original_path"])),rec["size_bytes"],jdump({"source_file_id":rec["file_id"]})))
        provenance(con,"artifact",aid,rec["source_id"],rec["file_id"],git_commit=rec.get("git_commit"),derivation_method="file_manifest",confidence=1.0)
        if rec["file_type"] not in ("codex_rollout","events_jsonl","jsonl"):
            parser="json" if rec["file_type"] in ("json","manifest","experiment","result","status") else "classified_artifact"
            parse="parsed" if parser=="json" else "referenced"
            if parser=="json":
                try:_load(rec["original_path"])
                except Exception as exc:parse="failed";_warning(con,rec["source_id"],"json_parse_error",str(exc),rec["relative_path"],node_id=node_id)
            con.execute("UPDATE source_files SET parse_status=?,parser_name=? WHERE file_id=?",(parse,parser,rec["file_id"]))
    # Duplicate copies: identical content hashes and thread IDs retain all provenance.
    by_hash=defaultdict(list)
    for rec in manifest["files"]:by_hash[rec["sha256"]].append(rec)
    for digest,recs in by_hash.items():
        if len(recs)>1:
            canonical=recs[0]["file_id"]
            for r in recs[1:]:
                did=stable_id("dup","file",canonical,r["file_id"]);con.execute("INSERT OR IGNORE INTO duplicate_candidates VALUES(?,?,?,?,?,?,?,?)",(did,"source_file",canonical,r["source_id"],r["file_id"],"sha256",digest,"canonical_plus_provenance"))
    for tid,occ in manifest.get("duplicate_thread_candidates",{}).items():
        sid=stable_id("session",tid)
        for x in occ[1:]:
            did=stable_id("dup","session",sid,x["file_id"]);con.execute("INSERT OR IGNORE INTO duplicate_candidates VALUES(?,?,?,?,?,?,?,?)",(did,"session",sid,files[x["file_id"]]["source_id"],x["file_id"],"thread_id",tid,"merged_session_multiple_sources"))
    con.commit();return con
