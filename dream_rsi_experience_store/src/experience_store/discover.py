from __future__ import annotations
import datetime as dt, hashlib, json, subprocess, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from .util import file_type, sha256_file, stable_id, walk_files


def git(repo: Path, *args: str, check=True) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True,
                          check=check).stdout


def materialize_nulla_logs(bare: Path, target: Path) -> list[dict]:
    target.mkdir(parents=True, exist_ok=True)
    branches = git(bare, "for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines()
    records = []
    seen = set()
    for branch in branches:
        names = git(bare, "ls-tree", "-r", "--name-only", branch).splitlines()
        for rel in names:
            if not (rel.startswith("logs/") or "/raw_sessions/" in f"/{rel}" or rel.endswith("events.jsonl")):
                continue
            key = (branch, rel)
            if key in seen: continue
            seen.add(key)
            safe_branch = branch.replace("/", "__")
            out = target / safe_branch / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            blob = subprocess.run(["git", "-C", str(bare), "show", f"{branch}:{rel}"], capture_output=True, check=True).stdout
            out.write_bytes(blob)
            commit = git(bare, "log", "-1", "--format=%H", branch, "--", rel).strip()
            records.append({"branch": branch, "relative_path": rel, "path": str(out.resolve()), "git_commit": commit})
    return records


def discover(workspace: Path, output: Path) -> dict:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    local_repo = workspace / "WorldModel_plants"
    bare = workspace / "_ingest_raw/git/WorldModel_plants.git"
    extracted = workspace / "_ingest_raw/codex_sessions"
    nulla_root = workspace / "_ingest_raw/nulla_git"
    nulla_files = materialize_nulla_logs(bare, nulla_root)
    sources = [
        {"source_id":"src_rafael_local", "source_type":"rafael_local", "source_root":str((local_repo/"research").resolve()),
         "repo_url":git(local_repo,"remote","get-url","origin").strip(), "repo_path":str(local_repo.resolve()),
         "git_branch":git(local_repo,"branch","--show-current").strip(), "git_commit":git(local_repo,"rev-parse","HEAD").strip(),
         "archive_path":None,"discovered_at":now,"metadata":{}},
        {"source_id":"src_nulla_git", "source_type":"nulla_git", "source_root":str(nulla_root.resolve()),
         "repo_url":git(bare,"remote","get-url","origin").strip(),"repo_path":str(bare.resolve()),"git_branch":None,
         "git_commit":None,"archive_path":None,"discovered_at":now,"metadata":{}},
    ]
    archives = sorted(Path.home().joinpath("Downloads").glob("codex_sessions*.zip"))
    for i, archive in enumerate(archives, 1):
        sources.append({"source_id":f"src_codex_zip_{i:03d}","source_type":"codex_sessions_zip",
            "source_root":str(extracted.resolve()),"repo_url":None,"repo_path":None,"git_branch":None,"git_commit":None,
            "archive_path":str(archive.resolve()),"discovered_at":now,"metadata":{"archive_sha256":sha256_file(archive)}})
    files=[]
    roots=[(sources[0],local_repo/"research"),(sources[1],nulla_root)]
    for source in sources[2:]: roots.append((source,extracted))
    seen_source_paths=set()
    for source,root in roots:
        for path in walk_files(root):
            rel=path.relative_to(root).as_posix()
            key=(source["source_id"],str(path.resolve()))
            if key in seen_source_paths: continue
            seen_source_paths.add(key)
            stat=path.stat(); digest=sha256_file(path)
            meta=next((x for x in nulla_files if x["path"]==str(path.resolve())),{})
            files.append({"file_id":stable_id("file",source["source_id"],rel,digest),"source_id":source["source_id"],
                "original_path":str(path.resolve()),"relative_path":rel,"file_type":file_type(path),"size_bytes":stat.st_size,
                "sha256":digest,"mtime":dt.datetime.fromtimestamp(stat.st_mtime,dt.timezone.utc).isoformat(),
                "git_commit":meta.get("git_commit"),"archived_copy_path":str(path.resolve()) if source["source_type"]!="rafael_local" else None})
    # The archive itself is immutable raw evidence in addition to its extracted
    # members.  Keep it in the same source manifest without parsing it as data.
    for source in sources[2:]:
        archive=Path(source["archive_path"]);stat=archive.stat();digest=sha256_file(archive)
        rel=f"__archive__/{archive.name}"
        files.append({"file_id":stable_id("file",source["source_id"],rel,digest),"source_id":source["source_id"],
            "original_path":str(archive.resolve()),"relative_path":rel,"file_type":"zip_archive","size_bytes":stat.st_size,
            "sha256":digest,"mtime":dt.datetime.fromtimestamp(stat.st_mtime,dt.timezone.utc).isoformat(),
            "git_commit":None,"archived_copy_path":str(archive.resolve())})
    # Zip entry counts include directories and prove archive coverage separately from extracted files.
    zip_entries=[]
    nested=[]
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                zip_entries.append({"archive":str(archive),"name":info.filename,"size":info.file_size,"crc":info.CRC,"is_dir":info.is_dir()})
                if info.filename.lower().endswith(".zip"): nested.append(info.filename)
    branches=git(bare,"for-each-ref","--format=%(refname:short)","refs/heads").splitlines()
    anchor="7d0999a73370223934e540adf3b2066ebda5b7ae"
    identity=git(bare,"show","-s","--format=%an%x09%ae",anchor).strip().split("\t")
    logfmt="%H%x09%an%x09%ae%x09%aI%x09%s"
    commits=[]
    for line in git(bare,"log","--all",f"--author={identity[1]}",f"--format={logfmt}").splitlines():
        sha,name,email,when,msg=line.split("\t",4)
        contained=git(bare,"for-each-ref",f"--contains={sha}","--format=%(refname:short)","refs/heads").splitlines()
        commits.append({"commit":sha,"author_name":name,"author_email":email,"timestamp":when,"message":msg,"branches":contained})
    thread_ids=defaultdict(list)
    types=Counter()
    for rec in files:
        if rec["file_type"] not in ("codex_rollout","events_jsonl","jsonl"): continue
        try:
            with Path(rec["original_path"]).open(errors="replace") as h:
                for n,line in enumerate(h,1):
                    try:o=json.loads(line)
                    except Exception:continue
                    types[o.get("type") or f"{o.get('actor','unknown')}.{o.get('phase','unknown')}.{o.get('state','unknown')}"]+=1
                    payload=o.get("payload") or {}
                    tid=payload.get("session_id") or payload.get("thread_id") or (o.get("thread_id") if o.get("type")=="thread.started" else None)
                    if tid:thread_ids[str(tid)].append({"file_id":rec["file_id"],"line":n})
        except OSError: pass
    duplicates={k:v for k,v in thread_ids.items() if len({x["file_id"] for x in v})>1}
    manifest={"schema_version":1,"discovered_at":now,"workspace":str(workspace.resolve()),"sources":sources,"files":files,
              "archives":zip_entries,"nested_archives":nested,"git":{"branches":branches,"nulla_identity":identity,"nulla_commits":commits},
              "thread_ids":dict(thread_ids),"duplicate_thread_candidates":duplicates,"event_type_counts":dict(types)}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    return manifest


def render_report(manifest: dict, path: Path):
    counts=Counter(f["source_id"] for f in manifest["files"]); sizes=Counter()
    for f in manifest["files"]:sizes[f["source_id"]]+=f["size_bytes"]
    ft=Counter(f["file_type"] for f in manifest["files"])
    lines=["# Source discovery","",f"Discovered: `{manifest['discovered_at']}`","","## Sources","",
           "| source | type | files | bytes | root/archive |","|---|---:|---:|---:|---|"]
    for s in manifest["sources"]:
        lines.append(f"| `{s['source_id']}` | {s['source_type']} | {counts[s['source_id']]} | {sizes[s['source_id']]} | `{s.get('archive_path') or s['source_root']}` |")
    lines += ["","## Required inventory","",f"- Rafael candidates: `{counts['src_rafael_local']}` files",
              f"- Git repositories: `1` (`{manifest['sources'][0]['repo_url']}`)",f"- Remote branches: `{len(manifest['git']['branches'])}`",
              f"- Nulla1202 commits: `{len(manifest['git']['nulla_commits'])}`",f"- Codex archives: `{len(manifest['archives']) and len(manifest['sources'])-2}`",
              f"- Archive entries: `{len(manifest['archives'])}`",f"- rollout JSONL: `{ft['codex_rollout']}`",
              f"- events JSONL: `{ft['events_jsonl']}`",f"- detected thread IDs: `{len(manifest['thread_ids'])}`",
              f"- duplicate thread candidates: `{len(manifest['duplicate_thread_candidates'])}`",f"- nested zip entries: `{len(manifest['nested_archives'])}`",
              "","## Nulla1202 commits","","| commit | branches | message |","|---|---|---|"]
    for c in manifest["git"]["nulla_commits"]:lines.append(f"| `{c['commit'][:12]}` | {', '.join(c['branches'])} | {c['message']} |")
    lines += ["","## Observed event types","", "```json", json.dumps(manifest["event_type_counts"],indent=2,ensure_ascii=False), "```",""]
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text("\n".join(lines),encoding="utf-8")
