"""Capture build provenance, or reconstruct public inputs in a NEW workspace only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent
DEST = ROOT/'outputs/rebuild'


def read(path):
    return json.loads(path.read_text())


def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture():
    public = ORIGINAL/'public_experience'
    repositories = []
    for name in ('inventory.json', 'additional_inventory.json'):
        for r in read(public/'reports'/name):
            repositories.append({k:r.get(k) for k in ('source', 'repo', 'sha')})
    for r in repositories:
        local = public/'repos'/r['source']
        r['actual_head'] = subprocess.check_output(['git','-C',str(local),'rev-parse','HEAD'], text=True).strip() if local.exists() else None
    downloads = {}
    for name in ('download_manifest.json','download_manifest_swe_additional.json'):
        for r in read(public/'reports'/name):
            if 'path' in r:
                downloads[r['path']] = r
    raw_files = []
    for p in sorted((public/'raw').rglob('*')):
        if p.is_file():
            raw_files.append({'path':str(p.relative_to(public)), 'bytes':p.stat().st_size,'sha256':hash_file(p)})
    used = {}
    for p in sorted((public/'normalized').glob('*.db')):
        c = sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True)
        used[p.stem] = [r[0] for r in c.execute('SELECT DISTINCT source_file FROM raw_events ORDER BY source_file')]
        c.close()
    code = {}
    for base in ('public_experience/src','dream_rsi_experience_store/src','dream_rsi_experience_store/scripts'):
        for p in sorted((ORIGINAL/base).rglob('*')):
            if p.is_file() and p.suffix in ('.py','.sql','.mjs'):
                code[str(p.relative_to(ORIGINAL))] = hash_file(p)
    result = {'repository_commit':subprocess.check_output(['git','-C',str(ORIGINAL),'rev-parse','HEAD'],text=True).strip(),
              'repositories':repositories,'downloads':list(downloads.values()),'raw_files':raw_files,
              'adapter_used_files':used,'code_sha256':code,
              'source_note':'Drive and issue attachments have no immutable revision recorded; SHA256 validates bytes.'}
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs/evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('captured',len(raw_files),'raw files',len(downloads),'downloads')


def prepare():
    if DEST.exists():
        raise FileExistsError('Rebuild directory already exists; do not overwrite: '+str(DEST))
    evidence = read(ROOT/'source_manifest.json')
    for name in ('public_experience/src','dream_rsi_experience_store/src','dream_rsi_experience_store/scripts'):
        shutil.copytree(ORIGINAL/name, DEST/name, ignore=shutil.ignore_patterns('__pycache__','node_modules'))
    for name in ('public_experience/requirements.txt','dream_rsi_experience_store/schema.sql'):
        shutil.copy2(ORIGINAL/name, DEST/name)
    for name in ('normalized','reports','raw','repos'):
        (DEST/'public_experience'/name).mkdir(exist_ok=True)
    (DEST/'experience_store').mkdir(exist_ok=True)
    (DEST/'evidence.json').write_text(json.dumps(evidence,indent=2))
    # Do not silently copy private/local experience.db or credentials.
    for rel, expected in evidence['code_sha256'].items():
        if hash_file(DEST/rel) != expected:
            raise ValueError('Code differs from captured version: '+rel)
    print(DEST)


def download():
    evidence = read(DEST/'evidence.json')
    public = DEST/'public_experience'
    for r in evidence['repositories']:
        p = public/'repos'/r['source']
        if not p.exists():
            subprocess.run(['git','clone','--no-checkout','https://github.com/'+r['repo']+'.git',str(p)],check=True)
            subprocess.run(['git','-C',str(p),'checkout','--detach',r['sha']],check=True)
        actual = subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
        if actual != r['sha']:
            raise ValueError('Unexpected existing checkout: '+str(p))
    def fetch(url,p,expected=None):
        p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            temp=p.with_suffix(p.suffix+'.partial')
            subprocess.run(['curl','--fail','--location','--retry','3',url,'-o',str(temp)],check=True)
            if expected and hash_file(temp)!=expected:
                raise ValueError('Download hash mismatch: '+url)
            temp.rename(p)
        if expected and hash_file(p)!=expected:
            raise ValueError('Existing file differs: '+str(p))
    for d in evidence['downloads']:
        fetch(d['url'],public/d['path'],d['sha256'])
    attachments={'88':'https://github.com/user-attachments/files/20842742/f170432e-eb4b-4e94-b045-c3745efacc49.json',
                 '141':'https://github.com/user-attachments/files/21194986/openevolve_20250712_015411.log'}
    for r in evidence['raw_files']:
        if '/issue_attachments/' in r['path']:
            issue=Path(r['path']).parent.name.removeprefix('issue_')
            fetch(attachments[issue],public/r['path'],r['sha256'])
    subprocess.run(['python3',str(public/'src/download_search_agents.py')],check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    archive=next(r for r in evidence['raw_files'] if r['path'].endswith('gpt4o_search_trajectories.zip'))
    if hash_file(public/archive['path'])!=archive['sha256']:
        raise ValueError('Drive file changed: exact snapshot unavailable')
    print('Inputs acquired. Run prepare_raw.py then verify-raw before normalization.')


def verify_raw():
    evidence=read(DEST/'evidence.json')
    failures=[]
    for r in evidence['raw_files']:
        p=DEST/'public_experience'/r['path']
        if not p.exists() or hash_file(p)!=r['sha256']:
            failures.append(r['path'])
    if failures:
        raise ValueError({'missing_or_changed_raw':failures})
    print('All raw files match captured SHA256')


def local_snapshot():
    """Logical rebuild input for the local store whose historical DDL is missing."""
    target=DEST/'experience_store/experience.db'
    if not target.parent.exists():raise ValueError('Run prepare first')
    if target.exists():raise FileExistsError(target)
    source=ORIGINAL/'experience_store/experience.db'
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(target) as dst:
        src.backup(dst)
    print('Read-only local snapshot:',target)


def sample_raw():
    """Read one actual record per source and retain format/field evidence only."""
    import sys
    sys.path.insert(0,str(ORIGINAL/'public_experience/src'))
    from common import json_records
    from adapters.search_agents import pages
    import pyarrow.parquet as pq
    evidence=read(ROOT/'source_manifest.json');samples={}
    for source,files in evidence['adapter_used_files'].items():
        if not files:continue
        selected=next((f for f in files if '/programs/' in f),files[0]) if source=='openevolve' else next((f for f in files if f.endswith('.parquet')),files[0]) if source=='sweagent' else files[0]
        p=ORIGINAL/'public_experience'/selected
        if p.suffix=='.parquet':d=next(pq.ParquetFile(p).iter_batches(batch_size=1)).to_pylist()[0]
        elif p.suffix in ('.json','.jsonl','.traj'):d=next(json_records(p))
        elif p.suffix=='.html':
            task,pgs=pages(p);d={'task':task,'first_page':pgs[0]}
        else:d={'first_lines':p.read_text().splitlines()[:3]}
        samples[source]={'file':selected,'sha256':hash_file(p),'top_type':type(d).__name__,
                         'fields':{k:type(v).__name__ for k,v in d.items()} if isinstance(d,dict) else {'first_record':list(d[0]) if d else []}}
    (ROOT/'outputs/raw_samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
    print('sampled',len(samples),'sources')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['capture','prepare','download','verify-raw','local-snapshot','sample-raw'])
    globals()[parser.parse_args().command.replace('-','_')]()
