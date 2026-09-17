"""Pinned public history download; never executes downloaded source."""
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import urllib.parse
from pathlib import Path
from inventory import ROOT

def clone(item):
    dst=ROOT/'repos'/item['source']
    if not dst.exists():
        subprocess.run(['gh','repo','clone',item['repo'],str(dst),'--','--depth','1'],check=True,capture_output=True)
    actual=subprocess.check_output(['git','-C',str(dst),'rev-parse','HEAD'],text=True).strip()
    if actual!=item['sha']:
        raise RuntimeError(f'HEAD changed since inventory: {item["repo"]}')
    print('CLONED',item['source'],actual,flush=True)
    return item['source']

def download(item):
    dataset,f,sha=item
    source='swe_agent' if dataset.startswith('SWE') else 'rest_mcts'
    dest=ROOT/'raw'/source/dataset.replace('/','__')/f['path']
    dest.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/datasets/{dataset}/resolve/{sha}/'+urllib.parse.quote(f['path'])
    if not dest.exists():
        tmp=dest.with_suffix(dest.suffix+'.partial')
        cmd=['curl','--fail','--location','--retry','3','--retry-delay','2','--connect-timeout','30','--max-time','1800','--silent','--show-error',url,'-o',str(tmp)]
        result=subprocess.run(cmd,capture_output=True,text=True)
        if result.returncode:
            return {'source':source,'dataset':dataset,'url':url,'error':result.stderr}
        tmp.rename(dest)
    size=dest.stat().st_size
    if size!=f['size']:
        raise RuntimeError(f'Size mismatch: {dest}: {size} != {f["size"]}')
    sha256=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()
    lfs=f.get('lfs',{}).get('oid')
    if lfs and sha256!=lfs:
        raise RuntimeError(f'LFS hash mismatch: {dest}')
    print('DOWNLOADED',source,f['path'],size,flush=True)
    return {'source':source,'dataset':dataset,'url':url,'path':str(dest.relative_to(ROOT)),
            'sha256':sha256,'bytes':size,'revision':sha}

if __name__=='__main__':
    inv=json.loads((ROOT/'reports/inventory.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as e:
        list(e.map(clone,inv))
    jobs=[]
    for x in json.loads((ROOT/'reports/hf_inventory.json').read_text()):
        for f in x.get('files',[]):
            if f.get('type')!='file':continue
            if x['dataset'].startswith('SWE'):
                keep=f['path'].endswith('.parquet')
            else: keep=f['path'].endswith(('.json','.jsonl'))
            if keep:jobs.append((x['dataset'],f,x['info']['sha']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as e:
        results=list(e.map(download,jobs))
    (ROOT/'reports/download_manifest.json').write_text(json.dumps(results,indent=2))
