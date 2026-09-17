"""Bounded parser experiment runner adapted from local csv_experiment/run.py.

Codex proposes adapter changes; this runner freezes evaluator/schema, snapshots
adapter code and compares deterministic development/held-out conversion metrics.
It never trains models, invents data, invokes an LLM or modifies raw inputs.
"""
import argparse
import hashlib
import itertools
import json
import shutil
import time
from pathlib import Path
from common import ROOT,digest
from audit import aggregate,evaluate,hard_failure,objective
from registry import ADAPTERS

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--source',choices=ADAPTERS);p.add_argument('--limit',type=int,default=1000)
    a=p.parse_args();dst=ROOT/'reports/autoresearch'/a.label
    dst.mkdir(parents=True,exist_ok=False)
    freeze=ROOT/'reports/autoresearch/fixed_hashes.json'
    fixed={str(x.relative_to(ROOT)):hashlib.file_digest(x.open('rb'),'sha256').hexdigest() for x in [ROOT/'src/schema.sql',ROOT/'src/audit.py',ROOT/'src/common.py']}
    if freeze.exists():
        assert json.loads(freeze.read_text())==fixed,'Fixed evaluator/schema/common changed'
    else:freeze.write_text(json.dumps(fixed,indent=2))
    result={}
    for name,cls in ADAPTERS.items():
        if a.source and a.source!=name:continue
        start=time.time();dev=[];held=[];errors=[]
        try:
            for b in itertools.islice(cls().bundles(),a.limit):
                metrics=evaluate(b)
                (held if int(digest(b['run']['run_id']),16)%5==0 else dev).append(metrics)
        except Exception as e:errors.append(f'{type(e).__name__}: {e}')
        result[name]={'development':aggregate(dev),'held_out':aggregate(held),'runs_evaluated':len(dev)+len(held),'errors':errors,'seconds':time.time()-start}
        for split in ('development','held_out'):
            m=result[name][split]
            if m:result[name][split]['objective']=objective(m)
        print(name,json.dumps(result[name]),flush=True)
    shutil.copytree(ROOT/'src/adapters',dst/'adapters',ignore=shutil.ignore_patterns('__pycache__'))
    (dst/'result.json').write_text(json.dumps(result,indent=2))
    with (ROOT/'reports/autoresearch/results.jsonl').open('a') as f:
        f.write(json.dumps({'label':a.label,'result':result})+'\n')

if __name__=='__main__':main()
