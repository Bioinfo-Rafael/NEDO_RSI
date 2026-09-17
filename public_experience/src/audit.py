"""Fixed, offline evaluator. Kept outside adapter optimization allowlist."""
import collections
import json
import math
import sqlite3
from common import ROOT,number

def evaluate(bundle):
    ns=bundle['nodes'];es=bundle['events'];by={n['node_id']:n for n in ns};evs={e['event_id']:e for e in es}
    out=dict(nodes=len(ns),edges=0,roots=0,action=0,result=0,score=0,provenance=0,cycles=0,
             duplicate_ids=len(ns)-len(by),invalid_parents=0,fake_scores=0,unknown_parents=0,raw_events=len(es),max_depth=0)
    for n in ns:
        m=json.loads(n['metadata_json']);par=n['parent_id']
        out['edges']+=par is not None;out['roots']+=par is None
        out['unknown_parents']+=m.get('parent_relation')=='unknown'
        out['action']+=bool(n.get('proposal'));out['result']+=bool(n.get('result_summary'));out['score']+=n.get('score') is not None
        out['provenance']+=bool(n.get('source_reference')) and all((ROOT/p).is_file() for p in json.loads(n['source_files_json']))
        out['max_depth']=max(out['max_depth'],n['depth'])
        if par is not None and (par not in by or by[par]['run_id']!=n['run_id']):out['invalid_parents']+=1
        seen=set();cur=n
        while cur is not None:
            if cur['node_id'] in seen:out['cycles']+=1;break
            seen.add(cur['node_id']);cur=by.get(cur['parent_id'])
        if n.get('score') is not None:
            evidence=m.get('score_evidence',{})
            try:
                v=json.loads(evs[evidence['event_id']]['raw_json'])
                for k in evidence['path']:v=v[k]
                if number(v)!=n['score']:out['fake_scores']+=1
            except (KeyError,IndexError,TypeError):out['fake_scores']+=1
    return out

def hard_failure(m):return any(m[k] for k in ('cycles','duplicate_ids','invalid_parents','fake_scores'))

def objective(m):
    if hard_failure(m):return None
    den=m['nodes'] or 1
    return sum(m[k]/den for k in ('action','result','provenance'))+(den-m['unknown_parents'])/den

def aggregate(items):
    total=collections.Counter()
    for m in items:
        for k,v in m.items():
            total[k]=max(total[k],v) if k=='max_depth' else total[k]+v
    return dict(total)
