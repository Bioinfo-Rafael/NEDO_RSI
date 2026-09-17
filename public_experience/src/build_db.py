"""Offline, atomic DB builds using the schema copied from the actual local DB."""
import argparse
import collections
import json
import os
import sqlite3
import time
from common import ROOT
from audit import evaluate,hard_failure
from registry import ADAPTERS

def insert(con,table,row):
    keys=tuple(row)
    con.execute(f'INSERT INTO {table} ({",".join(keys)}) VALUES ({",".join("?" for _ in keys)})',tuple(row.values()))

def build(source):
    dst=ROOT/'normalized'/f'{source}.db';tmp=dst.with_suffix('.building.db')
    if tmp.exists():raise RuntimeError(f'Build already exists: {tmp}')
    con=sqlite3.connect(tmp);con.execute('PRAGMA foreign_keys=ON');con.execute('PRAGMA journal_mode=MEMORY');con.execute('PRAGMA synchronous=OFF')
    con.execute('PRAGMA cache_size=-65536')
    schema=(ROOT/'src/schema.sql').read_text();tables=schema.split('CREATE INDEX')[0];con.executescript(tables)
    total=collections.Counter();splits={'development':collections.Counter(),'held_out':collections.Counter()};hist=collections.Counter();datasets=collections.Counter();start=time.time()
    for b in ADAPTERS[source]().bundles():
        m=evaluate(b)
        if hard_failure(m):raise ValueError(f'Invalid conversion: {b["run"]["run_id"]}: {m}')
        insert(con,'runs',b['run'])
        # Parent may be later in array; defer FK checks within each transaction.
        con.execute('PRAGMA defer_foreign_keys=ON')
        for n in b['nodes']:insert(con,'experiences',n)
        for e in b['events']:insert(con,'raw_events',e)
        total['runs']+=1
        for k,v in m.items():total[k]=max(total[k],v) if k=='max_depth' else total[k]+v
        from common import digest
        split='held_out' if int(digest(b['run']['run_id']),16)%5==0 else 'development'
        splits[split]['runs']+=1
        for k,v in m.items():splits[split][k]=max(splits[split][k],v) if k=='max_depth' else splits[split][k]+v
        children=collections.Counter(n['parent_id'] for n in b['nodes'] if n['parent_id'])
        hist.update(children.get(n['node_id'],0) for n in b['nodes'])
        datasets[json.loads(b['run']['metadata_json'])['source_dataset']]+=len(b['nodes'])
        if total['runs']%1000==0:
            con.commit()
            if total['runs']%10000==0:print(source,'runs',total['runs'],'nodes',total['nodes'],'seconds',round(time.time()-start),flush=True)
    if hasattr(ADAPTERS[source](),'unbound_events'):
        for e in ADAPTERS[source]().unbound_events():
            insert(con,'raw_events',e);total['raw_events']+=1;total['unbound_events']+=1
    con.commit()
    print(source,'creating indices',flush=True)
    for statement in schema.split(';'):
        if statement.strip().startswith('CREATE INDEX'):con.execute(statement)
    print(source,'checking foreign keys and SQLite integrity',flush=True)
    assert not con.execute('PRAGMA foreign_key_check').fetchone()
    assert con.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    con.commit();con.close();os.replace(tmp,dst)
    report={'source':source,**dict(total),'branching_histogram':dict(hist),'datasets_nodes':dict(datasets),'splits':splits,'seconds':time.time()-start}
    (ROOT/'reports'/f'{source}_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

def combine():
    for source in ADAPTERS:
        if (ROOT/'normalized'/f'{source}.building.db').exists():
            raise RuntimeError(f'Source build still in progress: {source}')
    public=ROOT/'public_experience.db';tmp=public.with_suffix('.building.db')
    if tmp.exists():raise RuntimeError(f'Build already exists: {tmp}')
    # Reuse the largest verified source, including its indexes. This avoids
    # inserting and reindexing millions of already validated rows a second time.
    base=max(ADAPTERS,key=lambda s:json.loads((ROOT/'reports'/f'{s}_audit.json').read_text()).get('nodes',0))
    base_path=ROOT/'normalized'/f'{base}.db'
    src_con=sqlite3.connect(f'file:{base_path}?mode=ro',uri=True);con=sqlite3.connect(tmp)
    print('backing up verified base',base,flush=True)
    src_con.backup(con);src_con.close()
    con.execute('PRAGMA journal_mode=MEMORY');con.execute('PRAGMA synchronous=OFF')
    con.execute('PRAGMA cache_size=-2097152')
    for src in ADAPTERS:
        if src==base:continue
        p=ROOT/'normalized'/f'{src}.db'
        if not p.exists():raise FileNotFoundError(p)
        con.execute('ATTACH DATABASE ? AS src',(f'file:{p}?mode=ro',))
        for table in ('runs','experiences','raw_events'):
            print('merging',src,table,flush=True)
            con.execute(f'INSERT INTO main.{table} SELECT * FROM src.{table}')
        con.commit();con.execute('DETACH DATABASE src')
        print('merged',src,flush=True)
    con.commit()
    print('public checking foreign keys',flush=True)
    assert not con.execute('PRAGMA foreign_key_check').fetchone()
    con.close();os.replace(tmp,public)
    combine_existing_public()

def combine_existing_public():
    # Backup the large public DB, then insert the small original store unchanged.
    # This preserves every index without recreating multi-million-row indexes.
    public=ROOT/'public_experience.db'
    original=ROOT.parent/'experience_store/experience.db';combined=ROOT/'combined_experience.db';ctmp=combined.with_suffix('.building.db')
    if ctmp.exists():raise RuntimeError(f'Build already exists: {ctmp}')
    src=sqlite3.connect(f'file:{public}?mode=ro',uri=True);out=sqlite3.connect(ctmp)
    print('combined backing up completed public DB',flush=True)
    src.backup(out);src.close()
    out.execute('PRAGMA cache_size=-65536')
    out.execute('PRAGMA journal_mode=MEMORY');out.execute('PRAGMA synchronous=OFF');out.execute('ATTACH DATABASE ? AS src',(f'file:{original}?mode=ro',))
    for table in ('runs','experiences','raw_events'):
        print('combined copying original',table,flush=True)
        out.execute(f'INSERT INTO main.{table} SELECT * FROM src.{table}');out.commit()
    out.execute('DETACH DATABASE src')
    out.close();os.replace(ctmp,combined)
    print('public and combined complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',choices=ADAPTERS);p.add_argument('--combine',action='store_true');p.add_argument('--combined-only',action='store_true');a=p.parse_args()
    if a.combine:combine()
    elif a.combined_only:combine_existing_public()
    elif a.source:build(a.source)
    else:p.error('--source or --combine is required')
