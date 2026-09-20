"""Two-table SQLite materialization, bounded export data, and integrity verification."""
import collections
import gzip
import json
import sqlite3
from .db import ROOT, WORKSPACE, connect, dumps, sha, write_json, output, verify_protected, source_paths
from .delex import cached, DEFAULT_MODEL, PROMPT_VERSION, validate

CAP=5_000_000_000
SCHEMA='''
CREATE TABLE runs (
 run_id TEXT PRIMARY KEY, source_type TEXT NOT NULL, task_name TEXT,
 source_db TEXT NOT NULL, source_reference TEXT,
 root_anchors_json TEXT NOT NULL, metadata_json TEXT NOT NULL
);
CREATE TABLE branch_points (
 run_id TEXT NOT NULL REFERENCES runs(run_id), branch_point_id TEXT PRIMARY KEY,
 original_node_id TEXT NOT NULL, previous_branch_point_id TEXT,
 next_branch_point_ids_json TEXT NOT NULL,
 incoming_node_ids_json TEXT NOT NULL, incoming_raw_json TEXT NOT NULL,
 terminal_branch_node_ids_json TEXT NOT NULL, terminal_branches_raw_json TEXT NOT NULL,
 branching_factor INTEGER NOT NULL CHECK(branching_factor>=2), depth INTEGER NOT NULL,
 source_type TEXT NOT NULL, source_reference TEXT NOT NULL, raw_context_json TEXT NOT NULL,
 delexicalized_json TEXT, delex_status TEXT NOT NULL CHECK(delex_status IN ('pending','completed','failed')),
 delex_model TEXT, delex_prompt_version TEXT NOT NULL, source_hash TEXT NOT NULL,
 prompt_sha256 TEXT NOT NULL, input_sha256 TEXT, codex_version TEXT, timestamp TEXT,
 output_sha256 TEXT, cache_key TEXT
);
CREATE INDEX branch_points_run ON branch_points(run_id);
CREATE INDEX branch_points_previous ON branch_points(previous_branch_point_id);
CREATE INDEX branch_points_status_source ON branch_points(delex_status,source_type);
'''


def build(model=DEFAULT_MODEL):
    units=json.loads((ROOT/'outputs/selected.json').read_text())
    runs={r['run']['run_id']:r for r in json.loads((ROOT/'outputs/runs.json').read_text())}
    temp=output(ROOT/'outputs/branch_memory.building.db')
    if temp.exists():raise FileExistsError('Unfinished build exists: '+str(temp))
    c=sqlite3.connect(temp)
    c.execute('PRAGMA foreign_keys=ON');c.execute('PRAGMA page_size=4096')
    c.execute(f'PRAGMA max_page_count={CAP//4096}')
    c.executescript(SCHEMA)
    for rid in sorted({u['run_id'] for u in units}):
        r=runs[rid];run=r['run'];source='nedo_rsi' if r['source_db']=='experience_store/experience.db' else run['source_type']
        wanted={u['raw_context']['root_anchor_id'] for u in units if u['run_id']==rid}
        anchors={k:v for k,v in r['root_anchors'].items() if k in wanted}
        c.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?)',(rid,source,run.get('task_name'),r['source_db'],run.get('source_reference'),dumps(anchors),
            dumps({'original_run':run,'original_branch_point_ids':r['branch_point_ids'],'source_node_count':r['node_count']})))
    statuses=collections.Counter()
    for u in units:
        result=cached(u,runs[u['run_id']],model)
        row={}
        for k,v in u.items():
            if k in ('next_branch_point_ids','incoming_node_ids','incoming_raw','terminal_branch_node_ids','terminal_branches_raw','raw_context'):
                row[k+'_json']=dumps(v)
            elif k=='source_reference':row[k]=dumps(v)
            else:row[k]=v
        row.update(delex_status='completed' if result else 'pending',delexicalized_json=dumps(result['delexicalized']) if result else None,
                   delex_model=model if result else None,delex_prompt_version=PROMPT_VERSION,
                   prompt_sha256=sha((ROOT/'prompts/delexicalize.md').read_text()))
        if result:
            for k in ('input_sha256','codex_version','timestamp','output_sha256','cache_key'):row[k]=result[k]
        c.execute(f'INSERT INTO branch_points ({",".join(row)}) VALUES({",".join("?" for _ in row)})',tuple(row.values()))
        statuses[row['delex_status']]+=1
    c.commit()
    assert not c.execute('PRAGMA foreign_key_check').fetchall()
    assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    c.close()
    if temp.stat().st_size>CAP:raise ValueError('5 GB cap exceeded')
    temp.replace(ROOT/'branch_memory.db')
    report={'rows':len(units),'runs':len({u['run_id'] for u in units}),'bytes':(ROOT/'branch_memory.db').stat().st_size,'statuses':dict(statuses),'cap_bytes':CAP}
    write_json(ROOT/'outputs/build.json',report)
    return report


def verify(full_hash=False):
    c=connect(ROOT/'branch_memory.db')
    assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert not c.execute('PRAGMA foreign_key_check').fetchall()
    assert (ROOT/'branch_memory.db').stat().st_size<=CAP
    units={u['branch_point_id']:u for u in json.loads((ROOT/'outputs/selected.json').read_text())}
    all_units={}
    for line in (ROOT/'outputs/units.jsonl').open():
        u=json.loads(line);all_units[u['branch_point_id']]=u
    for nid,u in all_units.items():
        original={k:v for k,v in u.items() if k!='source_hash'}
        assert sha(dumps(original))==u['source_hash']
        seen=set();current=nid
        while current is not None:
            assert current not in seen,'Compressed cycle'
            seen.add(current);current=all_units[current]['previous_branch_point_id']
        for child in u['next_branch_point_ids']:
            assert all_units[child]['previous_branch_point_id']==nid
            assert all_units[child]['run_id']==u['run_id']
    owners=set()
    for u in units.values():
        for nid in [u['branch_point_id']]+u['incoming_node_ids']+[x for p in u['terminal_branch_node_ids'] for x in p]:
            key=(u['run_id'],nid)
            assert key not in owners,'Duplicate node assignment'
            owners.add(key)
    for row in c.execute('SELECT * FROM branch_points'):
        u=units[row['branch_point_id']]
        assert row['source_hash']==u['source_hash']
        for key in ('raw_context','incoming_raw','terminal_branches_raw','incoming_node_ids','terminal_branch_node_ids','next_branch_point_ids'):
            assert json.loads(row[key+'_json'])==u[key]
        if row['delex_status']=='completed':
            value=json.loads(row['delexicalized_json']);validate(value,u)
            assert sha(dumps(value))==row['output_sha256']
    assert c.execute('SELECT count(*) FROM branch_points').fetchone()[0]==len(units)
    c.close()
    result={'compressed_tree_cycles':0,'duplicate_node_assignments':0,'rows':len(units),**verify_protected(full_hash)}
    write_json(ROOT/'outputs/verification.json',result)
    return result


def coverage(source_db, node_id):
    p=(WORKSPACE/source_db).resolve()
    if p not in {x.resolve() for x in source_paths()}:
        raise ValueError('Use an audited canonical source DB')
    source_db=str(p.relative_to(WORKSPACE))
    c=connect(p);node=c.execute('SELECT run_id FROM experiences WHERE node_id=?',(node_id,)).fetchone();c.close()
    if node is None:raise ValueError('Unknown node ID')
    selected=set(json.loads((ROOT/'outputs/selection.json').read_text())['selected_ids'])
    with gzip.open(ROOT/'outputs/coverage.jsonl.gz','rt') as f:
        for line in f:
            a=json.loads(line)
            if a['source_db']==source_db and a['node_id']==node_id:
                if a['owner'] and a['owner'] not in selected:
                    a['original_assignment']=a['assignment'];a['assignment']='discarded_sampling'
                return a
    return {'node_id':node_id,'run_id':node['run_id'],'assignment':'excluded_linear_run'}


def excel_data():
    c=connect(ROOT/'branch_memory.db')
    columns=['run_id','branch_point_id','previous_branch_point_id','next_branch_point_ids','source_type','branching_factor','depth',
             'incoming_raw','terminal_branches_raw','abstract_state','decision_context','terminal_branches_abstract','search_pattern','delex_status','model',
             'raw_context','source_reference','input_sha256','prompt_sha256']
    rows=[];long_cells=[]
    def safe(v,nid,col):
        if not isinstance(v,str):return v
        # Workbook is a readable view; SQLite remains the full-text, typed source.
        if len(v.encode('utf-16-le'))//2>32000:
            long_cells.append({'branch_point_id':nid,'column':col})
            v=f'[全文: branch_memory.db / branch_points / {nid} / {col}]\n'+v[:1000]
        if v.startswith(('=','+','-','@')):v="'"+v
        return ''.join(x for x in v if ord(x)>=32 or x in '\n\r\t')
    for r in c.execute('SELECT * FROM branch_points ORDER BY source_type,run_id,depth,branch_point_id'):
        d=json.loads(r['delexicalized_json']) if r['delexicalized_json'] else {}
        values=[r['run_id'],r['branch_point_id'],r['previous_branch_point_id'],r['next_branch_point_ids_json'],r['source_type'],r['branching_factor'],r['depth'],
                r['incoming_raw_json'],r['terminal_branches_raw_json'],d.get('abstract_state'),d.get('decision_context'),dumps(d.get('terminal_branches')),
                d.get('search_pattern'),r['delex_status'],r['delex_model'],r['raw_context_json'],r['source_reference'],r['input_sha256'],r['prompt_sha256']]
        rows.append([safe(v,r['branch_point_id'],col) for col,v in zip(columns,values)])
    run_cols=['run_id','source_type','task_name','source_db','source_reference','root_anchors_json']
    run_rows=[[safe(r[k],r['run_id'],k) for k in run_cols] for r in c.execute('SELECT * FROM runs ORDER BY source_type,run_id')]
    stats=[]
    selection=json.loads((ROOT/'outputs/selection.json').read_text())
    for source,d in selection['by_source'].items():
        complete=c.execute("SELECT count(*) FROM branch_points WHERE source_type=? AND delex_status='completed'",(source,)).fetchone()[0]
        stats.append([source,d['total'],d['selected'],d['discarded'],complete])
    c.close()
    result={'Branch_Points':{'columns':columns,'rows':rows},'Runs':{'columns':run_cols,'rows':run_rows},
            'Stats':{'columns':['source_type','total_branch_points','selected','discarded','completed'],'rows':stats}}
    write_json(ROOT/'outputs/excel_data.json',result)
    write_json(ROOT/'outputs/excel_long_cells.json',long_cells)
    return {k:len(v['rows']) for k,v in result.items()}
