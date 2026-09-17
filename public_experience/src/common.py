"""Shared three-table records. No network, inference or reward generation."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str, allow_nan=False)

def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:20]

def text(value):
    if value is None or value == '' or value == {} or value == []:
        return None
    return value if isinstance(value, str) else dumps(value)

def number(value):
    if value is None or isinstance(value, bool): return None
    try:
        n=float(value)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError):return None

def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))

def json_records(path):
    """JSON arrays, JSONL (including mislabeled .json), and columnar DPO JSON."""
    import ijson
    with Path(path).open('rb') as f:
        first=f.read(512).lstrip()[:1]
        f.seek(0)
        if first==b'[':
            yield from ijson.items(f,'item',use_float=True)
        else:
            # jsonl cannot be distinguished by extension in these releases.
            try:
                first_line=f.readline()
                initial=json.loads(first_line)
                if any(f.read(1)):
                    f.seek(0)
                    for line in f:
                        if line.strip():yield json.loads(line)
                else:
                    yield from columnar(initial)
            except (ValueError,TypeError):
                f.seek(0)
                yield from columnar(json.load(f))

def columnar(value):
    if isinstance(value,list):yield from value
    elif isinstance(value,dict) and isinstance(value.get('prompt'),list):
        lengths={len(v) for v in value.values() if isinstance(v,list)}
        if len(lengths)!=1:raise ValueError('Mismatched DPO arrays')
        for i in range(len(value['prompt'])):yield {k:v[i] for k,v in value.items()}
    else:yield value

def event(source,run_id,node_id,path,record,payload,event_type='source_record',sequence=0,locator=None):
    source_file=relative(path)
    return {'event_id':f'{source}::event::{digest((run_id,node_id,source_file,record,sequence,event_type))}',
            'run_id':run_id,'node_id':node_id,'sequence_no':sequence,'event_type':event_type,
            'source_file':source_file,'source_line':record,'binding_method':'source_record',
            'binding_confidence':1.0,'raw_json':dumps(payload)}

def node(source,run_id,original,parent,sequence,path,record,*,proposal=None,prompt=None,result=None,
         score=None,score_name=None,direction='unknown',metrics=None,metadata=None,node_type='search_node',status=None):
    meta={'original_node_id':original, 'original_parent_id':parent,
          'source_record':record,'source_dataset':source}
    meta.update(metadata or {})
    return {'run_id':run_id,'node_id':f'{run_id}::{original}',
            'parent_id':f'{run_id}::{parent}' if parent is not None else None,
            'sequence_index':sequence,'depth':0,'display_path':'', 'node_type':node_type,
            'proposal':text(proposal),'prompt':text(prompt),'result_summary':text(result),
            'status':status,'score':number(score),'score_name':score_name,
            'score_direction':direction,'metrics_json':dumps(metrics or {}),
            'source_type':source,'source_reference':f'{relative(path)}#record={record}',
            'source_files_json':dumps([relative(path)]),'metadata_json':dumps(meta)}

def finish(source,run_id,nodes,events,*,dataset,structure,reference,metadata=None):
    """Validate all edges and derive paths only from known parents; preserve forests."""
    by_id={n['node_id']:n for n in nodes}
    if len(by_id)!=len(nodes):raise ValueError('duplicate node IDs')
    paths={}
    def visit(n):
        chain=[];seen=set();cur=n
        while cur['node_id'] not in paths:
            nid=cur['node_id']
            if nid in seen:raise ValueError('cycle')
            seen.add(nid);chain.append(cur)
            par=cur['parent_id']
            if par is None:break
            if par not in by_id:raise ValueError(f'invalid parent {par}')
            cur=by_id[par]
        prefix=paths.get(cur['node_id'],[])
        for x in reversed(chain):
            prefix=prefix+[x['node_id']];paths[x['node_id']]=prefix
        return paths[n['node_id']]
    grouped={}
    for e in events:
        if e['node_id'] is not None:grouped.setdefault(e['node_id'],[]).append(e)
    roots=[]
    for n in nodes:
        p=visit(n);n['depth']=len(p)-1;n['display_path']='/'.join(p)
        if n['parent_id'] is None:roots.append(n['node_id'])
        ev=grouped.get(n['node_id'],[]);n['event_count']=len(ev)
        n['first_event_sequence']=min((e['sequence_no'] for e in ev),default=None)
        n['last_event_sequence']=max((e['sequence_no'] for e in ev),default=None)
        m=json.loads(n['metadata_json']);m.update(source_dataset=dataset,source_structure=structure)
        m.setdefault('parent_relation','unknown')
        m.setdefault('state',{'column':'prompt'})
        m.setdefault('action',{'column':'proposal'})
        m.setdefault('next_state',{'column':'result_summary'})
        m.setdefault('original_payload',{'file':json.loads(n['source_files_json'])[0],'record':m['source_record']})
        n['metadata_json']=dumps(m)
    meta={'source_dataset':dataset,'source_structure':structure,'root_node_ids':roots}
    meta.update(metadata or {})
    run={'run_id':run_id,'project_name':dataset,'task_name':meta.get('task_name',dataset),
         'root_node_id':roots[0] if len(roots)==1 else None,'source_type':source,
         'source_reference':reference,'replay_eligible':int(bool(nodes) and len(roots)==1 and not meta.get('incomplete')),
         'status':'recorded','metadata_json':dumps(meta)}
    return {'run':run,'nodes':nodes,'events':events}

class Adapter:
    source=''
    def discover(self):return []
    def load_raw(self,path):return json_records(path)
    def bundles(self):raise NotImplementedError
    def iter_runs(self):
        for b in self.bundles():yield b['run']
    def iter_experiences(self):
        for b in self.bundles():yield from b['nodes']
    def iter_raw_events(self):
        for b in self.bundles():yield from b['events']
