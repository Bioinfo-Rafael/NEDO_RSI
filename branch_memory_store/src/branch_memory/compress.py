"""Lossless assignment of observed nodes to decision-point units; no invented edges."""
import collections
import gzip
import json
from .db import ROOT, WORKSPACE, connect, dumps, sha, write_json, output

FIELDS = ('node_id','parent_id','sequence_index','depth','node_type','prompt','proposal','result_summary',
          'score','score_name','score_direction','status','source_reference','source_type')


def payload(n):
    result = {k:n.get(k) for k in FIELDS}
    result['metrics'] = json.loads(n.get('metrics_json') or '{}')
    m = json.loads(n.get('metadata_json') or '{}')
    result['evidence'] = {k:v for k,v in m.items() if k in (
        'parent_relation','parent_relation_type','parent_relation_confidence','parent_evidence','score_evidence',
        'selected','selected_sequence_parent','original_node_id','original_parent_id','missing_structural_root',
        'implicit_root','original_depth','task_results','score_direction_method','score_direction_evidence')}
    return result


def compress(run, records, source_db):
    nodes = {n['node_id']:n for n in records}
    if len(nodes)!=len(records):
        raise ValueError('Duplicate node IDs')
    children = collections.defaultdict(list)
    roots=[]
    for nid,n in nodes.items():
        if n['run_id']!=run['run_id']:
            raise ValueError('Cross-run node')
        parent=n['parent_id']
        if parent in nodes:
            children[parent].append(nid)
        else:
            roots.append(nid)
    for kids in children.values():
        kids.sort(key=lambda nid:(nodes[nid]['sequence_index'],nid))
    depth={}; stack=[(r,0) for r in roots]
    while stack:
        nid,d=stack.pop()
        if nid in depth:
            raise ValueError('Cycle or duplicate traversal')
        depth[nid]=d
        stack.extend((k,d+1) for k in children[nid])
    if len(depth)!=len(nodes):
        raise ValueError('Cycle in source graph')
    points={nid for nid in nodes if len(children[nid])>=2}
    if not points:
        return [],{},[{ 'node_id':nid,'assignment':'excluded_linear_run','owner':None} for nid in nodes]
    assignment={}; units=[]; anchors={}
    def own(nid,kind,owner):
        if nid in assignment:
            raise ValueError('Node assigned twice: '+nid)
        assignment[nid]={'node_id':nid,'assignment':kind,'owner':owner}
    for nid in sorted(points):
        own(nid,'branching_point',nid)
    for nid in sorted(points):
        current=nodes[nid]['parent_id']; incoming=[]; anchor=None
        while current in nodes and current not in points:
            if nodes[current]['parent_id'] not in nodes:
                anchor=current
                anchors[current]=payload(nodes[current])
                break
            incoming.append(current);current=nodes[current]['parent_id']
        previous=current if current in points else None
        incoming.reverse()
        for x in incoming:own(x,'incoming_segment',nid)
        terminals=[]; next_points=[]
        for child in children[nid]:
            path=[];current=child
            while current not in points:
                path.append(current)
                if not children[current]:break
                current=children[current][0]
            if current in points:
                next_points.append(current)
            else:
                terminals.append(path)
                for x in path:own(x,'terminal_segment',nid)
        record={'run_id':run['run_id'],'branch_point_id':nid,'original_node_id':nid,
                'previous_branch_point_id':previous,'next_branch_point_ids':next_points,
                'incoming_node_ids':incoming,'incoming_raw':[payload(nodes[x]) for x in incoming],
                'terminal_branch_node_ids':terminals,'terminal_branches_raw':[[payload(nodes[x]) for x in p] for p in terminals],
                'branching_factor':len(children[nid]),'depth':depth[nid],
                'source_type': 'nedo_rsi' if source_db=='experience_store/experience.db' else nodes[nid]['source_type'],
                'source_reference':{'source_db':source_db,'run_id':run['run_id'],'node_id':nid,'original_reference':nodes[nid].get('source_reference')},
                'raw_context':{'branch_point':payload(nodes[nid]),'root_anchor_id':anchor,
                               'run_structure':json.loads(run.get('metadata_json') or '{}').get('source_structure','local'),
                               'incomplete':bool(json.loads(run.get('metadata_json') or '{}').get('incomplete'))}}
        record['source_hash']=sha(dumps(record))
        units.append(record)
    for nid in nodes:
        if nid not in assignment:
            # Anchors are retained once in runs, outside the six requested memory assignment categories.
            assignment[nid]={'node_id':nid,'assignment':'unresolved','owner':None,
                             'reason':'root_anchor_in_runs' if nid in anchors else 'component_without_branch_or_missing_parent'}
    return units,anchors,list(assignment.values())


def extract():
    # Source audits suffice; hashing duplicate aggregate DBs can finish independently.
    audit=[json.loads(p.read_text()) for p in sorted((ROOT/'outputs/audit').glob('*.json'))]
    from .db import source_paths
    if not {str(p.relative_to(WORKSPACE)) for p in source_paths()}.issubset({r['path'] for r in audit}):
        raise ValueError('Complete every canonical source audit before extraction')
    target=output(ROOT/'outputs/units.jsonl')
    if target.exists():
        raise FileExistsError('Extraction already exists; reuse it or choose a new workspace')
    run_records=[]; count=0; coverage_counts=collections.Counter()
    with target.open('w') as out, gzip.open(output(ROOT/'outputs/coverage.jsonl.gz'),'wt') as coverage:
        for report in audit:
            if 'branch_points' not in report:continue
            c=connect(WORKSPACE/report['path'])
            for rid in sorted({p['run_id'] for p in report['branch_points']}):
                run=dict(c.execute('SELECT * FROM runs WHERE run_id=?',(rid,)).fetchone())
                records=[dict(r) for r in c.execute('SELECT * FROM experiences WHERE run_id=? ORDER BY sequence_index,node_id',(rid,))]
                units,anchors,assignments=compress(run,records,report['path'])
                run_records.append({'run':run,'source_db':report['path'],'root_anchors':anchors,'branch_point_ids':[u['branch_point_id'] for u in units],
                                    'node_count':len(records)})
                for u in units:out.write(dumps(u)+'\n');count+=1
                for a in assignments:
                    coverage.write(dumps({'run_id':rid,'source_db':report['path'],**a})+'\n')
                    coverage_counts[a['assignment']]+=1
            c.close()
    write_json(ROOT/'outputs/runs.json',run_records)
    write_json(ROOT/'outputs/extraction.json',{'branch_points':count,'runs_with_branch':len(run_records),'coverage':dict(coverage_counts),
        'linear_only_rule':'Any run absent from audit.branch_points has every node classified excluded_linear_run. Resolve via coverage CLI.'})
    return {'branch_points':count,'runs':len(run_records)}
