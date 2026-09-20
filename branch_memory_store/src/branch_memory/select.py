"""Deterministic stratified branch sampling, with a recorded finite inference budget."""
import collections
import json
from .db import ROOT, dumps, sha, write_json


def select(limit=100, seed=17, max_chars=100000):
    units=[json.loads(line) for line in (ROOT/'outputs/units.jsonl').open()]
    runs={r['run']['run_id']:r['run'] for r in json.loads((ROOT/'outputs/runs.json').read_text())}
    selected=[];excluded={}; groups=collections.defaultdict(list)
    for u in units:
        nid=u['branch_point_id'];n=u['raw_context']['branch_point'];e=n['evidence']
        if len(dumps(u))>max_chars:
            excluded[nid]='input_size_limit';continue
        if u['source_type']=='nedo_rsi':selected.append(u);continue
        stratum=(u['source_type'],sha(str(runs[u['run_id']].get('task_name'))),
                 min(u['depth']//3,3),min(u['branching_factor']//3,3),n['score'] is not None,
                 n.get('status') or 'unknown',e.get('parent_relation',e.get('parent_relation_type','unknown')))
        groups[stratum].append(u)
    if limit<len(selected):raise ValueError('limit must retain all eligible NEDO branch points')
    for items in groups.values():
        items.sort(key=lambda u:sha(f'{seed}:{u["source_hash"]}'))
    # Source-balanced rounds avoid a source with many task strata consuming all slots.
    by_source=collections.defaultdict(list)
    for key in sorted(groups,key=lambda k:sha(f'{seed}:{dumps(k)}')):
        by_source[key[0]].append(groups[key])
    sources=sorted(by_source, key=lambda s:({'openevolve':0,'evomcts':1}.get(s,2),s))
    while len(selected)<limit:
        progressed=False
        for s in sources:
            strata=by_source[s]
            while strata and not strata[0]:strata.pop(0)
            if strata:
                items=strata.pop(0);selected.append(items.pop(0));strata.append(items);progressed=True
                if len(selected)>=limit:break
        if not progressed:break
    ids={u['branch_point_id'] for u in selected}
    for u in units:
        if u['branch_point_id'] not in ids:excluded.setdefault(u['branch_point_id'],'stratified_sampling')
    counts={}
    for source in sorted({u['source_type'] for u in units}):
        rows=[u for u in units if u['source_type']==source]
        counts[source]={'total':len(rows),'selected':sum(u['branch_point_id'] in ids for u in rows),
                        'discarded':sum(u['branch_point_id'] not in ids for u in rows),
                        'reasons':dict(collections.Counter(excluded[u['branch_point_id']] for u in rows if u['branch_point_id'] not in ids))}
    write_json(ROOT/'outputs/selected.json',selected)
    report={'seed':seed,'limit':limit,'max_input_chars':max_chars,'selected':len(selected),'total':len(units),'by_source':counts,
            'selected_ids':[u['branch_point_id'] for u in selected],'discarded':excluded,
            'reason':'Finite semantic pilot/dataset budget; no raw padding. Preserve all eligible local, then source-balanced hash-sorted task/depth/factor/score/status/parent-quality strata.',
            'next_links':'Original branch IDs retained even if unselected; membership must be checked before joining.'}
    write_json(ROOT/'outputs/selection.json',report)
    return report
