import collections
import json
from common import Adapter,ROOT,dumps,digest,event,finish,node,relative,text

class OpenEvolve(Adapter):
    source='openevolve'
    def discover(self):return sorted((ROOT/'raw/openevolve').glob('**/programs/*.json'))
    def bundles(self):
        records=collections.defaultdict(list)
        for p in self.discover():
            d=json.loads(p.read_text());records[d['id']].append((p,d))
        def root(k):
            seen=set()
            while records[k][0][1].get('parent_id'):
                if k in seen:raise ValueError('cycle in programs')
                seen.add(k);k=records[k][0][1]['parent_id']
                if k not in records:raise ValueError('missing parent program')
            return k
        groups=collections.defaultdict(list)
        for k in records:groups[root(k)].append(k)
        for root_id,ids in sorted(groups.items()):
            rid=f'openevolve::MyOpenEvolve::{root_id}'
            ns=[];es=[]
            for i,k in enumerate(sorted(ids,key=lambda k:(records[k][0][1].get('generation',0),records[k][0][1].get('iteration_found',0),k))):
                versions=records[k]
                # Prefer prompt-rich snapshot; all snapshots remain in raw_events.
                p,d=max(versions,key=lambda v:len(dumps(v[1].get('prompts',{}))))
                pid=d.get('parent_id');parent=records[pid][0][1] if pid else None
                metrics=d.get('metrics',{})
                prompts=d.get('prompts')
                responses=[r for v in (prompts or {}).values() for r in v.get('responses',[]) if r]
                n=node(self.source,rid,k,pid,i,p,1,proposal=d.get('metadata',{}).get('changes_description') or text(responses),
                       prompt={'parent_code':parent['code'],'parent_metrics':parent.get('metrics')} if parent else None,
                       result=d.get('code'),score=metrics.get('combined_score'),score_name='combined_score' if 'combined_score' in metrics else None,
                       direction='maximize' if 'combined_score' in metrics else 'unknown',metrics=metrics,
                       metadata={'parent_relation':'explicit','generation':d.get('generation'),'iteration':d.get('iteration_found'),
                                 'snapshot_count':len(versions),'prompts_reference':{'file':relative(p),'pointer':'/prompts'} if prompts else None,
                                 'timestamp':d.get('timestamp')},status='recorded')
                ns.append(n)
                for j,(vp,vd) in enumerate(versions):
                    ev=event(self.source,rid,n['node_id'],vp,1,vd,'program_checkpoint',j);es.append(ev)
                    if vp==p and n['score'] is not None:
                        m=json.loads(n['metadata_json']);m['score_evidence']={'event_id':ev['event_id'],'path':['metrics','combined_score']};n['metadata_json']=dumps(m)
            yield finish(self.source,rid,ns,es,dataset='ypwang61/MyOpenEvolve/circle_packing',structure='tree',reference='https://github.com/ypwang61/MyOpenEvolve')
