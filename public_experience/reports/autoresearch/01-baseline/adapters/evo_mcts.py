import collections
import json
import re
from common import Adapter,ROOT,dumps,event,finish,node,relative,number

EVAL=re.compile(r'Eval_times: (\d+), Iteration (\d+), Depth (\d+), Operator ([^,]+), response_id [^:]+: Objective value: (\S+)')
ACTION=re.compile(r'Action: (\w+), Father Obj: ([^,]+), Now Obj: ([^,]+), Depth: (\d+)')

class EvoMCTS(Adapter):
    source='evomcts'
    def discover(self):return sorted((ROOT/'raw/evo_mcts').glob('**/merged_log.log'))
    def bundles(self):
        for p in self.discover():
            rid=f'evomcts::{p.parent.name}';records={};events=[];nodes=[]
            for ln,line in enumerate(p.read_text().splitlines(),1):
                m=EVAL.search(line)
                if not m:continue
                idx,it,dep,op,obj=m.groups()
                records[idx]={'eval_times':int(idx),'iteration':int(it),'depth':int(dep),'operator':op,'objective':obj,'line':line,'source_line':ln}
            paper={}
            if p.parent.name=='run1_april_2025':
                pp=ROOT/'raw/evo_mcts/repository/results/paper_data/mcts_tree_nodes_pt5_algorithm.jsonl'
                for ln,line in enumerate(pp.read_text().splitlines(),1):
                    d=json.loads(line);paper[str(d['eval_times'])]=(pp,ln,d)
            for i,(idx,d) in enumerate(sorted(records.items(),key=lambda kv:int(kv[0]))):
                n=node(self.source,rid,idx,None,i,p,d['source_line'],proposal=d['operator'],
                       result={'objective':d['objective']},score=d['objective'],score_name='objective',direction='minimize',
                       metadata={'parent_relation':'unknown','original_depth':d['depth'],'iteration':d['iteration']},status='evaluated')
                ev=event(self.source,rid,n['node_id'],p,d['source_line'],d,'evaluation',i);events.append(ev)
                md=json.loads(n['metadata_json'])
                if n['score'] is not None:md['score_evidence']={'event_id':ev['event_id'],'path':['objective']}
                if idx in paper:
                    pp,ln,payload=paper[idx]
                    n['result_summary']=payload['code'];n['proposal']=dumps({k:payload.get(k) for k in ('operator','thinking','reflection','algorithm')})
                    n['metrics_json']=dumps({'objective':d['objective'],'fitness':payload['fitness']})
                    md['paper_record']={'file':relative(pp),'line':ln}
                    events.append(event(self.source,rid,n['node_id'],pp,ln,payload,'paper_node',len(records)+ln))
                n['metadata_json']=dumps(md);nodes.append(n)
            yield finish(self.source,rid,nodes,events,dataset='iphysresearch/evo-mcts',structure='partial_tree',reference=relative(p),metadata={'incomplete':True,'paper_parent_ids_absent':True})
