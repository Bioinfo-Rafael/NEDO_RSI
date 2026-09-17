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
            lines=p.read_text().splitlines()
            for ln,line in enumerate(lines,1):
                m=EVAL.search(line)
                if not m:continue
                idx,it,dep,op,obj=m.groups()
                records[idx]={'eval_times':int(idx),'iteration':int(it),'depth':int(dep),'operator':op,'objective':obj,'line':line,'source_line':ln}
            parents={};parent_evidence={}
            # Official evolution_interface rounds stored objective to 5 decimals.
            # An Action line is logged only for an accepted child. Require unique
            # child and father matching score, depth and the earlier source lines.
            for ln,line in enumerate(lines,1):
                m=ACTION.search(line)
                if not m:continue
                op,father,child,dep=m.groups();dep=int(dep)
                cs=[k for k,v in records.items() if v['source_line']<ln and v['depth']==dep and v['operator']==op and number(v['objective']) is not None and round(float(v['objective']),5)==float(child)]
                if len(cs)!=1:continue
                ck=cs[0];cv=records[ck]
                ps=[k for k,v in records.items() if v['source_line']<cv['source_line'] and v['depth']==dep-1 and number(v['objective']) is not None and round(float(v['objective']),5)==float(father)]
                if len(ps)==1:
                    parents[ck]=ps[0];parent_evidence[ck]={'line':ln,'text':line,'father_objective':father,'child_objective':child,'round_digits':5}
            paper={}
            breakthroughs={}
            if p.parent.name=='run1_april_2025':
                pp=ROOT/'raw/evo_mcts/repository/results/paper_data/mcts_tree_nodes_pt5_algorithm.jsonl'
                for ln,line in enumerate(pp.read_text().splitlines(),1):
                    d=json.loads(line);paper[str(d['eval_times'])]=(pp,ln,d)
                for bp in sorted(p.parent.glob('breakthrough_nodes/*.json')):
                    payload=json.loads(bp.read_text());breakthroughs[str(payload['eval_times'])]=(bp,payload)
            for i,(idx,d) in enumerate(sorted(records.items(),key=lambda kv:int(kv[0]))):
                parent=parents.get(idx)
                n=node(self.source,rid,idx,parent,i,p,d['source_line'],proposal=d['operator'],
                       result={'objective':d['objective']},score=d['objective'],score_name='objective',direction='minimize',
                       metadata={'parent_relation':'reconstructed_from_official_code' if parent is not None or d['depth']==1 else 'unknown',
                                 'parent_evidence':parent_evidence.get(idx),'original_depth':d['depth'],'iteration':d['iteration'],
                                 'missing_structural_root':d['depth']==1},status='evaluated' if number(d['objective']) is not None else 'evaluation_failed')
                ev=event(self.source,rid,n['node_id'],p,d['source_line'],d,'evaluation',i);events.append(ev)
                md=json.loads(n['metadata_json'])
                if n['score'] is not None:md['score_evidence']={'event_id':ev['event_id'],'path':['objective']}
                if idx in paper:
                    pp,ln,payload=paper[idx]
                    n['result_summary']=payload['code'];n['proposal']=dumps({k:payload.get(k) for k in ('operator','thinking','reflection','algorithm')})
                    n['metrics_json']=dumps({'objective':d['objective'],'fitness':payload['fitness']})
                    md['paper_record']={'file':relative(pp),'line':ln}
                    events.append(event(self.source,rid,n['node_id'],pp,ln,payload,'paper_node',len(records)+ln))
                if idx in breakthroughs:
                    bp,payload=breakthroughs[idx]
                    n['prompt']=dumps({k:v for k,v in payload.items() if k in ('system_content','user_content','user_content_reflection','system_content_reflection')})
                    n['result_summary']=payload.get('code') or n['result_summary']
                    events.append(event(self.source,rid,n['node_id'],bp,1,payload,'breakthrough_record',len(records)+100+i))
                if idx in parent_evidence:
                    evd=parent_evidence[idx]
                    events.append(event(self.source,rid,n['node_id'],p,evd['line'],evd,'accepted_expansion',evd['line']))
                n['metadata_json']=dumps(md);nodes.append(n)
            yield finish(self.source,rid,nodes,events,dataset='iphysresearch/evo-mcts',structure='partial_tree',reference=relative(p),metadata={'incomplete':True,'paper_parent_ids_absent':True})
