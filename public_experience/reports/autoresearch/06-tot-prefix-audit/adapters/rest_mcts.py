import json
import re
from common import Adapter,ROOT,digest,dumps,event,finish,node,relative,number

class RestMCTS(Adapter):
    source='restmcts'
    def discover(self):return sorted((ROOT/'raw/rest_mcts').glob('**/*.json'))
    def parse_record(self,p,index,d):
        dataset=p.parent.name.replace('__','/',1)
        context=d.get('instruction',d.get('prompt',d.get('content')))
        paths=[]
        if 'prompt_answer' in d:
            context,sep,solution=d['prompt_answer'].partition('\nSolution:\n')
            if not sep:context,solution=None,d['prompt_answer']
            paths=[('prefix',solution,d.get('label'),'label')]
        elif 'output' in d:paths=[('policy',d['output'],None,None)]
        elif 'summary' in d:paths=[('policy',d['summary'],None,None)]
        elif 'chosen' in d and 'rejected' in d:
            paths=[('chosen',d['chosen'],None,None),('rejected',d['rejected'],None,None)]
        elif 'response_chosen' in d and 'response_rejected' in d:
            paths=[('chosen',d['response_chosen'],None,None),('rejected',d['response_rejected'],None,None)]
        else:raise ValueError(f'Unknown ReST record fields {list(d)}: {p}')
        for choice,solution,score,score_name in paths:
            if not isinstance(solution,str):raise ValueError('Non-text reasoning')
            rid=f'restmcts::{digest(relative(p))}::{index}-{choice}'
            # Split only explicit source step labels; never infer a tree from equal prefixes.
            starts=[m.start() for m in re.finditer(r'(?m)^\s*(?:Step\s+\d+\s*[:：]|步骤\s*\d+\s*[:：])',solution)]
            cuts=[0]+[x for x in starts[1:] if x>0]+[len(solution)]
            steps=[solution[a:b] for a,b in zip(cuts,cuts[1:]) if a<b] or ['']
            ns=[];es=[];prefix=''
            for si,step in enumerate(steps):
                last=si==len(steps)-1
                n=node(self.source,rid,str(si+1),str(si) if si else None,si,p,index,
                       proposal=step or None,prompt=(context or '')+('\n'+prefix if prefix else ''),result=prefix+step or None,
                       score=score if last else None,score_name=score_name if last else None,
                       metadata={'parent_relation':'linear_sequence','choice':choice,'step_index':si,'step_count':len(steps)})
                payload={'record':d} if last else {'step':step,'source_record':index,'step_index':si}
                ev=event(self.source,rid,n['node_id'],p,index,payload,'reasoning_record' if last else 'reasoning_step',si)
                if n['score'] is not None:
                    m=json.loads(n['metadata_json']);m['score_evidence']={'event_id':ev['event_id'],'path':['record','label']};n['metadata_json']=dumps(m)
                ns.append(n);es.append(ev);prefix+=step
            structure='linearized_mcts' if 'PRM' in dataset or 'ReST-MCTS_Policy' in dataset or re.search(r'ReST-MCTS_[12]nd|ReST-MCTS_1st',dataset) else 'linear_reasoning'
            yield finish(self.source,rid,ns,es,dataset=dataset,structure=structure,reference=f'https://huggingface.co/datasets/{dataset}',metadata={'record_index':index,'choice':choice,'task_name':context,'original_split':p.stem})
    def bundles(self):
        for p in self.discover():
            for i,d in enumerate(self.load_raw(p),1):yield from self.parse_record(p,i,d)
