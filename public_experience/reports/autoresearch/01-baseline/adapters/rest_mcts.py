import json
import re
from common import Adapter,ROOT,digest,dumps,event,finish,node,relative,number

class RestMCTS(Adapter):
    source='restmcts'
    def discover(self):return sorted((ROOT/'raw/rest_mcts').glob('**/*.json'))
    def parse_record(self,p,index,d):
        dataset=p.parent.name.replace('__','/',1)
        context=d.get('instruction',d.get('prompt'))
        paths=[]
        if 'prompt_answer' in d:
            context,sep,solution=d['prompt_answer'].partition('\nSolution:\n')
            if not sep:context,solution=None,d['prompt_answer']
            paths=[('prefix',solution,d.get('label'),'label')]
        elif 'output' in d:paths=[('policy',d['output'],None,None)]
        elif 'chosen' in d and 'rejected' in d:
            paths=[('chosen',d['chosen'],None,None),('rejected',d['rejected'],None,None)]
        else:raise ValueError(f'Unknown ReST record fields {list(d)}: {p}')
        for choice,solution,score,score_name in paths:
            if not isinstance(solution,str):raise ValueError('Non-text reasoning')
            rid=f'restmcts::{digest(relative(p))}::{index}-{choice}'
            n=node(self.source,rid,'1',None,0,p,index,proposal=solution,prompt=context,result=solution,
                   score=score,score_name=score_name,metadata={'parent_relation':'linear_sequence','choice':choice})
            ev=event(self.source,rid,n['node_id'],p,index,d,'reasoning_record')
            if n['score'] is not None:
                m=json.loads(n['metadata_json']);m['score_evidence']={'event_id':ev['event_id'],'path':['label']};n['metadata_json']=dumps(m)
            structure='linearized_mcts' if 'PRM' in dataset or 'ReST-MCTS_Policy' in dataset or re.search(r'ReST-MCTS_[12]nd|ReST-MCTS_1st',dataset) else 'linear_reasoning'
            yield finish(self.source,rid,[n],[ev],dataset=dataset,structure=structure,reference=f'https://huggingface.co/datasets/{dataset}',metadata={'record_index':index,'choice':choice,'task_name':context,'original_split':p.stem})
    def bundles(self):
        for p in self.discover():
            for i,d in enumerate(self.load_raw(p),1):yield from self.parse_record(p,i,d)
