import json
from common import Adapter,ROOT,digest,dumps,event,finish,node,relative

class SWEAgent(Adapter):
    source='sweagent'
    def discover(self):return sorted((ROOT/'raw/swe_agent/repository').glob('**/*.traj'))
    def bundles(self):
        for p in self.discover():
            d=json.loads(p.read_text());steps=d.get('trajectory',[])
            if not steps:continue
            rid=f'sweagent::demonstration::{digest(relative(p))}';ns=[];es=[]
            for i,s in enumerate(steps):
                n=node(self.source,rid,str(i),str(i-1) if i else None,i,p,1,proposal=s.get('action'),
                       prompt=steps[i-1].get('observation') if i else None,result=s.get('observation'),
                       metadata={'parent_relation':'linear_sequence','thought':s.get('thought'),'trajectory_index':i},node_type='agent_action')
                ns.append(n);es.append(event(self.source,rid,n['node_id'],p,1,s,'thought_action_observation',i))
            yield finish(self.source,rid,ns,es,dataset='SWE-agent/demonstrations',structure='linear',reference=relative(p),metadata={'info':d.get('info'),'demonstration':True})
