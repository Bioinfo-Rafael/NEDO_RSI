import json
from common import Adapter,ROOT,digest,dumps,event,finish,node,relative

class SWEAgent(Adapter):
    source='sweagent'
    def discover(self):return sorted((ROOT/'raw/swe_agent/repository').glob('**/*.traj'))
    def bundles(self):
        yield from self.demonstrations()
        yield from self.smith()
    def demonstrations(self):
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
    def smith(self):
        import pyarrow.parquet as pq
        files=sorted((ROOT/'raw/swe_agent').glob('**/*.parquet'),key=lambda p:(['tool','xml','ticks','train'].index(p.name.split('-')[0]),p.name))
        seen=set()
        for p in files:
            row_no=0
            for batch in pq.ParquetFile(p).iter_batches(batch_size=32):
                for d in batch.to_pylist():
                    row_no+=1
                    if d['traj_id'] in seen:continue
                    seen.add(d['traj_id'])
                    yield self.parse_messages(p,row_no,d)
    def parse_messages(self,p,row_no,d):
        messages=json.loads(d['messages']);rid=f'sweagent::SWE-smith::{d["traj_id"]}';ns=[];es=[]
        indices=[i for i,m in enumerate(messages) if m['role']=='assistant']
        for si,i in enumerate(indices):
            message=messages[i];nxt=indices[si+1] if si+1<len(indices) else len(messages)
            observations=messages[i+1:nxt]
            # One assistant message is one action turn, with all resulting tool observations.
            action=message.get('tool_calls') or message.get('content')
            state=messages[:i] if si==0 else messages[indices[si-1]+1:i]
            n=node(self.source,rid,str(si),str(si-1) if si else None,si,p,row_no,
                   proposal=action,prompt=state,result=observations,
                   metrics={'resolved':d['resolved']} if si+1==len(indices) else {},
                   metadata={'parent_relation':'linear_sequence','message_index':i,'instance_id':d['instance_id'],
                             'thought':message.get('content'),'trajectory_prefix_reference':{'file':relative(p),'row':row_no,'messages_before':i},
                             'observation':{'messages_start':i+1,'messages_end':nxt},'original_node_id':i},node_type='agent_action',
                   status=('resolved' if d['resolved'] else 'unresolved') if si+1==len(indices) else None)
            ns.append(n)
            payload={'assistant':message,'observations':observations}
            if si==0:payload['context']=messages[:i]
            if si+1==len(indices):payload['outcome']={k:v for k,v in d.items() if k!='messages'}
            es.append(event(self.source,rid,n['node_id'],p,row_no,payload,'assistant_action_observation',si))
        return finish(self.source,rid,ns,es,dataset='SWE-bench/SWE-smith-trajectories',structure='linear',
                      reference=f'https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories',
                      metadata={'traj_id':d['traj_id'],'instance_id':d['instance_id'],'resolved':d['resolved'],'model':d['model'],
                                'canonical_encoding':p.name.split('-')[0],'encoding_preference':['tool','xml','ticks','train']})
