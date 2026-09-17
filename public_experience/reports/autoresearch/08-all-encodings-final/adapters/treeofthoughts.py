"""Official ToT beam candidates and DFS action-stack logs; no inferred timing edges."""
import collections
import json
from common import Adapter,ROOT,digest,dumps,event,finish,node,relative

class TreeOfThoughts(Adapter):
    source='treeofthoughts'
    def discover(self):
        r=ROOT/'raw/treeofthoughts/logs'
        return sorted(list(r.glob('**/*greedy*.json'))+list(r.glob('crosswords/infoss_dfs*.json')))
    def bundles(self):
        for p in self.discover():
            for index,d in enumerate(json.loads(p.read_text()),1):
                if isinstance(d,list):yield self.dfs(p,index,d)
                else:yield self.beam(p,index,d)
    def beam(self,p,index,d):
        rid=f'treeofthoughts::{digest(relative(p))}::{d.get("idx",index)}';ns=[];es=[];previous=[]
        for si,step in enumerate(d['steps']):
            if len(step['values'])!=len(step['new_ys']):raise ValueError('ToT score alignment')
            current=[]
            for j,output in enumerate(step['new_ys']):
                possible=[(oid,old) for oid,old in previous if old in step['ys'] and output.startswith(old)]
                pid,prefix=possible[0] if len(possible)==1 else (None,'')
                relation='reconstructed_from_official_code' if len(possible)==1 or si==0 else 'unknown'
                oid=f'{si}-{j}'
                n=node(self.source,rid,oid,pid,len(ns),p,index,prompt={'question':step['x'],'prefix':prefix if relation!='unknown' else None},
                       proposal=output[len(prefix):] if relation!='unknown' else output,result=output,score=step['values'][j],
                       score_name='value',direction='maximize',metrics={'value':step['values'][j]},
                       metadata={'parent_relation':relation,'step':si,'candidate_index':j,'selected':output in step['select_new_ys'],'original_depth':si+1,'implicit_root':si==0})
                ev=event(self.source,rid,n['node_id'],p,index,{'step':si,'x':step['x'],'candidate':output,'value':step['values'][j],'selected_prefixes':step['ys']},'beam_candidate',len(es))
                m=json.loads(n['metadata_json']);m['score_evidence']={'event_id':ev['event_id'],'path':['value']};n['metadata_json']=dumps(m)
                ns.append(n);es.append(ev);current.append((oid,output))
            previous=current
        return finish(self.source,rid,ns,es,dataset='princeton-nlp/tree-of-thought-llm/'+p.parent.name,
                      structure='beam_search_forest',reference=relative(p),metadata={'task_index':d.get('idx'),'incomplete':True,'missing_prompt_root':True})
    def dfs(self,p,index,records):
        rid=f'treeofthoughts::{digest(relative(p))}::{index}';ns=[];es=[]
        paths=collections.defaultdict(list)
        for i,d in enumerate(records):paths[tuple(d['actions'])].append(i)
        for i,d in enumerate(records):
            parent=paths.get(tuple(d['actions'][:-1]),[])
            pid=str(parent[0]) if len(parent)==1 else None
            n=node(self.source,rid,str(i),pid,i,p,index,prompt={'action_stack':d['actions'][:-1]},proposal=d['actions'][-1],
                   result=d['info'],score=d['info'].get('r_word'),score_name='r_word',direction='maximize',metrics=d['info'],
                   metadata={'parent_relation':'reconstructed_from_official_code' if pid is not None or len(d['actions'])==1 else 'unknown',
                             'actions':d['actions'],'total_step':d['total_step'],'original_depth':d['env_step'],'original_node_id':d['total_step']})
            ev=event(self.source,rid,n['node_id'],p,index,d,'dfs_expansion',i);es.append(ev)
            m=json.loads(n['metadata_json']);m['score_evidence']={'event_id':ev['event_id'],'path':['info','r_word']};n['metadata_json']=dumps(m);ns.append(n)
        return finish(self.source,rid,ns,es,dataset='princeton-nlp/tree-of-thought-llm/crosswords',structure='dfs_search_forest',
                      reference=relative(p),metadata={'incomplete':True,'missing_prompt_root':True,'task_array_index':index-1})
