import json
import re
from bs4 import BeautifulSoup
from common import Adapter,ROOT,digest,dumps,event,finish,node,relative

def pages(path):
    # RenderHelper emits malformed nested divs; split at its exact page marker.
    content=path.read_text(errors='replace')
    chunks=content.split('<h2>New Page</h2>')
    task=BeautifulSoup(chunks[0],'html.parser').get_text()
    output=[]
    for chunk in chunks[1:]:
        chunk=re.sub(r"<img\b[^>]*>",'',chunk)
        soup=BeautifulSoup(chunk,'html.parser')
        ob=soup.select_one('.state_obv pre');act=soup.select_one('.predict_action')
        extra=[d.get_text() for d in soup.select('.additional_text')]
        output.append({'observation':ob.get_text() if ob else None,'action':act.get_text() if act else None,'additional_text':extra})
    return task,output

class SearchAgents(Adapter):
    source='searchagents'
    def discover(self):return sorted((ROOT/'raw/search_agents').glob('**/render_*.html'))
    def bundles(self):
        for p in self.discover():
            task,pgs=pages(p);rid=f'searchagents::{p.parent.name}::{p.stem}';ns=[];es=[]
            if not pgs:continue
            initial=node(self.source,rid,'initial_observation',None,0,p,1,prompt=task,result=pgs[0]['observation'],
                         metadata={'parent_relation':'linear_sequence','html_page_index':0},node_type='observation')
            ns.append(initial);es.append(event(self.source,rid,initial['node_id'],p,1,{'observation':pgs[0]['observation'],'task':task},'initial_observation',0))
            previous='initial_observation';batch=-1;last_signature=None;batch_nodes=[];base_parent=previous
            for i,page in enumerate(pgs):
                candidates=[];selected=[]
                for extra in page['additional_text']:
                    m=re.match(r'^#\d+: a_idx=(\d+),curr_a_idx=(-?\d+),depth=(\d+): (.*) \(score: ([\d.eE+-]+), time: ([^)]+)\)$',extra,re.S)
                    if m:
                        a,c,d,pred,score,elapsed=m.groups();candidates.append({'a_idx':int(a),'curr_a_idx':int(c),'depth':int(d),'prediction':pred,'score':score,'elapsed':elapsed,'text':extra})
                    m=re.match(r'^#\d+: Selected action (\d+): (.*)$',extra,re.S)
                    if m:selected.append((int(m[1]),m[2]))
                signature=digest(dumps(candidates))
                if signature!=last_signature or not selected or selected[-1][0]==0:
                    batch+=1;last_signature=signature;batch_nodes=[];base_parent=previous
                    for j,c in enumerate(candidates):
                        oid=f'b{batch}c{j}';parent=None
                        if c['depth']==0:parent=base_parent
                        else:
                            possible=[old for old in batch_nodes if old[0]['a_idx']==c['a_idx'] and old[0]['depth']==c['depth']-1]
                            if len(possible)==1:parent=possible[0][1]['node_id'].removeprefix(rid+'::')
                        n=node(self.source,rid,oid,parent,len(ns),p,i+1,proposal=c['prediction'],
                               prompt=page['observation'] if c['depth']==0 else None,score=c['score'],score_name='value',direction='maximize',
                               metadata={'parent_relation':'reconstructed_from_official_code' if parent is not None else 'unknown',
                                         'a_idx':c['a_idx'],'curr_a_idx':c['curr_a_idx'],'search_depth':c['depth'],'batch':batch,'html_page_index':i},node_type='search_candidate')
                        ev=event(self.source,rid,n['node_id'],p,i+1,c,'evaluated_candidate',len(es));es.append(ev)
                        md=json.loads(n['metadata_json']);md['score_evidence']={'event_id':ev['event_id'],'path':['score']};n['metadata_json']=dumps(md)
                        ns.append(n);batch_nodes.append((c,n))
                chosen=selected[-1][1] if selected else page['action']
                matches=[n for c,n in batch_nodes if c['prediction']==chosen]
                if len(matches)==1:
                    n=matches[0];md=json.loads(n['metadata_json']);md['selected']=True
                    if n['parent_id'] is None:
                        n['parent_id']=f'{rid}::{previous}';md['parent_relation']='reconstructed_from_official_code';md['selected_sequence_parent']=True
                    n['metadata_json']=dumps(md)
                else:
                    n=node(self.source,rid,f'executed{i}',previous,len(ns),p,i+1,proposal=chosen,
                           metadata={'parent_relation':'linear_sequence','selected':True,'html_page_index':i},node_type='browser_action');ns.append(n)
                n['prompt']=page['observation'];n['result_summary']=pgs[i+1]['observation'] if i+1<len(pgs) else None
                es.append(event(self.source,rid,n['node_id'],p,i+1,page,'rendered_step',len(es)))
                previous=n['node_id'].removeprefix(rid+'::')
            outcomes=[];result_file=p.parent/'results.txt'
            if result_file.exists():
                task_id=p.stem.removeprefix('render_')
                for line_no,line in enumerate(result_file.read_text().splitlines(),1):
                    match=re.search(r'\[Result\] \((PASS|FAIL)\).*?/(\d+)\.json',line)
                    if match and match[2]==task_id:
                        outcomes.append({'result':match[1],'line':line_no})
                        # A task-level result is not a per-candidate reward. Multiple
                        # attempts are retained without inventing an attempt match.
                        es.append(event(self.source,rid,None,result_file,line_no,{'text':line,'result':match[1]},'task_result',len(es)))
            yield finish(self.source,rid,ns,es,dataset='kohjingyu/search-agents/GPT-4o',structure='partial_search_tree',reference=relative(p),
                         metadata={'task_name':task,'incomplete':True,'unsaved_branch_observations':True,'task_results':outcomes})
