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
            for i,page in enumerate(pgs):
                n=node(self.source,rid,str(i),str(i-1) if i else None,i,p,i+1,
                       prompt=page['observation'],proposal=page['action'],result=pgs[i+1]['observation'] if i+1<len(pgs) else None,
                       metadata={'parent_relation':'linear_sequence','html_page_index':i},node_type='browser_action')
                ns.append(n);es.append(event(self.source,rid,n['node_id'],p,i+1,page,'rendered_step',i))
            if ns:yield finish(self.source,rid,ns,es,dataset='kohjingyu/search-agents/GPT-4o',structure='linear',reference=relative(p),metadata={'task_name':task,'incomplete':True})
