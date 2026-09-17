"""Download the exact public Drive link published by the official README."""
import html
import re
import subprocess
import urllib.parse
import urllib.request
from common import ROOT

def main():
    file_id='127GqJ19qxpAcWlUKXlr5zBeAIW5Pi_0H'
    url=f'https://drive.google.com/uc?export=download&id={file_id}'
    dest=ROOT/'raw/search_agents/gpt4o_search_trajectories.zip'
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():return
    with urllib.request.urlopen(url,timeout=60) as response:
        content=response.read()
    if content.startswith(b'PK'):
        dest.write_bytes(content);return
    page=content.decode()
    params=dict(re.findall(r'type="hidden" name="([^"]+)" value="([^"]*)"',page))
    action=re.search(r'<form[^>]*action="([^"]+)"',page)
    if not action:raise RuntimeError('Public Drive download form unavailable')
    target=html.unescape(action[1])+'?'+urllib.parse.urlencode(params)
    if not target.startswith('https://drive.usercontent.google.com/download?'):raise ValueError('Unexpected Drive target')
    tmp=dest.with_suffix('.zip.partial')
    subprocess.run(['curl','--fail','--location','--retry','3','--connect-timeout','30','--max-time','1800',target,'-o',str(tmp)],check=True)
    tmp.rename(dest)

if __name__=='__main__':main()
