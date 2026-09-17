import concurrent.futures
import json
import urllib.request
from inventory import ROOT

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'NEDO-RSI-public-history-inventory'}), timeout=60) as r:
        return json.load(r)

def inspect(dataset):
    try:
        info = get('https://huggingface.co/api/datasets/' + dataset)
        url = f'https://huggingface.co/api/datasets/{dataset}/tree/{info["sha"]}?recursive=true&expand=false&limit=1000'
        files = []
        while url:
            with urllib.request.urlopen(url, timeout=60) as r:
                files += json.load(r)
                import re
                m = re.search(r'<([^>]+)>; rel="next"', r.headers.get('Link', ''))
                url = m.group(1) if m else None
        result = {'dataset': dataset, 'info':info, 'files':files}
        print(dataset, len(files), sum(x.get('size',0) for x in files), [(x['path'],x.get('size')) for x in files[:8]], flush=True)
    except Exception as e:
        result = {'dataset': dataset, 'error':str(e)}
        print(dataset, str(e), flush=True)
    (ROOT / 'reports/inventory' / (dataset.replace('/','__')+'.json')).write_text(json.dumps(result,indent=2))
    return result

if __name__ == '__main__':
    links = json.loads((ROOT/'reports/inventory/rest_mcts/summary.json').read_text())['links']
    datasets = [x.split('/datasets/')[1] for x in links if '/datasets/' in x]
    datasets.append('SWE-bench/SWE-smith-trajectories')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as e:
        data=list(e.map(inspect,datasets))
    (ROOT/'reports/hf_inventory.json').write_text(json.dumps(data,indent=2))
