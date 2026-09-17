"""Collect public repository metadata before downloading history payloads."""
import concurrent.futures
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'openevolve': 'algorithmicsuperintelligence/openevolve',
    'search_agents': 'kohjingyu/search-agents',
    'evo_mcts': 'iphysresearch/evo-mcts',
    'rest_mcts': 'THUDM/ReST-MCTS',
    'swe_agent': 'SWE-agent/SWE-agent',
    'dream_rsi': 'zhengkid/Dream-RSI',
}

def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', path], text=True))

def inspect(item):
    name, repo = item
    dst = ROOT / 'reports/inventory' / name
    dst.mkdir(parents=True, exist_ok=True)
    info = api(f'repos/{repo}')
    sha = api(f'repos/{repo}/commits/{info["default_branch"]}')['sha']
    tree = api(f'repos/{repo}/git/trees/{sha}?recursive=1')
    releases = api(f'repos/{repo}/releases?per_page=100')
    import base64
    readme_data = api(f'repos/{repo}/readme')
    readme = base64.b64decode(readme_data['content']).decode()
    for filename, data in [('repository.json', info), ('tree.json', tree), ('releases.json', releases)]:
        (dst / filename).write_text(json.dumps(data, indent=2))
    (dst / 'README.md').write_text(readme)
    links = sorted(set(re.findall(r'https?://[^\s<>\)\]"\x27]+', readme)))
    candidates = [x for x in tree.get('tree', []) if x['type'] == 'blob' and
                  re.search(r'jsonl?$|parquet$|\.traj$|checkpoints|paper_data', x['path'])]
    summary = {'source': name, 'repo': repo, 'sha': sha, 'repo_kib': info['size'],
               'license': info.get('license'), 'tree_truncated': tree.get('truncated'),
               'files': len(tree.get('tree', [])), 'candidates': candidates,
               'links': links, 'release_assets': [a for r in releases for a in r['assets']]}
    (dst / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'source': name, 'sha': sha, 'files': summary['files'],
                      'repo_kib': info['size'], 'candidates': len(candidates),
                      'candidate_bytes': sum(x.get('size', 0) for x in candidates),
                      'links': [x for x in links if any(s in x for s in ('huggingface', 'drive.google', 'trajectory', 'trajector', 'data', 'dropbox'))],
                      'release_assets': [(x['name'], x['size']) for x in summary['release_assets']]}), flush=True)
    return summary

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        summaries = list(executor.map(inspect, SOURCES.items()))
    issue = api('repos/algorithmicsuperintelligence/openevolve/issues/156')
    comments = api('repos/algorithmicsuperintelligence/openevolve/issues/156/comments?per_page=100')
    (ROOT / 'reports/inventory/openevolve/issue156.json').write_text(json.dumps({'issue': issue, 'comments': comments}, indent=2))
    bodies = issue['body'] + '\n' + '\n'.join(x['body'] for x in comments)
    repos = sorted(set(re.findall(r'https://github.com/([^/\s]+/MyOpenEvolve)', bodies)))
    print('ISSUE_156_REPOS', repos, flush=True)
    for repo in repos:
        summaries.append(inspect(('myopenevolve', repo)))
    (ROOT / 'reports/inventory.json').write_text(json.dumps(summaries, indent=2))
