"""Copy acquired checkpoints and safely unpack the public browser traces."""
import json
import shutil
import zipfile
from common import ROOT

def main():
    selectors={
        'openevolve':('myopenevolve',['**/programs/*.json','**/checkpoints/**/metadata.json']),
        'evo_mcts':('evo_mcts',['results/paper_data/*.jsonl','execution_logs/evo-mcts/**/*.json','execution_logs/evo-mcts/**/*.log']),
        'swe_agent':('swe_agent',['trajectories/demonstrations/**/*.traj']),
        'treeofthoughts':('treeofthoughts',['logs/**/*.json']),
    }
    for source,(repo,patterns) in selectors.items():
        base=ROOT/'repos'/repo
        for src in sorted({p for pattern in patterns for p in base.glob(pattern) if p.is_file()}):
            target=ROOT/'raw'/source/('repository' if source!='treeofthoughts' else '')/src.relative_to(base)
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(src,target)
    archive=ROOT/'raw/search_agents/gpt4o_search_trajectories.zip'
    output=archive.parent/'extracted'
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target=(output/info.filename).resolve()
            if not target.is_relative_to(output.resolve()):raise ValueError('Unsafe archive member')
            if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with z.open(info) as a,target.open('wb') as b:shutil.copyfileobj(a,b)
            if target.stat().st_size!=info.file_size:raise ValueError(f'Incomplete member: {target}')

if __name__=='__main__':main()
