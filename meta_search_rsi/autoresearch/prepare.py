# ORIGIN: ORIGINAL
# IMPLEMENTATION_NOTE: Freeze selected DB records, memory and evaluator file hashes.
"""Preparation called by the single CLI entrypoint; no executable standalone script."""
import hashlib
import json
from pathlib import Path
from meta_search_rsi import db
from meta_search_rsi.tree import from_records

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Hash an existing file in a bounded-memory stream."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fixed_files(root: Path = ROOT) -> list[Path]:
    """Everything executable except the sole editable candidate is fixed."""
    paths = list((root / 'src').rglob('*.py')) + list((root / 'autoresearch').glob('*.py'))
    paths += list((root / 'tests').glob('*.py')) + list((root / 'configs').glob('*'))
    paths += [root / 'pyproject.toml']
    paths += [p for p in [root / 'autoresearch/program.md', root / 'AGENTS.md', root / 'autoresearch.sh'] if p.exists()]
    return sorted(p for p in paths if p.name != 'policy_candidate.py')


def prepare(db_path: Path, run_ids: list[str], memory: list[dict], config: dict,
            destination: Path, root: Path = ROOT) -> dict:
    """Snapshot complete selected evaluation runs, never mutate the source DB."""
    destination.mkdir(parents=True, exist_ok=False)
    cases = []
    tasks = set()
    for run_id in run_ids:
        run, nodes = db.read_run(db_path, run_id, config['max_nodes'])
        tree = from_records(run, nodes, str(db_path.resolve()))
        tasks.add(tree.run.task_name)
        cases.append({'run': run, 'nodes': nodes, 'source': tree.run.source})
    memory = [d for d in memory if d['source_run'] not in set(run_ids) and d['task_name'] not in tasks]
    for name, value in [('cases.json', cases), ('memory.json', memory), ('config.json', config)]:
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))
    stat = db_path.stat()
    manifest = {'source_db': str(db_path.resolve()), 'source_stat': [stat.st_size, stat.st_mtime_ns],
        'run_ids': run_ids, 'memory_documents': len(memory),
        'fixed': {str(p.relative_to(root)): digest(p) for p in fixed_files(root)},
        'inputs': {p.name: digest(p) for p in destination.glob('*.json')},
        'scope': 'Selected full run snapshots, not a full-database copy or blind holdout'}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def verify(destination: Path, root: Path = ROOT) -> dict:
    """Reject modified fixed files, added executable files or changed frozen inputs."""
    manifest = json.loads((destination / 'manifest.json').read_text())
    current = {str(p.relative_to(root)): digest(p) for p in fixed_files(root)}
    if current != manifest['fixed']:
        raise ValueError('Fixed evaluator/config/tests changed; start a new study explicitly')
    for name, expected in manifest['inputs'].items():
        if digest(destination / name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    source = Path(manifest['source_db'])
    stat = source.stat()
    if [stat.st_size, stat.st_mtime_ns] != manifest['source_stat']:
        raise ValueError('Source database changed since preparation')
    return manifest
