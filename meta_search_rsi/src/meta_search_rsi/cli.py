# ORIGIN: ORIGINAL
# IMPLEMENTATION_NOTE: Single CLI entrypoint; all generated files stay in outputs/.
"""Human interface for tree, memory, retrieval, replay and policy evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from . import db
from .atoms import build_atoms, atom_document
from .skills import build_skills, skill_document
from .tree import from_records, SearchTree
from .retrieval import Retriever
from .policy import choose
from .replay import replay

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / 'outputs'


def configuration(path: Path | None = None) -> dict:
    """Load JSON-form YAML (a YAML 1.2 subset) using only the standard library."""
    config = json.loads((ROOT / 'configs/default.yaml').read_text())
    if path:
        config.update(json.loads(path.read_text()))
    return config


def output_path(value: str) -> Path:
    """Require an output path inside this package, including symlink resolution."""
    path = Path(value)
    path = (ROOT / path if not path.is_absolute() else path).resolve()
    if not path.is_relative_to(OUTPUTS.resolve()):
        raise ValueError('Generated files must be under meta_search_rsi/outputs/')
    return path


def write_json(path: Path, value: object) -> None:
    """Write a complete JSON artifact in outputs/ using atomic replacement."""
    output_path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def load_memory(path: Path) -> list[dict]:
    """Read an explicitly selected memory artifact; no implicit whole-DB index."""
    value = json.loads(path.read_text())
    return value['documents'] if isinstance(value, dict) else value


def load_tree(path: Path, run_id: str, config: dict) -> SearchTree:
    """Load one complete bounded source run through the read-only DB adapter."""
    run, nodes = db.read_run(path, run_id, config['max_nodes'])
    return from_records(run, nodes, str(path.resolve()))


def main(argv: list[str] | None = None) -> None:
    """Dispatch offline operations and the explicit optional Codex proposal loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ['runs', 'tree', 'build-memory', 'retrieve', 'replay', 'evaluate', 'autoresearch']:
        sub = commands.add_parser(command)
        sub.add_argument('--db', type=Path, default=ROOT.parent / 'public_experience/public_experience.db')
        sub.add_argument('--config', type=Path)
        sub.add_argument('--run-id', action='append', default=[])
        sub.add_argument('--memory', type=Path, default=OUTPUTS / 'memory/memory.json')
        sub.add_argument('--out')
        if command == 'runs':
            sub.add_argument('--pattern', default='*'); sub.add_argument('--limit', type=int, default=10)
        if command == 'retrieve':
            sub.add_argument('--query', required=True); sub.add_argument('--kind', choices=['atom', 'skill'])
            sub.add_argument('--structure'); sub.add_argument('--source')
        if command in ('replay', 'evaluate', 'autoresearch'):
            sub.add_argument('--budget', type=int); sub.add_argument('--goal', default='')
        if command == 'replay':
            sub.add_argument('--root-id', help='Select a component root in a forest')
            sub.add_argument('--policy', choices=['random', 'dfs', 'bfs', 'current_best', 'meta_free', 'meta_memory'], default='meta_memory')
        if command in ('evaluate', 'autoresearch'):
            sub.add_argument('--study', default='outputs/autoresearch/demo')
            sub.add_argument('--prepare-only', action='store_true')
            sub.add_argument('--label', default='v1')
        if command == 'autoresearch':
            sub.add_argument('--iterations', type=int, default=3)
            sub.add_argument('--dry-run', action='store_true', help='Prepare prompt/argv without calling Codex')
            sub.add_argument('--proposal-file', type=Path, help='Evaluate a manually supplied JSON proposal offline')
            sub.add_argument('--model', help='Optional Codex model override')
            sub.add_argument('--codex-bin', default='codex')
            sub.add_argument('--codex-timeout', type=float, default=300)
    args = parser.parse_args(argv)
    config = configuration(args.config)
    if getattr(args, 'budget', None) is not None:
        config['budget'] = args.budget
    if args.command == 'runs':
        print(json.dumps(db.list_runs(args.db, args.pattern, args.limit), ensure_ascii=False, indent=2)); return
    if args.command in ('evaluate', 'autoresearch'):
        sys.path.insert(0, str(ROOT))
        from autoresearch.prepare import prepare
        from autoresearch.evaluate import evaluate_study
        study = output_path(args.study)
        if not study.exists():
            if not args.run_id:
                parser.error('--run-id required to prepare a study')
            manifest = prepare(args.db, args.run_id, load_memory(args.memory), config, study)
            print(json.dumps({'prepared': str(study), 'memory_documents': manifest['memory_documents']}))
        if args.prepare_only:
            from autoresearch.prepare import verify
            verify(study)
            return
        if args.command == 'autoresearch':
            from autoresearch.codex_loop import run_loop
            result = run_loop(study, args.iterations, dry_run=args.dry_run,
                              proposal_file=args.proposal_file, executable=args.codex_bin,
                              model=args.model, timeout=args.codex_timeout)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if any(r['status'] in ('codex_error', 'interrupted') for r in result['iterations']):
                raise SystemExit(1)
            return
        record = evaluate_study(study, args.label, ROOT / 'autoresearch/policy_candidate.py')
        print(json.dumps({'status': record['status'], 'objective': record['objective'],
              'baselines': {k: v['summary'] for k, v in record['results'].items()}}, indent=2)); return
    if args.command in ('tree', 'build-memory', 'replay') and not args.run_id:
        parser.error('--run-id required; whole-database operations are never implicit')
    if args.command == 'build-memory':
        documents = []
        for run_id in args.run_id:
            tree = load_tree(args.db, run_id, config)
            documents.extend(atom_document(a) for a in build_atoms(tree, config['entity_slots'], config['history_nodes']))
            documents.extend(skill_document(s) for s in build_skills(tree, config['entity_slots']))
        result = {'documents': documents, 'source_db': str(args.db.resolve()), 'run_ids': args.run_id}
        destination = output_path(args.out or 'outputs/memory/memory.json')
        write_json(destination, result)
        print(json.dumps({'path': str(destination), 'atoms': sum(d['kind'] == 'atom' for d in documents),
                          'skills': sum(d['kind'] == 'skill' for d in documents)})); return
    if args.command == 'retrieve':
        retriever = Retriever(load_memory(args.memory), config['bm25_k1'], config['bm25_b'])
        result = [asdict(m) for m in retriever.search(args.query, config['retrieval_k'],
                  exclude_runs=set(args.run_id), source=args.source, kind=args.kind, structure=args.structure)]
        destination = output_path(args.out or 'outputs/memory/retrieval.json')
        write_json(destination, result)
        print(json.dumps({'path': str(destination), 'matches': len(result),
              'memory_ids': [m['memory_id'] for m in result]}, indent=2)); return
    if len(args.run_id) != 1:
        parser.error('tree/replay takes exactly one --run-id')
    tree = load_tree(args.db, args.run_id[0], config)
    if args.command == 'tree':
        result = {'run': asdict(tree.run), 'missing_parents': tree.missing_parents,
                  'nodes': [asdict(n) for n in tree.nodes.values()]}
        write_json(output_path(args.out or 'outputs/tree.json'), result)
        for node in tree.nodes.values():
            depth = len(tree.path_to(node.node_id)) - 1
            print('  ' * depth + node.node_id + ' | score=' + str(node.score))
    else:
        retriever = Retriever(load_memory(args.memory), config['bm25_k1'], config['bm25_b']) if args.policy == 'meta_memory' else None
        result = replay(tree, lambda c: choose(c, config, args.policy), config, retriever, args.goal, args.policy, args.root_id)
        destination = output_path(args.out or 'outputs/replay/replay.json')
        write_json(destination, asdict(result))
        print(json.dumps({'path': str(destination), 'rank': result.rank_attainment,
                          'raw_best': result.raw_best, 'rounds': result.rounds}, indent=2))


if __name__ == '__main__':
    main()
