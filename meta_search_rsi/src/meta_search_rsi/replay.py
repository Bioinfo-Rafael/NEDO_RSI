# ORIGIN: PAPER_INSPIRED
# PAPER: Dream-RSI, Sec. 3; Appendix B.2
# PAPER_URL: https://arxiv.org/html/2609.14858v1
# UPSTREAM_CODE: none used
# IMPLEMENTATION_NOTE: Sequential generalized-tree replay; evaluation is ORIGINAL.
"""Separate the hidden historical environment from policy-visible JSON."""
from dataclasses import dataclass, asdict
from collections import Counter
import json
from statistics import mean
from typing import Callable
from .policy import PolicyDecision, validate_decision
from .retrieval import Retriever
from .tree import SearchTree, score_key, utility


@dataclass(frozen=True)
class ReplayResult:
    """Origin: ORIGINAL. Decisions and final evaluator-only diagnostics."""
    run_id: str
    policy: str
    root_node_id: str
    component_nodes: int
    excluded_component_nodes: int
    observed_ids: list[str]
    rounds: int
    reveals: int
    empty_probes: int
    raw_best: dict
    rank_attainment: float | None
    normalized_attainment: float | None
    rank_auc: float | None
    objective: float | None
    trace: list[dict]
    limitations: list[str]


def policy_context(tree: SearchTree, observed: list[str], exhausted: set[str],
                   attempts: Counter, config: dict, round_no: int, retriever: Retriever | None,
                   goal: str, plateau: int) -> dict:
    """Origin: ORIGINAL. Whitelist observed fields; no full-tree stats or source IDs."""
    prefix = tree.prefix(set(observed))
    alias = {key: f'P{i:04d}' for i, key in enumerate(observed)}
    observed_keys = {score_key(n) for n in prefix.nodes.values()} - {None}
    scored = [n for n in prefix.nodes.values() if score_key(n) is not None]
    values = sorted({utility(n) for n in scored}) if len(observed_keys) == 1 else []
    records, evidence = [], []
    for i, key in enumerate(observed):
        node = prefix.nodes[key]
        text = '\n'.join(v or '' for v in [node.prompt, node.result_summary])[:config['context_chars']]
        rank = values.index(utility(node)) / max(1, len(values) - 1) if values and score_key(node) else 0.0
        path = prefix.path_to(key)
        signs = []
        for a, b in zip(path, path[1:]):
            if score_key(a) is not None and score_key(a) == score_key(b):
                signs.append((utility(b) > utility(a)) - (utility(b) < utility(a)))
        memories = retriever.search(text + '\n' + goal, config['retrieval_k'],
                    exclude_runs={tree.run.run_id}, exclude_tasks={tree.run.task_name},
                    structure='tree' if any(len(prefix.children(k)) > 1 for k in observed) else 'linear') if retriever else []
        support = sum((1 if m.outcome_label == 'success' else -1 if m.outcome_label == 'failure' else 0)
                      * m.relevance / (1 + m.relevance) for m in memories) / max(1, len(memories))
        records.append({'node_id': alias[key], 'parent_id': alias.get(node.parent_id), 'observed_index': i,
            'depth': len(path) - 1, 'children_count': len(prefix.children(key)), 'score': node.score,
            'score_name': node.score_name, 'score_direction': node.score_direction, 'observed_rank': rank,
            'improvement_history': signs, 'status': node.status, 'result_summary': text,
            'attempts': attempts[key], 'memory_support': support})
        for memory in memories:
            doc = asdict(memory)
            doc['raw_text'] = doc['raw_text'][:config['context_chars']]
            doc['abstracted_text'] = doc['abstracted_text'][:config['context_chars']]
            evidence.append({'target_node_id': alias[key], **doc})
    # Availability is learned only via past empty probes, not hidden child counts.
    frontier = [n for key, n in zip(observed, records) if key not in exhausted]
    return {'nodes': records, 'frontier': frontier, 'memories': evidence,
            'current_node_id': alias[observed[-1]] if observed else None, 'goal': goal,
            'round': round_no, 'budget_remaining': config['budget'] - round_no,
            'budget_total': config['budget'], 'seed': config['seed'], 'plateau_rounds': plateau}


def attainment(tree: SearchTree, observed: list[str]) -> dict:
    """Origin: ORIGINAL. Evaluation-only full-run normalization, never policy input."""
    groups = {}
    for node in tree.nodes.values():
        if score_key(node):
            groups.setdefault(score_key(node), []).append(node)
    raw = {}
    for key, nodes in groups.items():
        seen = [n for n in nodes if n.node_id in observed]
        raw[':'.join(key)] = max(seen, key=utility).score if seen else None
    if len(groups) != 1:
        return {'raw_best': raw, 'rank': None, 'normalized': None}
    nodes = next(iter(groups.values()))
    values = sorted({utility(n) for n in nodes})
    seen = [utility(n) for n in nodes if n.node_id in observed]
    if not seen:
        return {'raw_best': raw, 'rank': 0.0, 'normalized': 0.0}
    best = max(seen)
    rank = values.index(best) / (len(values) - 1) if len(values) > 1 else 1.0
    normal = (best - values[0]) / (values[-1] - values[0]) if values[-1] != values[0] else 1.0
    return {'raw_best': raw, 'rank': rank, 'normalized': normal}


def replay(tree: SearchTree, policy: Callable[[dict], PolicyDecision], config: dict,
           retriever: Retriever | None = None, goal: str = '', name: str = 'candidate', root_id: str | None = None) -> ReplayResult:
    """Reset one component to root only; reveal actual children after decisions."""
    if config['budget'] < 1:
        raise ValueError('budget must be positive')
    roots = tree.root_nodes()
    if not roots:
        raise ValueError('Cannot replay an empty tree')
    total_nodes = len(tree.nodes)
    root_id = root_id or roots[0].node_id
    tree = tree.component(root_id)
    observed = [root_id]
    exhausted, attempts, trace, ranks = set(), Counter(), [], []
    reveals = empty = plateau = 0
    for round_no in range(config['budget']):
        context = policy_context(tree, observed, exhausted, attempts, config, round_no, retriever, goal, plateau)
        decision = policy(json.loads(json.dumps(context)))  # no shared mutable objects
        validate_decision(decision, context)
        if decision.action == 'STOP':
            trace.append({'round': round_no, 'decision': asdict(decision), 'revealed': None})
            break
        target = observed[int(decision.target_node_id[1:])]
        attempts[target] += 1
        child = next((n for n in tree.children(target) if n.node_id not in observed), None)
        if child:
            parent = tree.nodes[target]
            improved = score_key(parent) is not None and score_key(parent) == score_key(child) and utility(child) > utility(parent)
            plateau = 0 if improved else plateau + 1
            observed.append(child.node_id); reveals += 1
        else:
            exhausted.add(target); empty += 1; plateau += 1
        trace.append({'round': round_no, 'decision': asdict(decision),
                      'revealed': child.node_id if child else None,
                      'memory_ids': sorted({m['memory_id'] for m in context['memories']})})
        ranks.append(attainment(tree, observed)['rank'])
    final = attainment(tree, observed)
    rounds = reveals + empty
    padded = ranks + [final['rank']] * (config['budget'] - len(ranks))
    objective = final['rank'] - config['rank_cost'] * rounds / config['budget'] if final['rank'] is not None else None
    return ReplayResult(tree.run.run_id, name, root_id, len(tree.nodes), total_nodes - len(tree.nodes),
        observed, rounds, reveals, empty, final['raw_best'],
        final['rank'], final['normalized'], mean(padded) if all(v is not None for v in padded) else None,
        objective, trace, ['Recorded transitions only; no counterfactual action prediction',
        'Only the selected real component is evaluated; unresolved parent may be a missing-history boundary',
        'Mixed/unknown score cohorts excluded from scalar rank aggregation'])


def summarize(results: list[ReplayResult]) -> dict:
    """Origin: ORIGINAL. Macro-average normalized results, never raw cross-task scores."""
    valid = [r for r in results if r.objective is not None]
    return {'runs': len(results), 'scored_runs': len(valid), 'unscored_runs': len(results) - len(valid),
            'mean_rank': mean(r.rank_attainment for r in valid) if valid else None,
            'mean_objective': mean(r.objective for r in valid) if valid else None,
            'mean_rounds': mean(r.rounds for r in results) if results else 0}
