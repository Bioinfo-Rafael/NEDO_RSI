# ORIGIN: PAPER_INSPIRED
# PAPER: Dream-RSI, Sec. 3 (programmable exploration policy)
# PAPER_URL: https://arxiv.org/html/2609.14858v1
# UPSTREAM_CODE: none used
# IMPLEMENTATION_NOTE: All rankings/actions are original Phase 1 heuristics.
"""Budget allocation over a JSON observed prefix; never a task-action generator."""
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class PolicyDecision:
    """Origin: ORIGINAL. Choose an observed target or stop, using replay-local IDs."""
    action: str
    target_node_id: str | None = None
    reason: str = ''


def validate_decision(decision: PolicyDecision, context: dict) -> None:
    """Origin: ORIGINAL. Reject invalid actions and hidden/exhausted targets."""
    if decision.action not in ('DEEPEN', 'WIDEN', 'REVISIT', 'STOP'):
        raise ValueError('Unknown policy action')
    if decision.action == 'STOP':
        if decision.target_node_id is not None:
            raise ValueError('STOP cannot have a target')
    elif decision.target_node_id not in {n['node_id'] for n in context['frontier']}:
        raise ValueError('Target must be an observed, non-exhausted node')


def decision_for(node: dict, context: dict, reason: str) -> PolicyDecision:
    """Origin: ORIGINAL. Name the allocation intent from observed topology only."""
    action = 'WIDEN' if node['children_count'] else 'DEEPEN'
    if not node['children_count'] and node['node_id'] != context['current_node_id']:
        action = 'REVISIT'
    return PolicyDecision(action, node['node_id'], reason)


def choose(context: dict, config: dict, mode: str = 'meta_memory') -> PolicyDecision:
    """Origin: ORIGINAL. Six baselines over the same revealed observations."""
    nodes = context['frontier']
    if context['budget_remaining'] <= 0 or not nodes:
        return PolicyDecision('STOP', reason='budget or observed opportunities exhausted')
    order = lambda n: (n['observed_index'], n['node_id'])
    if mode == 'random':
        picked = random.Random(context['seed'] + context['round']).choice(nodes)
    elif mode == 'dfs':
        picked = max(nodes, key=lambda n: (n['depth'], -n['observed_index']))
    elif mode == 'bfs':
        picked = min(nodes, key=lambda n: (n['depth'], *order(n)))
    elif mode == 'current_best':
        picked = max(nodes, key=lambda n: (n['observed_rank'], -n['observed_index']))
    elif mode in ('meta_free', 'meta_memory'):
        def rank(n: dict) -> tuple:
            plateau = context['plateau_rounds'] >= config['plateau_rounds']
            diversity = 1 / (1 + n['children_count']) if plateau else 0
            evidence = n['memory_support'] if mode == 'meta_memory' else 0
            value = (config['score_weight'] * n['observed_rank']
                     + config['depth_weight'] * n['depth']
                     + config['diversity_weight'] * diversity
                     + config['memory_weight'] * evidence)
            return value, -n['attempts'], -n['observed_index']
        picked = max(nodes, key=rank)
    else:
        raise ValueError(f'Unknown policy: {mode}')
    return decision_for(picked, context, mode)
