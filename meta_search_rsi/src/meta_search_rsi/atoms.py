# ORIGIN: PAPER_INSPIRED
# PAPER: SGA-MCTS, Findings ACL 2026, Sec. 3.3.2
# PAPER_URL: https://arxiv.org/html/2604.14712v1
# UPSTREAM_CODE: none used
# IMPLEMENTATION_NOTE: Newly written edge abstraction, with raw fields and outcomes.
"""Deterministic SGA-style atoms; no LLM, MCTS generation or guessed goals."""
from dataclasses import dataclass, asdict
import hashlib
import json
import re
from .tree import SearchTree, ExperienceNode, finite_number, score_key, utility


@dataclass(frozen=True)
class ExperienceAtom:
    """Origin: ORIGINAL. Evidence container for one actual parent→child transition."""
    memory_id: str
    source: str
    source_run: str
    source_nodes: list[str]
    task_name: str
    structure: str
    raw_state: dict
    raw_goal: str | None
    raw_action: str | None
    raw_outcome: dict
    abstract_state: str
    abstract_goal: str | None
    abstract_action: str | None
    abstract_outcome: str
    outcome_label: str
    abstraction: str = 'explicit_slots_only'


def delexicalize(text: str | None, slots: dict[str, str]) -> str | None:
    """Origin: PAPER_INSPIRED, SGA-MCTS §3.3.2; typed slots, no source code copied.

    Only explicitly configured entities are replaced. Control numbers stay intact.
    """
    if text is None:
        return None
    for entity in sorted(slots, key=len, reverse=True):
        if not entity or not re.fullmatch(r'<[A-Z_]+>', slots[entity]):
            raise ValueError('entity_slots must map nonempty entities to <TYPED_SLOT>')
        text = re.sub(r'(?<!\w)' + re.escape(entity) + r'(?!\w)', lambda _: slots[entity], text)
    return text


def outcome_label(parent: ExperienceNode | None, node: ExperienceNode) -> str:
    """Origin: ORIGINAL. Explicit status, then comparable within-run improvement."""
    status = (node.status or '').lower()
    if status in ('failed', 'failure', 'error', 'crashed', 'rejected'):
        return 'failure'
    if status in ('success', 'successful', 'solved', 'accepted'):
        return 'success'
    if parent and score_key(parent) is not None and score_key(parent) == score_key(node):
        delta = utility(node) - utility(parent)
        return 'success' if delta > 0 else 'failure' if delta < 0 else 'unknown'
    return 'unknown'  # completed is not proof of success


def build_atoms(tree: SearchTree, slots: dict, history_nodes: int = 2) -> list[ExperienceAtom]:
    """Origin: PAPER_INSPIRED. Extract only real edges, keeping every raw field."""
    atoms = []
    for child in tree.nodes.values():
        parent = tree.parent(child.node_id)
        if parent is None:
            continue
        history = tree.path_to(parent.node_id)[-history_nodes:] if history_nodes else []
        state = {'prompt': parent.prompt, 'result': parent.result_summary, 'metrics': parent.metrics,
                 'history': [{'node_id': n.node_id, 'proposal': n.proposal} for n in history]}
        goal = child.metadata.get('goal', tree.run.metadata.get('goal'))
        goal = goal if isinstance(goal, str) else None
        comparable = score_key(parent) is not None and score_key(parent) == score_key(child)
        metric_delta = {k: v - parent.metrics[k] for k, v in child.metrics.items()
                        if finite_number(v) and finite_number(parent.metrics.get(k))}
        outcome = {'result': child.result_summary, 'status': child.status, 'score': child.score,
                   'score_name': child.score_name, 'score_direction': child.score_direction,
                   'score_delta': child.score - parent.score if comparable else None,
                   'metrics': child.metrics, 'metrics_delta': metric_delta,
                   'metrics_delta_scope': 'same run and metric key; units not inferred'}
        identity = json.dumps([tree.run.source, tree.run.run_id, parent.node_id, child.node_id])
        memory_id = 'atom:' + hashlib.sha256(identity.encode()).hexdigest()[:24]
        atoms.append(ExperienceAtom(memory_id, tree.run.source, tree.run.run_id,
                     [parent.node_id, child.node_id], tree.run.task_name,
                     tree.run.metadata.get('source_structure', 'unknown'), state, goal,
                     child.proposal, outcome, delexicalize(json.dumps(state, ensure_ascii=False), slots),
                     delexicalize(goal, slots), delexicalize(child.proposal, slots),
                     delexicalize(json.dumps(outcome, ensure_ascii=False), slots), outcome_label(parent, child)))
    return atoms


def atom_document(atom: ExperienceAtom) -> dict:
    """Origin: ORIGINAL. Serialize complete evidence and searchable representations."""
    data = asdict(atom)
    data.update(kind='atom', raw_text=json.dumps([atom.raw_state, atom.raw_goal, atom.raw_action,
                atom.raw_outcome], ensure_ascii=False), abstracted_text='\n'.join(
                v or '' for v in [atom.abstract_state, atom.abstract_goal, atom.abstract_action, atom.abstract_outcome]))
    return data
