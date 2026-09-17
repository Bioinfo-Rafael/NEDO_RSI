# ORIGIN: PAPER_INSPIRED
# PAPER: LifeMem, Sec. 4.1-4.3; Appendix L
# PAPER_URL: https://arxiv.org/html/2609.12655v1
# UPSTREAM_CODE: none used
# IMPLEMENTATION_NOTE: Exact structural workflow fallback, not LLM skill distillation.
"""Group identical abstract step sequences and preserve all source paths."""
from dataclasses import dataclass, asdict
import hashlib
import json
from .atoms import delexicalize, outcome_label
from .tree import SearchTree


@dataclass(frozen=True)
class Skill:
    """Origin: ORIGINAL. Structural workflow evidence, not a claimed causal rule."""
    skill_id: str
    name: str
    preconditions: list[str]
    workflow_steps: list[str | None]
    success_pattern: list[str]
    failure_pattern: list[str]
    source_run_ids: list[str]
    source_node_ids: list[str]
    source: str
    task_name: str
    structure: str
    trajectories: list[dict]


def build_skills(tree: SearchTree, slots: dict) -> list[Skill]:
    """Group root→leaf paths by exact abstract actions; never invent a workflow step."""
    groups = {}
    for leaf in tree.nodes.values():
        if tree.children(leaf.node_id):
            continue
        path = tree.path_to(leaf.node_id)
        steps = [delexicalize(n.proposal, slots) for n in path[1:]]
        # Empty/missing actions must not collapse unrelated trajectories.
        signature = json.dumps(steps if any(steps) else ['missing_actions', leaf.node_id])
        groups.setdefault(signature, []).append((path, steps))
    skills = []
    for signature, entries in groups.items():
        identity = json.dumps([tree.run.source, tree.run.run_id, signature])
        skill_id = 'skill:' + hashlib.sha256(identity.encode()).hexdigest()[:24]
        success, failure, trajectories = [], [], []
        for path, _ in entries:
            label = outcome_label(path[-2] if len(path) > 1 else None, path[-1])
            evidence = path[-1].result_summary
            if evidence and label in ('success', 'failure'):
                (success if label == 'success' else failure).append(evidence)
            trajectories.append({'node_ids': [n.node_id for n in path],
                'raw_actions': [n.proposal for n in path[1:]], 'label': label,
                'raw_result': path[-1].result_summary})
        first = entries[0][0][0]
        skills.append(Skill(skill_id, 'Structural workflow ' + skill_id[-8:],
            [first.prompt] if first.prompt else [], entries[0][1], success, failure,
            [tree.run.run_id], sorted({n.node_id for path, _ in entries for n in path}),
            tree.run.source, tree.run.task_name, tree.run.metadata.get('source_structure', 'unknown'), trajectories))
    return skills


def skill_document(skill: Skill) -> dict:
    """Origin: ORIGINAL. Normalize skill metadata for the shared lexical retriever."""
    labels = {path['label'] for path in skill.trajectories}
    return {**asdict(skill), 'memory_id': skill.skill_id, 'kind': 'skill',
            'source_run': skill.source_run_ids[0], 'source_nodes': skill.source_node_ids,
            'outcome_label': next(iter(labels)) if len(labels) == 1 else 'mixed',
            'raw_text': json.dumps(skill.trajectories, ensure_ascii=False),
            'abstracted_text': '\n'.join(step or '[unrecorded action]' for step in skill.workflow_steps)}
