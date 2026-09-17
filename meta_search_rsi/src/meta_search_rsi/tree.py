# ORIGIN: ORIGINAL
# IMPLEMENTATION_NOTE: Existing schema mapping and observed-only tree operations.
"""Tree reconstruction; no guessed edges and no hidden-prefix metadata."""
from dataclasses import dataclass, field
import json
import math
from typing import Any


@dataclass(frozen=True)
class Run:
    """Source identity and task context, never passed wholesale to a policy."""
    run_id: str
    task_name: str = ''
    project_name: str = ''
    source: str = ''
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExperienceNode:
    """One stored observation; sequence is environment order, not an edge."""
    node_id: str
    parent_id: str | None
    sequence_index: int = 0
    prompt: str | None = None
    proposal: str | None = None
    result_summary: str | None = None
    status: str | None = None
    score: float | None = None
    score_name: str | None = None
    score_direction: str = 'unknown'
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def finite_number(value: Any) -> bool:
    """Accept finite numeric observations, excluding booleans."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def score_key(node: ExperienceNode) -> tuple[str, str] | None:
    """Identify a comparable score; unknown directions are never guessed."""
    if finite_number(node.score) and node.score_name and node.score_direction in ('minimize', 'maximize'):
        return node.score_name, node.score_direction
    return None


def utility(node: ExperienceNode) -> float:
    """Orient a known score so that larger is better, within its cohort only."""
    if score_key(node) is None:
        raise ValueError('Score has no known name/direction/value')
    return node.score if node.score_direction == 'maximize' else -node.score


class SearchTree:
    """A validated tree/forest. Missing parents remain unresolved references."""

    def __init__(self, run: Run, nodes: list[ExperienceNode]):
        self.run = run
        self.nodes = {node.node_id: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError('Duplicate node_id')
        self._children: dict[str, list[str]] = {key: [] for key in self.nodes}
        self.missing_parents = {}
        for node in nodes:
            if node.parent_id in self.nodes:
                self._children[node.parent_id].append(node.node_id)
            elif node.parent_id is not None:
                self.missing_parents[node.node_id] = node.parent_id
        for children in self._children.values():
            children.sort(key=lambda key: (self.nodes[key].sequence_index, key))
        self._validate_cycles()

    def _validate_cycles(self) -> None:
        done = set()
        for key in self.nodes:
            trail = set()
            while key in self.nodes and key not in done:
                if key in trail:
                    raise ValueError(f'Cycle at {key}')
                trail.add(key)
                key = self.nodes[key].parent_id
            done.update(trail)

    def root_nodes(self) -> list[ExperienceNode]:
        """Return component roots, including nodes with unresolved parents."""
        return sorted((n for n in self.nodes.values() if n.parent_id not in self.nodes),
                      key=lambda n: (n.sequence_index, n.node_id))

    def children(self, node_id: str) -> list[ExperienceNode]:
        """Return stored children in deterministic recorded order (environment only)."""
        return [self.nodes[key] for key in self._children[node_id]]

    def parent(self, node_id: str) -> ExperienceNode | None:
        """Return a known parent, without fabricating missing references."""
        return self.nodes.get(self.nodes[node_id].parent_id)

    def path_to(self, node_id: str) -> list[ExperienceNode]:
        """Follow actual parent links from the component root to a node."""
        path = []
        node = self.nodes[node_id]
        while node is not None:
            path.append(node)
            node = self.parent(node.node_id)
        return list(reversed(path))

    def prefix(self, observed_ids: set[str]) -> 'SearchTree':
        """Build a closed observed subtree, excluding all hidden nodes and metadata."""
        if not observed_ids <= self.nodes.keys():
            raise ValueError('Unknown observed node')
        for key in observed_ids:
            parent = self.nodes[key].parent_id
            if parent in self.nodes and parent not in observed_ids:
                raise ValueError('Prefix is not ancestor-closed')
        run = Run(self.run.run_id)  # no final-run status/metadata in a prefix
        return SearchTree(run, [n for n in self.nodes.values() if n.node_id in observed_ids])

    def component(self, root_id: str) -> 'SearchTree':
        """Select one actual rooted component for root-only replay, without fake edges."""
        if root_id not in {n.node_id for n in self.root_nodes()}:
            raise ValueError('Replay root must be a component root')
        pending, selected = [root_id], []
        while pending:
            key = pending.pop()
            selected.append(self.nodes[key])
            pending.extend(reversed(self._children[key]))
        return SearchTree(self.run, selected)

    def frontier(self, observed_ids: set[str]) -> list[ExperienceNode]:
        """Observed roots plus observed leaves; never inspect hidden child existence."""
        prefix = self.prefix(observed_ids)
        roots = {n.node_id for n in prefix.root_nodes()}
        return [n for n in prefix.nodes.values() if n.node_id in roots or not prefix.children(n.node_id)]


def from_records(run: dict, rows: list[dict], source: str) -> SearchTree:
    """Map the existing SQLite schema into a tree, preserving missing data."""
    info = Run(run['run_id'], run.get('task_name') or '', run.get('project_name') or '',
               source, json.loads(run.get('metadata_json') or '{}'))
    nodes = []
    for row in rows:
        values = {key: row[key] for key in ExperienceNode.__dataclass_fields__ if key in row}
        values['metrics'] = json.loads(row.get('metrics_json') or '{}')
        values['metadata'] = json.loads(row.get('metadata_json') or '{}')
        nodes.append(ExperienceNode(**values))
    return SearchTree(info, nodes)
