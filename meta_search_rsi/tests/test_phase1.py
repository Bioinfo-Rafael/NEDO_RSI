# ORIGIN: ORIGINAL
# IMPLEMENTATION_NOTE: Synthetic regression fixtures are isolated under outputs/.
"""Boundary, provenance and deterministic offline regression tests."""
from collections import Counter
from dataclasses import replace, asdict
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meta_search_rsi import db
from meta_search_rsi.atoms import build_atoms, atom_document, delexicalize
from meta_search_rsi.skills import build_skills, skill_document
from meta_search_rsi.tree import Run, ExperienceNode, SearchTree
from meta_search_rsi.retrieval import Retriever
from meta_search_rsi.policy import choose, PolicyDecision, validate_decision
from meta_search_rsi.replay import replay, policy_context, attainment, summarize
from meta_search_rsi.cli import configuration, output_path
from autoresearch.evaluate import Candidate
from autoresearch.prepare import prepare, verify


def fixture(direction='maximize'):
    """Make a tree with siblings, a deeper branch and a failed action."""
    def node(key, parent, order, score, status='completed'):
        return ExperienceNode(key, parent, order, prompt='evaluate model',
            proposal='reduce forecast_loss weight 0.2 to 0.1', result_summary='evaluate loss',
            status=status, score=score, score_name='loss', score_direction=direction,
            metrics={'loss': score, 'resolved': True})
    return SearchTree(Run('run', 'task', 'project', 'fixture', {'source_structure': 'tree'}),
        [node('r', None, 0, 1), node('a', 'r', 1, 2), node('b', 'r', 2, 0, 'failed'), node('c', 'a', 3, 4)])


class Phase1Tests(unittest.TestCase):
    """Every test targets a behavior or isolation boundary, not implementation text."""

    def setUp(self):
        self.config = configuration()
        self.config['budget'] = 6
        self.tree = fixture()

    def test_tree_reconstruction(self):
        self.assertEqual([n.node_id for n in self.tree.root_nodes()], ['r'])
        self.assertEqual([n.node_id for n in self.tree.children('r')], ['a', 'b'])
        self.assertEqual([n.node_id for n in self.tree.path_to('c')], ['r', 'a', 'c'])
        self.assertEqual(self.tree.parent('a').node_id, 'r')
        self.assertEqual({n.node_id for n in self.tree.frontier({'r', 'a'})}, {'r', 'a'})

    def test_cycles_duplicates_and_prefix_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Cycle'):
            SearchTree(Run('x'), [ExperienceNode('a', 'b'), ExperienceNode('b', 'a')])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            SearchTree(Run('x'), [ExperienceNode('a', None)] * 2)
        with self.assertRaisesRegex(ValueError, 'ancestor'):
            self.tree.prefix({'r', 'c'})

    def test_forest_missing_parent_preserved(self):
        tree = SearchTree(Run('x'), [ExperienceNode('a', 'missing'), ExperienceNode('b', None)])
        self.assertEqual(len(tree.root_nodes()), 2)
        self.assertEqual(tree.missing_parents, {'a': 'missing'})
        self.assertEqual(tree.nodes['a'].parent_id, 'missing')

    def test_forest_replay_starts_with_one_root(self):
        tree = SearchTree(Run('forest'), [ExperienceNode('r1', None), ExperienceNode('a', 'r1'),
                                        ExperienceNode('r2', None), ExperienceNode('b', 'r2')])
        result = replay(tree, lambda c: choose(c, self.config, 'dfs'), self.config, root_id='r2')
        self.assertEqual(result.root_node_id, 'r2')
        self.assertEqual(set(result.observed_ids), {'r2', 'b'})
        self.assertEqual(result.excluded_component_nodes, 2)

    def test_memory_has_effect_without_dictating_actions(self):
        context = {'frontier': [
            {'node_id': 'P0000', 'observed_rank': 0.5, 'depth': 1, 'children_count': 0,
             'memory_support': -1, 'attempts': 0, 'observed_index': 0},
            {'node_id': 'P0001', 'observed_rank': 0.5, 'depth': 1, 'children_count': 0,
             'memory_support': 1, 'attempts': 0, 'observed_index': 1}],
            'budget_remaining': 2, 'plateau_rounds': 0, 'current_node_id': 'P0001'}
        self.assertEqual(choose(context, self.config, 'meta_free').target_node_id, 'P0000')
        self.assertEqual(choose(context, self.config, 'meta_memory').target_node_id, 'P0001')

    def test_atom_edge_mapping_and_deltas(self):
        atoms = build_atoms(self.tree, {'forecast_loss': '<OBJECTIVE>'})
        self.assertEqual(len(atoms), 3)
        a = atoms[0]
        self.assertEqual(a.source_nodes, ['r', 'a'])
        self.assertEqual(a.raw_state['result'], self.tree.nodes['r'].result_summary)
        self.assertEqual(a.raw_action, self.tree.nodes['a'].proposal)
        self.assertIsNone(a.raw_goal)
        self.assertEqual(a.raw_outcome['score_delta'], 1)
        self.assertNotIn('resolved', a.raw_outcome['metrics_delta'])
        self.assertIn('<OBJECTIVE>', a.abstract_action)
        self.assertIn('0.2 to 0.1', a.abstract_action)
        self.assertIn('forecast_loss', a.raw_action)

    def test_unknown_scores_and_cross_metric_not_compared(self):
        nodes = list(self.tree.nodes.values())
        nodes[1] = replace(nodes[1], score_name='different')
        atoms = build_atoms(SearchTree(self.tree.run, nodes), {})
        self.assertIsNone(atoms[0].raw_outcome['score_delta'])
        self.assertIsNone(attainment(SearchTree(self.tree.run, nodes), ['r', 'a'])['rank'])
        unknown = SearchTree(Run('u'), [ExperienceNode('r', None, score=8, score_direction='unknown')])
        self.assertIsNone(attainment(unknown, ['r'])['rank'])

    def test_skill_provenance(self):
        skills = build_skills(self.tree, {})
        self.assertEqual(len(skills), 2)
        paths = [p['node_ids'] for s in skills for p in s.trajectories]
        self.assertIn(['r', 'a', 'c'], paths)
        self.assertIn(['r', 'b'], paths)
        for s in skills:
            self.assertEqual(s.source_run_ids, ['run'])
            self.assertEqual(s.source, 'fixture')
            self.assertTrue(set(s.source_node_ids) <= self.tree.nodes.keys())
            self.assertEqual(skill_document(s)['kind'], 'skill')

    def test_retrieval_provenance_diversity_and_exclusion(self):
        docs = [atom_document(a) for a in build_atoms(self.tree, {})]
        retriever = Retriever(docs)
        memories = retriever.search('evaluate loss model', 3, structure='linear')
        self.assertEqual({m.outcome_label for m in memories}, {'success', 'failure'})
        self.assertTrue(all(m.source_run == 'run' and m.raw_text for m in memories))
        self.assertTrue(any(m.selection_reason == 'structural_alternative' for m in memories))
        self.assertEqual(retriever.search('loss', exclude_runs={'run'}), [])
        self.assertEqual(retriever.search('loss', exclude_tasks={'task'}), [])
        self.assertEqual(retriever.search('loss', source='unrelated'), [])

    def test_hidden_score_id_metadata_and_child_count_do_not_leak(self):
        nodes = [self.tree.nodes['r'], replace(self.tree.nodes['a'], score=99999,
            node_id='secret-999', result_summary='SECRET'), ExperienceNode('new-hidden', 'r', 5)]
        other = SearchTree(replace(self.tree.run, metadata={'source_structure': 'partial_tree', 'best': 99999}), nodes)
        args = (['r'], set(), Counter(), self.config, 0, None, '', 0)
        first, second = policy_context(self.tree, *args), policy_context(other, *args)
        self.assertEqual(first, second)
        self.assertEqual(first['nodes'][0]['children_count'], 0)
        self.assertEqual(first['nodes'][0]['node_id'], 'P0000')
        self.assertNotIn('SECRET', json.dumps(second))

    def test_same_run_memory_cannot_leak(self):
        documents = [atom_document(a) for a in build_atoms(self.tree, {})]
        args = (['r'], set(), Counter(), self.config, 0)
        context = policy_context(self.tree, *args, Retriever(documents), '', 0)
        self.assertEqual(context['memories'], [])

    def test_prefix_reveal_only_real_edges(self):
        snapshots = []
        def policy(ctx):
            snapshots.append(ctx)
            return choose(ctx, self.config, 'dfs')
        result = replay(self.tree, policy, self.config)
        self.assertEqual(len(snapshots[0]['nodes']), 1)
        revealed = {'r'}
        for step in result.trace:
            if step['revealed']:
                self.assertIn(self.tree.nodes[step['revealed']].parent_id, revealed)
                revealed.add(step['revealed'])
        self.assertTrue(set(result.observed_ids) <= self.tree.nodes.keys())
        self.assertGreater(result.empty_probes, 0)

    def test_score_direction_and_metrics(self):
        self.assertEqual(attainment(self.tree, ['r', 'a', 'c'])['rank'], 1)
        self.assertEqual(attainment(fixture('minimize'), ['r', 'b'])['rank'], 1)
        self.assertEqual(attainment(fixture('minimize'), ['r', 'a', 'c'])['raw_best'], {'loss:minimize': 1})

    def test_stop_and_invalid_policy(self):
        result = replay(self.tree, lambda c: PolicyDecision('STOP'), self.config)
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.observed_ids, ['r'])
        context = policy_context(self.tree, ['r'], set(), Counter(), self.config, 0, None, '', 0)
        for decision in [PolicyDecision('BOGUS'), PolicyDecision('DEEPEN', 'hidden'), PolicyDecision('STOP', 'P0000')]:
            with self.assertRaises(ValueError):
                validate_decision(decision, context)

    def test_all_baselines_deterministic_and_no_network(self):
        docs = [atom_document(a) for a in build_atoms(self.tree, {})]
        for mode in ['random', 'dfs', 'bfs', 'current_best', 'meta_free', 'meta_memory']:
            with patch('socket.socket', side_effect=AssertionError('Network forbidden')):
                first = replay(self.tree, lambda c: choose(c, self.config, mode), self.config, Retriever(docs))
                second = replay(self.tree, lambda c: choose(c, self.config, mode), self.config, Retriever(docs))
                self.assertEqual(asdict(first), asdict(second))
        self.assertEqual(delexicalize('keep control=1', {}), 'keep control=1')

    def test_candidate_sandbox(self):
        candidate = Candidate((ROOT / 'autoresearch/policy_candidate.py').read_text())
        result = replay(self.tree, candidate, self.config)
        self.assertGreater(result.reveals, 0)
        for source in ["import os\ndef choose(ctx): return {}", "def choose(ctx): return open('x').read()",
                       "def choose(ctx): return ctx.__class__", "def choose(ctx):\n while True: pass",
                       "def choose(ctx):\n ctx['nodes'] = []\n return {}"]:
            with self.assertRaises(ValueError):
                Candidate(source)

    def test_output_boundary(self):
        with self.assertRaises(ValueError):
            output_path('../experience_store/experience.db')

    def test_readonly_db_and_immutable_evaluator(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            root = Path(folder)
            source = root / 'fixture.db'
            with sqlite3.connect(source) as con:
                con.executescript('CREATE TABLE runs(run_id TEXT,task_name TEXT,metadata_json TEXT);'
                    'CREATE TABLE experiences(node_id TEXT,parent_id TEXT,run_id TEXT,sequence_index INTEGER);'
                    "INSERT INTO runs VALUES('x','task','{}'); INSERT INTO experiences VALUES('r',NULL,'x',0);")
            with db.connect(source) as con:
                with self.assertRaises(sqlite3.OperationalError):
                    con.execute("INSERT INTO runs VALUES('y','task','{}')")
            for directory in ['src', 'autoresearch', 'configs', 'tests']:
                (root / directory).mkdir()
            (root / 'pyproject.toml').write_text('')
            fixed = root / 'src/fixed.py'; fixed.write_text('original')
            study = root / 'study'
            prepare(source, ['x'], [], self.config, study, root)
            verify(study, root)
            fixed.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Fixed evaluator'):
                verify(study, root)
            fixed.write_text('original')
            (study / 'memory.json').write_text('[{"leak":true}]')
            with self.assertRaisesRegex(ValueError, 'Frozen input'):
                verify(study, root)


if __name__ == '__main__':
    unittest.main()
