"""Regression checks against public records, with independent expectations."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,finish,node
from audit import evaluate,hard_failure
from adapters.openevolve import OpenEvolve
from adapters.rest_mcts import RestMCTS
from adapters.evo_mcts import EvoMCTS
from adapters.swe_agent import SWEAgent
from adapters.search_agents import SearchAgents,pages
from adapters.treeofthoughts import TreeOfThoughts

class PublicRecords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.open=list(OpenEvolve().bundles())
        cls.evo=list(EvoMCTS().bundles())

    def test_openevolve_roots_and_nodes(self):
        self.assertEqual(len(self.open),5)
        self.assertEqual(sum(len(b['nodes']) for b in self.open),587)
        self.assertEqual(sum(n['parent_id'] is None for b in self.open for n in b['nodes']),5)

    def test_openevolve_parents_are_explicit(self):
        for b in self.open:
            self.assertFalse(hard_failure(evaluate(b)))
            for n in b['nodes']:
                p=ROOT/json.loads(n['source_files_json'])[0];d=json.loads(p.read_text())
                expected=b['run']['run_id']+'::'+d['parent_id'] if d['parent_id'] else None
                self.assertEqual(n['parent_id'],expected)
                self.assertEqual(n['result_summary'],d['code'])
                self.assertEqual(n['score'],d['metrics'].get('combined_score'))

    def test_evo_known_parent_and_paper_records(self):
        b=next(b for b in self.evo if b['run']['run_id']=='evomcts::run1_april_2025')
        by={json.loads(n['metadata_json'])['original_node_id']:n for n in b['nodes']}
        self.assertEqual(by['28']['parent_id'],'evomcts::run1_april_2025::12')
        self.assertEqual(sum(e['event_type']=='paper_node' for e in b['events']),38)
        for n in b['nodes']:
            m=json.loads(n['metadata_json'])
            if n['parent_id']:
                self.assertIsNotNone(m['parent_evidence'])
        self.assertTrue(any(n['parent_id'] is None and json.loads(n['metadata_json'])['parent_relation']=='unknown' for n in b['nodes']))

    def test_rest_all_file_formats(self):
        parser=RestMCTS()
        for p in parser.discover():
            d=next(parser.load_raw(p));bundles=list(parser.parse_record(p,1,d))
            self.assertTrue(bundles,p)
            for b in bundles:
                self.assertFalse(hard_failure(evaluate(b)),p)
                for n in b['nodes'][:-1]:self.assertIsNone(n['score'])
                if 'label' not in d:
                    self.assertTrue(all(n['score'] is None for n in b['nodes']))
                for i,n in enumerate(b['nodes']):
                    self.assertEqual(n['parent_id'],b['nodes'][i-1]['node_id'] if i else None)

    def test_rest_split_preserves_reasoning_exactly(self):
        p=next((ROOT/'raw/rest_mcts').glob('*Llama3*Policy_1st/*.json'))
        parser=RestMCTS();d=next(parser.load_raw(p));b=next(parser.parse_record(p,1,d))
        self.assertGreater(len(b['nodes']),1)
        self.assertEqual(''.join(n['proposal'] or '' for n in b['nodes']),d['output'])
        self.assertEqual(b['nodes'][-1]['result_summary'],d['output'])

    def test_swe_no_invented_numeric_reward(self):
        b=next(SWEAgent().smith())
        self.assertGreater(len(b['nodes']),1)
        self.assertTrue(all(n['score'] is None for n in b['nodes']))
        self.assertIn('resolved',json.loads(b['nodes'][-1]['metrics_json']))
        self.assertFalse(hard_failure(evaluate(b)))

    def test_swe_all_four_encodings(self):
        import pyarrow.parquet as pq
        for encoding in ('tool','xml','ticks','train'):
            p=next((ROOT/'raw/swe_agent').glob('**/'+encoding+'-*.parquet'))
            d=next(pq.ParquetFile(p).iter_batches(batch_size=1)).to_pylist()[0]
            b=SWEAgent().parse_messages(p,1,d)
            messages=json.loads(d['messages']) if isinstance(d['messages'],str) else d['messages']
            self.assertEqual(len(b['nodes']),sum(m['role']=='assistant' for m in messages))
            self.assertFalse(hard_failure(evaluate(b)))

    def test_search_preserves_unselected_candidates(self):
        b=next(SearchAgents().bundles());m=evaluate(b)
        self.assertGreater(m['nodes'],m['result'])
        self.assertTrue(any(n['node_type']=='search_candidate' for n in b['nodes']))
        self.assertFalse(hard_failure(m))

    def test_tot_siblings_and_branching(self):
        b=next(TreeOfThoughts().bundles());counts={}
        for n in b['nodes']:
            if n['parent_id']:counts[n['parent_id']]=counts.get(n['parent_id'],0)+1
        self.assertGreater(max(counts.values()),1)
        self.assertFalse(hard_failure(evaluate(b)))

    def test_invented_score_is_hard_failure(self):
        b=copy.deepcopy(self.open[0]);b['nodes'][0]['score']=123456789
        self.assertTrue(hard_failure(evaluate(b)))

    def test_invalid_parent_is_hard_failure(self):
        b=copy.deepcopy(self.open[0]);b['nodes'][0]['parent_id']='missing'
        self.assertTrue(hard_failure(evaluate(b)))

    def test_cycle_is_hard_failure(self):
        b=copy.deepcopy(self.open[0]);b['nodes'][0]['parent_id']=b['nodes'][0]['node_id']
        self.assertTrue(hard_failure(evaluate(b)))

    def test_duplicate_is_hard_failure(self):
        b=copy.deepcopy(self.open[0]);b['nodes'].append(b['nodes'][0])
        self.assertTrue(hard_failure(evaluate(b)))

    def test_parsers_are_offline(self):
        with patch.object(socket,'create_connection',side_effect=AssertionError('network forbidden')),patch.object(socket.socket,'connect',side_effect=AssertionError('network forbidden')):
            for cls in (OpenEvolve,RestMCTS,EvoMCTS,SWEAgent,SearchAgents,TreeOfThoughts):
                b=next(cls().bundles());self.assertTrue(b['nodes'])

if __name__=='__main__':unittest.main()
