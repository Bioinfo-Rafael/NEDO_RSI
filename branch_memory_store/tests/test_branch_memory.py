import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from branch_memory.compress import compress
from branch_memory.delex import validate, cache_key, outcome, SCHEMA
from branch_memory.db import ROOT, connect, dumps, sha
from branch_memory.build import SCHEMA as DB_SCHEMA, CAP


def fixture(edges=None):
    edges=edges or [('root',None),('A','root'),('B','A'),('C','B'),('D','B'),('E','C'),('F','D'),('G','E'),('H','F'),('I','F')]
    return [{'run_id':'r','node_id':n,'parent_id':p,'sequence_index':i,'depth':0,'prompt':None,'proposal':'observe '+n,
             'result_summary':None,'score':None,'score_name':None,'score_direction':'unknown','metadata_json':'{}','source_type':'fixture'} for i,(n,p) in enumerate(edges)]


def result_for(unit):
    return {'abstract_state':None,'incoming_history':[{'node_id':n,'abstract_action':None,'abstract_outcome':None} for n in unit['incoming_node_ids']],
            'decision_context':None,'terminal_branches':[{'node_ids':p,'abstract_strategy':None,'abstract_trajectory':['unknown']*len(p),'abstract_outcome':None,'outcome_type':'unknown'} for p in unit['terminal_branch_node_ids']],
            'continuations':[{'next_branch_point_id':n,'relationship':'continues_to_next_decision_point'} for n in unit['next_branch_point_ids']],
            'search_pattern':None,'uncertainties':['test fixture']}


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.units,self.anchors,self.coverage=compress({'run_id':'r','metadata_json':'{}'},fixture(),'fixture.db')
        self.by={u['branch_point_id']:u for u in self.units}

    def test_exact_bf_fixture(self):
        self.assertEqual(set(self.by),{'B','F'})
        self.assertEqual(self.by['B']['incoming_node_ids'],['A'])
        self.assertEqual(self.by['F']['incoming_node_ids'],['D'])
        self.assertEqual(self.by['B']['terminal_branch_node_ids'],[['C','E','G']])
        self.assertEqual(self.by['F']['terminal_branch_node_ids'],[['H'],['I']])
        self.assertEqual(self.by['B']['next_branch_point_ids'],['F'])
        self.assertEqual(self.by['F']['previous_branch_point_id'],'B')
        self.assertEqual(set(self.anchors),{'root'})

    def test_single_assignment_full_coverage(self):
        assigned=[x['node_id'] for x in self.coverage]
        self.assertEqual(len(assigned),10)
        self.assertEqual(len(set(assigned)),10)
        self.assertEqual(next(x for x in self.coverage if x['node_id']=='D')['owner'],'F')

    def test_root_branch(self):
        units,anchors,_=compress({'run_id':'r'},fixture([('R',None),('A','R'),('B','R')]),'f')
        self.assertEqual([u['original_node_id'] for u in units],['R'])
        self.assertEqual(anchors,{})

    def test_linear_excluded(self):
        u,_,cov=compress({'run_id':'r'},fixture([('R',None),('A','R')]),'f')
        self.assertEqual(u,[])
        self.assertEqual({r['assignment'] for r in cov},{'excluded_linear_run'})

    def test_cycle_rejected(self):
        with self.assertRaises(ValueError):compress({'run_id':'r'},fixture([('A','B'),('B','A')]),'f')

    def test_coverage_rejects_duplicate_aggregate_input(self):
        from branch_memory.build import coverage
        with self.assertRaisesRegex(ValueError,'canonical source'):
            coverage('public_experience/combined_experience.db','any-node')

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):compress({'run_id':'r'},fixture()+[fixture()[0]],'f')

    def test_partial_forest_no_fabrication(self):
        u,a,c=compress({'run_id':'r'},fixture([('R','missing'),('A','R'),('B','R'),('X',None)]),'f')
        self.assertEqual(u[0]['raw_context']['branch_point']['parent_id'],'missing')
        self.assertEqual(u[0]['next_branch_point_ids'],[])
        self.assertEqual(len(c),4)

    def test_raw_ids_unchanged_and_hash(self):
        for u in self.units:
            self.assertEqual(u['original_node_id'],u['branch_point_id'])
            self.assertEqual(sha(dumps({k:v for k,v in u.items() if k!='source_hash'})),u['source_hash'])

    def test_malformed_or_invented_json_rejected(self):
        u=self.by['B'];valid=result_for(u)
        self.assertEqual(validate(valid,u),valid)
        for key,value in [('extra','bad'),('terminal_branches',[]),('continuations',[]),('incoming_history',[])]:
            wrong=copy.deepcopy(valid);wrong[key]=value
            with self.assertRaises(ValueError):validate(wrong,u)
        wrong=copy.deepcopy(valid);wrong['terminal_branches'][0]['outcome_type']='improved'
        with self.assertRaises(ValueError):validate(wrong,u)

    def test_cache_hash_input_prompt_model(self):
        base=cache_key({'x':1},'v1','model')
        self.assertEqual(base,cache_key({'x':1},'v1','model'))
        for data,prompt,model in [({'x':2},'v1','model'),({'x':1},'v2','model'),({'x':1},'v1','other')]:
            self.assertNotEqual(base,cache_key(data,prompt,model))

    def test_resume_retry_and_completed_skip(self):
        from branch_memory import delex
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root=Path(directory);(root/'outputs').mkdir();(root/'prompts').mkdir()
            (root/'prompts/delexicalize.md').write_text('test prompt')
            (root/'outputs/selected.json').write_text(dumps(self.units))
            (root/'outputs/runs.json').write_text(dumps([{'run':{'run_id':'r'},'root_anchors':self.anchors}]))
            with patch.object(delex,'ROOT',root),patch.object(delex.subprocess,'check_output',return_value='fake-version'):
                calls=[]
                def invoke(data,*args):
                    calls.append(data)
                    if len(calls)==1:return {},{}
                    return result_for(data['unit']),{}
                with patch.object(delex,'invoke',side_effect=invoke):
                    first=delex.process(limit=1)
                    self.assertEqual((first['attempts'],first['success']),(2,1))
                    second=delex.process(limit=1)
                    self.assertEqual((second['success'],second['cache_hit']),(1,1))
                with patch.object(delex,'invoke',side_effect=AssertionError('cache must avoid model')):
                    third=delex.process(limit=2)
                    self.assertEqual((third['success'],third['cache_hit']),(0,2))

    def test_score_comparison_requires_cohort(self):
        b={'score':.2,'score_name':'loss','score_direction':'minimize'}
        self.assertEqual(outcome(b,{**b,'score':.1}),'improved')
        self.assertEqual(outcome(b,{**b,'score':.3}),'degraded')
        self.assertEqual(outcome(b,{**b,'score_direction':'unknown'}),'unknown')
        self.assertEqual(outcome(b,{'status':'failed'}),'failed')

    def test_parallel_checkpoint(self):
        from branch_memory import delex
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root=Path(directory);(root/'outputs').mkdir();(root/'prompts').mkdir()
            (root/'prompts/delexicalize.md').write_text('parallel fixture')
            (root/'outputs/selected.json').write_text(dumps(self.units))
            (root/'outputs/runs.json').write_text(dumps([{'run':{'run_id':'r'},'root_anchors':self.anchors}]))
            with patch.object(delex,'ROOT',root),patch.object(delex.subprocess,'check_output',return_value='fake-version'),patch.object(delex,'invoke',side_effect=lambda data,*args:(result_for(data['unit']),{})):
                result=delex.process(limit=2,workers=2)
                self.assertEqual((result['success'],result['failed']),(2,0))
                self.assertEqual(len(list((root/'outputs/cache').glob('*.json'))),2)

    def test_timeout_retry_then_success(self):
        import subprocess
        from branch_memory import delex
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root=Path(directory);(root/'outputs').mkdir();(root/'prompts').mkdir()
            (root/'prompts/delexicalize.md').write_text('timeout fixture')
            (root/'outputs/selected.json').write_text(dumps(self.units[:1]))
            (root/'outputs/runs.json').write_text(dumps([{'run':{'run_id':'r'},'root_anchors':self.anchors}]))
            with patch.object(delex,'ROOT',root),patch.object(delex.subprocess,'check_output',return_value='fake-version'),patch.object(delex,'invoke',side_effect=[subprocess.TimeoutExpired('fake',1),(result_for(self.units[0]),{})]):
                result=delex.process(limit=1)
                self.assertEqual((result['success'],result['attempts']),(1,2))

    def test_original_db_read_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as d:
            p=Path(d)/'original.db';c=sqlite3.connect(p);c.execute('CREATE TABLE x(id)');c.commit();c.close()
            before=p.read_bytes();c=connect(p)
            with self.assertRaises(sqlite3.OperationalError):c.execute('INSERT INTO x VALUES(1)')
            c.close();self.assertEqual(before,p.read_bytes())

    def test_two_tables_and_cap(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as d:
            p=Path(d)/'branch.db';c=sqlite3.connect(p);c.executescript(DB_SCHEMA)
            self.assertEqual({r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")},{'runs','branch_points'})
            # Exercise SQLite's actual allocation guard at a small test cap.
            c.execute('PRAGMA max_page_count=12');c.execute('CREATE TABLE large_data(x)')
            with self.assertRaises(sqlite3.DatabaseError):c.execute('INSERT INTO large_data VALUES(zeroblob(100000))')
            c.close();self.assertLess(p.stat().st_size,CAP)


if __name__=='__main__':unittest.main()
