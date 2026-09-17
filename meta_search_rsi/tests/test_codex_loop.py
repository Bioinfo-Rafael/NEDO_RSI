# ORIGIN: ORIGINAL
# IMPLEMENTATION_NOTE: Offline integration tests; no real Codex/model invocation.
"""Exercise proposal transport, evaluation, recovery and immutable boundaries."""
import contextlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meta_search_rsi.cli import configuration
from autoresearch.prepare import prepare
from autoresearch.evaluate import evaluate_study
from autoresearch.codex_loop import run_loop, best_source

STOP = "# ORIGIN: ORIGINAL\ndef choose(ctx):\n    return {'action': 'STOP'}\n"
EXPLORE = """# ORIGIN: ORIGINAL
def choose(ctx):
    if not ctx['frontier']:
        return {'action': 'STOP'}
    return {'action': 'WIDEN', 'target_node_id': ctx['frontier'][0]['node_id']}
"""


class CodexLoopTests(unittest.TestCase):
    """Test the real host pipeline with tiny read-only DBs and a fake executable."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'outputs')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ['src', 'autoresearch', 'tests', 'configs', 'outputs']:
            (self.root / directory).mkdir()
        (self.root / 'pyproject.toml').write_text('')
        (self.root / 'autoresearch/program.md').write_text('Use only observed frontier.')
        (self.root / 'autoresearch/policy_candidate.py').write_text(STOP)
        self.source = self.root / 'source.db'
        with sqlite3.connect(self.source) as con:
            con.executescript('CREATE TABLE runs(run_id TEXT,task_name TEXT,metadata_json TEXT);'
                'CREATE TABLE experiences(node_id TEXT,parent_id TEXT,run_id TEXT,sequence_index INTEGER,'
                'score REAL,score_name TEXT,score_direction TEXT);'
                "INSERT INTO runs VALUES('x','task','{}');"
                "INSERT INTO experiences VALUES('r',NULL,'x',0,0,'quality','maximize');"
                "INSERT INTO experiences VALUES('a','r','x',1,1,'quality','maximize');"
                "INSERT INTO experiences VALUES('b','r','x',2,2,'quality','maximize');")
        self.study = self.root / 'outputs/study'
        config = configuration(); config['budget'] = 3
        prepare(self.source, ['x'], [], config, self.study, self.root)
        self.proposal = self.root / 'outputs/proposal.json'

    def run_quiet(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_loop(self.study, root=self.root, **kwargs)

    def propose(self, code):
        self.proposal.write_text(json.dumps({'hypothesis': 'Try recorded alternatives', 'code': code}))
        return self.run_quiet(iterations=1, proposal_file=self.proposal)

    def test_dry_run_does_not_start_process(self):
        with patch('subprocess.Popen', side_effect=AssertionError('must remain offline')):
            result = self.run_quiet(dry_run=True)
        attempt = result['iterations'][0]
        self.assertEqual(attempt['status'], 'dry_run')
        self.assertIn('--sandbox', attempt['command'])
        self.assertIn('read-only', attempt['command'])
        self.assertNotIn('--dangerously-bypass-approvals-and-sandbox', attempt['command'])
        prompt = (Path(attempt['directory']) / 'prompt.md').read_text()
        self.assertNotIn(str(self.source), prompt)
        self.assertIn('ADOPTED CODE', prompt)

    def test_manual_keep_discard_crash_and_resume(self):
        first = self.propose(EXPLORE)
        self.assertEqual(first['iterations'][0]['status'], 'keep')
        objective = first['best']['objective']
        self.assertEqual(self.propose(EXPLORE)['iterations'][0]['status'], 'discard')
        broken = '# ORIGIN: ORIGINAL\ndef choose(ctx):\n    return ctx["nonexistent"]\n'
        self.assertEqual(self.propose(broken)['iterations'][0]['status'], 'crash')
        self.assertEqual(best_source(self.study), EXPLORE)
        self.assertEqual(json.loads((self.study / 'best.json').read_text())['objective'], objective)
        self.assertEqual((self.root / 'autoresearch/policy_candidate.py').read_text(), STOP)
        self.assertEqual(len(list((self.study / 'codex').iterdir())), 3)

    def test_invalid_json_preserves_best(self):
        self.run_quiet(dry_run=True)
        self.proposal.write_text('not json')
        result = self.run_quiet(iterations=1, proposal_file=self.proposal)
        self.assertEqual(result['iterations'][0]['status'], 'invalid_proposal')
        self.assertEqual(best_source(self.study), STOP)

    def test_syntax_error_is_recorded_as_trial(self):
        result = self.propose('# ORIGIN: ORIGINAL\ndef broken(\n')
        self.assertEqual(result['iterations'][0]['status'], 'crash')
        self.assertIn('SyntaxError', result['iterations'][0]['error'])

    def fake_codex(self, mode='success'):
        path = self.root / 'outputs/fake-codex'
        source = f'''#!{sys.executable}
import json,os,sys,time
from pathlib import Path
args=sys.argv[1:]
json.loads(Path(args[args.index('--output-schema')+1]).read_text())
sys.stdin.read()
Path('environment.json').write_text(json.dumps({{'home':os.environ['CODEX_HOME'],
 'has_api_key':'CODEX_API_KEY' in os.environ or 'OPENAI_API_KEY' in os.environ}}))
if {mode!r}=='timeout': time.sleep(10)
if {mode!r}=='exit': sys.exit(7)
Path(args[args.index('--output-last-message')+1]).write_text({json.dumps({'hypothesis':'Fake test only','code':EXPLORE})!r})
print('{{"type":"turn.completed"}}')
'''
        path.write_text(source); path.chmod(0o755)
        return str(path)

    def test_fake_cli_end_to_end_and_auth_isolation(self):
        with patch.dict(os.environ, {'CODEX_HOME':'/do-not-touch', 'OPENAI_API_KEY':'fake-test', 'CODEX_API_KEY':'fake-test'}):
            result = self.run_quiet(iterations=2, executable=self.fake_codex())
        self.assertEqual([r['status'] for r in result['iterations']], ['keep', 'discard'])
        attempt = Path(result['iterations'][0]['directory'])
        environment = json.loads((attempt / 'environment.json').read_text())
        self.assertEqual(environment['home'], str(self.root / 'outputs/codex-home'))
        self.assertFalse(environment['has_api_key'])
        self.assertTrue((attempt / 'events.jsonl').exists())
        self.assertEqual(best_source(self.study), EXPLORE)

    def test_timeout_stops_loop_and_keeps_seed(self):
        result = self.run_quiet(iterations=3, executable=self.fake_codex('timeout'), timeout=0.1)
        self.assertEqual(len(result['iterations']), 1)
        self.assertEqual(result['iterations'][0]['status'], 'codex_error')
        self.assertIn('TimeoutExpired', result['iterations'][0]['error'])
        self.assertEqual(best_source(self.study), STOP)

    def test_codex_exit_stops_loop(self):
        result = self.run_quiet(iterations=3, executable=self.fake_codex('exit'))
        self.assertEqual(len(result['iterations']), 1)
        self.assertEqual(result['iterations'][0]['status'], 'codex_error')
        self.assertIn('exited 7', result['iterations'][0]['error'])

    def test_fixed_file_tampering_rejected(self):
        self.run_quiet(dry_run=True)
        (self.root / 'src/new.py').write_text('# added evaluator code')
        with self.assertRaisesRegex(ValueError, 'Fixed evaluator'):
            self.propose(EXPLORE)

    def test_changed_best_snapshot_rejected(self):
        self.run_quiet(dry_run=True)
        best = json.loads((self.study / 'best.json').read_text())
        (self.study / 'trials' / best['label'] / 'policy_candidate.py').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            self.run_quiet(dry_run=True)

    def test_reject_unbounded_count_and_parallel_loop(self):
        with self.assertRaises(ValueError):
            self.run_quiet(iterations=0)
        import fcntl
        with (self.study / '.codex-loop.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, 'already running'):
                self.run_quiet(dry_run=True)


if __name__ == '__main__':
    unittest.main()
