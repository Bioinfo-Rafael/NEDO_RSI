# ORIGIN: ORIGINAL
# IMPLEMENTATION_NOTE: Fixed AST interpreter, evaluation and keep/discard bookkeeping.
"""No exec/eval/import of candidate code; it cannot access filesystem or evaluator."""
import ast
import fcntl
from dataclasses import asdict
import hashlib
import json
import math
import operator
from pathlib import Path
import time
from meta_search_rsi.policy import PolicyDecision, choose
from meta_search_rsi.replay import replay, summarize
from meta_search_rsi.retrieval import Retriever
from meta_search_rsi.tree import from_records
from .prepare import ROOT, verify


class Candidate:
    """Interpret one choose(ctx) function with finite fuel and plain JSON values."""

    def __init__(self, source: str, timeout: float = 2):
        self.timeout = timeout
        if len(source) > 20000:
            raise ValueError('Candidate too large')
        module = ast.parse(source)
        funcs = [n for n in module.body if isinstance(n, ast.FunctionDef)]
        if len(module.body) != 1 or len(funcs) != 1 or funcs[0].name != 'choose':
            raise ValueError('Exactly one choose(ctx) function required')
        self.function = funcs[0]
        args = self.function.args
        if ([a.arg for a in args.args] != ['ctx'] or args.defaults or args.vararg or args.kwarg
                or args.posonlyargs or args.kwonlyargs or self.function.decorator_list):
            raise ValueError('Expected undecorated choose(ctx)')
        allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Expr, ast.Constant,
                   ast.Assign, ast.Name, ast.Load, ast.Store, ast.If, ast.For, ast.Return,
                   ast.Subscript, ast.Dict, ast.List, ast.Tuple, ast.BinOp, ast.UnaryOp,
                   ast.BoolOp, ast.Compare, ast.IfExp, ast.Add, ast.Sub, ast.Mult, ast.Div,
                   ast.Mod, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq,
                   ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn)
        for node in ast.walk(module):
            if not isinstance(node, allowed):
                raise ValueError(f'Unsupported candidate syntax: {type(node).__name__}')
            if isinstance(node, ast.Name) and node.id.startswith('_'):
                raise ValueError('Private names forbidden')
            if isinstance(node, ast.Assign) and (len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name)):
                raise ValueError('Only local-name assignment allowed')
            if isinstance(node, ast.For) and (not isinstance(node.target, ast.Name) or node.orelse):
                raise ValueError('Only simple finite for loops allowed')

    def __call__(self, context: dict) -> PolicyDecision:
        """Evaluate with no globals, no Python calls, and a fresh local environment."""
        self.fuel = 100000
        self.deadline = time.monotonic() + self.timeout
        self.env = {'ctx': context}
        returned, result = self._block(self.function.body)
        if not returned or not isinstance(result, dict):
            raise ValueError('Candidate must return a decision dictionary')
        return PolicyDecision(**result)

    def _tick(self) -> None:
        self.fuel -= 1
        if self.fuel < 0 or time.monotonic() > self.deadline:
            raise ValueError('Candidate computation budget exceeded')

    def _block(self, statements: list[ast.stmt]) -> tuple[bool, object]:
        for node in statements:
            self._tick()
            if isinstance(node, ast.Return):
                return True, self._expr(node.value)
            if isinstance(node, ast.Assign):
                self.env[node.targets[0].id] = self._expr(node.value)
            elif isinstance(node, ast.If):
                result = self._block(node.body if self._expr(node.test) else node.orelse)
                if result[0]:
                    return result
            elif isinstance(node, ast.For):
                values = self._expr(node.iter)
                if not isinstance(values, (list, tuple)) or len(values) > 10000:
                    raise ValueError('Loop must iterate over a bounded JSON array')
                for value in values:
                    self.env[node.target.id] = value
                    result = self._block(node.body)
                    if result[0]:
                        return result
            elif isinstance(node, ast.Expr):
                self._expr(node.value)  # docstring only; call nodes are not supported
        return False, None

    def _expr(self, node: ast.AST) -> object:
        self._tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.env[node.id]
        if isinstance(node, ast.Subscript):
            return self._expr(node.value)[self._expr(node.slice)]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._expr(x) for x in node.elts]
        if isinstance(node, ast.Dict):
            return {self._expr(k): self._expr(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.IfExp):
            return self._expr(node.body if self._expr(node.test) else node.orelse)
        if isinstance(node, ast.BoolOp):
            value = self._expr(node.values[0])
            for part in node.values[1:]:
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    break
                value = self._expr(part)
            return value
        if isinstance(node, ast.UnaryOp):
            return {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}[type(node.op)](self._expr(node.operand))
        if isinstance(node, ast.BinOp):
            left, right = self._expr(node.left), self._expr(node.right)
            if type(left) not in (int, float) or type(right) not in (int, float):
                raise ValueError('Arithmetic is numeric only')
            op = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.Mod: operator.mod}[type(node.op)]
            value = op(left, right)
            if not math.isfinite(value) or abs(value) > 1e12:
                raise ValueError('Arithmetic outside finite bounds')
            return value
        if isinstance(node, ast.Compare):
            ops = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                   ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                   ast.Is: operator.is_, ast.IsNot: operator.is_not,
                   ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
            left = self._expr(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = self._expr(right_node)
                if not ops[type(op)](left, right):
                    return False
                left = right
            return True
        raise ValueError(f'Unsupported expression: {type(node).__name__}')


def evaluate_study(study: Path, label: str, candidate_path: Path, root: Path = ROOT,
                   hypothesis: str = '') -> dict:
    """Serialize trials; record invalid candidates and retain only strict improvements."""
    verify(study, root)
    with (study / '.evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another evaluation is running in this study') from error
        return _evaluate_trial(study, label, candidate_path, root, hypothesis)


def _evaluate_trial(study: Path, label: str, candidate_path: Path, root: Path,
                    hypothesis: str) -> dict:
    if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
        raise ValueError('Use a simple alphanumeric label')
    source = candidate_path.read_text()
    config = json.loads((study / 'config.json').read_text())
    destination = study / 'trials' / label
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'policy_candidate.py').write_text(source)
    cases = json.loads((study / 'cases.json').read_text())
    memory = Retriever(json.loads((study / 'memory.json').read_text()), config['bm25_k1'], config['bm25_b'])
    results = {}
    record = {'label': label, 'hypothesis': hypothesis, 'status': 'crash', 'objective': None,
              'candidate_sha256': hashlib.sha256(source.encode()).hexdigest(), 'results': results}
    # Baseline errors are evaluator errors, never attributed to candidate code.
    for mode in ['random', 'dfs', 'bfs', 'current_best', 'meta_free', 'meta_memory']:
        results[mode] = _evaluate_cases(cases, config, memory, mode,
                                     lambda c, m=mode: choose(c, config, m))
    try:
        candidate = Candidate(source, config['candidate_timeout'])
        results['candidate'] = _evaluate_cases(cases, config, memory, 'candidate', candidate)
        record['objective'] = results['candidate']['summary']['mean_objective']
        best_path = study / 'best.json'
        previous = json.loads(best_path.read_text()) if best_path.exists() else None
        objective = record['objective']
        record['status'] = ('keep' if objective is not None and
                            (previous is None or objective > previous['objective']) else 'discard')
    except (ValueError, SyntaxError, KeyError, TypeError, IndexError, ArithmeticError) as error:
        record['error'] = f'{type(error).__name__}: {error}'
    # Changes to fixed inputs must raise, even when the candidate crashes.
    verify(study, root)
    _atomic_json(destination / 'result.json', record)
    if record['status'] == 'keep':
        # The immutable trial snapshot is authoritative; best_policy.py is a convenience copy.
        best_copy = study / 'best_policy.py.tmp'
        best_copy.write_text(source)
        best_copy.replace(study / 'best_policy.py')
        _atomic_json(study / 'best.json', {'label': label, 'objective': record['objective'],
                     'candidate_sha256': record['candidate_sha256']})
    with (study / 'results.jsonl').open('a') as log:
        log.write(json.dumps({k: v for k, v in record.items() if k != 'results'}) + '\n')
    return record


def _evaluate_cases(cases: list[dict], config: dict, memory: Retriever, mode: str, policy) -> dict:
    values = []
    for case in cases:
        tree = from_records(case['run'], case['nodes'], case['source'])
        values.append(replay(tree, policy, config,
                            memory if mode in ('meta_memory', 'candidate') else None, name=mode))
    return {'summary': summarize(values), 'runs': [asdict(v) for v in values]}


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)
