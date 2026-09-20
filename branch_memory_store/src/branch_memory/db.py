"""Source access, evidence manifests, and exact SQLite/storage audits."""
import collections
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
PROTECTED = [WORKSPACE / x for x in ('experience_store', 'public_experience', 'dream_rsi_experience_store')]
TABLES = ('runs', 'experiences', 'raw_events')


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def output(path):
    p = Path(path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('All writes must remain under branch_memory_store')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path, value):
    p = output(path)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(dumps(value) + '\n')
    temp.replace(p)


def connect(path):
    c = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA query_only=ON')
    c.execute('PRAGMA temp_store=MEMORY')
    return c


def source_paths():
    return [WORKSPACE / 'experience_store/experience.db'] + sorted((WORKSPACE / 'public_experience/normalized').glob('*.db'))


def protected_snapshot():
    result = {}
    for root in PROTECTED:
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in files:
                p = Path(directory) / name
                s = p.lstat()
                result[str(p.relative_to(WORKSPACE))] = [s.st_size, s.st_mtime_ns]
    return result


def audit():
    out = ROOT / 'outputs'
    before = out / 'protected_before.json'
    if not before.exists():
        write_json(before, protected_snapshot())
    reports = []
    for p in source_paths() + [WORKSPACE/'public_experience/public_experience.db', WORKSPACE/'public_experience/combined_experience.db']:
        destination = out / 'audit' / (p.stem + ('_local' if p.parent.name == 'experience_store' else '') + '.json')
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved['stat'] == [p.stat().st_size, p.stat().st_mtime_ns]:
                reports.append(saved)
                continue
            raise ValueError('Source changed since audit: ' + str(p))
        print('AUDIT', str(p.relative_to(WORKSPACE)), flush=True)
        start = time.monotonic()
        c = connect(p)
        report = {'path': str(p.relative_to(WORKSPACE)), 'bytes': p.stat().st_size,
                  'stat': [p.stat().st_size, p.stat().st_mtime_ns], 'sha256': file_hash(p)}
        report['sqlite_master'] = [dict(r) for r in c.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name')]
        report['schema'] = {t: {k: [dict(r) for r in c.execute(f'PRAGMA {k}("{t}")')]
                                  for k in ('table_info', 'foreign_key_list', 'index_list')} for t in TABLES}
        report['counts'] = {t: c.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in TABLES}
        try:
            report['storage_pages'] = [dict(r) for r in c.execute('SELECT name,sum(pgsize) bytes,sum(payload) payload,sum(unused) unused FROM dbstat GROUP BY name')]
        except sqlite3.OperationalError as e:
            report['storage_pages_error'] = str(e)
        report['page_size'] = c.execute('PRAGMA page_size').fetchone()[0]
        report['freelist_count'] = c.execute('PRAGMA freelist_count').fetchone()[0]
        if p.parent.name in ('normalized', 'experience_store'):
            # Parent must actually exist in the same run. No artificial shared root.
            points = [dict(r) for r in c.execute('''SELECT p.run_id,p.node_id,p.depth,p.source_type,
                (SELECT count(*) FROM experiences c WHERE c.parent_id=p.node_id AND c.run_id=p.run_id) children
                FROM (SELECT parent_id FROM experiences WHERE parent_id IS NOT NULL
                      GROUP BY parent_id HAVING count(*)>=2) g
                JOIN experiences p ON p.node_id=g.parent_id
                WHERE children>=2 ORDER BY p.run_id,p.node_id''')]
            report['branch_points'] = points
            report['runs_with_branch'] = len({r['run_id'] for r in points})
            report['linear_only_runs'] = report['counts']['runs'] - report['runs_with_branch']
            report['run_sources'] = [dict(r) for r in c.execute('SELECT source_type,count(*) count FROM runs GROUP BY source_type')]
            # FK validates missing parents without repeatedly loading large TEXT payload pages.
            report['foreign_key_violations'] = [tuple(r) for r in c.execute('PRAGMA foreign_key_check')]
            report['columns'] = {}
            for table in TABLES:
                cols = [r['name'] for r in report['schema'][table]['table_info'] if r['type'] == 'TEXT']
                expressions = [f'coalesce(sum(length(cast("{col}" AS BLOB))),0) AS "{col}"' for col in cols]
                # Exact UTF-8 payload lengths; includes NULL as zero, excludes SQLite record/index overhead.
                report['columns'][table] = dict(c.execute(f'SELECT {",".join(expressions)} FROM {table}').fetchone())
        c.close()
        report['elapsed_seconds'] = time.monotonic() - start
        write_json(destination, report)
        reports.append(report)
        print('DONE', p.stem, report['counts'], 'branches', len(report.get('branch_points', [])), flush=True)
    write_json(out / 'audit.json', reports)
    return reports


def verify_protected(full_hash=False):
    before = json.loads((ROOT/'outputs/protected_before.json').read_text())
    now = protected_snapshot()
    changed = [p for p in before.keys() | now.keys() if before.get(p) != now.get(p)]
    hashed = 0
    if full_hash:
        for report in json.loads((ROOT/'outputs/audit.json').read_text()):
            if file_hash(WORKSPACE/report['path']) != report['sha256']:
                changed.append(report['path'])
            hashed += 1
    result = {'original_files_modified_count': len(set(changed)), 'original_db_modified_count': len({p for p in changed if p.endswith('.db')}),
              'changed': sorted(set(changed)), 'protected_files': len(before), 'db_full_hash_checks': hashed}
    write_json(ROOT/'outputs/protected_verification.json', result)
    if changed:
        raise ValueError(result)
    return result
