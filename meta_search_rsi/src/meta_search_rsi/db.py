# ORIGIN: ORIGINAL
# IMPLEMENTATION_NOTE: SQL-only adapter for the existing three-table SQLite DB.
"""Read records without creating or modifying an SQLite file."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator


@contextmanager
def connect(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open an existing database read-only; close even if a query fails."""
    uri = Path(path).resolve().as_uri() + '?mode=ro'
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    try:
        yield connection
    finally:
        connection.close()


def list_runs(path: str | Path, pattern: str = '*', limit: int = 10) -> list[dict]:
    """List bounded run metadata; GLOB can use the run_id index."""
    if limit < 1:
        raise ValueError('limit must be positive')
    with connect(path) as db:
        rows = db.execute('SELECT run_id,task_name,source_type FROM runs '
                          'WHERE run_id GLOB ? ORDER BY run_id LIMIT ?', (pattern, limit))
        return [dict(row) for row in rows]


def read_run(path: str | Path, run_id: str, max_nodes: int) -> tuple[dict, list[dict]]:
    """Read a complete bounded run; refuse truncation of its tree."""
    if max_nodes < 1:
        raise ValueError('max_nodes must be positive')
    with connect(path) as db:
        row = db.execute('SELECT * FROM runs WHERE run_id=?', (run_id,)).fetchone()
        if row is None:
            raise KeyError(f'Unknown run: {run_id}')
        nodes = db.execute('SELECT * FROM experiences WHERE run_id=? '
                           'ORDER BY sequence_index,node_id LIMIT ?', (run_id, max_nodes + 1)).fetchall()
        if len(nodes) > max_nodes:
            raise ValueError(f'Run exceeds max_nodes={max_nodes}; increase explicitly')
        return dict(row), [dict(n) for n in nodes]
