PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  project_name TEXT,
  task_name TEXT,
  root_node_id TEXT,
  status TEXT,
  start_time TEXT,
  end_time TEXT,
  replay_eligible INTEGER NOT NULL DEFAULT 0 CHECK (replay_eligible IN (0, 1)),
  source_type TEXT,
  source_reference TEXT,
  git_branch TEXT,
  codex_thread_ids_json TEXT NOT NULL DEFAULT '[]',
  original_session_ids_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS experiences (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  node_id TEXT PRIMARY KEY,
  parent_id TEXT REFERENCES experiences(node_id),
  branch_id TEXT,
  attempt_index INTEGER,
  sequence_index INTEGER NOT NULL,
  depth INTEGER NOT NULL,
  display_path TEXT NOT NULL,
  node_type TEXT NOT NULL DEFAULT 'unknown',
  proposal TEXT,
  prompt TEXT,
  result_summary TEXT,
  status TEXT,
  score REAL,
  score_name TEXT,
  score_direction TEXT NOT NULL DEFAULT 'unknown'
    CHECK (score_direction IN ('minimize', 'maximize', 'unknown')),
  metrics_json TEXT NOT NULL DEFAULT '{}',
  artifacts_json TEXT NOT NULL DEFAULT '[]',
  commit_hash TEXT,
  commit_hashes_json TEXT NOT NULL DEFAULT '[]',
  commit_message TEXT,
  git_commit_before TEXT,
  git_commit_after TEXT,
  event_count INTEGER NOT NULL DEFAULT 0,
  trace_start_time TEXT,
  trace_end_time TEXT,
  first_event_sequence INTEGER,
  last_event_sequence INTEGER,
  source_type TEXT,
  source_reference TEXT,
  source_files_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS raw_events (
  event_id TEXT PRIMARY KEY,
  run_id TEXT,
  node_id TEXT REFERENCES experiences(node_id),
  sequence_no INTEGER,
  timestamp TEXT,
  event_type TEXT,
  role TEXT,
  tool_name TEXT,
  command TEXT,
  cwd TEXT,
  stdout TEXT,
  stderr TEXT,
  exit_code INTEGER,
  message TEXT,
  source_file TEXT NOT NULL,
  source_line INTEGER NOT NULL,
  binding_method TEXT,
  binding_confidence REAL,
  raw_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiences_run_id ON experiences(run_id);
CREATE INDEX IF NOT EXISTS idx_experiences_parent_id ON experiences(parent_id);
CREATE INDEX IF NOT EXISTS idx_raw_events_node_id ON raw_events(node_id);
CREATE INDEX IF NOT EXISTS idx_raw_events_run_id ON raw_events(run_id);
CREATE INDEX IF NOT EXISTS idx_raw_events_sequence_no ON raw_events(sequence_no);
