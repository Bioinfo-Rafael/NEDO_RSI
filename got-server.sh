#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/Graph-of-Trace"
export GOT_OUTPUT_BASE_DIR="$PWD/runs"
export GOT_OUTPUT_PATH_TEMPLATE='{base_dir}/{project_name}/{session_id}/got.json'
exec .venv/bin/python server.py "$@"
