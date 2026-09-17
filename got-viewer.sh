#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/Graph-of-Trace"
exec .venv/bin/python local-setup/viewer.py "$@"
