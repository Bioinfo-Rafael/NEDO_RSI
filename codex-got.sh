#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export CODEX_HOME="$ROOT/.codex-got"
exec codex "$@"
