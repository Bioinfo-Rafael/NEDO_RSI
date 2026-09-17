#!/bin/sh
# ORIGIN: ORIGINAL
# IMPLEMENTATION_NOTE: Convenience wrapper for the single Python CLI.
set -eu
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
exec python3 -m meta_search_rsi.cli autoresearch "$@"
