#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    if command -v python3.12 >/dev/null 2>&1; then
        csv_python=$(command -v python3.12)
    elif [ -x "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12" ]; then
        csv_python="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12"
    else
        echo 'Python 3.12 is required. Install it, then rerun: bash setup.sh' >&2
        exit 1
    fi
    "$csv_python" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
echo 'Ready. Prepare data: .venv/bin/python prepare.py'
