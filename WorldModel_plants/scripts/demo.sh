#!/usr/bin/env bash
# One-command demo: physics checks -> data -> baseline -> score -> pictures.
# Small on purpose; scripts/gen_dataset.py with the defaults is the real run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

echo "==> golden tests"
$PY -m pytest tests/ -q -x --ignore=tests/test_stability.py

echo; echo "==> dataset (small)"
$PY scripts/gen_dataset.py --out data --n-train 8 --n-val 3 --n-ood 3 --frames 40 --batch-size 8

echo; echo "==> prepare"
$PY integration/prepare.py

echo; echo "==> train baseline (1 min, deliberately weak)"
(cd integration && ../$PY train_baseline.py --minutes 1)

echo; echo "==> evaluate"
(cd integration && ../$PY evaluate.py --max-eps 3)

echo; echo "==> render"
$PY -m wmf.viz.render "$(ls data/train/*.npz | head -1)" --out out

echo; echo "done. pictures in out/, metric above is val_nrmse (lower is better)."
