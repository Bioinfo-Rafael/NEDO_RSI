#!/usr/bin/env bash
# Everything after `pip install -e .`, in dependency order. Each step is
# idempotent and can be re-run on its own. Times are for an 8-core CPU.
#
#   1. identification data      ~35 min with 4 processes (skip if data_ctrl_v2/ is populated)
#   2. setpoints + PI gains      seconds   (derived from the data; run.py and run_control.py read them)
#   3. baseline experiment       ~6 min    (trains M for 150 s, then C drives the plant)
#   4. hold / PI / MPC demo run  ~5 min    (the target-tracking figure)
#   5. Codex loop (optional)     ~1 h      scripts/run_control_research.sh <tag> <n_experiments>
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

if [ -z "$(ls data_ctrl_v2/train/*.npz 2>/dev/null)" ]; then
  echo "==> 1. identification data (4 processes)"
  mkdir -p data_ctrl_v2
  for i in 0 1 2; do
    n=11; [ $i -eq 2 ] && n=10
    $PY -u scripts/gen_control_data.py --out data_ctrl_v2 --plant a \
        --train-start $((i*11)) --n-train $n --n-val 0 > data_ctrl_v2/gen_train_$i.log 2>&1 &
  done
  $PY -u scripts/gen_control_data.py --out data_ctrl_v2 --plant a --n-train 0 --n-val 8 \
      > data_ctrl_v2/gen_val.log 2>&1 &
  wait
fi
echo "    $(ls data_ctrl_v2/train/*.npz | wc -l) train / $(ls data_ctrl_v2/val/*.npz | wc -l) val episodes"

echo "==> 2. setpoints and PI gains from the data"
$PY scripts/pick_setpoints.py
$PY scripts/identify_pi.py

echo "==> 3. baseline experiment (M = linear state space, the floor Codex starts from)"
(cd control_research && ../$PY run.py > run.log 2>&1 && grep -E "^control_cost:|_tracking:" run.log)

echo "==> 4. hold / PI / MPC on the same setpoint step"
$PY scripts/run_control.py --model figs/cv_model.pkl --steps 400 --out figs/control_run.json

echo "done."
