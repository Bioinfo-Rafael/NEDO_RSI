#!/usr/bin/env bash
# Every figure the report embeds, in dependency order. Run after
# scripts/reproduce.sh (data, setpoints, gains, baseline, demo run) and,
# optionally, after the Codex loop.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p figs

echo "==> speed (run alone; timings next to other jobs measure contention)"
$PY scripts/measure_speed.py

echo "==> flow diagram, plant population, dead time"
$PY scripts/fig_flow.py
$PY scripts/fig_plants.py
$PY scripts/fig_deadtime.py

echo "==> control: tracking, actions, constraints"
$PY scripts/fig_control.py
$PY - <<'PYEOF'
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import fig_report
print(fig_report.fig_constraints(Path("figs/constraints.png")))
PYEOF

if ls control_research/.*/git >/dev/null 2>&1; then
  echo "==> Codex history"
  $PY scripts/extract_codex_history.py
fi

echo "==> movie (slow: a full closed-loop run)"
$PY scripts/make_control_movie.py --steps 400 --out figs/control_movie.gif
$PY - <<'PYEOF'
# web-sized copy for the report
from PIL import Image, ImageSequence
im = Image.open("figs/control_movie.gif")
frames = [f.copy().convert("P", palette=Image.ADAPTIVE) for i, f in enumerate(ImageSequence.Iterator(im)) if i % 2 == 0]
frames[0].save("figs/control_movie_web.gif", save_all=True, append_images=frames[1:], loop=0,
               duration=im.info.get("duration", 80) * 2, optimize=True)
import os; print("figs/control_movie_web.gif", os.path.getsize("figs/control_movie_web.gif") // 1024, "KB")
PYEOF

echo "==> report"
$PY scripts/build_report.py
