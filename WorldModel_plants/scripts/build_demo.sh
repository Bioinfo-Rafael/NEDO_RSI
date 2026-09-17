#!/usr/bin/env bash
# Everything the demo needs, in order. Run after the autoresearch loop ends.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p figs

echo "==> A/B: does the geometry input help on unseen plants?"
$PY scripts/ab_geometry.py --seconds 180 --out figs/ab_geometry.json

echo
echo "==> train the final model and roll it out against the simulator"
$PY scripts/make_demo.py --seconds 180 --horizon 40 --out figs

echo
echo "==> simulator speed, for a like-for-like comparison"
$PY - <<'PY'
import time, numpy as np, jax.numpy as jnp
from wmf.plants.schema import PlantConfig
from wmf.plants.encode import build_geometry, encode_action
from wmf.sim.rollout import build_static, rollout, make_initial
from wmf.sim.physics import SimParams
cfg = PlantConfig.from_yaml("configs/plants/plant_a.yaml")
g = build_geometry(cfg); st = build_static(g); prm = SimParams.from_plant(cfg)
a = jnp.asarray(encode_action(g, dict(stoker_speed=.18, waste_feed=2.5,
        primary_air=[.8]*cfg.n_zones, secondary_air=[.5]*cfg.n_nozzles)))
N = 40
q0, b0 = make_initial(st)
rollout(q0, b0, jnp.broadcast_to(a,(N,)+a.shape), st, prm, 100, N)[0].block_until_ready()
t0 = time.time()
rollout(q0, b0, jnp.broadcast_to(a,(N,)+a.shape), st, prm, 100, N)[0].block_until_ready()
ms = (time.time()-t0)/N*1000
import json, pathlib
pathlib.Path("figs/speed.json").write_text(json.dumps({"sim_ms_per_frame": ms}))
print(f"  simulator: {ms:.1f} ms per control frame (100 gas steps)")
PY

echo
echo "==> figures"
$PY scripts/make_demo_figs.py --figs figs
echo
echo "done. assets in figs/"
