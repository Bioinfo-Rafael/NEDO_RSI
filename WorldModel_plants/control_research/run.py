#!/usr/bin/env python
"""One experiment: train M, then let C drive the plant and score the tracking.

    cd control_research && ../.venv/bin/python run.py > run.log 2>&1
    grep "^control_cost:" run.log
"""
from __future__ import annotations

import argparse, pickle, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import prepare
import worldmodel as wm
from evaluate import SCENARIOS, score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=prepare.TRAIN_SECONDS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    data = prepare.load_split("train")
    stats = prepare.norm_stats(data)

    t1 = time.time()
    params = wm.train(data, stats, a.seconds, a.seed)
    train_seconds = time.time() - t1

    n_params = sum(int(np.asarray(v).size) for p in params for v in
                   __import__("jax").tree.flatten(p)[0])

    # Keep the trained ensemble so run_control.py / make_control_movie.py can
    # plan inside exactly this M. The module path travels with it, because the
    # params only mean something to the worldmodel.py that produced them.
    out = Path(__file__).resolve().parent.parent / "figs" / "wm_bundle.pkl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"params": params, "stats": stats, "warmup": wm.WARMUP,
                     "wm_path": "control_research/worldmodel.py"}, f)

    t2 = time.time()
    res = score(params, stats, wm, warmup=wm.WARMUP)
    eval_seconds = time.time() - t2

    print("---")
    print(f"control_cost:     {res['control_cost']:.6f}")
    for s in SCENARIOS:
        r = res[s["name"]]
        print(f"{s['name']+'_tracking:':<18s}{r['tracking']:.6f}")
        print(f"{s['name']+'_overshoot:':<18s}{r['overshoot']:.6f}")
    print(f"ensemble:         {len(params)}")
    print(f"num_params_K:     {n_params/1000:.1f}")
    print(f"training_seconds: {train_seconds:.1f}")
    print(f"eval_seconds:     {eval_seconds:.1f}")
    print(f"total_seconds:    {time.time()-t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
