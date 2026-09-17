#!/usr/bin/env python
"""Wall-clock cost of one control step, for the plant and for the controller.

Run it alone: JAX on CPU shares cores with whatever else is running, and a
timing taken next to a data-generation job is a timing of the contention."""
from __future__ import annotations

import json, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from run_control import Plant, load_wm, CONTROL_INTERVAL  # noqa: E402


def main() -> int:
    p = Plant("a"); u = p.u_init.copy()
    p.step(u)                                        # compile
    t0 = time.perf_counter(); n = 20
    for _ in range(n): p.step(u)
    plant_ms = (time.perf_counter() - t0) / n * 1e3

    out = {"plant_step_ms": plant_ms, "control_interval_s": CONTROL_INTERVAL,
           "sim_faster_than_realtime": CONTROL_INTERVAL * 1e3 / plant_ms,
           "substeps_per_control_step": p.save_every, "dt_ms": p.prm.dt * 1e3}

    bundle = ROOT / "figs" / "wm_bundle.pkl"
    if bundle.exists():
        from wmf.control import model as cvm
        from wmf.control.mpc import MPC, MPCConfig
        b = pickle.load(open(bundle, "rb"))
        sc = cvm.Scaler(**{k: np.asarray(b["stats"][k]) for k in ("y_mean", "y_std", "u_mean", "u_std")})
        ctl = MPC(b["params"], sc, p.u_lo, p.u_hi, MPCConfig(), wm=load_wm(b)); ctl.reset(u)
        w = b["warmup"]; Y, U = [], []
        for _ in range(w + 1):
            y = p.step(u); Y.append(y); U.append(u.copy())
        tgt = np.asarray(Y[-1]); ctl.update_bias(Y[-1])
        ctl.act(np.array(Y[-w:]), np.array(U[-w:]), tgt, u)       # compile
        t0 = time.perf_counter(); n = 10
        for _ in range(n): ctl.act(np.array(Y[-w:]), np.array(U[-w:]), tgt, u)
        out["mpc_act_ms"] = (time.perf_counter() - t0) / n * 1e3
        c = MPCConfig()
        out["futures_per_decision"] = c.population * c.iterations * len(b["params"])
        out["horizon_s"] = c.horizon * CONTROL_INTERVAL
        out["ensemble"] = len(b["params"])
    (ROOT / "figs").mkdir(exist_ok=True)
    (ROOT / "figs" / "speed.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items(): print(f"  {k:28s} {v:.3g}" if isinstance(v, float) else f"  {k:28s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
