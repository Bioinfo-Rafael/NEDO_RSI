"""`python -m wmf.selfcheck` / `wmf-selfcheck`: does the simulator work here?

Thirty seconds, no data, no GPU required. It reports the JAX backend, then for
every plant config: builds the furnace, applies a bad input to show that it is
rejected, applies an out-of-envelope input to show that it is projected, runs
20 s at the nominal input and 20 s at a random corner, and checks that every
measurement is finite and that the velocity clamp never bound. Exit code 0
means a teammate can start.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    import jax
    from .furnace import Furnace
    from .sim.physics import U_CLAMP

    print(f"JAX {jax.__version__}  backend: {jax.default_backend()}  devices: {jax.devices()}")
    ok = True
    for yaml in sorted((ROOT / "configs" / "plants").glob("*.yaml")):
        t0 = time.time()
        f = Furnace(yaml)
        # 1. a malformed input is refused with a message, not a crash later
        try:
            f.step(stoker_speed=0.02, waste_feed=1.3, primary_air=[1.0, 1.0], secondary_air=0.4)
            print(f"  {yaml.stem}: FAILED - bad zone count was accepted"); ok = False
        except ValueError:
            pass
        # 2. an out-of-envelope input is projected, not applied
        a = f.project(f._as_action(stoker_speed=5.0, waste_feed=-1.0, primary_air=99.0, secondary_air=0.0))
        e = f.envelope
        inside = (e["stoker_speed"][0] <= a["stoker_speed"] <= e["stoker_speed"][1]
                  and e["waste_feed"][0] <= a["waste_feed"] <= e["waste_feed"][1]
                  and all(e["primary_air"][0] <= v <= e["primary_air"][1] for v in a["primary_air"]))
        if not inside:
            print(f"  {yaml.stem}: FAILED - projection left the envelope: {a}"); ok = False
        # 3. it advances, at the nominal input and at a corner, and stays finite
        f.reset()
        rng = np.random.default_rng(0)
        corner = [e[k][int(rng.integers(2))] for k in ("stoker_speed", "waste_feed", "primary_air", "secondary_air")]
        ys = []
        for k in range(40):
            y = f.step(f.u_nominal if k < 20 else corner)
            ys.append([y["t_exit"], y["o2_exit"], y["unburnt_bed"]])
        ys = np.asarray(ys)
        vmax = float(np.sqrt(f.field[..., 1] ** 2 + f.field[..., 2] ** 2).max())
        fin = bool(np.isfinite(ys).all())
        clamp = vmax >= 0.999 * U_CLAMP
        status = "ok" if (fin and not clamp) else "FAILED"
        ok &= (fin and not clamp)
        print(f"  {yaml.stem:8s} {status:6s}  40 steps in {time.time()-t0:4.1f}s  "
              f"T_exit {ys[-1,0]:6.0f} K  O2 {ys[-1,1]:.3f}  unburnt {ys[-1,2]:.4f}  |v|max {vmax:4.1f} m/s"
              + ("  (clamp bound!)" if clamp else ""))
    print("SELFCHECK PASSED" if ok else "SELFCHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
