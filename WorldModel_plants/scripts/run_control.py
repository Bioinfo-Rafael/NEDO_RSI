#!/usr/bin/env python
"""Close the loop: C plans inside M, the plant is the simulator.

    M (learned)  ->  candidate futures  ->  C picks a plan
                                              |  first move only
                                              v
                                     simulator (the plant)
                                              |  measurement
                                              +-> back into M's warm-up

The controller never sees the simulator's internal state - only the three flue
measurements a real plant would have. Everything it knows about what is burning
on the grate is inferred by the recurrent state from the history of those three
numbers, which is the only way this transfers to a plant that cannot measure
its own bed.

Baselines it is scored against:
  hold      keep the initial action forever (what "no control" looks like)
  pi        a per-loop PI controller on the two dominant pairings
"""
from __future__ import annotations

import argparse, importlib.util, json, pickle, time
from pathlib import Path

import functools

import jax
import jax.numpy as jnp
import numpy as np

from wmf import reward as rewardmod
from wmf.actions import RANGES, bounds, clip_action, nominal_action_vector
from wmf.control import model as cvm
from wmf.control.mpc import MPC, MPCConfig
from wmf.plants.encode import build_geometry, encode_action
from wmf.plants.schema import PlantConfig
from wmf.sim.physics import SimParams
from wmf.sim.rollout import build_static, coupled_step, make_initial

ROOT = Path(__file__).resolve().parent.parent   # resolve configs from the repo,
                                                 # not from the caller's cwd
CV = ["t_exit", "o2_exit", "yf_exit", "unburnt_bed"]
CONTROL_INTERVAL = 1.0      # s
# Reference-plant bounds, kept for callers that have no cfg in hand. Anything
# that knows its plant should use Plant.u_lo / u_hi / u_init, which scale with
# the grate (wmf.actions.envelope).
_REF_CFG = PlantConfig.from_yaml(ROOT / "configs" / "plants" / "plant_a.yaml")
U_LO, U_HI = bounds(_REF_CFG)
U_INIT = nominal_action_vector(_REF_CFG)


class Plant:
    """The simulator, wrapped so it exposes only what an operator would see."""

    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def __init__(self, plant="a", save_every=None):
        self.cfg = PlantConfig.from_yaml(ROOT / "configs" / "plants" / f"plant_{plant}.yaml")
        self.geom = build_geometry(self.cfg)
        self.static = build_static(self.geom)
        self.prm = SimParams.from_plant(self.cfg)
        # control interval 1 s (INTERFACES.md 1.1): 333 sub-steps at dt = 3 ms
        self.save_every = save_every or int(round(CONTROL_INTERVAL / self.prm.dt))
        self.u_lo, self.u_hi = bounds(self.cfg)
        self.u_init = nominal_action_vector(self.cfg)
        self.reset()

    def reset(self):
        self.q, self.bed = make_initial(self.static)
        return self._measure()

    def _measure(self, unburnt=0.0):
        pr = rewardmod.proxies(self.q, unburnt, self.static)
        return np.array([float(pr[k]) for k in CV], np.float32)

    def expand(self, u_agg):
        """Four aggregate knobs -> the plant's actual actuators."""
        return clip_action(self.cfg, {
            "stoker_speed": float(u_agg[0]), "waste_feed": float(u_agg[1]),
            "primary_air": [float(u_agg[2])] * self.cfg.n_zones,
            "secondary_air": [float(u_agg[3])] * self.cfg.n_nozzles})

    @functools.partial(jax.jit, static_argnums=(0,))
    def _advance(self, q, bed, a):
        """One control interval as a single compiled scan.

        Calling coupled_step in a Python loop costs 17.8 s per control step -
        every one of the 100 sub-steps pays JAX dispatch. The same work inside
        lax.scan takes ~25 ms, which is the difference between a control loop
        that can run and one that cannot.
        """
        def body(carry, _):
            return coupled_step(carry, a, self.static, self.prm)

        (q, bed), (_, ub) = jax.lax.scan(body, (q, bed), None, length=self.save_every)
        return q, bed, ub.mean()

    def step(self, u_agg):
        a = jnp.asarray(encode_action(self.geom, self.expand(u_agg)))
        self.q, self.bed, ub = self._advance(self.q, self.bed, a)
        pr = rewardmod.proxies(self.q, ub, self.static)
        return np.array([float(pr[k]) for k in CV], np.float32)


PI_GAINS_FILE = ROOT / "figs" / "pi_gains.json"


def load_pi_gains():
    """SIMC gains identified from the current dataset by scripts/identify_pi.py.
    Hard-coding them here is how the PI baseline silently went stale when the
    plant's time scales changed by a factor of forty."""
    g = json.loads(PI_GAINS_FILE.read_text())
    out = {k: np.asarray(v, np.float64) for k, v in g.items() if k in ("kp", "ti", "bias")}
    # which knob each loop moves and which CV it watches, in loop order
    mv_idx = {"stoker_speed": 0, "waste_feed": 1, "primary_level": 2, "secondary_level": 3}
    cv_idx = {"t_exit": 0, "o2_exit": 1, "yf_exit": 2, "unburnt_bed": 3}
    out["mv"] = np.array([mv_idx[m] for m, _ in g["pairs"]])
    out["cv"] = np.array([cv_idx[c] for _, c in g["pairs"]])
    return out


def pi_controller(y, target, integ, u, gains, u_lo=U_LO, u_hi=U_HI, dt=CONTROL_INTERVAL):
    """Three SISO PI loops, paired by the measured RGA:

        primary air   -> furnace exit temperature
        secondary air -> exit oxygen
        grate speed   -> unburnt solids leaving the grate

    Gains follow SIMC (Kc ~ 0.5/K, Ti ~ min(tau, 4L)) from the process gains and
    dead times measured on the identification data; see scripts/identify_pi.py.
    The third loop has a dead time of one grate residence (~600 s), which is the
    whole reason a PI loop cannot do much with it inside a scenario - and that
    is a fact about the plant, not about the tuning.
    """
    e = target - y
    u = u.copy()
    kp, ti, bias, mv, cv = gains["kp"], gains["ti"], gains["bias"], gains["mv"], gains["cv"]
    lo, hi = u_lo[mv], u_hi[mv]
    err = e[cv]

    raw = bias + kp * (err + integ / ti)
    sat = np.clip(raw, lo, hi)
    # conditional anti-windup: stop integrating any loop that is against a limit
    active = (raw == sat) | (np.sign(err) != np.sign(raw - sat))
    integ = integ + np.where(active, err * dt, 0.0)
    integ = np.clip(integ, -1e4, 1e4)

    u[mv] = sat
    return u, integ


def load_wm(bundle):
    """The world-model module the bundle's params belong to. Bundles written by
    control_research/run.py carry ``wm_path``; older ones (train_cv_model.py)
    are for the GRU in wmf.control.model."""
    path = bundle.get("wm_path")
    if not path:
        return cvm
    spec = importlib.util.spec_from_file_location("wm_bundle_module", ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.path.insert(0, str((ROOT / path).parent))     # worldmodel.py imports prepare
    spec.loader.exec_module(mod)
    return mod


def run(mode, plant, target, n_steps, bundle, warmup_steps, seed=0, target1=None, t_step=None):
    """``target`` holds until ``t_step``, then ``target1`` (a setpoint step) if given."""
    p = Plant(plant)
    y = p.reset()
    u = p.u_init.copy()
    gains = load_pi_gains() if mode == "pi" else None

    ctl = None
    if mode == "mpc":
        sc = cvm.Scaler(**{k: bundle["stats"][k] for k in
                           ("y_mean", "y_std", "u_mean", "u_std")})
        ctl = MPC(bundle["params"], sc, p.u_lo, p.u_hi, MPCConfig(seed=seed), wm=load_wm(bundle))
        ctl.reset(u)

    y_hist, u_hist, log = [], [], []
    integ = np.zeros(3)
    t0 = time.time()
    r = np.asarray(target, np.float32)
    for k in range(n_steps):
        if target1 is not None and t_step is not None and k == t_step:
            r = np.asarray(target1, np.float32)
        target = r
        y = p.step(u)
        y_hist.append(y.copy()); u_hist.append(u.copy())
        log.append({"t": k * CONTROL_INTERVAL, "y": y.tolist(), "u": u.tolist(), "r": r.tolist()})

        if mode == "hold":
            pass
        elif mode == "pi":
            u, integ = pi_controller(y, target, integ, u, gains, p.u_lo, p.u_hi)
        elif mode == "mpc":
            if len(y_hist) >= warmup_steps:
                ctl.update_bias(y)
                u = ctl.act(np.array(y_hist[-warmup_steps:]),
                            np.array(u_hist[-warmup_steps:]), target, u)
    return {"mode": mode, "log": log, "seconds": time.time() - t0,
            "target": np.asarray(target).tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="figs/wm_bundle.pkl",
                    help="written by control_research/run.py")
    ap.add_argument("--plant", default="a")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--t-target", type=float, default=None,
                    help="default: the 'temp_step' scenario's stepped setpoint")
    ap.add_argument("--o2-target", type=float, default=None)
    ap.add_argument("--burnout-target", type=float, default=None,
                    help="unburnt solids leaving the grate; lower is cleaner ash")
    ap.add_argument("--modes", default="hold,pi,mpc")
    ap.add_argument("--scenario", default="temp_step",
                    help="setpoint step from figs/setpoints.json (r0 then r1 at --t-step); "
                         "'none' holds the --*-target values throughout")
    ap.add_argument("--t-step", type=int, default=150)
    ap.add_argument("--out", default="figs/control_run.json")
    a = ap.parse_args()

    with open(a.model, "rb") as f:
        bundle = pickle.load(f)
    sps = json.loads((ROOT / "figs" / "setpoints.json").read_text())
    if a.scenario != "none":
        r0 = np.asarray(sps[a.scenario]["r0"], np.float32)
        r1 = np.asarray(sps[a.scenario]["r1"], np.float32)
    else:
        r0 = r1 = np.asarray(sps["temp_step"]["r1"], np.float32)
    for i, v in ((0, a.t_target), (1, a.o2_target), (3, a.burnout_target)):
        if v is not None:
            r0[i] = r1[i] = v
    target = r1

    runs = {}
    for mode in a.modes.split(","):
        r = run(mode, a.plant, r0, a.steps, bundle, bundle["warmup"],
                target1=r1 if a.scenario != "none" else None, t_step=a.t_step)
        y = np.array([s["y"] for s in r["log"]])
        tail = y[-40:]
        print(f"{mode:5s} 最終40step平均  T={tail[:,0].mean():7.1f}K(目標{target[0]:.0f}) "
              f"O2={tail[:,1].mean():.4f}(目標{target[1]:.3f}) "
              f"灰未燃={tail[:,3].mean():.3f}(目標{target[3]:.3f})  "
              f"|ΔT|={abs(tail[:,0].mean()-target[0]):6.1f}K  [{r['seconds']:.0f}s]", flush=True)
        runs[mode] = r

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(runs))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
