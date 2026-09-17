"""Scoring harness. READ-ONLY — this is the ground truth of the experiment.

The world model is NOT scored on prediction error. It is scored on how well the
plant tracks its setpoints when the controller plans inside it. A model can be
excellent at prediction and still make a poor controller: if it is confidently
wrong in exactly the region the optimiser is drawn to, C will steer there.

    control_cost = mean over scenarios of
                     sum_k  w_k * IAE_k / sigma_k          (tracking)
                   + overshoot penalty
                   + move penalty

IAE is the integral of absolute error, measured after a settling allowance so a
controller is not punished for the transport delay it cannot beat. Each channel
is divided by that channel's spread in the identification data, so degrees and
mass fractions are comparable.

Scenarios are setpoint *steps*, and every setpoint has been checked against the
identification data for joint reachability - asking for a temperature and an
oxygen level that never co-occur measures nothing but the weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

SETTLE = 40          # control steps excluded from IAE (40 s at a 1 s interval)
RUN_STEPS = 400      # 400 s per scenario: the gas settles, the grate does not - by design
W = np.array([1.0, 1.0, 0.0, 0.7])       # t_exit, o2_exit, yf_exit, unburnt_bed

# (initial setpoint, stepped setpoint, step time). Setpoints are chosen from the
# identification data by scripts/pick_setpoints.py so that every (T, O2, burnout)
# triple is jointly reachable: asking for a pair that never co-occurs in the
# data measures nothing but the weights.
import json as _json
_SP = _json.loads((ROOT / "figs" / "setpoints.json").read_text())
SCENARIOS = [
    {"name": "temp_step", "r0": _SP["temp_step"]["r0"], "r1": _SP["temp_step"]["r1"], "t_step": 150},
    {"name": "o2_step",   "r0": _SP["o2_step"]["r0"],   "r1": _SP["o2_step"]["r1"],   "t_step": 150},
]


def _mpc(params_list, stats, wm, plant):
    from wmf.control.mpc import MPC, MPCConfig
    from wmf.control import model as cvm
    sc = cvm.Scaler(**{k: np.asarray(stats[k]) for k in
                       ("y_mean", "y_std", "u_mean", "u_std")})
    ctl = MPC(params_list, sc, plant.u_lo, plant.u_hi, MPCConfig(), wm=wm)
    return ctl, sc


_PLANT = None


def _plant():
    """One Plant for the whole evaluation. Constructing it per scenario makes
    JAX recompile the 100-substep advance every time, which costs more than the
    scenario itself."""
    global _PLANT
    if _PLANT is None:
        from run_control import Plant
        _PLANT = Plant("a")
    return _PLANT


def run_scenario(params_list, stats, wm, sc_def, warmup, seed=0):
    p = _plant()
    ctl, sc = _mpc(params_list, stats, wm, p)
    y = p.reset()
    u = p.u_init.copy()
    ctl.reset(u)

    r0 = np.asarray(sc_def["r0"], np.float32); r1 = np.asarray(sc_def["r1"], np.float32)
    ys, us, rs = [], [], []
    for k in range(RUN_STEPS):
        r = r0 if k < sc_def["t_step"] else r1
        y = p.step(u)
        ys.append(y.copy()); us.append(u.copy()); rs.append(r.copy())
        if len(ys) >= warmup:
            ctl.update_bias(y)
            u = ctl.act(np.array(ys[-warmup:]), np.array(us[-warmup:]), r, u)
    return np.array(ys), np.array(us), np.array(rs)


def score(params_list, stats, wm, warmup=20) -> dict:
    sd = np.asarray(stats["y_std"])
    out, per = {}, []
    for s in SCENARIOS:
        ys, us, rs = run_scenario(params_list, stats, wm, s, warmup)
        # IAE after the settling allowance, on each side of the step
        m = np.ones(len(ys), bool)
        m[:SETTLE] = False
        m[s["t_step"]:s["t_step"] + SETTLE] = False
        iae = (np.abs(ys - rs)[m] / sd).mean(axis=0)
        tracking = float((iae * W).sum())
        # overshoot past the new setpoint, on the stepped channel
        ch = int(np.argmax(np.abs(np.asarray(s["r1"]) - np.asarray(s["r0"])) / sd))
        seg = ys[s["t_step"]:, ch]; r1 = s["r1"][ch]; r0 = s["r0"][ch]
        over = float(max(0.0, (seg.max() - r1) if r1 > r0 else (r1 - seg.min())) / sd[ch])
        from wmf.control.mpc import MPCConfig
        move = float((np.abs(np.diff(us, axis=0)) / np.asarray(MPCConfig().du_max)).mean())
        out[s["name"]] = {"tracking": tracking, "overshoot": over, "move": move,
                          "iae_per_channel": iae.tolist()}
        per.append(tracking + 0.5 * over + 0.1 * move)
    out["control_cost"] = float(np.mean(per))
    return out
