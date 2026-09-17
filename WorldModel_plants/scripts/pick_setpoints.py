#!/usr/bin/env python
"""Choose jointly reachable setpoints from the identification data.

A setpoint triple (T, O2, burnout) is usable only if the plant actually visits
its neighbourhood. (T = 900, O2 = 0.060) looked reasonable on paper and
co-occurred in 0.5% of steady states - scoring a controller against it
measured the cost weights and nothing else.

Quasi-steady samples are the last ``TAIL`` seconds of every slow dwell of the
excitation signal; on those the joint distribution is what the plant can hold.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wmf.control.data import load_split  # noqa: E402

TAIL = 60


def quasi_steady(d):
    """Samples where none of the MVs has moved for TAIL steps."""
    u, y = d["u"], d["y"]
    # "moved" relative to each knob's spread: the excitation jitters zones by a
    # few percent every frame, so an absolute threshold sees motion everywhere
    sd_u = u.reshape(-1, u.shape[-1]).std(0)
    # grate speed and feed are piecewise constant; the two air levels carry a
    # per-frame zone jitter of a few percent, so only a real level change
    # (at least half a std) counts as a move there
    thr = np.array([1e-9, 1e-9, 0.5 * sd_u[2], 0.5 * sd_u[3]])
    keep = []
    for e in range(u.shape[0]):
        moved = (np.abs(np.diff(u[e], axis=0)) > thr).any(1)
        since = np.zeros(len(u[e]), int)
        c = 0
        for t in range(1, len(u[e])):
            c = 0 if moved[t - 1] else c + 1
            since[t] = c
        keep.append(y[e][since >= TAIL])
    return np.concatenate(keep)


def reachable_fraction(ys, r, tol):
    return float((np.all(np.abs(ys[:, [0, 1, 3]] - r[[0, 1, 3]]) <= tol, axis=1)).mean())


def main() -> int:
    d = load_split("train", ROOT / "data_ctrl_v2")
    ys = quasi_steady(d)
    if len(ys) < 200:            # excitation too fast for the tail rule: use everything
        print(f"  quasi-steady samples: {len(ys)} < 200, falling back to all samples")
        ys = d["y"].reshape(-1, d["y"].shape[-1])
    sd = ys.std(0)
    tol = 0.35 * sd[[0, 1, 3]]
    q = lambda p, ch: float(np.percentile(ys[:, ch], p))

    # a central operating point and one-sigma-ish steps on one channel at a time
    # 30th -> 60th percentile: a step the controller has to work for, without
    # landing on the state the plant drifts to by itself at the nominal action
    # (measured: holding the nominal action rests at the ~75th percentile of T,
    # which made "do nothing" score perfectly against a q75 target)
    base = np.array([q(30, 0), q(30, 1), 0.0, q(50, 3)])
    t_hi = np.array([q(60, 0), q(30, 1), 0.0, q(50, 3)])
    o2_hi = np.array([q(30, 0), q(70, 1), 0.0, q(50, 3)])

    # snap each setpoint to the nearest quasi-steady sample so it is a state
    # the plant has actually held, then report how often that neighbourhood is visited
    def snap(r):
        z = (ys[:, [0, 1, 3]] - r[[0, 1, 3]]) / sd[[0, 1, 3]]
        k = int(np.argmin((z ** 2).sum(1)))
        return np.array([ys[k, 0], ys[k, 1], 0.0, ys[k, 3]])

    sp = {"temp_step": {"r0": snap(base).tolist(), "r1": snap(t_hi).tolist()},
          "o2_step": {"r0": snap(base).tolist(), "r1": snap(o2_hi).tolist()}}
    for name, s in sp.items():
        for k in ("r0", "r1"):
            r = np.asarray(s[k]); f = reachable_fraction(ys, r, tol)
            print(f"{name:10s} {k}: T={r[0]:7.1f}  O2={r[1]:.4f}  burnout={r[3]:.4f}   近傍到達率 {100*f:5.1f}%")
    sp["quasi_steady_samples"] = int(len(ys))
    sp["y_std_quasi_steady"] = sd.tolist()
    (ROOT / "figs").mkdir(exist_ok=True)
    (ROOT / "figs" / "setpoints.json").write_text(json.dumps(sp, indent=1))
    print("-> figs/setpoints.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
