"""Data for the control model: the controlled variables, not the field.

A controller does not need 64x64x7. It needs the handful of quantities the
operator is actually holding to a setpoint, and it needs them far enough ahead
to cover the plant's dead time. Those are two different modelling problems, and
conflating them is why the full-field model cannot drive a control loop: it is
accurate for ~1-2 s, and the grate alone has ~17 s of transport delay.

Controlled variables (CV), all already recorded by the simulator:
    t_exit    flue-gas temperature   [K]   <- the regulated variable
    o2_exit   flue-gas O2            [-]   <- the other regulated variable
    yf_exit   unburnt fuel at exit   [-]   <- constraint / quality
    unburnt_bed  unburnt solids leaving the grate [kg/m2/s] <- what the grate is for

Manipulated variables (MV), the four aggregate knobs an operator actually turns:
    stoker_speed, waste_feed, primary_level, secondary_level

The dataset holds per-zone air, which is richer than four knobs. The model is
trained on the full per-zone vector so it stays general; the controller searches
over the four aggregates and expands them to every zone. That keeps the search
space small enough for real-time use without narrowing what the model knows.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
# Controlled variables. unburnt_bed is here because it is the only thing the
# grate speed actually commands: measured over the identification data, the
# stoker has 1.45 sigma of authority over it and below 0.35 sigma over every
# flue-gas quantity. A CV set of flue measurements alone makes the grate look
# like a redundant actuator, which is an artefact of the choice, not the plant.
# On a real furnace this is the ignition loss of the ash - a regulated quantity.
CV_NAMES = ["t_exit", "o2_exit", "yf_exit", "unburnt_bed"]
CV_IDX = [0, 1, 2, 5]         # columns of the `proxies` array
MV_NAMES = ["stoker_speed", "waste_feed", "primary_level", "secondary_level"]


def load_split(split: str, data_dir: str | Path = None) -> dict:
    """Return CV trajectories, the full action vectors, and the aggregates."""
    d = Path(data_dir or ROOT / "data") / split
    files = sorted(glob.glob(str(d / "*.npz")))
    if not files:
        raise FileNotFoundError(f"no episodes in {d}")

    y, u_full, u_agg, meta = [], [], [], []
    for f in files:
        with np.load(f) as z:
            pr = z["proxies"][:, CV_IDX].astype(np.float32)
            a = z["a_scalar"].astype(np.float32)
            m = json.loads(str(z["meta"]))
        nz, nn = m["n_zones"], m["n_nozzles"]
        agg = np.stack([
            a[:, 0],                       # stoker_speed
            a[:, 1],                       # waste_feed
            a[:, 2:2 + nz].mean(axis=1),   # mean primary air per zone
            a[:, 2 + nz:2 + nz + nn].mean(axis=1),
        ], axis=-1)
        y.append(pr); u_full.append(a); u_agg.append(agg); meta.append(m)

    return {
        "y": np.stack(y),            # [E, T, 3]
        "u": np.stack(u_agg),        # [E, T, 4]
        "u_full": u_full,            # ragged across plants
        "meta": meta,
    }


def norm_stats(d: dict) -> dict:
    """Scaling taken from the training split only."""
    y = d["y"].reshape(-1, d["y"].shape[-1])
    u = d["u"].reshape(-1, d["u"].shape[-1])
    return {
        "y_mean": y.mean(0), "y_std": y.std(0) + 1e-8,
        "u_mean": u.mean(0), "u_std": u.std(0) + 1e-8,
    }


def windows(d: dict, stats: dict, horizon: int, batch: int, rng, warmup: int = 20):
    """Yield (y_warm, u_warm, u_future, y_future) for recurrent training.

    ``warmup`` frames are fed in before the prediction starts. The bed state is
    not observable, so the recurrent state has to infer it from recent history -
    that warm-up is the only place it can come from, and the controller does the
    same thing online.
    """
    y, u = d["y"], d["u"]
    E, T, _ = y.shape
    span = warmup + horizon
    while True:
        e = rng.integers(0, E, batch)
        t = rng.integers(0, T - span - 1, batch)
        idx = t[:, None] + np.arange(span)[None, :]
        yy = (y[e[:, None], idx] - stats["y_mean"]) / stats["y_std"]
        uu = (u[e[:, None], idx] - stats["u_mean"]) / stats["u_std"]
        yield (yy[:, :warmup], uu[:, :warmup], uu[:, warmup:], yy[:, warmup:])
