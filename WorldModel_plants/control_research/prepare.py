"""Data preparation. READ-ONLY.

Identification data from the simulator, driven by amplitude-modulated
pseudo-random inputs so that every knob is excited across its whole range. That
matters more than the volume: operating data from the real plant has a
condition number above 5000 on its inputs - the grate sections move in fixed
ratios under the existing controller - so no amount of it can tell you what
happens if you move them independently. This dataset can.
"""
from __future__ import annotations

import glob, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CV_NAMES = ["t_exit", "o2_exit", "yf_exit", "unburnt_bed"]
CV_IDX = [0, 1, 2, 5]
MV_NAMES = ["stoker_speed", "waste_feed", "primary_level", "secondary_level"]

# Fixed budget for training the world model, in seconds. Every experiment gets
# the same one, so ideas are compared rather than machines.
TRAIN_SECONDS = 150.0


def load_split(split: str, data_dir: str | Path = None) -> dict:
    d = Path(data_dir or ROOT / "data_ctrl_v2") / split
    files = sorted(glob.glob(str(d / "*.npz")))
    if not files:
        raise FileNotFoundError(f"no episodes in {d}")
    y, u = [], []
    for f in files:
        with np.load(f) as z:
            pr = z["proxies"][:, CV_IDX].astype(np.float32)
            a = z["a_scalar"].astype(np.float32)
            m = json.loads(str(z["meta"]))
        nz, nn = m["n_zones"], m["n_nozzles"]
        u.append(np.stack([a[:, 0], a[:, 1],
                           a[:, 2:2 + nz].mean(1),
                           a[:, 2 + nz:2 + nz + nn].mean(1)], -1))
        y.append(pr)
    return {"y": np.stack(y), "u": np.stack(u)}


def norm_stats(d: dict) -> dict:
    y = d["y"].reshape(-1, d["y"].shape[-1]); u = d["u"].reshape(-1, d["u"].shape[-1])
    return {"y_mean": y.mean(0), "y_std": y.std(0) + 1e-8,
            "u_mean": u.mean(0), "u_std": u.std(0) + 1e-8}


def windows(d, stats, horizon, batch, rng, warmup=20):
    y, u = d["y"], d["u"]
    E, T, _ = y.shape
    span = warmup + horizon
    while True:
        e = rng.integers(0, E, batch); t = rng.integers(0, T - span - 1, batch)
        idx = t[:, None] + np.arange(span)[None, :]
        yy = (y[e[:, None], idx] - stats["y_mean"]) / stats["y_std"]
        uu = (u[e[:, None], idx] - stats["u_mean"]) / stats["u_std"]
        yield yy[:, :warmup], uu[:, :warmup], uu[:, warmup:], yy[:, warmup:]
