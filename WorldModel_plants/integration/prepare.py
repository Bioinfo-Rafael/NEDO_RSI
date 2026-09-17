"""Data preparation. IMMUTABLE - the agent edits train.py, never this file.

Mirrors the role of prepare.py in karpathy/autoresearch: it turns the raw
episodes into arrays and fixes the normalisation, so that every experiment is
scored on the same data in the same units.

Normalisation statistics come from the TRAIN split only.  Taking them over the
whole dataset would leak the held-out plants into training and quietly flatter
every generalisation number the demo is built on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA / "prepared.npz"
SPLITS = ("train", "val_iid", "val_b", "val_c")
# Fixed wall-clock training budget, in seconds.  Every experiment gets the
# same one so that ideas are compared, not machines.  Read-only.
TRAIN_SECONDS = 180.0
Q_CH, A_CH, G_CH = 7, 4, 4


def _load_split(d: Path):
    files = sorted(d.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no episodes in {d} - run scripts/gen_dataset.py first")
    q, a, g, meta = [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            q.append(z["q"].astype(np.float32))
            a.append(z["a"].astype(np.float32))
            g.append(z["g"].astype(np.float32))
            meta.append(json.loads(str(z["meta"])))
    return np.stack(q), np.stack(a), np.stack(g), meta


def build(force: bool = False) -> dict:
    if CACHE.exists() and not force:
        return load()

    out: dict = {}
    for s in SPLITS:
        q, a, g, meta = _load_split(DATA / s)
        out[f"{s}_q"], out[f"{s}_a"], out[f"{s}_g"] = q, a, g
        # dx is the reason a 9 m and a 15 m furnace are not interchangeable on a
        # fixed 64x64 grid; carry it so train.py can use it as a channel
        out[f"{s}_dx"] = np.array([m["dx"] for m in meta], dtype=np.float32)
        out[f"{s}_dy"] = np.array([m["dy"] for m in meta], dtype=np.float32)
        print(f"  {s:8s} {q.shape[0]:4d} episodes  q{q.shape}")

    tr = out["train_q"]
    out["q_mean"] = tr.reshape(-1, Q_CH).mean(0)
    out["q_std"] = tr.reshape(-1, Q_CH).std(0) + 1e-6
    d = np.diff(tr, axis=1).reshape(-1, Q_CH)
    out["dq_std"] = d.std(0) + 1e-8
    out["a_std"] = out["train_a"].reshape(-1, A_CH).std(0) + 1e-8

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"  -> {CACHE}  ({CACHE.stat().st_size / 1e6:.0f} MB)")
    return out


def load() -> dict:
    if not CACHE.exists():
        return build()
    with np.load(CACHE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def batches(d: dict, split: str, batch_size: int, rng: np.random.Generator):
    """Yield ((q_t, a_t, g, dx), dq) pairs for one-step training, forever."""
    q, a, g = d[f"{split}_q"], d[f"{split}_a"], d[f"{split}_g"]
    dx = d[f"{split}_dx"]
    n_ep, n_t = q.shape[0], q.shape[1]
    qm, qs, dqs = d["q_mean"], d["q_std"], d["dq_std"]
    while True:
        ep = rng.integers(0, n_ep, batch_size)
        t = rng.integers(0, n_t - 1, batch_size)
        x = (q[ep, t] - qm) / qs
        y = (q[ep, t + 1] - q[ep, t]) / dqs
        yield (x, a[ep, t] / d["a_std"], g[ep], dx[ep]), y


if __name__ == "__main__":
    build(force=True)
