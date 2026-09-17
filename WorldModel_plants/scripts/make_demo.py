#!/usr/bin/env python
"""Build the demo assets: what the world model actually predicts, next to the truth.

Trains the current integration/train.py once, keeps the parameters (train.py
itself does not save them), then produces the one thing a world-model demo has
to show: the model fed its own output for many steps, side by side with the
simulator it is imitating.

    python scripts/make_demo.py --out figs/
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INTEG = ROOT / "integration"


def _load(name: str):
    """Import integration/<name>.py, whatever the agent has left in it."""
    sys.path.insert(0, str(INTEG))
    spec = importlib.util.spec_from_file_location(name, INTEG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def train_once(seconds: float, seed: int = 0):
    """Run train.py exactly as the loop left it, and capture what it trained.

    Replicating the training here would go stale the moment the agent changes
    the loss signature or the batching - it already did once. Instead the real
    main() runs, and the scoring call it makes on the way out is intercepted to
    keep the parameters, which train.py otherwise discards.
    """
    import sys as _sys

    train = _load("train")
    captured = {}
    original_score = train.score

    def capture(params, d, build_input, *a, **k):
        captured["params"] = params
        captured["d"] = d
        return original_score(params, d, build_input, *a, **k)

    train.score = capture
    argv = _sys.argv
    _sys.argv = ["train.py", "--seconds", str(seconds), "--seed", str(seed)]
    try:
        train.main()
    finally:
        _sys.argv = argv
        train.score = original_score

    if "params" not in captured:
        raise RuntimeError("train.main() did not reach the scoring call")
    return captured["params"], captured["d"], train


def rollout(params, d, train, split: str, ep: int, start: int, horizon: int):
    """Feed the model its own output `horizon` times. Returns (pred, truth)."""
    import jax
    import jax.numpy as jnp

    q, a, g, dx = d[f"{split}_q"], d[f"{split}_a"], d[f"{split}_g"], d[f"{split}_dx"]
    qm, qs, dqs, as_ = d["q_mean"], d["q_std"], d["dq_std"], d["a_std"]

    @jax.jit
    def advance(state, a_t, g_e, dx_e):
        inp = train.build_input((state - qm) / qs, a_t / as_, g_e, dx_e)
        return state + train.forward(params, inp) * dqs

    state = jnp.asarray(q[ep, start][None])
    preds = [np.asarray(state[0])]
    t0 = time.time()
    for k in range(horizon):
        state = advance(state, jnp.asarray(a[ep, start + k][None]),
                        jnp.asarray(g[ep][None]), jnp.asarray(dx[ep][None]))
        preds.append(np.asarray(state[0]))
    ms_per_step = (time.time() - t0) / horizon * 1000
    truth = q[ep, start:start + horizon + 1]
    return np.stack(preds), np.asarray(truth), ms_per_step


def busiest_window(d, split, ep, horizon):
    """Pick the window where the furnace actually changes.

    Actions are piecewise constant, so most of an episode sits at quasi-steady
    state; predicting that well is not evidence of much. This finds the window
    with the largest change in the true temperature field, which is where the
    model is genuinely being asked something.
    """
    q = d[f"{split}_q"][ep, ..., 0]
    n_t = q.shape[0]
    # Skip the startup transient. Ignition from a cold-ish furnace is the most
    # violent thing in the episode, but it is not what the model would be used
    # for - the question in operation is how the furnace responds to a change
    # in the controls, and that is what the rest of the episode contains.
    warmup = 40
    best, best_s = -1.0, warmup
    for s in range(warmup, n_t - horizon - 1, 5):
        delta = float(np.abs(q[s + horizon] - q[s]).mean())
        if delta > best:
            best, best_s = delta, s
    return best_s, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    params, d, train = train_once(args.seconds)
    import jax
    flat = jax.tree.flatten(params)[0]
    np.savez(out / "params_final.npz", n=len(flat),
             **{f"p{i}": np.asarray(v) for i, v in enumerate(flat)})

    for split, ep in (("val_iid", 0), ("val_b", 0)):
        start, delta = busiest_window(d, split, ep, args.horizon)
        print(f"  {split}: window t={start} (mean |dT| = {delta:.0f} K over the horizon)")
        pred, truth, ms = rollout(params, d, train, split, ep, start, args.horizon)
        np.savez_compressed(out / f"rollout_{split}.npz",
                            pred=pred.astype(np.float32),
                            truth=truth.astype(np.float32),
                            g=d[f"{split}_g"][ep], ms_per_step=ms, start=start)
        err = np.abs(pred - truth)[..., 0].mean(axis=(1, 2))
        print(f"  {split:8s} {ms:6.2f} ms/step   |T error| "
              f"{err[1]:6.1f} K after 1 step -> {err[-1]:6.1f} K after {args.horizon}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
