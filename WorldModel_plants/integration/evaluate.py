"""Scoring harness. READ-ONLY - this is the ground-truth metric.

    val_nrmse = mean over (held-out episodes, channels) of
                  RMSE of a HORIZON-step open-loop rollout / channel std

Open loop, not one step: a model can look excellent at one step and diverge as
soon as it is fed its own output, and the rollout is what the downstream needs.
Normalised per channel so temperature (~1000 K) does not drown out mass
fractions (~0.1).

The headline number is val_iid.  val_b and val_c are unseen plants - a model
that merely memorised plant_a scores well on val_iid and badly on those.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import prepare

HORIZON = 16
MAX_EPS = 8
SPLITS = ("val_iid", "val_b", "val_c")
CHANNELS = ["T", "u", "v", "p", "Y_F", "Y_O2", "Y_P"]


def _starts(n_t: int, horizon: int) -> np.ndarray:
    return np.arange(0, n_t - horizon, max(1, (n_t - horizon) // 8))


def rollout_nrmse(params, d, split, build_input, forward, horizon=HORIZON, max_eps=MAX_EPS):
    q, a, g, dx = d[f"{split}_q"], d[f"{split}_a"], d[f"{split}_g"], d[f"{split}_dx"]
    qm, qs, dqs, as_ = d["q_mean"], d["q_std"], d["dq_std"], d["a_std"]
    n_ep = q.shape[0] if max_eps is None else min(max_eps, q.shape[0])

    @jax.jit
    def advance(state, a_t, g_e, dx_e):
        inp = build_input((state - qm) / qs, a_t / as_, g_e, dx_e)
        return state + forward(params, inp) * dqs

    se, cnt = np.zeros(q.shape[-1]), 0
    for e in range(n_ep):
        for s in _starts(q.shape[1], horizon):
            state = jnp.asarray(q[e, s][None])
            for k in range(horizon):
                state = advance(state, jnp.asarray(a[e, s + k][None]),
                                jnp.asarray(g[e][None]), jnp.asarray(dx[e][None]))
            se += ((np.asarray(state[0]) - q[e, s + horizon]) ** 2).reshape(-1, q.shape[-1]).mean(0)
            cnt += 1
    return np.sqrt(se / cnt) / qs


def persistence_nrmse(d, split, horizon=HORIZON, max_eps=MAX_EPS):
    """Reference: predict that nothing changes. A model above this line is
    worse than a constant, which val_nrmse alone would not tell you."""
    q, qs = d[f"{split}_q"], d["q_std"]
    n_ep = q.shape[0] if max_eps is None else min(max_eps, q.shape[0])
    se, cnt = np.zeros(q.shape[-1]), 0
    for e in range(n_ep):
        for s in _starts(q.shape[1], horizon):
            se += ((q[e, s] - q[e, s + horizon]) ** 2).reshape(-1, q.shape[-1]).mean(0)
            cnt += 1
    return np.sqrt(se / cnt) / qs


def score(params, d, build_input, forward=None, horizon=HORIZON, max_eps=MAX_EPS) -> dict:
    """Called by train.py at the end of a run. Returns the metric dict."""
    if forward is None:
        from train import forward as fwd
        forward = fwd
    out = {}
    for s in SPLITS:
        per_ch = rollout_nrmse(params, d, s, build_input, forward, horizon, max_eps)
        out[s] = float(per_ch.mean())
        out[f"{s}_per_channel"] = {n: float(v) for n, v in zip(CHANNELS, per_ch)}
    out["persistence_iid"] = float(persistence_nrmse(d, "val_iid", horizon, max_eps).mean())
    return out


# This module is imported by train.py; there is nothing to run standalone.
# Keeping it import-only means there is exactly one entry point to the
# experiment, so a run cannot be scored by a path the loop does not use.
