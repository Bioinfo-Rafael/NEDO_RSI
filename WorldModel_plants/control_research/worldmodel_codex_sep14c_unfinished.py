"""Bootstrapped autoregressive identification in shared observation coordinates."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

N_CV = 4
N_MV = 4
WARMUP = 20
ENSEMBLE = 5
RIDGE = 1e-2


def init_params(key):
    return {"coef": jnp.zeros((17, N_CV))}


def warm_up(params, y_hist, u_hist):
    return jnp.concatenate([y_hist[:, -2], u_hist[:, -1]], axis=-1)


def _features(y, dy, u, du, xp):
    return xp.concatenate([y, dy, u, du, xp.ones_like(y[..., :1])], axis=-1)


def predict(params, state, y_last, u, bias=None):
    import jax

    def step(carry, action):
        y, dy, prev_u = carry
        delta = _features(y, dy, prev_u, action - prev_u, jnp) @ params["coef"]
        yn = y + delta
        return (yn, delta, action), yn if bias is None else yn + bias

    _, ys = jax.lax.scan(step, (y_last, y_last - state[:, :4], state[:, 4:]),
                         jnp.swapaxes(u, 0, 1))
    return jnp.swapaxes(ys, 0, 1)


def train(data, stats, seconds, seed=0, plant=None):
    y = (data["y"] - stats["y_mean"]) / stats["y_std"]
    u = (data["u"] - stats["u_mean"]) / stats["u_std"]
    x = _features(y[:, 1:-1], y[:, 1:-1] - y[:, :-2], u[:, 1:-1],
                  u[:, 2:] - u[:, 1:-1], np).astype(np.float64)
    target = (y[:, 2:] - y[:, 1:-1]).astype(np.float64)
    out = []
    for k in range(ENSEMBLE):
        rng = np.random.default_rng(1000 + seed * 17 + k)
        episodes = rng.integers(0, len(y), len(y))
        xx, yy = x[episodes].reshape(-1, x.shape[-1]), target[episodes].reshape(-1, N_CV)
        reg = RIDGE * np.eye(xx.shape[-1])
        coef = np.linalg.solve(xx.T @ xx + reg, xx.T @ yy)
        out.append({"coef": jnp.asarray(coef, dtype=jnp.float32)})
    return out
