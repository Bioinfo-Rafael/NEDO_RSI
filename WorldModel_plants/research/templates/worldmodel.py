"""Initial linear state-space baseline; editable by the research agent.

Fixed interface, shared raw-history state, synchronous timed updates.
This is a new benchmark baseline: synchronous training changes the number
of updates completed within a wall-time budget relative to the legacy runner.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

N_CV = 4
N_MV = 4

# Everything below this line is the floor, not the design.
STATE = 8
ENSEMBLE = 5
WARMUP = 20
TRAIN_HORIZON = 30
BATCH = 64
LR = 3e-3
TRAINING_UPDATES = 0


def init_params(key):
    k = jax.random.split(key, 4)
    return {
        # A is initialised near the identity so the untrained model says
        # "nothing changes", which is a safe thing for a controller to be told.
        "A": jnp.eye(STATE) * 0.95 + jax.random.normal(k[0], (STATE, STATE)) * 0.01,
        "B": jax.random.normal(k[1], (N_MV, STATE)) * 0.05,
        "C": jax.random.normal(k[2], (STATE, N_CV)) * 0.05,
        "E": jax.random.normal(k[3], (N_CV, STATE)) * 0.05,   # observation -> state
        "x0": jnp.zeros(STATE),
    }


def warm_up(p, y_hist, u_hist):
    """Shared-coordinate state: the raw recent history, flattened."""
    return jnp.concatenate([y_hist, u_hist], axis=-1).reshape(y_hist.shape[0], -1)


def _filter(p, y_hist, u_hist):
    """Run the linear recursion over observed history to estimate the state."""
    b = y_hist.shape[0]
    x = jnp.broadcast_to(p["x0"], (b, STATE))

    def step(x, t):
        x = x @ p["A"] + u_hist[:, t] @ p["B"] + y_hist[:, t] @ p["E"]
        return x, None

    x, _ = jax.lax.scan(step, x, jnp.arange(y_hist.shape[1]))
    return x


def predict(p, state, y_last, u_future, bias=None):
    hist = state.reshape(state.shape[0], -1, N_CV + N_MV)
    x = _filter(p, hist[:, :, :N_CV], hist[:, :, N_CV:])

    def step(carry, t):
        x, y = carry
        x = x @ p["A"] + u_future[:, t] @ p["B"]
        y = y + x @ p["C"]
        return (x, y), (y if bias is None else y + bias)

    _, ys = jax.lax.scan(step, (x, y_last), jnp.arange(u_future.shape[1]))
    return jnp.swapaxes(ys, 0, 1)


def loss_fn(p, y_warm, u_warm, u_future, y_future):
    pred = predict(p, warm_up(p, y_warm, u_warm), y_warm[:, -1], u_future)
    return jnp.mean((pred - y_future) ** 2)


def _adam_init(p):
    z = lambda: jax.tree.map(jnp.zeros_like, p)
    return ((z(), z()), 0)


def _adam(p, g, st, lr, b1=.9, b2=.999, eps=1e-8):
    (m, v), t = st; t += 1
    m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
    v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
    mh = jax.tree.map(lambda a: a / (1 - b1 ** t), m)
    vh = jax.tree.map(lambda a: a / (1 - b2 ** t), v)
    return jax.tree.map(lambda a, b, c: a - lr * b / (jnp.sqrt(c) + eps), p, mh, vh), ((m, v), t)


def train(data, stats, seconds: float, seed: int = 0):
    from prepare import windows
    global TRAINING_UPDATES
    TRAINING_UPDATES = 0

    per_member = seconds / ENSEMBLE
    out = []
    for k in range(ENSEMBLE):
        rng = np.random.default_rng(1000 + seed * 17 + k)
        gen = windows(data, stats, TRAIN_HORIZON, BATCH, rng, WARMUP)
        p = init_params(jax.random.PRNGKey(seed * 17 + k))
        opt = _adam_init(p)

        @jax.jit
        def step(p, opt, yw, uw, uf, yf):
            l, g = jax.value_and_grad(loss_fn)(p, yw, uw, uf, yf)
            p, opt = _adam(p, g, opt, LR)
            return p, opt, l

        t_end = time.time() + per_member
        while time.time() < t_end:
            yw, uw, uf, yf = next(gen)
            p, opt, loss = step(p, opt, jnp.asarray(yw), jnp.asarray(uw),
                             jnp.asarray(uf), jnp.asarray(yf))
            jax.block_until_ready(loss)
            TRAINING_UPDATES += 1
        out.append(p)
    return out
