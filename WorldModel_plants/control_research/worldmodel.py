"""THE FILE YOU REWRITE. There is no intended architecture here.

What this file must do is fixed by four function signatures. What it does it
with is not decided, and nothing below should be read as a recommendation.

The implementation you inherit is a **linear state-space model** - the classical
system-identification answer, chosen precisely because it is not a neural
architecture and therefore does not point at one. It is a floor to beat, not a
starting point to refine. Deleting all of it is a normal move.

    x_{t+1} = A x_t + B u_t
    y_t     = C x_t

Nobody has decided whether the replacement should be recurrent, convolutional,
attentional, an operator, an ensemble of small linear models, a Gaussian
process, physics with a learned residual, or something with no accepted name.
That is the open question, and it is yours.

The contract
------------
    init_params(key)                        -> params for one ensemble member
    warm_up(params, y_hist, u_hist)         -> state  [B, ...]   (any shape)
    predict(params, state, y_last, u, bias) -> [B, H, N_CV]
    train(data, stats, seconds, seed)       -> list of params

Also keep `N_CV = 4`, `WARMUP`, `ENSEMBLE`. Everything else - the state
representation, how long a history you look at, what you optimise, how many
members and whether they even share an architecture - is free.

`warm_up` returns whatever your model needs as its state; the controller treats
it as opaque and hands it straight back to `predict`. One caveat learned the
hard way: the controller averages that state across ensemble members, so it must
live in coordinates the members share. Raw history does. Independently-learned
latents do not.

What you are scored on
----------------------
`control_cost`: how well the plant tracks its setpoints when the controller
plans inside your model. Not prediction error. The two have already been
measured moving in opposite directions on this plant.
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
            p, opt, _ = step(p, opt, jnp.asarray(yw), jnp.asarray(uw),
                             jnp.asarray(uf), jnp.asarray(yf))
        out.append(p)
    return out
