"""THE ONLY FILE YOU EDIT.

Model, optimiser and training loop for the incinerator world model.  Runs for a
fixed wall-clock budget (prepare.TRAIN_SECONDS), then scores itself with the
read-only harness in evaluate.py and prints a summary.

Metric: val_nrmse, lower is better.  See program.md.

This baseline is deliberately weak.  Known weaknesses, left for you to find:
  * `build_input` throws `g` and `dx` away, so the model cannot see the walls,
    the flue, the air inlets, or how big a metre is on this grid.
  * two conv layers, 16 channels - a receptive field of 5 cells, smaller than
    the structures it has to predict.
  * two-step training, though it is scored on a 16-step rollout.
  * plain L2 on every channel equally, so temperature drowns out the species.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import prepare
from evaluate import score

# ----------------------------------------------------------------- config
WIDTH = 16
DEPTH = 2
LR = 1e-3
BATCH = 16
TRAIN_HORIZON = 2


# ------------------------------------------------------------------ model
def init_params(key, in_ch: int, out_ch: int):
    ks = jax.random.split(key, DEPTH + 1)
    params, c_in = [], in_ch
    for d in range(DEPTH):
        w = jax.random.normal(ks[d], (3, 3, c_in, WIDTH)) * np.sqrt(2.0 / (9 * c_in))
        params.append((w, jnp.zeros(WIDTH)))
        c_in = WIDTH
    w = jax.random.normal(ks[DEPTH], (3, 3, c_in, out_ch)) * 0.01
    params.append((w, jnp.zeros(out_ch)))
    return params


def _conv(x, w, b, dilation=1):
    y = jax.lax.conv_general_dilated(
        x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"),
        rhs_dilation=(dilation, dilation),
    )
    return y + b


def forward(params, x):
    for depth, (w, b) in enumerate(params[:-1]):
        x = jax.nn.gelu(_conv(x, w, b, dilation=2 ** depth))
    w, b = params[-1]
    return _conv(x, w, b)


def build_input(x, a, g, dx):
    """State and actuators only.  `g` (geometry) and `dx` (grid spacing) are
    accepted and discarded - that is one of the things left to fix."""
    del g, dx
    return jnp.concatenate([x, a], axis=-1)


def loss_fn(params, x, a, g, dx, y, dq_scale):
    """Train on open-loop predictions, including the model's own feedback."""
    cumulative = jnp.zeros_like(x)
    loss = 0.0
    for k in range(TRAIN_HORIZON):
        inp = build_input(x + cumulative * dq_scale, a[:, k], g, dx)
        cumulative = cumulative + forward(params, inp)
        loss = loss + jnp.mean((cumulative - y[:, k]) ** 2)
    return loss / TRAIN_HORIZON


def rollout_batches(d, batch_size, rng):
    """Sample consecutive training frames with cumulative increment targets."""
    q, a = d["train_q"], d["train_a"]
    while True:
        ep = rng.integers(0, q.shape[0], batch_size)
        t = rng.integers(0, q.shape[1] - TRAIN_HORIZON, batch_size)
        times = t[:, None] + np.arange(TRAIN_HORIZON)[None, :]
        initial = q[ep, t]
        x = (initial - d["q_mean"]) / d["q_std"]
        actions = a[ep[:, None], times] / d["a_std"]
        y = (q[ep[:, None], times + 1] - initial[:, None]) / d["dq_std"]
        yield (x, actions, d["train_g"][ep], d["train_dx"][ep]), y


# -------------------------------------------------------------- optimiser
def adam_init(params):
    zeros = lambda: jax.tree.map(jnp.zeros_like, params)
    return ((zeros(), zeros()), 0)   # m and v must be separate trees


def adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    (m, v), t = state
    t += 1
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
    mh = jax.tree.map(lambda z: z / (1 - b1 ** t), m)
    vh = jax.tree.map(lambda z: z / (1 - b2 ** t), v)
    params = jax.tree.map(lambda p, aa, bb: p - lr * aa / (jnp.sqrt(bb) + eps), params, mh, vh)
    return params, ((m, v), t)


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=prepare.TRAIN_SECONDS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    d = prepare.load()
    rng = np.random.default_rng(args.seed)
    stream = rollout_batches(d, BATCH, rng)
    dq_scale = jnp.asarray(d["dq_std"] / d["q_std"])

    (x0, a0, g0, dx0), y0 = next(stream)
    inp0 = build_input(jnp.asarray(x0), jnp.asarray(a0[:, 0]), jnp.asarray(g0), jnp.asarray(dx0))
    params = init_params(jax.random.PRNGKey(args.seed), inp0.shape[-1], y0.shape[-1])
    opt = adam_init(params)

    @jax.jit
    def step(params, opt, x, a, g, dx, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, a, g, dx, y, dq_scale)
        params, opt = adam_update(params, grads, opt, LR)
        return params, opt, loss

    t_train = time.time()
    t_end = t_train + args.seconds
    n, running = 0, None
    while time.time() < t_end:
        (x, a, g, dx), y = next(stream)
        params, opt, loss = step(params, opt, jnp.asarray(x), jnp.asarray(a),
                                 jnp.asarray(g), jnp.asarray(dx), jnp.asarray(y))
        running = float(loss) if running is None else 0.99 * running + 0.01 * float(loss)
        n += 1
    train_seconds = time.time() - t_train

    res = score(params, d, build_input)
    n_params = sum(int(np.asarray(w).size + np.asarray(b).size) for w, b in params)

    print("---")
    print(f"val_nrmse:        {res['val_iid']:.6f}")
    print(f"val_b_nrmse:      {res['val_b']:.6f}")
    print(f"val_c_nrmse:      {res['val_c']:.6f}")
    print(f"persistence_iid:  {res['persistence_iid']:.6f}")
    print(f"gen_gap_b:        {res['val_b'] / max(res['val_iid'], 1e-9):.3f}")
    print(f"train_loss:       {running:.6f}")
    print(f"training_seconds: {train_seconds:.1f}")
    print(f"total_seconds:    {time.time() - t_start:.1f}")
    print(f"num_steps:        {n}")
    print(f"num_params_K:     {n_params / 1000:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
