"""Recurrent model of the controlled variables, for use inside MPC.

Why recurrent here, when the full-field model is not
---------------------------------------------------
The field model sees the whole gas state, so it is close to Markovian in what
it observes and a feed-forward residual step is defensible. The controller sees
only three scalars at the flue. The thing that decides what happens next - how
much fuel is lying where on the grate - is invisible to it, and it arrives with
~17 s of transport delay. A memoryless map from (y_t, u_t) to y_{t+1} therefore
cannot be right: the same flue reading with a full grate and an empty one leads
somewhere different. The recurrent state is where that hidden bed state gets
inferred, from the recent history the controller already has.

Trained by unrolling over the *control* horizon, not one step. A one-step model
is fit to a job it is never asked to do; MPC only ever queries it open loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

N_CV = 4
N_MV = 4


def init_params(key, hidden: int = 48):
    """Minimal GRU + linear read-out, written out so it stays inspectable."""
    k = jax.random.split(key, 6)
    din = N_CV + N_MV

    def glorot(kk, shape):
        lim = np.sqrt(6.0 / (shape[0] + shape[1]))
        return jax.random.uniform(kk, shape, minval=-lim, maxval=lim)

    return {
        # update / reset / candidate gates, each over [input, hidden]
        "Wz": glorot(k[0], (din + hidden, hidden)), "bz": jnp.zeros(hidden),
        "Wr": glorot(k[1], (din + hidden, hidden)), "br": jnp.zeros(hidden),
        "Wh": glorot(k[2], (din + hidden, hidden)), "bh": jnp.zeros(hidden),
        # read-out predicts the *increment* in y, which keeps the scale small
        # and makes "nothing changes" the zero-parameter default
        "Wo": glorot(k[3], (hidden, N_CV)) * 0.1, "bo": jnp.zeros(N_CV),
        "h0": jnp.zeros(hidden),
    }


def gru_cell(p, x, h):
    xh = jnp.concatenate([x, h], axis=-1)
    z = jax.nn.sigmoid(xh @ p["Wz"] + p["bz"])
    r = jax.nn.sigmoid(xh @ p["Wr"] + p["br"])
    xhr = jnp.concatenate([x, r * h], axis=-1)
    hh = jnp.tanh(xhr @ p["Wh"] + p["bh"])
    return (1.0 - z) * h + z * hh


def warm_up(p, y_hist, u_hist):
    """Run the recurrence over observed history to infer the hidden state."""
    b = y_hist.shape[0]
    h = jnp.broadcast_to(p["h0"], (b, p["h0"].shape[0]))

    def step(h, t):
        x = jnp.concatenate([y_hist[:, t], u_hist[:, t]], axis=-1)
        return gru_cell(p, x, h), None

    h, _ = jax.lax.scan(step, h, jnp.arange(y_hist.shape[1]))
    return h


def predict(p, h, y_last, u_future, bias=None):
    """Open-loop forecast of the CVs for a given action plan.

    ``bias`` is the offset-free correction: the controller's running estimate of
    what the model is getting wrong, added to every predicted step. Without it a
    model that is 20 K low leaves 20 K of permanent setpoint error.
    """
    def step(carry, t):
        h, y = carry
        x = jnp.concatenate([y, u_future[:, t]], axis=-1)
        h = gru_cell(p, x, h)
        y = y + h @ p["Wo"] + p["bo"]
        out = y if bias is None else y + bias
        return (h, y), out

    (_, _), ys = jax.lax.scan(step, (h, y_last), jnp.arange(u_future.shape[1]))
    return jnp.swapaxes(ys, 0, 1)          # [B, H, N_CV]


def rollout_loss(p, y_warm, u_warm, u_future, y_future):
    h = warm_up(p, y_warm, u_warm)
    pred = predict(p, h, y_warm[:, -1], u_future)
    # Weight early steps a little more: MPC applies only the first move, so
    # near-term accuracy is what actually closes the loop.
    w = jnp.exp(-jnp.arange(pred.shape[1]) / (pred.shape[1] / 2.0))[None, :, None]
    return jnp.mean(w * (pred - y_future) ** 2) / jnp.mean(w)


@dataclass(frozen=True)
class Scaler:
    y_mean: np.ndarray
    y_std: np.ndarray
    u_mean: np.ndarray
    u_std: np.ndarray

    def ny(self, y): return (y - self.y_mean) / self.y_std
    def dy(self, y): return y * self.y_std + self.y_mean
    def nu(self, u): return (u - self.u_mean) / self.u_std
