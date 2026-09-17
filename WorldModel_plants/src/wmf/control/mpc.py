"""Controller C: cross-entropy-method MPC inside the learned world model.

This is the "what should I do?" half. It never touches the plant to decide - it
rolls candidate action plans forward inside M and keeps the ones that reach the
setpoint, which is the whole point of having a world model in the first place.

Four things that are not optional if the loop is to close on a real plant:

**Receding horizon.** Optimise over H steps, apply only the first move, then
re-optimise from the new measurement. The plan is always wrong; applying only
its first step is what keeps the error from compounding.

**Offset-free correction.** Any model mismatch leaves a permanent setpoint
error - a model that reads 20 K low parks the furnace 20 K low forever. The
running estimate d_t = y_measured - y_predicted is added to every future
prediction, which moves the steady state onto the target without a model that
is exactly right.

**Pessimism and a trust region.** A controller optimising inside a learned model
will find and exploit its errors: if M has not learned that more air also cools,
C will ask for maximum air forever. Scoring a plan by the *worst* ensemble
member removes the incentive, and penalising ensemble disagreement keeps plans
inside the region the data actually covered.

**Rate limits.** A grate does not change speed instantly. Plans are sampled in
increments, so every candidate respects the slew limit by construction rather
than being clipped into something that was never scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from . import model as cvm


@dataclass
class MPCConfig:
    horizon: int = 30              # 3.0 s at a 0.1 s control interval
    population: int = 256
    elites: int = 32
    iterations: int = 4
    gamma: float = 1.0
    # tracking weights on [t_exit, o2_exit, yf_exit, unburnt_bed], normalised.
    # The grate only earns its place in the plan if burnout is on the scorecard.
    q_weight: tuple = (1.0, 1.0, 0.0, 0.7)
    # move suppression per knob [stoker, feed, primary, secondary]
    r_weight: tuple = (0.30, 0.30, 0.05, 0.05)
    # slew limit per control step, in physical units
    # Per control step (1 s), each ~3% of its own span: an air damper takes
    # 30-60 s for full stroke and a grate ramps slower still. The old values
    # were set against a much narrower air range and would now traverse the
    # whole envelope in a few seconds.
    du_max: tuple = (0.0010, 0.060, 0.015, 0.018)
    # penalty on ensemble disagreement - the trust region
    lambda_disagree: float = 2.0
    init_std: tuple = (0.04, 0.40, 0.15, 0.15)
    bias_decay: float = 0.85       # EWMA on the offset-free correction
    seed: int = 0


class MPC:
    def __init__(self, params_list, scaler: cvm.Scaler, u_lo, u_hi, cfg: MPCConfig = None,
                 wm=None):
        """``wm`` is the world model module. It must expose warm_up() and
        predict() with the documented signatures; everything else about it -
        architecture, state size, how it is trained - is free to change."""
        self.wm = wm if wm is not None else cvm
        self.ps = params_list                 # ensemble
        self.sc = scaler
        self.cfg = cfg or MPCConfig()
        self.u_lo = np.asarray(u_lo, np.float32)
        self.u_hi = np.asarray(u_hi, np.float32)
        self.bias = np.zeros(self.wm.N_CV, np.float32)     # in normalised y units
        self._key = jax.random.PRNGKey(self.cfg.seed)
        self._mean = None
        self._pred_next = None                          # for the bias update
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        c = self.cfg
        q = jnp.asarray(c.q_weight); r = jnp.asarray(c.r_weight)
        disc = jnp.asarray(c.gamma ** np.arange(c.horizon))[None, :, None]
        u_std = jnp.asarray(self.sc.u_std); u_mean = jnp.asarray(self.sc.u_mean)

        def rollout_one(p, h0, y0, u_seq, bias):
            h = jnp.broadcast_to(h0, (u_seq.shape[0],) + h0.shape[-1:])
            y = jnp.broadcast_to(y0, (u_seq.shape[0], self.wm.N_CV))
            return self.wm.predict(p, h, y, u_seq, bias=bias)

        def score(h0, y0, u_phys, target_n, bias):
            """u_phys [P, H, 4] physical units -> cost [P]."""
            u_n = (u_phys - u_mean) / u_std
            preds = jnp.stack([rollout_one(p, h0, y0, u_n, bias) for p in self.ps])
            # pessimism: judge a plan by its worst ensemble member
            err = ((preds - target_n[None, None, :]) ** 2 * q) * disc
            per_model = err.sum(axis=(2, 3))               # [K, P]
            track = per_model.max(axis=0)
            # trust region: disagreement means the plan has left the data
            disagree = preds.var(axis=0).mean(axis=(1, 2))
            # move suppression on the increments actually applied
            du = jnp.diff(u_phys, axis=1) / jnp.asarray(self.cfg.du_max)
            move = (du ** 2 * r).sum(axis=(1, 2))
            return track + self.cfg.lambda_disagree * disagree + move

        self._score = jax.jit(score)

        def warm(y_hist, u_hist):
            return jnp.stack([self.wm.warm_up(p, y_hist, u_hist) for p in self.ps]).mean(0)

        self._warm = jax.jit(warm)
        self._sample_j = jax.jit(self._sample_impl)

        def one_step(h0, y0, u_n, bias):
            return jnp.stack([self.wm.predict(p, h0, y0, u_n, bias=bias)
                              for p in self.ps]).mean(0)

        self._one_step = jax.jit(one_step)

    # ------------------------------------------------------------------
    def _sample_impl(self, key, mean, std, u_prev):
        """Sample plans as increments so the slew limit holds by construction."""
        c = self.cfg
        P, H = c.population, c.horizon
        du_max = jnp.asarray(c.du_max)
        # first move is relative to what is currently applied
        base = jnp.concatenate([jnp.asarray(u_prev)[None, :], jnp.asarray(mean)], axis=0)
        dmean = jnp.diff(base, axis=0)                                  # [H, 4]
        noise = jax.random.normal(key, (P, H, 4)) * jnp.asarray(std)
        du = jnp.clip(dmean[None] + noise, -du_max, du_max)
        u = jnp.asarray(u_prev)[None, None, :] + jnp.cumsum(du, axis=1)
        return jnp.clip(u, jnp.asarray(self.u_lo), jnp.asarray(self.u_hi))

    def reset(self, u0):
        self._mean = np.tile(np.asarray(u0, np.float32), (self.cfg.horizon, 1))
        self.bias[:] = 0.0
        self._pred_next = None

    # ------------------------------------------------------------------
    def update_bias(self, y_measured):
        """Offset-free correction from the previous step's one-step prediction."""
        if self._pred_next is None:
            return
        yn = self.sc.ny(np.asarray(y_measured, np.float32))
        innovation = yn - self._pred_next
        self.bias = self.cfg.bias_decay * self.bias + (1 - self.cfg.bias_decay) * innovation

    def act(self, y_hist, u_hist, target, u_prev):
        """One receding-horizon solve. Histories are in physical units."""
        c = self.cfg
        yn = jnp.asarray(self.sc.ny(np.asarray(y_hist, np.float32)))[None]
        un = jnp.asarray(self.sc.nu(np.asarray(u_hist, np.float32)))[None]
        h0 = self._warm(yn, un)                  # shared warm start
        y0 = yn[:, -1]
        target_n = jnp.asarray(self.sc.ny(np.asarray(target, np.float32)))
        bias = jnp.asarray(self.bias)

        if self._mean is None:
            self.reset(u_prev)
        mean = jnp.asarray(self._mean)
        std = jnp.asarray(c.init_std)

        for _ in range(c.iterations):
            self._key, k = jax.random.split(self._key)
            plans = self._sample_j(k, mean, std, jnp.asarray(u_prev))   # [P, H, 4]
            cost = self._score(h0, y0, plans, target_n, bias)
            idx = jnp.argsort(cost)[: c.elites]
            elite = plans[idx]
            mean = elite.mean(axis=0)
            std = jnp.maximum(elite.std(axis=0), jnp.asarray(c.init_std) * 0.1)

        self._mean = np.asarray(mean)
        u = np.asarray(mean[0], np.float32)
        u = np.clip(u, self.u_lo, self.u_hi)

        # remember what the model expects next, so the bias can be updated
        un_plan = (mean[None, :1] - jnp.asarray(self.sc.u_mean)) / jnp.asarray(self.sc.u_std)
        self._pred_next = np.asarray(self._one_step(h0, y0, un_plan, bias)[0, 0])

        # shift the plan for a warm start next step
        self._mean = np.concatenate([self._mean[1:], self._mean[-1:]], axis=0)
        return u
