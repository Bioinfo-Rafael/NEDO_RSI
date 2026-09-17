"""Objective and its proxy indicators.

Nothing in the demo's first stage consumes the reward - the task is prediction,
not control.  It is computed and written to the dataset anyway so that adding a
planner later does not mean regenerating every episode.

PROXIES, NOT SPECIES.  The chemistry is one step, F + s O2 -> P, so there is no
CO and no NOx in this model at all.  What is recorded are surrogates, and they
must be labelled as such wherever they are reported:

    unburnt   integral of Y_F leaving through the flue
    co_proxy  volume of the locally fuel-rich region (equivalence ratio > 1)
    nox_proxy residence-time integral of gas above NOX_T
"""

from __future__ import annotations

import jax.numpy as jnp

from .sim.physics import S_STOICH, Y_O2_AIR

T_TARGET = 1123.0      # K, the 850 degC that incineration regulation fixes on
O2_TARGET = 0.06       # flue O2 mass fraction
NOX_T = 1500.0         # K, threshold of the thermal-NOx surrogate

W_T, W_O2, W_CO, W_UNB, W_E = 1.0e-5, 20.0, 0.5, 2.0, 0.02


def proxies(q, bed_unburnt, static) -> dict:
    """Per-frame scalar indicators. ``q`` is [H, W, 7] for a single frame."""
    mask = static["mask"]
    outlet = static["outlet"]
    n_out = jnp.maximum(outlet.sum(), 1.0)

    probe = static["probe"]
    t_exit = (q[..., 0] * probe).sum() / jnp.maximum(probe.sum(), 1.0)
    o2_exit = (q[..., 5] * outlet).sum() / n_out
    yf_exit = (q[..., 4] * outlet).sum() / n_out

    # local equivalence ratio: fuel-rich where Y_F * s exceeds the O2 present
    phi = (q[..., 4] * S_STOICH) / jnp.maximum(q[..., 5], 1e-6)
    co_proxy = ((phi > 1.0) & (mask > 0) & (q[..., 4] > 1e-4)).sum() / jnp.maximum(mask.sum(), 1.0)
    nox_proxy = ((q[..., 0] > NOX_T) & (mask > 0)).sum() / jnp.maximum(mask.sum(), 1.0)

    return {
        "t_exit": t_exit,
        "o2_exit": o2_exit,
        "yf_exit": yf_exit,
        "co_proxy": co_proxy,
        "nox_proxy": nox_proxy,
        "unburnt_bed": bed_unburnt,
    }


def reward(pr: dict, action_energy) -> jnp.ndarray:
    """Scalar objective built from the proxies. Higher is better."""
    return -(
        W_T * (pr["t_exit"] - T_TARGET) ** 2
        + W_O2 * (pr["o2_exit"] - O2_TARGET) ** 2
        + W_CO * pr["co_proxy"]
        + W_UNB * (pr["unburnt_bed"] + pr["yf_exit"])
        + W_E * action_energy
    )
