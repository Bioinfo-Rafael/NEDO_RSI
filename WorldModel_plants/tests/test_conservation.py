"""Golden tests on the operators and the chemistry.

These are the checks a proposed solver change has to survive.  They assert the
invariants directly rather than eyeballing a full coupled run, so a regression
points at the term that broke.
"""

import jax.numpy as jnp
import numpy as np

from wmf.sim.physics import (
    S_STOICH, SimParams, _lap, _upwind, react, reaction_rate,
)

RNG = np.random.default_rng(0)


def _field(h=32, w=32):
    return jnp.asarray(RNG.random((h, w)), dtype=jnp.float32)


def test_laplacian_conserves_total():
    """Diffusion may move a scalar around but must not create or destroy it."""
    f = _field()
    assert abs(float(_lap(f, 0.1, 0.2).sum())) < 1e-3


def test_uniform_advection_conserves_total():
    """With a uniform velocity the advective and conservative forms coincide,
    so the domain integral must be unchanged."""
    f = _field()
    u = jnp.full_like(f, 2.0)
    v = jnp.full_like(f, -1.5)
    assert abs(float(_upwind(f, u, v, 0.1, 0.1).sum())) < 1e-2


def test_reaction_stoichiometry_is_exact():
    """F + s O2 -> (1+s) P.  Every atom the fuel loses must show up."""
    prm = SimParams()
    t0 = jnp.asarray([1400.0, 1200.0, 900.0])
    yf0 = jnp.asarray([0.05, 0.10, 0.02])
    yo0 = jnp.asarray([0.20, 0.15, 0.23])
    yp0 = jnp.zeros(3)
    (t1, yf1, yo1, yp1), _ = react(t0, yf0, yo0, yp0, prm.dt, prm)

    d_f = yf0 - yf1
    assert (d_f >= 0).all()
    # rtol is 1e-3, not tighter: these are float32 differences of nearly equal
    # numbers (0.23 - 0.2298), so the subtraction itself costs ~4 digits.  The
    # measured ratios are 3.0000033 / 3.0000038 / 3.0002348 against an exact 3.
    np.testing.assert_allclose(yo0 - yo1, S_STOICH * d_f, rtol=1e-3, atol=1e-9)
    np.testing.assert_allclose(yp1 - yp0, (1.0 + S_STOICH) * d_f, rtol=1e-3, atol=1e-9)


def test_reaction_energy_release():
    """Temperature rise must equal q_cp times the fuel actually burnt."""
    prm = SimParams()
    t0 = jnp.asarray([1400.0])
    yf0, yo0 = jnp.asarray([0.05]), jnp.asarray([0.20])
    (t1, yf1, _, _), _ = react(t0, yf0, yo0, jnp.zeros(1), prm.dt, prm)
    np.testing.assert_allclose(t1 - t0, prm.q_cp * (yf0 - yf1), rtol=1e-3)


def test_reaction_never_over_consumes():
    """Starve the cell of O2 and the limiter, not the timestep, must stop it."""
    prm = SimParams()
    (_, yf1, yo1, _), _ = react(
        jnp.asarray([2000.0]), jnp.asarray([0.5]), jnp.asarray([1e-4]),
        jnp.zeros(1), prm.dt, prm,
    )
    assert float(yo1.min()) >= -1e-9
    assert float(yf1.min()) >= -1e-9


def test_reaction_rate_is_negligible_below_ignition():
    """The regimes depend on there being a real ignition threshold."""
    cold = float(reaction_rate(jnp.asarray(600.0), jnp.asarray(0.05), jnp.asarray(0.2)))
    hot = float(reaction_rate(jnp.asarray(1400.0), jnp.asarray(0.05), jnp.asarray(0.2)))
    assert hot > 1000 * cold
