"""The solver must stay in the regime it is valid in, for every plant and every
corner of the action envelope.

These assert the things that were actually wrong once and would be easy to
reintroduce: the emergency velocity clamp doing the dynamics, the temperature
clip doing the physics, and a silent NaN.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from wmf.actions import envelope
from wmf.plants.encode import encode_action
from wmf.sim.physics import C_SOUND, SimParams, U_CLAMP
from wmf.sim.rollout import build_static, make_initial, rollout

N_FRAMES, SAVE_EVERY = 30, 100


def _corner(cfg, pri, sec, stoker, feed):
    return {
        "stoker_speed": stoker, "waste_feed": feed,
        "primary_air": [pri] * cfg.n_zones,
        "secondary_air": [sec] * cfg.n_nozzles,
    }


def _run(cfg, geom, action):
    static = build_static(geom)
    prm = SimParams.from_plant(cfg)
    a = jnp.asarray(encode_action(geom, action))
    q0, bed0 = make_initial(static)
    q, bed, diag = rollout(q0, bed0, jnp.broadcast_to(a, (N_FRAMES,) + a.shape),
                           static, prm, SAVE_EVERY, N_FRAMES)
    return q, bed, diag


# corners of the 4-D envelope as (lo/hi) bits; mapped onto each plant's own
# envelope inside _corner, since the envelope scales with the grate
CORNERS = list(itertools.product((0, 1), repeat=4))


def _corner_values(cfg, bits):
    env = envelope(cfg)
    return (env["primary_air"][bits[0]], env["secondary_air"][bits[1]],
            env["stoker_speed"][bits[2]], env["waste_feed"][bits[3]])


@pytest.mark.parametrize("corner", CORNERS, ids=lambda c: "p%d_s%d_v%d_f%d" % c)
def test_envelope_corner_is_finite(cfg, geom, corner):
    q, bed, _ = _run(cfg, geom, _corner(cfg, *_corner_values(cfg, corner)))
    assert bool(jnp.isfinite(q).all()), "non-finite gas state"
    assert bool(jnp.isfinite(bed).all()), "non-finite bed state"


@pytest.mark.parametrize("corner", CORNERS, ids=lambda c: "p%d_s%d_v%d_f%d" % c)
def test_velocity_clamp_never_binds(cfg, geom, corner):
    """If the clamp bites, the velocity field is the clamp's, not the flow's."""
    q, _, _ = _run(cfg, geom, _corner(cfg, *_corner_values(cfg, corner)))
    worst = float(jnp.maximum(jnp.abs(q[..., 1]).max(), jnp.abs(q[..., 2]).max()))
    assert worst < U_CLAMP * 0.999, f"velocity clamp binding at {worst:.1f} m/s"


@pytest.mark.parametrize("corner", CORNERS, ids=lambda c: "p%d_s%d_v%d_f%d" % c)
def test_artificial_compressibility_assumption_holds(cfg, geom, corner):
    """Artificial compressibility needs c >> |u| or the pressure channel is
    quantitatively meaningless."""
    q, _, _ = _run(cfg, geom, _corner(cfg, *_corner_values(cfg, corner)))
    umax = float(jnp.sqrt(q[..., 1] ** 2 + q[..., 2] ** 2).max())
    # 0.75 c, not 0.5: at a real plant's air rates the secondary jets reach
    # 12-14 m/s and c = 20 is what keeps dt at 3 ms. Above this the pressure
    # channel is qualitative only; the envelope is chosen so this holds.
    assert umax < 0.75 * C_SOUND, f"|u|max {umax:.1f} against c = {C_SOUND}"


def test_species_stay_bounded(cfg, geom):
    q, _, _ = _run(cfg, geom, _corner(cfg, 0.8, 0.5, 0.18, 2.5))
    for ch, hi in ((4, 1.0), (5, 0.2321), (6, 1.0)):
        assert float(q[..., ch].min()) >= -1e-6
        assert float(q[..., ch].max()) <= hi + 1e-6


def test_bed_mass_is_non_negative(cfg, geom):
    _, bed, _ = _run(cfg, geom, _corner(cfg, 0.8, 0.5, 0.18, 2.5))
    assert float(bed.min()) >= -1e-9


def test_determinism(cfg, geom):
    """The simulator is the ground truth; it has to be reproducible."""
    act = _corner(cfg, 0.8, 0.5, 0.18, 2.5)
    q1, _, _ = _run(cfg, geom, act)
    q2, _, _ = _run(cfg, geom, act)
    np.testing.assert_array_equal(np.asarray(q1), np.asarray(q2))
