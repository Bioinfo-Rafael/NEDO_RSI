"""Geometry rasterisation: the contract that lets one model serve every plant."""

import numpy as np

from wmf.actions import sample_action
from wmf.plants.encode import (
    A_FUEL, A_PRIMARY, A_SECONDARY, A_STOKER, H_GRID, W_GRID,
    check_validity, encode_action,
)


def test_grid_shape(geom):
    assert geom.mask.shape == (H_GRID, W_GRID)
    assert geom.g.shape == (H_GRID, W_GRID, 4)


def test_solid_halo(geom):
    """The stencils use periodic jnp.roll, so a fluid cell on the outer ring
    would silently couple opposite ends of the furnace."""
    m = geom.mask
    assert m[0].sum() == 0 and m[-1].sum() == 0
    assert m[:, 0].sum() == 0 and m[:, -1].sum() == 0


def test_validity_passes(geom):
    check_validity(geom)          # connectivity, passage width, buried actuators


def test_sdf_sign(geom):
    """Positive in the fluid, negative in the solid, and zero nowhere in between."""
    fluid = geom.mask > 0
    assert (geom.sdf[fluid] > 0).all()
    assert (geom.sdf[~fluid] < 0).all()


def test_actuators_and_sensors_in_fluid(geom):
    for cells in list(geom.zone_cells) + list(geom.nozzle_cells) + [geom.feed_cells]:
        for (j, i) in cells:
            assert geom.mask[j, i] > 0
    assert not ((geom.sensor > 0) & (geom.mask == 0)).any()


def test_action_flux_is_exactly_preserved(cfg, geom):
    """The reason a 4-zone and an 8-zone plant are interchangeable: the field
    stores a flux density, so integrating it over the injection area returns
    the scalar it was built from, whatever the cell count."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        act = sample_action(cfg, rng)
        a = encode_action(geom, act)
        assert np.isclose((a[..., A_PRIMARY] * geom.dx).sum(), np.sum(act["primary_air"]), rtol=1e-5)
        assert np.isclose((a[..., A_SECONDARY] * geom.dy).sum(), np.sum(act["secondary_air"]), rtol=1e-5)
        assert np.isclose((a[..., A_FUEL] * geom.dx).sum(), act["waste_feed"], rtol=1e-5)
        assert np.isclose(a[..., A_STOKER].max(), act["stoker_speed"], rtol=1e-5)


def test_action_field_never_lands_in_a_wall(cfg, geom):
    a = encode_action(geom, sample_action(cfg, np.random.default_rng(1)))
    assert not (np.abs(a).sum(-1) * (geom.mask == 0)).any()


def test_wrong_actuator_count_is_rejected(cfg, geom):
    import pytest
    bad = sample_action(cfg, np.random.default_rng(2))
    bad["primary_air"] = bad["primary_air"][:-1] if cfg.n_zones > 1 else [0.5, 0.5]
    with pytest.raises(ValueError):
        encode_action(geom, bad)
