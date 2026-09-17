"""Glue: Geometry (numpy, per plant) -> static jnp fields -> coupled rollout.

The coupled step is gas + bed with the bed acting as the bottom boundary
source of the gas.  Everything is inside ``lax.scan`` so a whole episode is a
single compiled call, and ``rollout_batch`` ``vmap``s it over plants.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ..plants.encode import Geometry, H_GRID, W_GRID, _NOZZLE_LO
from ..actions import envelope
from .. import reward as rewardmod
from . import bed as bedmod
from .physics import (
    SimParams, T, U, V, P, YF, YO2, YP, N_Q,
    RHO0, RHO_AIR, CP, Y_O2_AIR, gas_step, initial_state,
)
from .physics import T_AIR as T_AIR_AMBIENT  # noqa: F401  (documented in coupled_step)


# ------------------------------------------------------------------ static
# Hot-face temperature of the lower-furnace refractory lining, and how much of
# the waterwall's heat-transfer coefficient it has. Refractory exists to keep
# heat in, so it is close to adiabatic; the waterwall above it is what actually
# removes the heat to the steam side. Giving the refractory zone the waterwall's
# coefficient would make it a 1250 K heat bath feeding the gas, which is the
# opposite of what a lining does.
T_REFRACTORY = 1250.0    # K
REFRACTORY_LOSS = 0.10   # fraction of the waterwall's lam_loss


def build_static(geom: Geometry) -> dict:
    """Everything the compiled step needs, as fixed-shape jnp arrays."""
    cfg = geom.cfg
    mask = geom.mask.astype(np.float32)

    # One-hot of the first gas cell above the grate, used to *read* the gas
    # state that drives the bed.
    surf = np.zeros((H_GRID, W_GRID), dtype=np.float32)
    grate = np.zeros(W_GRID, dtype=np.float32)
    for z_cells in geom.zone_cells:
        for (j, i) in z_cells:
            surf[j, i] = 1.0
            grate[i] = 1.0

    # Separate, vertically spread map used to *write* the bed fluxes back into
    # the gas.  Dumping the whole volatile and char-combustion release into one
    # cell layer makes that layer run away (the temperature clip ends up doing
    # the physics); a real bed releases into a freeboard layer, not a plane.
    # Weights sum to 1 per column so the total flux is unchanged.
    release = np.zeros((H_GRID, W_GRID), dtype=np.float32)
    w_prof = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    for i in range(W_GRID):
        j0 = int(np.flatnonzero(surf[:, i])[0]) if surf[:, i].any() else -1
        if j0 < 0:
            continue
        acc = 0.0
        for k, wgt in enumerate(w_prof):
            j = j0 + k
            if j < H_GRID and mask[j, i] > 0:
                release[j, i] = wgt
                acc += wgt
        if acc > 0:
            release[:, i] /= acc

    # secondary-air injection direction: +1 blows right (left wall), -1 left
    sec_dir = np.zeros((H_GRID, W_GRID), dtype=np.float32)
    for cells in geom.nozzle_cells:
        for (j, i) in cells:
            sec_dir[j, i] = 1.0 if i < W_GRID // 2 else -1.0

    outlet = np.zeros((H_GRID, W_GRID), dtype=np.float32)
    outlet[H_GRID - 2, geom.flue_cols] = 1.0

    # Furnace-exit measurement plane. The flue slot itself is a 16-cell hole in
    # the top wall and its temperature is dominated by the boundary treatment
    # (it read 520-540 K whether the furnace averaged 740 K or 1040 K), so the
    # furnace temperature is taken where a plant's FEGT thermocouples sit: a
    # plane above the secondary-air zone and below the flue. Species are still
    # read at the flue, where the analysers are.
    probe = np.zeros((H_GRID, W_GRID), dtype=np.float32)
    j_probe = int(0.90 * H_GRID)
    probe[j_probe, :] = mask[j_probe, :]

    # feed profile: normalised so that sum(profile * dx) == 1
    feed = np.zeros(W_GRID, dtype=np.float32)
    for (j, i) in geom.feed_cells:
        feed[i] = 1.0
    tot = feed.sum() * geom.dx
    feed = feed / tot if tot > 0 else feed

    # Wall temperature is a field, not a number. A mass-burn boiler is built in
    # two parts: the lower furnace is refractory-lined with a hot face at
    # 1100-1400 K, and only above the secondary-air nozzles does it become
    # membrane waterwall at the steam-side temperature. Using the waterwall
    # value everywhere makes the combustion zone a cold surface, and that
    # removes the furnace's continuous ignition source: measured at lambda 1.73
    # the freeboard mixed to 539 K, where exp(-Ea/RT) is 1.6e-6 and a mixture
    # simply does not autoignite, so the gas stayed on a cold branch that a
    # real furnace's refractory would never allow.
    t_wall = np.full((H_GRID, W_GRID), float(cfg.wall.temperature), dtype=np.float32)
    refractory_top = int(_NOZZLE_LO * H_GRID)
    t_wall[:refractory_top, :] = T_REFRACTORY
    loss_scale = np.ones((H_GRID, W_GRID), dtype=np.float32)
    loss_scale[:refractory_top, :] = REFRACTORY_LOSS

    grate_idx = np.flatnonzero(grate > 0)
    last_col = int(grate_idx[-1]) if grate_idx.size else 0

    # as-received mass split of the incoming waste (bed channels)
    w = cfg.waste
    w_frac = w.moisture
    dry = 1.0 - w.moisture
    v_frac = dry * w.volatile_frac
    c_frac = dry * w.char_frac

    return {
        "mask": jnp.asarray(mask),
        "surf": jnp.asarray(surf),
        "release": jnp.asarray(release),
        "sec_dir": jnp.asarray(sec_dir),
        "outlet": jnp.asarray(outlet),
        "probe": jnp.asarray(probe),
        "t_wall": jnp.asarray(t_wall),
        "loss_scale": jnp.asarray(loss_scale),
        "grate_cols": jnp.asarray(grate),
        "feed_profile": jnp.asarray(feed),
        "last_grate_col": last_col,
        "dx": float(geom.dx),
        "dy": float(geom.dy),
        "feed_nominal": float(np.mean(envelope(cfg)["waste_feed"])),
        "w_frac": float(w_frac),
        "v_frac": float(v_frac),
        "c_frac": float(c_frac),
    }


# ------------------------------------------------------------ coupled step
def coupled_step(carry, a_field, static, prm: SimParams):
    q, bed = carry
    surf, dy = static["surf"], static["dy"]

    # Radiation reaches the bed from the whole freeboard, not from the one cell
    # the cold primary air enters. Using that cell as the bed's driver is what
    # made the furnace go out at realistic residence times.
    mask = static["mask"]
    # Radiosity is linear in T^4, so the column average has to be taken of T^4,
    # not of T. The freeboard is strongly non-uniform - a thin flame sheet at
    # 1750-1940 K sitting in gas whose volume mean is 600 K - and averaging T
    # first understates the radiation to the bed by about 7x (1.6e11 against
    # 1.1e12 K^4, i.e. an effective 630 K against 1026 K). With the wrong one
    # the bed's balance came out at -71 kW/m2 and the hot branch did not exist
    # at any air setting; with the right one it is +275 kW/m2.
    # ...and the bed sees the whole furnace, not the column directly above it.
    # A freeboard loaded with soot and CO2 is an optically thick enclosure, so
    # the bed at the feed end is radiated by the flame downstream of it - which
    # is what an ignition arch is built to do, and the only thing that lights
    # fresh waste. Per-column radiation leaves the feed end dark: the hot bed
    # the episode starts with washes off the end in one residence time and the
    # waste replacing it never ignites, which is exactly the 600 s decay we
    # measured.
    t_gas4 = ((q[..., T] ** 4 * mask).sum() / jnp.maximum(mask.sum(), 1.0)) * jnp.ones(q.shape[1])
    o2_surf = (q[..., YO2] * surf).sum(axis=0)

    stoker = (a_field[..., 3] * surf).sum() / jnp.maximum(surf.sum(), 1.0)
    waste = (a_field[..., 2] * static["dx"]).sum()

    # primary air through each column [kg/m2/s], and the O2 it carries
    air_mass = a_field[..., 0].sum(axis=0) * RHO_AIR
    o2_supply = air_mass * Y_O2_AIR

    bed, r_pyro, char_heat, unburnt, t_bed, q_rad = bedmod.bed_step(
        bed, t_gas4, o2_surf, o2_supply, air_mass, stoker, waste, static, prm.dt
    )

    # bed fluxes -> gas sources in the first cell above the grate
    inv_mass = 1.0 / (RHO0 * dy)                       # [m2/kg]
    rel = static["release"]
    fuel_src = rel * (r_pyro * inv_mass)[None, :]      # [1/s]
    heat_src = rel * (char_heat * inv_mass)[None, :]   # [K/s]

    # Whatever the bed loses by radiation, the freeboard gains. The furnace is
    # optically thick, so it is absorbed through the column rather than in the
    # first cell above the grate.
    n_col = jnp.maximum(mask.sum(axis=0), 1.0)
    heat_src = heat_src - (mask * (q_rad / (RHO0 * CP * dy * n_col))[None, :])

    # The primary air leaves the bed at the bed's temperature, not at ambient.
    # ALL of it does: it enters underneath and passes through the fuel bed, so
    # every cell of the primary-air injection field is fed preheated air, not
    # just the ones that coincide with the bed surface. Gating this on ``surf``
    # left 60% of the primary air entering the freeboard at 300 K, which cooled
    # the gas by -52 K/s on average against +35 K/s of reaction heat, and the
    # flame went out in 12 s with the bed still at 1107 K. The secondary air is
    # separately handled inside ``gas_step`` and does arrive at ambient.
    t_air_in = jnp.broadcast_to(t_bed[None, :], q.shape[:2])
    q, omega = gas_step(q, a_field, fuel_src, heat_src, static, prm,
                        t_air_in=t_air_in)

    return (q, bed), (omega, unburnt)


# ------------------------------------------------------------------ episode
@functools.partial(jax.jit, static_argnames=("save_every", "n_frames", "prm"))
def rollout(q0, bed0, a_seq, static, prm: SimParams, save_every: int, n_frames: int):
    """Run ``n_frames`` control frames, each ``save_every`` gas steps.

    ``a_seq`` [n_frames, H, W, 4] - the action is held constant within a frame,
    so the control interval is ``save_every * prm.dt``.
    Returns q [n_frames, H, W, 7], bed [n_frames, W, 3], diag dict.
    """

    def frame(carry, a_field):
        def inner(c, _):
            return coupled_step(c, a_field, static, prm)

        carry, (omega, unburnt) = jax.lax.scan(inner, carry, None, length=save_every)
        q, bed = carry
        pr = rewardmod.proxies(q, unburnt.mean(), static)
        energy = (a_field[..., 0].sum() * static["dx"]
                  + a_field[..., 1].sum() * static["dy"])
        diag = dict(pr)
        diag["omega"] = omega.mean(axis=0)      # mean reaction rate field over the frame
        diag["reward"] = rewardmod.reward(pr, energy)
        return carry, (q, bed, diag)

    _, (q_hist, bed_hist, diag) = jax.lax.scan(frame, (q0, bed0), a_seq)
    return q_hist, bed_hist, diag


def make_initial(static, t_init: float = 1100.0, waste_feed: float | None = None,
                 t_bed: float = 1150.0):
    """Initial (gas, bed) state. ``waste_feed`` [kg/s] sets the bed inventory,
    so the episode starts at the firing rate its own feed implies; by default
    the plant's nominal feed (middle of its envelope)."""
    if waste_feed is None:
        waste_feed = static["feed_nominal"]
    return (initial_state(static, t_init),
            bedmod.initial_bed(static, waste_feed, t_bed))


# ------------------------------------------------------------------ batched
def rollout_batch(q0, bed0, a_seq, statics, prm: SimParams, save_every: int, n_frames: int):
    """``rollout`` vmapped over plants (API.md 2, Level 3).

    All plants share the 64x64 grid, so their static fields stack and one
    compiled kernel covers the whole batch; that is the reason the geometry is
    carried as a mask rather than as a per-plant mesh.
    ``statics`` is a dict of stacked [B, ...] arrays.
    """
    fn = jax.vmap(
        lambda q, b, a, st: rollout(q, b, a, st, prm, save_every, n_frames),
        in_axes=(0, 0, 0, 0),
    )
    return fn(q0, bed0, a_seq, statics)


def stack_statics(statics: list[dict]) -> dict:
    """Stack per-plant static dicts into one [B, ...] pytree for rollout_batch."""
    out = {}
    for k in statics[0]:
        vals = [s[k] for s in statics]
        out[k] = jnp.stack([jnp.asarray(v) for v in vals])
    return out
