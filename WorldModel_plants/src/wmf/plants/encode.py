"""Rasterise a PlantConfig onto the fixed 64x64 grid.

This module is the single place where "設備固有の情報" becomes "固定形状の配列".
Everything downstream (solver, dataset, NN) sees only fixed-shape arrays, so a
plant with 4 air zones and a plant with 8 are handled by identical code.

Grid convention
---------------
Arrays are indexed ``[j, i]`` with ``j = 0`` at the **bottom** (y = 0) and
``i = 0`` at the left (x = 0).  ``dx = width / W``, ``dy = height / H``.

See API.md sections 3 and 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .schema import PlantConfig

H_GRID = 64
W_GRID = 64

# wall thickness in cells
_WALL = 2
# stoker surface height at the feed end, as a fraction of furnace height
_SURF_FEED_FRAC = 0.30
# lowest the stoker surface may descend to, as a fraction of furnace height
_SURF_MIN_FRAC = 0.10
# secondary air nozzles live between these heights (fraction of furnace height)
_NOZZLE_LO, _NOZZLE_HI = 0.50, 0.85

# Vertical spreading profile for injected air and for the bed->gas release.
# Must sum to 1.
INJECT_PROFILE = (0.4, 0.3, 0.2, 0.1)

# How far a secondary-air jet reaches into the furnace, as a fraction of the
# local width. A real nozzle discharges at 40-60 m/s specifically so the jets
# from opposite walls penetrate to the core and collide there - that is what
# secondary air is for. Injecting the whole nozzle flow into the single wall
# cell instead gives a local dilution rate of ~157 1/s, which pins that cell at
# the inlet temperature and builds a cold curtain down the wall that the flame
# cannot cross: measured, secondary_air 0.6 -> 1.2 took the furnace from lit
# (T_gas 1036 K) to completely out (T_max = wall temperature).
JET_PENETRATION = 0.35

# channel order of the actuator field `a` (API.md 3)
A_PRIMARY, A_SECONDARY, A_FUEL, A_STOKER = 0, 1, 2, 3
N_A_CHANNELS = 4

# channel order of the static geometry field `g` (API.md 3)
G_MASK, G_SDF, G_LAYOUT, G_SENSOR = 0, 1, 2, 3
N_G_CHANNELS = 4


@dataclass
class Geometry:
    """Everything the solver and the dataset need to know about one plant."""

    cfg: PlantConfig
    dx: float
    dy: float
    mask: np.ndarray          # [H, W] 1 = fluid, 0 = solid
    sdf: np.ndarray           # [H, W] signed distance to wall / furnace height
    layout: np.ndarray        # [H, W] static actuator identity map
    sensor: np.ndarray        # [H, W] 1 where a sensor sits
    surf_row: np.ndarray      # [W] first fluid row above the stoker (-1 if none)
    zone_of_col: np.ndarray   # [W] primary-air zone index per column (-1 if none)
    zone_cells: list          # n_zones lists of (j, i)
    nozzle_cells: list        # n_nozzles lists of (j, i)
    feed_cells: list          # cells where fresh waste enters
    flue_cols: np.ndarray     # columns of the flue opening

    @property
    def g(self) -> np.ndarray:
        """The [H, W, 4] static geometry tensor written to the dataset."""
        out = np.zeros((H_GRID, W_GRID, N_G_CHANNELS), dtype=np.float32)
        out[..., G_MASK] = self.mask
        out[..., G_SDF] = self.sdf
        out[..., G_LAYOUT] = self.layout
        out[..., G_SENSOR] = self.sensor
        return out


# ----------------------------------------------------------------------
def _stoker_surface(cfg: PlantConfig, dy: float) -> np.ndarray:
    """Row index of the stoker surface for each column."""
    f, s = cfg.furnace, cfg.stoker
    x = (np.arange(W_GRID) + 0.5) * (f.width / W_GRID)
    y_feed = _SURF_FEED_FRAC * f.height
    y_min = _SURF_MIN_FRAC * f.height
    y_surf = np.maximum(y_feed - x * math.tan(math.radians(s.inclination)), y_min)
    # stoker only spans its own length; beyond it the floor stays flat
    y_surf = np.where(x <= s.length, y_surf, y_surf[min(W_GRID - 1, int(s.length / (f.width / W_GRID)))])
    return np.floor(y_surf / dy).astype(int)


def _signed_distance(mask: np.ndarray, cell: float) -> np.ndarray:
    """Brute-force signed distance to the fluid/solid interface, in metres.

    64x64 is small enough that the exact O(N*M) computation is instant and
    avoids a scipy dependency.  Positive inside the fluid.
    """
    jj, ii = np.mgrid[0:H_GRID, 0:W_GRID]
    pts = np.stack([jj.ravel(), ii.ravel()], axis=1).astype(np.float32)
    fluid = mask.ravel() > 0

    def _min_dist(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        if dst.size == 0:
            return np.full(len(src), np.inf, dtype=np.float32)
        d = np.sqrt(((src[:, None, :] - dst[None, :, :]) ** 2).sum(-1))
        return d.min(axis=1)

    solid_pts = pts[~fluid]
    fluid_pts = pts[fluid]
    out = np.zeros(len(pts), dtype=np.float32)
    out[fluid] = _min_dist(fluid_pts, solid_pts)
    out[~fluid] = -_min_dist(solid_pts, fluid_pts)
    return (out * cell).reshape(H_GRID, W_GRID)


def _flue_columns(cfg: PlantConfig) -> np.ndarray:
    w = max(3, int(round(cfg.flue.diameter_ratio * W_GRID)))
    if cfg.flue.position == "top_left":
        start = _WALL + 1
    elif cfg.flue.position == "top_center":
        start = (W_GRID - w) // 2
    else:  # top_right
        start = W_GRID - _WALL - 1 - w
    start = int(np.clip(start, _WALL, W_GRID - _WALL - w))
    return np.arange(start, start + w)


# ----------------------------------------------------------------------
def build_geometry(cfg: PlantConfig) -> Geometry:
    f = cfg.furnace
    dx = f.width / W_GRID
    dy = f.height / H_GRID

    mask = np.ones((H_GRID, W_GRID), dtype=np.float32)
    mask[:, :_WALL] = 0.0
    mask[:, -_WALL:] = 0.0
    mask[-_WALL:, :] = 0.0

    # everything at or below the stoker surface is solid (bed + grate)
    surf = _stoker_surface(cfg, dy)
    rows = np.arange(H_GRID)[:, None]
    mask[rows <= surf[None, :]] = 0.0

    # Open the flue through the top wall, but keep the outermost ring solid:
    # the finite-difference stencils use periodic jnp.roll, so a fluid cell on
    # row H-1 would wrap onto row 0 and silently couple the flue to the grate.
    flue_cols = _flue_columns(cfg)
    mask[H_GRID - _WALL, flue_cols] = 1.0

    # first fluid row above the stoker, per column
    surf_row = np.full(W_GRID, -1, dtype=int)
    for i in range(W_GRID):
        col = np.flatnonzero(mask[:, i] > 0)
        if col.size:
            surf_row[i] = int(col[0])

    # --- primary air zones along the stoker ---------------------------
    stoker_cols = np.flatnonzero(
        (np.arange(W_GRID) + 0.5) * dx <= cfg.stoker.length
    )
    stoker_cols = np.array([i for i in stoker_cols if surf_row[i] >= 0])
    zone_of_col = np.full(W_GRID, -1, dtype=int)
    zone_cells: list[list[tuple[int, int]]] = [[] for _ in range(cfg.n_zones)]
    if stoker_cols.size:
        edges = np.linspace(0, stoker_cols.size, cfg.n_zones + 1).astype(int)
        for z in range(cfg.n_zones):
            cols = stoker_cols[edges[z]:edges[z + 1]]
            for i in cols:
                zone_of_col[i] = z
                zone_cells[z].append((int(surf_row[i]), int(i)))

    # --- secondary air nozzles on the side walls ----------------------
    j_lo = int(_NOZZLE_LO * H_GRID)
    j_hi = int(_NOZZLE_HI * H_GRID)
    heights = np.linspace(j_lo, j_hi, cfg.n_nozzles).astype(int)
    nozzle_cells: list[list[tuple[int, int]]] = []
    for n, j in enumerate(heights):
        i = _WALL if n % 2 == 0 else W_GRID - _WALL - 1   # alternate walls
        j = int(np.clip(j, 0, H_GRID - _WALL - 1))
        if mask[j, i] == 0.0:                              # step inward if walled
            cand = np.flatnonzero(mask[j] > 0)
            i = int(cand[0]) if n % 2 == 0 else int(cand[-1])
        nozzle_cells.append([(int(j), int(i))])

    # --- waste feed cells (first zone, at the surface) ----------------
    feed_cells = zone_cells[0][: max(1, len(zone_cells[0]) // 2)] if zone_cells[0] else []

    # --- static layout map --------------------------------------------
    layout = np.zeros((H_GRID, W_GRID), dtype=np.float32)
    for z, cells in enumerate(zone_cells):
        for (j, i) in cells:
            layout[j, i] = 0.05 + 0.45 * (z + 1) / cfg.n_zones
    for n, cells in enumerate(nozzle_cells):
        for (j, i) in cells:
            layout[j, i] = 0.50 + 0.50 * (n + 1) / cfg.n_nozzles

    # --- sensors: a rake across the gas region + two near the flue ----
    sensor = np.zeros((H_GRID, W_GRID), dtype=np.float32)
    for frac_y in (0.45, 0.70):
        j = int(frac_y * H_GRID)
        cols = np.flatnonzero(mask[j] > 0)
        if cols.size:
            for i in np.linspace(cols[0], cols[-1], 4).astype(int):
                sensor[j, i] = 1.0
    for i in flue_cols[:: max(1, len(flue_cols) // 2)]:
        sensor[H_GRID - _WALL - 1, i] = 1.0

    sdf = _signed_distance(mask, cell=0.5 * (dx + dy)) / f.height

    geom = Geometry(
        cfg=cfg, dx=dx, dy=dy, mask=mask, sdf=sdf.astype(np.float32),
        layout=layout, sensor=sensor, surf_row=surf_row,
        zone_of_col=zone_of_col, zone_cells=zone_cells,
        nozzle_cells=nozzle_cells, feed_cells=feed_cells, flue_cols=flue_cols,
    )
    check_validity(geom)
    return geom


# ----------------------------------------------------------------------
def check_validity(geom: Geometry, min_passage_cells: int = 3) -> None:
    """Reject geometries that would make the solver meaningless or unstable.

    Catches the failure modes we care about: a furnace with no outlet, a
    disconnected pocket the air can never reach, and a flow passage so narrow
    it is unresolved on the grid.
    """
    mask = geom.mask
    n_fluid = int(mask.sum())
    if n_fluid < 0.15 * mask.size:
        raise ValueError(f"fluid region too small: {n_fluid} cells")

    # flood fill from the flue opening -> every fluid cell must be reachable
    seen = np.zeros_like(mask, dtype=bool)
    stack = [(H_GRID - _WALL - 1, int(i)) for i in geom.flue_cols if mask[H_GRID - _WALL - 1, i] > 0]
    if not stack:
        raise ValueError("flue opening is not in the fluid region")
    while stack:
        j, i = stack.pop()
        if seen[j, i]:
            continue
        seen[j, i] = True
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nj, ni = j + dj, i + di
            if 0 <= nj < H_GRID and 0 <= ni < W_GRID and mask[nj, ni] > 0 and not seen[nj, ni]:
                stack.append((nj, ni))
    if seen.sum() != n_fluid:
        raise ValueError(
            f"fluid region is not connected: {n_fluid - int(seen.sum())} unreachable cells"
        )

    # narrowest horizontal passage in the gas space
    widths = [int(mask[j].sum()) for j in range(H_GRID) if mask[j].sum() > 0]
    if min(widths) < min_passage_cells:
        raise ValueError(f"passage narrower than {min_passage_cells} cells: {min(widths)}")

    # every actuator must sit in the fluid
    for cells in list(geom.zone_cells) + list(geom.nozzle_cells):
        for (j, i) in cells:
            if mask[j, i] == 0.0:
                raise ValueError(f"actuator at ({j},{i}) is inside a wall")


# ----------------------------------------------------------------------
def encode_action(geom: Geometry, action: dict) -> np.ndarray:
    """Scalar, plant-specific action -> fixed-shape [H, W, 4] field.

    The value stored is a **flux density** (per unit injection area), so that
    integrating the field over the injection area recovers the scalar exactly,
    regardless of how many grid cells the actuator happens to cover.  That is
    what makes a 4-zone plant and an 8-zone plant interchangeable to the NN.
    """
    cfg = geom.cfg
    a = np.zeros((H_GRID, W_GRID, N_A_CHANNELS), dtype=np.float32)

    primary = np.atleast_1d(action["primary_air"]).astype(float)
    secondary = np.atleast_1d(action["secondary_air"]).astype(float)
    if primary.size != cfg.n_zones:
        raise ValueError(f"primary_air has {primary.size} entries, expected {cfg.n_zones}")
    if secondary.size != cfg.n_nozzles:
        raise ValueError(f"secondary_air has {secondary.size} entries, expected {cfg.n_nozzles}")

    # Primary air enters upward through the grate.  It is spread over a few
    # cells in height with the same profile as the bed release: a 12 1/s volume
    # source concentrated in a single cell row radiates acoustic noise that
    # dominates the velocity field.  Weights sum to 1, so the area integral -
    # and therefore the scalar it decodes back to - is unchanged.
    for z, cells in enumerate(geom.zone_cells):
        if not cells:
            continue
        area = len(cells) * geom.dx
        for (j, i) in cells:
            acc = 0.0
            for k, wgt in enumerate(INJECT_PROFILE):
                if j + k < H_GRID and geom.mask[j + k, i] > 0:
                    acc += wgt
            for k, wgt in enumerate(INJECT_PROFILE):
                if j + k < H_GRID and geom.mask[j + k, i] > 0:
                    a[j + k, i, A_PRIMARY] += primary[z] / area * (wgt / acc)

    # secondary air enters horizontally through a nozzle: area per cell = dy * 1 m.
    # It is laid along the jet's path, not dumped in the wall cell. Spreading is
    # in the flow direction, so the area integral still decodes to the scalar.
    for n, cells in enumerate(geom.nozzle_cells):
        if not cells:
            continue
        area = len(cells) * geom.dy
        for (j, i) in cells:
            row = np.flatnonzero(geom.mask[j] > 0)
            if row.size == 0:
                continue
            step = 1 if i <= (row[0] + row[-1]) // 2 else -1
            n_pen = max(2, int(JET_PENETRATION * row.size))
            wgt = np.exp(-np.arange(n_pen) / max(n_pen / 2.0, 1.0))
            reach = [i + step * k for k in range(n_pen)]
            ok = [k for k, ii in enumerate(reach)
                  if 0 <= ii < W_GRID and geom.mask[j, ii] > 0]
            if not ok:
                continue
            norm = wgt[ok].sum()
            for k in ok:
                a[j, reach[k], A_SECONDARY] += secondary[n] / area * (wgt[k] / norm)

    # fresh waste is dropped at the feed end
    if geom.feed_cells:
        area = len(geom.feed_cells) * geom.dx
        for (j, i) in geom.feed_cells:
            a[j, i, A_FUEL] = float(action["waste_feed"]) / area

    # stoker speed is uniform along the grate; giving it its own channel keeps
    # it visible to the NN instead of hiding it in a scalar side-input
    for cells in geom.zone_cells:
        for (j, i) in cells:
            a[j, i, A_STOKER] = float(action["stoker_speed"])

    return a


def action_to_vector(geom: Geometry, action: dict) -> np.ndarray:
    """Flat, human-readable action vector for ``a_scalar`` (API.md 3)."""
    return np.concatenate([
        [float(action["stoker_speed"]), float(action["waste_feed"])],
        np.atleast_1d(action["primary_air"]).astype(float),
        np.atleast_1d(action["secondary_air"]).astype(float),
    ]).astype(np.float32)


# ----------------------------------------------------------------------
def dimensionless(geom: Geometry, action: dict) -> dict:
    """Dimensionless groups written to ``meta`` (API.md 3).

    Passing raw dimensional values (a 12 m furnace, 5 zones) to a network does
    not transfer across plants; these do, and they are what makes a 9 m and a
    15 m furnace comparable.  Evaluated at the nominal action of the episode.
    """
    from ..sim.physics import NU, PR, SC, GRAV, T_REF, RHO_AIR, RHO0, S_STOICH, Y_O2_AIR, A_ARR, EA_R

    cfg = geom.cfg
    f = cfg.furnace
    l_ref = f.height

    # characteristic velocity: total injected volume flow through the flue area
    q_air = float(np.sum(action["primary_air"]) + np.sum(action["secondary_air"]))
    hot = q_air * RHO_AIR / RHO0                       # expands on heating
    a_flue = max(len(geom.flue_cols) * geom.dx, 1e-6)
    u_ref = hot / a_flue

    d_t = 800.0                                        # representative excursion
    t_chem = 1.0 / max(A_ARR * 0.05 * 0.1 * np.exp(-EA_R / 1400.0), 1e-12)

    # equivalence ratio of the whole furnace: volatile supply vs O2 supply
    w = cfg.waste
    vol = float(action["waste_feed"]) * (1.0 - w.moisture) * w.volatile_frac
    o2 = q_air * RHO_AIR * Y_O2_AIR
    phi = (vol * S_STOICH) / max(o2, 1e-9)

    return {
        "Re": u_ref * l_ref / NU,
        "Pe": u_ref * l_ref / (NU / PR),
        "Ri": GRAV * (d_t / T_REF) * l_ref / max(u_ref**2, 1e-9),
        "Da": (l_ref / max(u_ref, 1e-9)) / t_chem,
        "phi": phi,
    }
