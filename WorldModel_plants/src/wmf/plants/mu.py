"""mu_known: the plant specification a drawing already tells you.

Split deliberately from anything that has to be inferred. A latent should not
spend capacity rediscovering the furnace width when the width is written on the
equipment sheet; it should be spent on what the sheet does not say - effective
heat loss, waste calorific variation, air-delivery efficiency.

Everything here is dimensionless or normalised to a fixed reference, so that a
9 m furnace and a 16 m furnace produce vectors a network can compare.
"""

from __future__ import annotations

import numpy as np

W_REF, H_REF = 12.0, 7.0          # reference furnace, for normalisation
N_MU = 12


def names() -> list[str]:
    return ["w_norm", "h_norm", "aspect", "incl_sin", "n_zones_norm", "n_nozzles_norm",
            "flue_ratio", "flue_pos", "dx_norm", "moisture", "volatile", "lhv_norm"]


def mu_known(cfg, geom=None) -> np.ndarray:
    f, s, w = cfg.furnace, cfg.stoker, cfg.waste
    pos = {"top_left": -1.0, "top_center": 0.0, "top_right": 1.0}[cfg.flue.position]
    dx = (geom.dx if geom is not None else f.width / 64.0)
    return np.array([
        f.width / W_REF,
        f.height / H_REF,
        f.width / f.height,
        np.sin(np.radians(s.inclination)),
        cfg.n_zones / 6.0,
        cfg.n_nozzles / 6.0,
        cfg.flue.diameter_ratio,
        pos,
        dx / (W_REF / 64.0),
        w.moisture,
        w.volatile_frac,
        w.lhv / 9.0,
    ], dtype=np.float32)


mu_known.names = names
