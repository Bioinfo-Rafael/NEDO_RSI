"""Procedural plant generation.

Three hand-written plants are not enough to claim anything about unseen
furnaces: with n=3 a held-out result is an anecdote. This samples the stoker
design space under the constraints a real furnace satisfies, then rejects what
the solver cannot represent - the validity checks are the same ones the
hand-written plants pass.

Families
--------
Plants are grouped into families so that evaluation can hold out a *family*
rather than a single plant. Holding out one plant from a family whose siblings
are in the training set measures interpolation between near-duplicates, which
is not the question. A family here is (size class, zone count): those two
change the furnace's response most, and a model that has never seen a large
6-zone furnace has genuinely never seen anything like it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encode import build_geometry
from .schema import PlantConfig

# size class -> (width range [m], height range [m])
SIZE_CLASSES = {
    "small":  ((8.0, 10.5), (5.0, 7.0)),
    "medium": ((10.5, 13.0), (6.0, 8.5)),
    "large":  ((13.0, 16.0), (7.0, 10.0)),
}
ZONE_COUNTS = (4, 5, 6, 7)
FLUE_POSITIONS = ("top_left", "top_center", "top_right")


@dataclass(frozen=True)
class Family:
    size: str
    zones: int

    @property
    def name(self) -> str:
        return f"{self.size}{self.zones}"


def families() -> list[Family]:
    return [Family(s, z) for s in SIZE_CLASSES for z in ZONE_COUNTS]


def sample(family: Family, rng: np.random.Generator, idx: int) -> PlantConfig:
    """One plant from a family. Raises ValueError if the draw is unusable."""
    (w_lo, w_hi), (h_lo, h_hi) = SIZE_CLASSES[family.size]
    width = float(rng.uniform(w_lo, w_hi))
    height = float(rng.uniform(h_lo, h_hi))

    # the grate spans the furnace; inclination is limited by how far the bed
    # surface may descend before it reaches the floor
    incl = float(rng.uniform(3.0, 14.0))
    n_zones = family.zones
    n_nozzles = int(rng.integers(max(2, n_zones - 2), n_zones + 3))

    moisture = float(rng.uniform(0.20, 0.50))
    volatile = float(rng.uniform(0.50, 0.70))
    ash = float(rng.uniform(0.06, 0.16))
    lhv = float(rng.uniform(6.5, 11.5))

    d = {
        "plant_id": f"{family.name}_{idx:03d}",
        "furnace": {"width": width, "height": height},
        "stoker": {"length": width, "inclination": incl, "zones": n_zones},
        "primary_air": {"zones": n_zones},
        "secondary_air": {"nozzles": n_nozzles},
        "flue": {"diameter_ratio": float(rng.uniform(0.18, 0.34)),
                 "position": str(rng.choice(FLUE_POSITIONS))},
        "waste": {"moisture": moisture, "lhv": lhv,
                  "volatile_frac": volatile, "ash_frac": ash},
        "wall": {"heat_transfer": float(rng.uniform(15.0, 35.0)),
                 "temperature": float(rng.uniform(560.0, 640.0))},
    }
    cfg = PlantConfig.from_dict(d)       # raises on out-of-range fields
    build_geometry(cfg)                  # raises on unusable geometry
    return cfg


def generate(n_per_family: int, seed: int = 0, hold_out: str | None = None
             ) -> dict[str, list[PlantConfig]]:
    """Return {family_name: [PlantConfig, ...]}, rejecting invalid draws."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[PlantConfig]] = {}
    for fam in families():
        if hold_out and fam.name == hold_out:
            continue
        got, tries = [], 0
        while len(got) < n_per_family and tries < n_per_family * 20:
            tries += 1
            try:
                got.append(sample(fam, rng, len(got)))
            except (ValueError, IndexError):
                continue
        if got:
            out[fam.name] = got
    return out
