"""Plant configuration schema.

A plant config is the "設計図" side of the interface: everything that is fixed
for a given incinerator (geometry, actuator layout, waste properties).  It is
deliberately separate from ``action``, which is what changes every timestep.

See API.md section 4.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Furnace:
    width: float   # m
    height: float  # m


@dataclass(frozen=True)
class Stoker:
    length: float       # m
    inclination: float  # deg
    zones: int


@dataclass(frozen=True)
class Flue:
    diameter_ratio: float  # fraction of furnace width
    position: str          # top_left | top_center | top_right


@dataclass(frozen=True)
class Waste:
    """Waste properties.

    ``moisture`` is on an as-received basis.  ``volatile_frac`` and ``ash_frac``
    are on a **dry basis**, so the char (fixed carbon) fraction is the remainder::

        char_frac = 1 - volatile_frac - ash_frac        (dry basis)

    These three feed the 1D bed model (drying -> pyrolysis -> char burnout).
    """

    moisture: float
    lhv: float            # MJ/kg, as received
    volatile_frac: float  # dry basis
    ash_frac: float       # dry basis

    @property
    def char_frac(self) -> float:
        return 1.0 - self.volatile_frac - self.ash_frac


@dataclass(frozen=True)
class Wall:
    heat_transfer: float  # W/m2K
    temperature: float    # K


@dataclass(frozen=True)
class PlantConfig:
    plant_id: str
    furnace: Furnace
    stoker: Stoker
    flue: Flue
    waste: Waste
    wall: Wall
    n_zones: int      # primary air zones
    n_nozzles: int    # secondary air nozzles

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "PlantConfig":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "PlantConfig":
        cfg = cls(
            plant_id=d["plant_id"],
            furnace=Furnace(**d["furnace"]),
            stoker=Stoker(**d["stoker"]),
            flue=Flue(**d["flue"]),
            waste=Waste(**d["waste"]),
            wall=Wall(**d["wall"]),
            n_zones=int(d["primary_air"]["zones"]),
            n_nozzles=int(d["secondary_air"]["nozzles"]),
        )
        cfg.validate()
        return cfg

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Cheap sanity checks. Geometric validity (connectivity, minimum
        passage width) is checked in plants.encode after rasterisation."""
        f, s = self.furnace, self.stoker
        if not (4.0 <= f.width <= 20.0):
            raise ValueError(f"furnace.width out of range: {f.width}")
        if not (3.0 <= f.height <= 12.0):
            raise ValueError(f"furnace.height out of range: {f.height}")
        if not (0.0 <= s.inclination <= 25.0):
            raise ValueError(f"stoker.inclination out of range: {s.inclination}")
        if s.length > f.width * 1.001:
            raise ValueError("stoker.length exceeds furnace.width")
        if self.n_zones != s.zones:
            raise ValueError("primary_air.zones must equal stoker.zones")
        if not (1 <= self.n_zones <= 10):
            raise ValueError(f"n_zones out of range: {self.n_zones}")
        if not (1 <= self.n_nozzles <= 12):
            raise ValueError(f"n_nozzles out of range: {self.n_nozzles}")
        if self.flue.position not in ("top_left", "top_center", "top_right"):
            raise ValueError(f"unknown flue.position: {self.flue.position}")
        if not (0.05 <= self.flue.diameter_ratio <= 0.6):
            raise ValueError(f"flue.diameter_ratio out of range: {self.flue.diameter_ratio}")
        w = self.waste
        if not (0.0 <= w.moisture < 0.8):
            raise ValueError(f"waste.moisture out of range: {w.moisture}")
        if w.volatile_frac + w.ash_frac > 1.0:
            raise ValueError(
                "volatile_frac + ash_frac must be <= 1.0 (dry basis); "
                f"got {w.volatile_frac} + {w.ash_frac}"
            )
        if not (2.0 <= w.lhv <= 20.0):
            raise ValueError(f"waste.lhv out of range: {w.lhv}")

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        """Stable hash of the config, recorded in meta so a dataset can always
        be traced back to the plant that produced it."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    # -- action vector layout ------------------------------------------
    @property
    def action_keys(self) -> list[str]:
        """Flat, human-readable names for ``a_scalar`` columns (see API.md 3)."""
        keys = ["stoker_speed", "waste_feed"]
        keys += [f"primary_air_{i}" for i in range(self.n_zones)]
        keys += [f"secondary_air_{i}" for i in range(self.n_nozzles)]
        return keys

    @property
    def action_dim(self) -> int:
        return 2 + self.n_zones + self.n_nozzles
