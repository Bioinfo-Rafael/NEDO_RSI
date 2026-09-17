"""wmf - a stoker incinerator you can drive with one call.

    from wmf import Furnace
    f = Furnace("plant_a")
    y = f.step(stoker_speed=0.02, waste_feed=1.3, primary_air=1.25, secondary_air=0.4)

See README.md; `python -m wmf.selfcheck` verifies the installation.
"""
from .furnace import Furnace  # noqa: F401
from .actions import envelope  # noqa: F401
from .plants.schema import PlantConfig  # noqa: F401

__all__ = ["Furnace", "envelope", "PlantConfig"]
