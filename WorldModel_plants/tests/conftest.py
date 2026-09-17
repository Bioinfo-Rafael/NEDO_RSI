import pytest

from wmf.plants.encode import build_geometry
from wmf.plants.schema import PlantConfig

PLANTS = ["a", "b", "c"]


@pytest.fixture(scope="session", params=PLANTS)
def cfg(request):
    return PlantConfig.from_yaml(f"configs/plants/plant_{request.param}.yaml")


@pytest.fixture(scope="session")
def geom(cfg):
    return build_geometry(cfg)
