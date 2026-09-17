"""Episode assembly and the on-disk format (API.md 3).

One episode = one .npz.  The format is the contract with the training side, so
it is defined here once and nowhere else.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..actions import sample_sequence
from ..plants.encode import build_geometry, encode_action, action_to_vector, dimensionless
from ..plants.schema import PlantConfig
from .physics import SimParams
from .rollout import build_static, make_initial, rollout

# 600 frames x 333 gas steps x 3 ms = 600 s, control interval 1 s.
#
# The length is set by the grate, not by the gas. One residence at the nominal
# grate speed is L / v = 12 / 0.020 = 600 s, and the response to a change in
# waste feed or grate speed only appears once that material has traversed the
# grate. An episode shorter than a residence puts a dead control variable in
# the dataset; the identification runs (scripts/gen_control_data.py) use two
# residences.
N_FRAMES = 600
SAVE_EVERY = 333


def run_episode(
    cfg: PlantConfig,
    seed: int = 0,
    n_frames: int = N_FRAMES,
    save_every: int = SAVE_EVERY,
    actions: list[dict] | None = None,
) -> dict:
    """Simulate one episode and return everything the dataset format needs."""
    rng = np.random.default_rng(seed)
    geom = build_geometry(cfg)
    static = build_static(geom)
    prm = SimParams.from_plant(cfg)

    if actions is None:
        actions = sample_sequence(cfg, rng, n_frames)

    a_fields = np.stack([encode_action(geom, a) for a in actions])
    a_scalar = np.stack([action_to_vector(geom, a) for a in actions])

    q0, bed0 = make_initial(static)
    q, bed, diag = rollout(q0, bed0, jnp.asarray(a_fields), static, prm, save_every, n_frames)

    meta = {
        "plant_id": cfg.plant_id,
        "plant_hash": cfg.hash(),
        "seed": int(seed),
        "dx": geom.dx,
        "dy": geom.dy,
        "dt": prm.dt,
        "save_every": int(save_every),
        "control_interval": prm.dt * save_every,
        "n_zones": cfg.n_zones,
        "n_nozzles": cfg.n_nozzles,
        "dimensionless": dimensionless(geom, actions[0]),
        "a_scalar_keys": cfg.action_keys,
        "sim_params": {"q_cp": prm.q_cp, "lam_loss": prm.lam_loss, "t_wall": prm.t_wall},
        # ground-truth values a plant latent would have to recover; kept out of
        # the inputs, used only to score identification
        "hidden": {
            "lhv": cfg.waste.lhv,
            "moisture": cfg.waste.moisture,
            "wall_heat_transfer": cfg.wall.heat_transfer,
            "wall_temperature": cfg.wall.temperature,
        },
    }

    return {
        "q": np.asarray(q, dtype=np.float16),
        "a": a_fields.astype(np.float16),
        "g": geom.g.astype(np.float32),
        "bed": np.asarray(bed, dtype=np.float32),
        "a_scalar": a_scalar.astype(np.float32),
        "reward": np.asarray(diag["reward"], dtype=np.float32),
        "proxies": np.stack(
            [np.asarray(diag[k]) for k in
             ("t_exit", "o2_exit", "yf_exit", "co_proxy", "nox_proxy", "unburnt_bed")],
            axis=-1,
        ).astype(np.float32),
        "meta": json.dumps(meta),
    }


PROXY_KEYS = ("t_exit", "o2_exit", "yf_exit", "co_proxy", "nox_proxy", "unburnt_bed")


def save_episode(ep: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **ep)
    return path


def load_episode(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    ep["meta"] = json.loads(str(ep["meta"]))
    return ep
