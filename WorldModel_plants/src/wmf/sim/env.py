"""Level 2 gym-like API (API.md 2).

Not used by the offline dataset path; it exists so that an online loop - a
planner, or an agent that wants to probe the simulator directly - can drive the
same ground truth one control interval at a time.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..actions import clip_action, sample_action
from ..plants.encode import build_geometry, encode_action
from ..plants.schema import PlantConfig
from .. import reward as rewardmod
from .episode import SAVE_EVERY
from .physics import SimParams
from .rollout import build_static, make_initial, rollout


class IncineratorEnv:
    def __init__(self, config: str | PlantConfig, seed: int = 0,
                 save_every: int = SAVE_EVERY, params: SimParams | None = None):
        self.cfg = config if isinstance(config, PlantConfig) else PlantConfig.from_yaml(config)
        self.geom = build_geometry(self.cfg)
        self.static = build_static(self.geom)
        self.prm = params or SimParams.from_plant(self.cfg)
        self.save_every = save_every
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> dict:
        self.q, self.bed = make_initial(self.static)
        self.t = 0
        return self._obs()

    def sample_action(self) -> dict:
        return sample_action(self.cfg, self.rng)

    def _obs(self) -> dict:
        return {
            "q": np.asarray(self.q),
            "bed": np.asarray(self.bed),
            "g": self.geom.g,
            "t": self.t,
        }

    def step(self, action: dict):
        action = clip_action(self.cfg, action)
        a_field = jnp.asarray(encode_action(self.geom, action))
        # Reuse the compiled offline solver for one frame. A Python loop here
        # dispatches hundreds of individual JAX operations per control step.
        qs, beds, diag = rollout(self.q, self.bed, a_field[None, ...],
                                 self.static, self.prm, self.save_every, 1)
        self.q, self.bed = qs[-1], beds[-1]
        unburnt = diag["unburnt_bed"][0]
        self.t += 1

        pr = rewardmod.proxies(self.q, unburnt, self.static)
        energy = (a_field[..., 0].sum() * self.static["dx"]
                  + a_field[..., 1].sum() * self.static["dy"])
        r = float(rewardmod.reward(pr, energy))
        info = {k: float(v) for k, v in pr.items()}
        return self._obs(), r, info
