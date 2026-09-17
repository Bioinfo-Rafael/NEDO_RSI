"""The simulator behind one class: give it an input, it advances.

    from wmf import Furnace

    f = Furnace("plant_a")                 # or a path to any plant YAML
    y = f.reset()                          # the operator's measurements
    y = f.step(stoker_speed=0.02, waste_feed=1.3, primary_air=1.25, secondary_air=0.4)
    y = f.step([0.02, 1.3, 1.25, 0.4])     # same thing as the 4 aggregate knobs
    y = f.step({"stoker_speed": 0.02, "waste_feed": 1.3,
                "primary_air": [1.2, 1.3, 1.25, 1.2, 1.3],      # per zone, if you want
                "secondary_air": [0.4] * 6})

Every call is one control interval (1 s). Whatever you pass is validated,
projected onto the plant's operating envelope (``f.envelope``), and applied
with the plant's rate limits if you ask (``rate_limited=True``); the returned
state is checked to be finite. Nothing here can be misused into a NaN.

``y`` is a dict of what a plant can actually measure: ``t_exit`` [K],
``o2_exit`` [-], ``yf_exit`` [-], ``unburnt_bed`` [kg/m2/s], plus ``t``. The
full 64x64 field is available as ``f.field`` for plotting, and the bed as
``f.bed`` - the controller should not look at them, a real plant cannot.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .actions import clip_action, envelope, nominal_action_vector
from .plants.encode import build_geometry, encode_action
from .plants.schema import PlantConfig
from .sim.env import IncineratorEnv

ROOT = Path(__file__).resolve().parents[2]
KNOBS = ("stoker_speed", "waste_feed", "primary_air", "secondary_air")
MEASURED = ("t_exit", "o2_exit", "yf_exit", "unburnt_bed")
CONTROL_INTERVAL = 1.0      # s


def _resolve(plant) -> PlantConfig:
    if isinstance(plant, PlantConfig):
        return plant
    p = Path(plant)
    if p.suffix != ".yaml":
        name = p.name if p.name.startswith("plant_") else f"plant_{p.name}"
        p = ROOT / "configs" / "plants" / f"{name}.yaml"
    if not p.exists():
        avail = sorted(x.stem for x in (ROOT / "configs" / "plants").glob("*.yaml"))
        raise FileNotFoundError(f"no plant config at {p}. Available: {avail}, or pass a YAML path.")
    return PlantConfig.from_yaml(p)


class Furnace:
    def __init__(self, plant="plant_a", seed: int = 0, rate_limited: bool = False):
        self.cfg = _resolve(plant)
        self.env = IncineratorEnv(self.cfg, seed=seed)
        self.geom = build_geometry(self.cfg)
        self.envelope = envelope(self.cfg)
        self.n_zones, self.n_nozzles = self.cfg.n_zones, self.cfg.n_nozzles
        self.rate_limited = rate_limited
        self.u_nominal = nominal_action_vector(self.cfg)
        self._last = None
        self.t = 0.0
        self.reset()

    # ------------------------------------------------------------ inputs
    def _as_action(self, u=None, **kw) -> dict:
        """Accept a dict, a 4-vector, or keywords; expand aggregate levels to zones."""
        if u is None and not kw:
            raise ValueError("step() needs an input: a dict, a 4-vector, or keywords "
                             f"{KNOBS}. Nominal for this plant is {np.round(self.u_nominal, 4).tolist()}.")
        if u is None:
            a = dict(kw)
        elif isinstance(u, dict):
            a = dict(u); a.update(kw)
        else:
            v = np.asarray(u, dtype=float).ravel()
            if v.shape != (4,):
                raise ValueError(f"a vector input must have 4 entries {KNOBS}, got shape {v.shape}")
            a = dict(zip(KNOBS, v.tolist()))
        missing = [k for k in KNOBS if k not in a]
        if missing:
            raise ValueError(f"missing {missing}; every step needs all of {KNOBS}")
        for k in KNOBS:
            arr = np.asarray(a[k], dtype=float)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{k} contains NaN/inf: {a[k]}")
        for k, n in (("primary_air", self.n_zones), ("secondary_air", self.n_nozzles)):
            arr = np.atleast_1d(np.asarray(a[k], dtype=float))
            if arr.size == 1:
                arr = np.repeat(arr, n)
            if arr.size != n:
                raise ValueError(f"{k} has {arr.size} entries; this plant has {n}. "
                                 f"Pass one number (same for all) or exactly {n}.")
            a[k] = arr.tolist()
        a["stoker_speed"] = float(np.asarray(a["stoker_speed"]).ravel()[0])
        a["waste_feed"] = float(np.asarray(a["waste_feed"]).ravel()[0])
        return a

    def project(self, action: dict) -> dict:
        """The action the plant will actually see: clipped to the envelope and,
        if ``rate_limited``, to how far the hardware can move in one interval."""
        a = clip_action(self.cfg, action)
        if self.rate_limited and self._last is not None:
            from .control.mpc import MPCConfig
            du = np.asarray(MPCConfig().du_max)
            lim = {"stoker_speed": du[0], "waste_feed": du[1], "primary_air": du[2], "secondary_air": du[3]}
            for k in ("stoker_speed", "waste_feed"):
                a[k] = float(np.clip(a[k], self._last[k] - lim[k], self._last[k] + lim[k]))
            for k in ("primary_air", "secondary_air"):
                prev = np.asarray(self._last[k]); cur = np.asarray(a[k])
                a[k] = np.clip(cur, prev - lim[k], prev + lim[k]).tolist()
        return a

    # ------------------------------------------------------------ dynamics
    def reset(self) -> dict:
        self.env.reset()
        self.t = 0.0
        self._last = None
        return self.measure()

    def step(self, u=None, **kw) -> dict:
        action = self.project(self._as_action(u, **kw))
        _, reward, info = self.env.step(action)
        self._last = action
        self.t += CONTROL_INTERVAL
        if not (np.isfinite(self.env.q).all() and np.isfinite(self.env.bed).all()):
            raise FloatingPointError(
                "the simulator state became non-finite. This should be impossible inside the "
                "envelope; please report the plant config and the action sequence.")
        y = {k: float(info[k]) for k in MEASURED}
        y["reward"] = float(reward); y["t"] = self.t
        return y

    def run(self, actions, callback=None) -> dict:
        """Apply a sequence of inputs (list of dicts / vectors), return arrays."""
        ys = []
        for k, a in enumerate(actions):
            y = self.step(a); ys.append([y[m] for m in MEASURED])
            if callback: callback(k, y)
        return {"t": np.arange(1, len(ys) + 1) * CONTROL_INTERVAL,
                "y": np.asarray(ys, np.float32), "names": list(MEASURED)}

    # ------------------------------------------------------------ views
    def measure(self) -> dict:
        from . import reward as rewardmod
        pr = rewardmod.proxies(self.env.q, 0.0, self.env.static)
        y = {k: float(pr[k]) for k in MEASURED}; y["t"] = self.t
        return y

    @property
    def field(self) -> np.ndarray:
        """[64, 64, 7] gas state: T, u, v, p, Y_F, Y_O2, Y_P. For plotting, not for control."""
        return np.asarray(self.env.q)

    @property
    def bed(self) -> np.ndarray:
        """[64, 4] grate: moisture, volatiles, char [kg/m2], bed temperature [K]."""
        return np.asarray(self.env.bed)

    @property
    def mask(self) -> np.ndarray:
        return self.geom.mask

    def __repr__(self):
        e = self.envelope
        return (f"Furnace({self.cfg.plant_id}: {self.cfg.furnace.width:g}x{self.cfg.furnace.height:g} m, "
                f"{self.n_zones} zones, {self.n_nozzles} nozzles; envelope stoker {e['stoker_speed']}, "
                f"feed {tuple(round(v,3) for v in e['waste_feed'])}, primary {tuple(round(v,3) for v in e['primary_air'])}, "
                f"secondary {tuple(round(v,3) for v in e['secondary_air'])}; t={self.t:g}s)")
