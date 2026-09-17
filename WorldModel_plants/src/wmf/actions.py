"""Action sampling.

Action ranges (API.md 1).  These are the operating envelope the ground truth is
valid over: outside them the solver is not guaranteed to stay clean, and the
dataset should not contain points the simulator cannot be trusted on.

The air limits are set by the flue exit velocity: total injected volume divided
by the flue area has to stay well under the artificial sound speed, or the
pressure channel stops being quantitatively meaningful.  tests/test_stability.py
asserts this at every corner of the box.
"""

from __future__ import annotations

import numpy as np

from .plants.schema import PlantConfig

RANGES = {
    # Chosen so the grate residence L/v spans ~400-1200 s on a 12 m grate,
    # bracketing the 660-720 s measured from the real plant's log. The old
    # 0.05-0.35 gave 34-240 s, which is a different machine.
    "stoker_speed":  (0.010, 0.030),  # m/s
    # 80-130% of the 1.25 kg/s nominal, i.e. a real plant's load range. The air
    # limits below are absolute, so a wider feed range would put lambda outside
    # anything the furnace can burn: measured at 1.0 kg/s with the air box at
    # nominal, lambda reaches 2.2-3.5 and the furnace goes out - correctly, a
    # real plant turns its air down with load.
    # Specific grate loading: 1.25 kg/s over the 12 m x 1 m slice is
    # 0.10 kg/m2/s, which is where real stoker grates sit (0.05-0.11, i.e.
    # 180-400 kg/m2/h). The 2.5 kg/s this used to run at was 0.21 - twice a
    # real grate - and it showed up as a flue velocity of 17 m/s against the
    # 8-15 m/s of a real plant, with the solver's velocity clamp binding on
    # 0.4% of cells. The 2D domain is one metre deep, so per-area loading is
    # the quantity that has to match, not the tonnage.
    "waste_feed":    (1.0, 1.6),     # kg/s
    # Measured operating envelope, not a guess. A 15-point sweep of the air box
    # at nominal feed gives a coherent map: the furnace stays lit across
    # primary 2.0-3.0 for every secondary setting, with T_gas 714-1295 K,
    # O2 6.5-15.9% and unburnt ash 3.5-7.9% - a real plant's operating window
    # and enough spread for the controller to have something to do. Outside it
    # the plant genuinely goes out: at primary 1.0-1.5 with high secondary the
    # bed is starved of the air its char needs, drops to 330 K, and 58% of the
    # feed leaves the grate unburnt. That failure is real and worth keeping
    # reachable, but not worth putting inside the box the dataset is drawn from.
    # Halved together with the feed, so every lambda in the measured map above
    # is preserved exactly while the velocities come back to a real furnace's.
    "primary_air":   (1.0, 1.5),     # Nm3/s per zone
    # Upper limit set by the artificial sound speed: at 0.8 the jet region
    # reached 16.2 m/s against U_CLAMP = 16, i.e. the clamp was the flow.
    "secondary_air": (0.2, 0.6),     # Nm3/s per nozzle
}


# The reference plant the numbers in RANGES were measured on.
REF_GRATE_LENGTH = 12.0     # m   (plant_a)
REF_LHV = 9.0               # MJ/kg
REF_ZONES = 5               # primary-air zones on plant_a
REF_NOZZLES = 6             # secondary-air nozzles on plant_a


def envelope(cfg: PlantConfig) -> dict:
    """Operating envelope of one plant.

    ``RANGES`` is the reference plant's. Feeders and fans on a real plant are
    sized to the grate, so the envelope scales with it: thermal load per grate
    area is held constant (feed ~ L / LHV) and air per grate area is held
    constant (air ~ L). Stoichiometric air per kg is itself ~ LHV, so lambda is
    preserved as well. Measured before this existed: plant_b (15 m, 7.5 MJ/kg)
    cooled to 430-630 K and plant_c (9 m, 11 MJ/kg) over-fired to 1950 K on the
    reference plant's absolute numbers.
    """
    k_area = cfg.stoker.length / REF_GRATE_LENGTH
    k_feed = k_area * REF_LHV / cfg.waste.lhv
    # The air numbers are per zone and per nozzle, so the *total* scales with
    # the grate only if the per-unit value is divided by the unit count.
    # Without this, plant_b (6 zones, 8 nozzles) got 1.5-1.7x the air per unit
    # of grate and ran cold at 490-830 K; plant_c (4 and 4) got 0.5-0.6x and
    # over-fired to 1950 K at 1.3% O2.
    k_pri = k_area * REF_ZONES / cfg.n_zones
    k_sec = k_area * REF_NOZZLES / cfg.n_nozzles
    return {
        "stoker_speed": tuple(RANGES["stoker_speed"]),
        "waste_feed": tuple(v * k_feed for v in RANGES["waste_feed"]),
        "primary_air": tuple(v * k_pri for v in RANGES["primary_air"]),
        "secondary_air": tuple(v * k_sec for v in RANGES["secondary_air"]),
    }


def nominal_action_vector(cfg: PlantConfig) -> np.ndarray:
    """Middle of the envelope as the 4 aggregate knobs
    (stoker_speed, waste_feed, primary_level, secondary_level)."""
    e = envelope(cfg)
    return np.array([np.mean(e["stoker_speed"]), np.mean(e["waste_feed"]),
                     np.mean(e["primary_air"]), np.mean(e["secondary_air"])], np.float32)


def bounds(cfg: PlantConfig):
    """(u_lo, u_hi) for the 4 aggregate knobs."""
    e = envelope(cfg)
    keys = ("stoker_speed", "waste_feed", "primary_air", "secondary_air")
    return (np.array([e[k][0] for k in keys], np.float32),
            np.array([e[k][1] for k in keys], np.float32))


def sample_action(cfg: PlantConfig, rng: np.random.Generator) -> dict:
    env = envelope(cfg)
    lo, hi = env["primary_air"]
    lo2, hi2 = env["secondary_air"]
    return {
        "stoker_speed": float(rng.uniform(*env["stoker_speed"])),
        "waste_feed": float(rng.uniform(*env["waste_feed"])),
        "primary_air": rng.uniform(lo, hi, size=cfg.n_zones).tolist(),
        "secondary_air": rng.uniform(lo2, hi2, size=cfg.n_nozzles).tolist(),
    }


def sample_sequence(
    cfg: PlantConfig,
    rng: np.random.Generator,
    n_frames: int,
    n_segments: int | None = None,
) -> list[dict]:
    """Piecewise-constant action sequence.

    Holding the action fixed for a whole episode would teach the world model
    only what a steady state looks like.  Stepping it a few times per episode is
    what makes the *response* to an action observable, which is the thing the
    model actually has to learn and the agent has to exploit.
    """
    if n_segments is None:
        n_segments = int(rng.integers(3, 7))
    edges = np.sort(rng.choice(np.arange(1, n_frames), size=n_segments - 1, replace=False))
    edges = np.concatenate([[0], edges, [n_frames]])
    seq: list[dict] = []
    for k in range(n_segments):
        act = sample_action(cfg, rng)
        seq.extend([act] * int(edges[k + 1] - edges[k]))
    return seq[:n_frames]


def clip_action(cfg: PlantConfig, action: dict) -> dict:
    """Project an arbitrary action onto the valid envelope."""
    env = envelope(cfg)
    out = dict(action)
    for k in ("stoker_speed", "waste_feed"):
        out[k] = float(np.clip(action[k], *env[k]))
    out["primary_air"] = np.clip(
        np.atleast_1d(action["primary_air"]).astype(float), *env["primary_air"]
    ).tolist()
    out["secondary_air"] = np.clip(
        np.atleast_1d(action["secondary_air"]).astype(float), *env["secondary_air"]
    ).tolist()
    return out


# ----------------------------------------------------------------------
# Identification input design
#
# sample_sequence() randomises every air zone independently, which is fine for
# learning the field but useless for identifying a controller: averaging five
# independent uniforms concentrates the *aggregate* air level near its mean, so
# the data covers only ~65% of the primary-air range and almost none of the
# extremes - exactly the region a controller has to reach to cool the furnace
# or to add oxygen.
#
# For identification the input has to be persistently exciting in the variables
# the controller actually moves. This is an amplitude-modulated pseudo-random
# sequence (APRBS): piecewise-constant levels drawn across the whole envelope,
# with dwell times drawn from two bands so that both the ~1 s gas response and
# the ~17 s grate transport are excited.
FAST_DWELL = (10, 40)      # frames
SLOW_DWELL = (200, 600)    # frames - long enough to excite the grate
ZONE_JITTER = 0.10         # keeps some spatial variety around the common level


def sample_identification_sequence(cfg: PlantConfig, rng, n_frames: int) -> list[dict]:
    """Piecewise-constant aggregate levels spanning the full envelope."""

    def levels(lo, hi):
        out = np.empty(n_frames)
        t = 0
        while t < n_frames:
            band = FAST_DWELL if rng.random() < 0.5 else SLOW_DWELL
            n = int(rng.integers(*band))
            out[t:t + n] = rng.uniform(lo, hi)
            t += n
        return out[:n_frames]

    env = envelope(cfg)
    stoker = levels(*env["stoker_speed"])
    feed = levels(*env["waste_feed"])
    pri = levels(*env["primary_air"])
    sec = levels(*env["secondary_air"])

    seq = []
    for k in range(n_frames):
        jz = 1.0 + rng.uniform(-ZONE_JITTER, ZONE_JITTER, cfg.n_zones)
        jn = 1.0 + rng.uniform(-ZONE_JITTER, ZONE_JITTER, cfg.n_nozzles)
        seq.append({
            "stoker_speed": float(stoker[k]),
            "waste_feed": float(feed[k]),
            "primary_air": np.clip(pri[k] * jz, *env["primary_air"]).tolist(),
            "secondary_air": np.clip(sec[k] * jn, *env["secondary_air"]).tolist(),
        })
    return seq
