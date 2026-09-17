"""Trusted simulator data generation. Candidate code cannot replace the solver."""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np

CV = ["t_exit", "o2_exit", "yf_exit", "unburnt_bed"]
PROXIES = ["t_exit", "o2_exit", "yf_exit", "co_proxy", "nox_proxy", "unburnt_bed"]
MV = ["stoker_speed", "waste_feed", "primary_level", "secondary_level"]
DEFAULT_POLICY = {"bounds_fraction": {k: [0.0, 1.0] for k in MV},
                  "slow_probability": {k: 0.5 for k in MV}}


def validate_policy(policy):
    if set(policy) != {"bounds_fraction", "slow_probability"}:
        raise ValueError("sampling must contain bounds_fraction and slow_probability only")
    if any(set(policy[k]) != set(MV) for k in policy):
        raise ValueError("sampling must specify the four aggregate actions")
    for k in MV:
        b = policy["bounds_fraction"][k]
        p = policy["slow_probability"][k]
        if (not isinstance(b, list) or len(b) != 2 or
                not all(isinstance(x, (int, float)) and np.isfinite(x) for x in b) or
                not 0 <= b[0] < b[1] <= 1):
            raise ValueError(f"invalid fractional bounds for {k}")
        if not isinstance(p, (int, float)) or not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"invalid slow_probability for {k}")
    return policy


def actions(cfg, seed, frames, policy):
    from wmf.actions import (sample_identification_sequence, envelope,
                             FAST_DWELL, SLOW_DWELL, ZONE_JITTER)
    validate_policy(policy)
    rng = np.random.default_rng(seed)
    if policy == DEFAULT_POLICY:
        return sample_identification_sequence(cfg, rng, frames)
    env = envelope(cfg)
    values = {}
    for k, ek in zip(MV, ["stoker_speed", "waste_feed", "primary_air", "secondary_air"]):
        lo, hi = env[ek]
        f0, f1 = policy["bounds_fraction"][k]
        low, high = lo + (hi - lo) * f0, lo + (hi - lo) * f1
        v = np.empty(frames)
        t = 0
        while t < frames:
            band = FAST_DWELL if rng.random() < 1 - policy["slow_probability"][k] else SLOW_DWELL
            n = int(rng.integers(*band))
            v[t:t+n] = rng.uniform(low, high)
            t += n
        values[k] = v
    seq = []
    for t in range(frames):
        jz = 1 + rng.uniform(-ZONE_JITTER, ZONE_JITTER, cfg.n_zones)
        jn = 1 + rng.uniform(-ZONE_JITTER, ZONE_JITTER, cfg.n_nozzles)
        seq.append({"stoker_speed": float(values[MV[0]][t]),
                    "waste_feed": float(values[MV[1]][t]),
                    "primary_air": np.clip(values[MV[2]][t] * jz, *env["primary_air"]).tolist(),
                    "secondary_air": np.clip(values[MV[3]][t] * jn, *env["secondary_air"]).tolist()})
    return seq


def generate(reference, output, seeds, policy, frames=1200):
    import jax
    import jax.numpy as jnp
    from wmf.plants.schema import PlantConfig
    from wmf.plants.encode import build_geometry, encode_action, action_to_vector
    from wmf.sim.physics import SimParams
    from wmf.sim.rollout import build_static, make_initial, rollout
    cfg = PlantConfig.from_yaml(Path(reference) / "configs/plants/plant_a.yaml")
    geom = build_geometry(cfg)
    static = build_static(geom)
    prm = SimParams.from_plant(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in seeds:
        start = time.monotonic()
        seq = actions(cfg, seed, frames, policy)
        fields = jnp.asarray(np.stack([encode_action(geom, a) for a in seq]))
        q0, b0 = make_initial(static)
        q, bed, diag = rollout(q0, b0, fields, static, prm, 333, frames)
        jax.block_until_ready((q, bed, diag))
        pr = np.stack([np.asarray(diag[k]) for k in PROXIES], -1)
        if not np.isfinite(pr).all():
            raise ValueError(f"non-finite simulator output for seed {seed}")
        meta = {"plant_id": cfg.plant_id, "plant_hash": cfg.hash(), "seed": seed,
                "n_zones": cfg.n_zones, "n_nozzles": cfg.n_nozzles,
                "dt": prm.dt, "save_every": 333, "control_interval": 333 * prm.dt,
                "frames": frames, "sampling_policy": policy,
                "alignment": "u[k] is applied before recorded y[k]; y[k] is post-action"}
        path = output / f"plant_a_{seed:06d}.npz"
        np.savez_compressed(path, proxies=pr.astype(np.float32),
                            a_scalar=np.array([action_to_vector(geom, a) for a in seq], np.float32),
                            reward=np.asarray(diag["reward"], np.float32), meta=json.dumps(meta))
        records.append({"file": str(path), "seed": seed, "frames": frames,
                        "physical_seconds": frames * 333 * prm.dt,
                        "wall_seconds": time.monotonic() - start})
        print(json.dumps({"generated": records[-1]}), flush=True)
        del q, bed, diag
    return records


def load_episodes(paths):
    ys, us = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            a = z["a_scalar"]
            nz, nn = meta["n_zones"], meta["n_nozzles"]
            ys.append(z["proxies"][:, [0, 1, 2, 5]].astype(np.float32))
            us.append(np.stack([a[:, 0], a[:, 1], a[:, 2:2+nz].mean(1),
                                a[:, 2+nz:2+nz+nn].mean(1)], -1).astype(np.float32))
    if not ys:
        raise ValueError("empty dataset")
    d = {"y": np.stack(ys), "u": np.stack(us)}
    if not all(np.isfinite(v).all() for v in d.values()):
        raise ValueError("non-finite dataset")
    return d


def action_coverage(data, u_lo, u_hi):
    """Descriptive coverage, not an estimate of causal information gain."""
    u = data["u"].reshape(-1, 4).astype(float)
    span = np.asarray(u_hi) - np.asarray(u_lo)
    un = (u - u_lo) / span
    bins = np.clip((un * 10).astype(int), 0, 9)
    cov = np.cov(un.T)
    return {"range_fraction": np.ptp(un, axis=0).tolist(),
            "occupied_deciles": [int(len(np.unique(bins[:, k]))) for k in range(4)],
            "input_covariance_eigenvalues": np.linalg.eigvalsh(cov).tolist()}
