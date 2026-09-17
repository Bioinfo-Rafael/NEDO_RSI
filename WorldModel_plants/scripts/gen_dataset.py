#!/usr/bin/env python
"""Generate the episode dataset.

The split is **by plant**, which is the whole point of the exercise:

    train     plant_a, seeds 0..N-1          seen plant, seen conditions
    val_iid   plant_a, disjoint seeds        seen plant, unseen conditions
    val_b     plant_b                        unseen plant, inside the design range
    val_c     plant_c                        unseen plant, outside it

A model that only memorises plant_a will look fine on val_iid and fall over on
val_b/val_c, which is exactly the distinction the demo has to be able to show.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from wmf.actions import sample_sequence
from wmf.plants.encode import action_to_vector, build_geometry, dimensionless, encode_action
from wmf.plants.schema import PlantConfig
from wmf.sim.episode import N_FRAMES, SAVE_EVERY, save_episode
from wmf.sim.physics import SimParams
from wmf.sim.rollout import build_static, make_initial, rollout_batch


def generate(cfg, seeds, out_dir, n_frames, save_every, batch_size):
    """Simulate a plant's episodes, vmapped over seeds."""
    geom = build_geometry(cfg)
    static = build_static(geom)
    prm = SimParams.from_plant(cfg)
    out_dir = Path(out_dir)
    written = []

    for start in range(0, len(seeds), batch_size):
        chunk = seeds[start:start + batch_size]
        acts, fields, scalars = [], [], []
        for sd in chunk:
            a = sample_sequence(cfg, np.random.default_rng(sd), n_frames)
            acts.append(a)
            fields.append(np.stack([encode_action(geom, x) for x in a]))
            scalars.append(np.stack([action_to_vector(geom, x) for x in a]))

        b = len(chunk)
        q0, bed0 = make_initial(static)
        q0 = jnp.broadcast_to(q0, (b,) + q0.shape)
        bed0 = jnp.broadcast_to(bed0, (b,) + bed0.shape)
        stacked = {k: jnp.broadcast_to(jnp.asarray(v), (b,) + jnp.asarray(v).shape)
                   for k, v in static.items()}

        q, bed, diag = rollout_batch(q0, bed0, jnp.asarray(np.stack(fields)),
                                     stacked, prm, save_every, n_frames)
        q = np.asarray(q); bed = np.asarray(bed)

        for i, sd in enumerate(chunk):
            if not np.isfinite(q[i]).all():
                print(f"  !! dropping seed {sd}: non-finite")
                continue
            meta = {
                "plant_id": cfg.plant_id, "plant_hash": cfg.hash(), "seed": int(sd),
                "dx": geom.dx, "dy": geom.dy, "dt": prm.dt,
                "save_every": int(save_every), "control_interval": prm.dt * save_every,
                "n_zones": cfg.n_zones, "n_nozzles": cfg.n_nozzles,
                "dimensionless": dimensionless(geom, acts[i][0]),
                "a_scalar_keys": cfg.action_keys,
                "sim_params": {"q_cp": prm.q_cp, "lam_loss": prm.lam_loss, "t_wall": prm.t_wall},
                "norm": {"q_mean": q[i].reshape(-1, 7).mean(0).tolist(),
                         "q_std": q[i].reshape(-1, 7).std(0).tolist()},
                "hidden": {"lhv": cfg.waste.lhv, "moisture": cfg.waste.moisture,
                           "wall_heat_transfer": cfg.wall.heat_transfer,
                           "wall_temperature": cfg.wall.temperature},
            }
            ep = {
                "q": q[i].astype(np.float16),
                "a": np.asarray(fields[i], dtype=np.float16),
                "g": geom.g.astype(np.float32),
                "bed": bed[i].astype(np.float32),
                "a_scalar": np.asarray(scalars[i], dtype=np.float32),
                "reward": np.asarray(diag["reward"][i], dtype=np.float32),
                "proxies": np.stack([np.asarray(diag[k][i]) for k in
                    ("t_exit", "o2_exit", "yf_exit", "co_proxy", "nox_proxy", "unburnt_bed")],
                    axis=-1).astype(np.float32),
                "meta": json.dumps(meta),
            }
            written.append(save_episode(ep, out_dir / f"{cfg.plant_id}_{sd:05d}.npz"))
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-ood", type=int, default=25)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--save-every", type=int, default=SAVE_EVERY)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--configs", default="configs/plants")
    args = ap.parse_args()

    cfgs = {p: PlantConfig.from_yaml(f"{args.configs}/plant_{p}.yaml") for p in "abc"}
    out = Path(args.out)

    # seeds are disjoint across splits so val_iid is genuinely unseen
    jobs = [
        ("train",   cfgs["a"], list(range(0, args.n_train))),
        ("val_iid", cfgs["a"], list(range(10_000, 10_000 + args.n_val))),
        ("val_b",   cfgs["b"], list(range(20_000, 20_000 + args.n_ood))),
        ("val_c",   cfgs["c"], list(range(30_000, 30_000 + args.n_ood))),
    ]
    for name, cfg, seeds in jobs:
        t0 = time.time()
        files = generate(cfg, seeds, out / name, args.frames, args.save_every, args.batch_size)
        mb = sum(f.stat().st_size for f in files) / 1e6
        print(f"{name:9s} {cfg.plant_id}  {len(files):4d} episodes  {mb:8.1f} MB  "
              f"{time.time() - t0:6.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
