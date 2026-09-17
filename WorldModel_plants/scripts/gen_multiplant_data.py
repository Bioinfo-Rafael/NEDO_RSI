#!/usr/bin/env python
"""Identification data across many procedurally generated plants.

Each plant gets its own excitation sequence and its own normalised
specification vector mu_known, saved alongside the trajectory so that a
plant-conditioned model has something to condition on.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from wmf.actions import sample_identification_sequence
from wmf.plants.encode import action_to_vector, build_geometry, encode_action
from wmf.plants.generator import generate
from wmf.plants.mu import mu_known
from wmf.sim.physics import SimParams
from wmf.sim.rollout import build_static, make_initial, rollout_batch

PROXY_KEYS = ("t_exit", "o2_exit", "yf_exit", "co_proxy", "nox_proxy", "unburnt_bed")


def run_plant(cfg, seeds, out_dir, n_frames, save_every):
    geom = build_geometry(cfg); static = build_static(geom)
    prm = SimParams.from_plant(cfg)
    mu = mu_known(cfg, geom)
    b = len(seeds)
    fields, scalars = [], []
    for sd in seeds:
        acts = sample_identification_sequence(cfg, np.random.default_rng(sd), n_frames)
        fields.append(np.stack([encode_action(geom, a) for a in acts]))
        scalars.append(np.stack([action_to_vector(geom, a) for a in acts]))
    q0, bed0 = make_initial(static)
    q0 = jnp.broadcast_to(q0, (b,) + q0.shape); bed0 = jnp.broadcast_to(bed0, (b,) + bed0.shape)
    st = {k: jnp.broadcast_to(jnp.asarray(v), (b,) + jnp.asarray(v).shape)
          for k, v in static.items()}
    q, bed, diag = rollout_batch(q0, bed0, jnp.asarray(np.stack(fields)), st,
                                 prm, save_every, n_frames)
    prox = np.stack([np.asarray(diag[k]) for k in PROXY_KEYS], axis=-1)
    written = 0
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for i, sd in enumerate(seeds):
        if not np.isfinite(prox[i]).all():
            continue
        np.savez_compressed(out_dir / f"{cfg.plant_id}_{sd:04d}.npz",
            proxies=prox[i].astype(np.float32),
            a_scalar=np.asarray(scalars[i], np.float32),
            mu=mu.astype(np.float32),
            meta=json.dumps({"plant_id": cfg.plant_id, "family": cfg.plant_id.split("_")[0],
                             "n_zones": cfg.n_zones, "n_nozzles": cfg.n_nozzles,
                             "dt": prm.dt, "save_every": save_every, "seed": int(sd),
                             "mu_names": mu_known.names()}))
        written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_multi")
    ap.add_argument("--per-family", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--save-every", type=int, default=333)
    ap.add_argument("--families", default="",
                    help="comma-separated family names to run (default all); lets processes split the work")
    a = ap.parse_args()

    plants = generate(a.per_family, seed=0)
    if a.families:
        keep = set(a.families.split(","))
        plants = {k: v for k, v in plants.items() if k in keep}
    total_p = sum(len(v) for v in plants.values())
    print(f"{len(plants)} 族 / {total_p} プラント / 各 {a.episodes} エピソード "
          f"= {total_p*a.episodes} 本", flush=True)

    t0 = time.time(); n = 0
    for fam, cfgs in plants.items():
        for c in cfgs:
            seeds = list(range(n * 100, n * 100 + a.episodes))
            w = run_plant(c, seeds, Path(a.out) / fam, a.frames, a.save_every)
            n += 1
            print(f"  [{n:3d}/{total_p}] {c.plant_id:16s} {w} eps  "
                  f"({time.time()-t0:5.0f}s)", flush=True)
    print(f"完了 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
