#!/usr/bin/env python
"""Identification dataset for the controller.

Only the controlled variables and the actions are kept - not the 64x64x7 field.
That is ~2000x smaller per episode, so the run is bounded by compute rather than
disk and many more operating points can be covered.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from wmf.actions import sample_identification_sequence
from wmf.plants.encode import action_to_vector, build_geometry, encode_action
from wmf.plants.schema import PlantConfig
from wmf.sim.physics import SimParams
from wmf.sim.rollout import build_static, make_initial, rollout

PROXY_KEYS = ("t_exit", "o2_exit", "yf_exit", "co_proxy", "nox_proxy", "unburnt_bed")


def generate(cfg, seeds, out_dir, n_frames, save_every, batch):
    geom = build_geometry(cfg); static = build_static(geom)
    prm = SimParams.from_plant(cfg)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for s0 in range(0, len(seeds), batch):
        chunk = seeds[s0:s0 + batch]
        fields, scalars = [], []
        for sd in chunk:
            acts = sample_identification_sequence(cfg, np.random.default_rng(sd), n_frames)
            fields.append(np.stack([encode_action(geom, a) for a in acts]))
            scalars.append(np.stack([action_to_vector(geom, a) for a in acts]))
        b = len(chunk)
        q0, bed0 = make_initial(static)
        q0 = jnp.broadcast_to(q0, (b,) + q0.shape); bed0 = jnp.broadcast_to(bed0, (b,) + bed0.shape)
        st = {k: jnp.broadcast_to(jnp.asarray(v), (b,) + jnp.asarray(v).shape)
              for k, v in static.items()}
        from wmf.sim.rollout import rollout_batch
        q, bed, diag = rollout_batch(q0, bed0, jnp.asarray(np.stack(fields)), st,
                                     prm, save_every, n_frames)
        prox = np.stack([np.asarray(diag[k]) for k in PROXY_KEYS], axis=-1)  # [B,T,6]
        rew = np.asarray(diag["reward"])
        for i, sd in enumerate(chunk):
            if not np.isfinite(prox[i]).all():
                print(f"  !! seed {sd} non-finite, dropped"); continue
            np.savez_compressed(out_dir / f"{cfg.plant_id}_{sd:05d}.npz",
                proxies=prox[i].astype(np.float32),
                a_scalar=np.asarray(scalars[i], np.float32),
                reward=rew[i].astype(np.float32),
                meta=json.dumps({"plant_id": cfg.plant_id, "n_zones": cfg.n_zones,
                                 "n_nozzles": cfg.n_nozzles, "dt": prm.dt,
                                 "save_every": save_every, "seed": int(sd),
                                 "control_interval": prm.dt * save_every}))
            written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_ctrl")
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--train-start", type=int, default=0,
                    help="first train seed; lets several processes split the work")
    ap.add_argument("--val-start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=1200, help="1 s control frames; two grate residences")
    ap.add_argument("--save-every", type=int, default=333, help="gas steps per frame: 1 s at dt = 3 ms")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--plant", default="a")
    a = ap.parse_args()
    cfg = PlantConfig.from_yaml(f"configs/plants/plant_{a.plant}.yaml")
    for name, seeds in (("train", range(a.train_start, a.train_start + a.n_train)),
                        ("val", range(50_000 + a.val_start, 50_000 + a.val_start + a.n_val))):
        if len(seeds) == 0:
            continue
        t0 = time.time()
        n = generate(cfg, list(seeds), Path(a.out) / name, a.frames, a.save_every, a.batch)
        print(f"{name:6s} {n:3d} episodes  {time.time()-t0:6.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
