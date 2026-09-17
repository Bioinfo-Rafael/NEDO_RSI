"""Level 1 CLI: config in, episode .npz out (API.md 2).

    python -m wmf.sim.run --plant configs/plants/plant_a.yaml \\
        --actions random --steps 200 --seed 0 --out data/train/ep_0001.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..actions import clip_action, sample_sequence
from ..plants.schema import PlantConfig
from .episode import N_FRAMES, SAVE_EVERY, run_episode, save_episode


def _load_actions(path: str, cfg: PlantConfig, n_frames: int) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):          # one action, held for the episode
        raw = [raw] * n_frames
    if len(raw) < n_frames:            # hold the last one
        raw = raw + [raw[-1]] * (n_frames - len(raw))
    return [clip_action(cfg, a) for a in raw[:n_frames]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plant", required=True, help="plant config YAML")
    ap.add_argument("--actions", default="random",
                    help="'random', 'hold', or a path to an action JSON file")
    ap.add_argument("--steps", type=int, default=N_FRAMES, help="control frames")
    ap.add_argument("--save-every", type=int, default=SAVE_EVERY,
                    help="gas steps per control frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = PlantConfig.from_yaml(args.plant)
    rng = np.random.default_rng(args.seed)

    if args.actions == "random":
        actions = None                                     # sampled in run_episode
    elif args.actions == "hold":
        actions = sample_sequence(cfg, rng, args.steps, n_segments=1)
    else:
        actions = _load_actions(args.actions, cfg, args.steps)

    ep = run_episode(cfg, seed=args.seed, n_frames=args.steps,
                     save_every=args.save_every, actions=actions)
    path = save_episode(ep, args.out)

    meta = json.loads(ep["meta"])
    finite = bool(np.isfinite(ep["q"]).all())
    print(f"{path}  q{ep['q'].shape} a{ep['a'].shape} g{ep['g'].shape} "
          f"bed{ep['bed'].shape}  {path.stat().st_size / 1e6:.1f} MB")
    print(f"  plant={meta['plant_id']} dx={meta['dx']:.4f} dt={meta['dt']} "
          f"control_interval={meta['control_interval']:.3f}s  finite={finite}")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
