#!/usr/bin/env python
"""Controlled A/B: does conditioning on the geometry help on an unseen plant?

The autoresearch loop scores val_nrmse on the *seen* plant, so it discards the
geometry input - correctly, by that metric. This asks the question the project
actually cares about: hold everything else fixed, change only whether the model
can see the furnace it is predicting, and measure on plants it never trained on.

Same seed, same budget, same architecture. One variable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INTEG = ROOT / "integration"
sys.path.insert(0, str(INTEG))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, INTEG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


def run(use_geometry: bool, seconds: float, seed: int):
    import jax, jax.numpy as jnp
    prepare = _load("prepare"); train = _load("train"); ev = _load("evaluate")

    def build_input(x, a, g, dx):
        if not use_geometry:
            return jnp.concatenate([x, a], axis=-1)
        spacing = jnp.broadcast_to(dx[:, None, None, None] / 0.1875, x.shape[:-1] + (1,))
        return jnp.concatenate([x, a, g, spacing], axis=-1)

    d = prepare.load()
    rng = np.random.default_rng(seed)
    stream = prepare.batches(d, "train", train.BATCH, rng)
    (x0, a0, g0, dx0), y0 = next(stream)
    inp0 = build_input(jnp.asarray(x0), jnp.asarray(a0), jnp.asarray(g0), jnp.asarray(dx0))
    params = train.init_params(jax.random.PRNGKey(seed), inp0.shape[-1], y0.shape[-1])
    opt = train.adam_init(params)

    def loss_fn(p, inp, y):
        return jnp.mean((train.forward(p, inp) - y) ** 2)

    @jax.jit
    def step(params, opt, x, a, g, dx, y):
        inp = build_input(x, a, g, dx)
        loss, grads = jax.value_and_grad(loss_fn)(params, inp, y)
        params, opt = train.adam_update(params, grads, opt, train.LR)
        return params, opt, loss

    t_end = time.time() + seconds
    n = 0
    while time.time() < t_end:
        (x, a, g, dx), y = next(stream)
        params, opt, _ = step(params, opt, jnp.asarray(x), jnp.asarray(a),
                              jnp.asarray(g), jnp.asarray(dx), jnp.asarray(y))
        n += 1

    res = ev.score(params, d, build_input, train.forward)
    res["steps"] = n
    res["n_params"] = sum(int(np.asarray(w).size + np.asarray(b).size) for w, b in params)
    return res, params, build_input, d, train


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figs/ab_geometry.json")
    args = ap.parse_args()

    out = {}
    for label, use_g in (("no_geometry", False), ("with_geometry", True)):
        print(f"--- {label} ---", flush=True)
        res, params, bi, d, train = run(use_g, args.seconds, args.seed)
        out[label] = {k: v for k, v in res.items() if not k.endswith("_per_channel")}
        out[label]["per_channel_b"] = res["val_b_per_channel"]
        print(f"  val_iid {res['val_iid']:.4f}   val_b {res['val_b']:.4f}   "
              f"val_c {res['val_c']:.4f}   gap_b {res['val_b']/res['val_iid']:.2f}", flush=True)
        if use_g:
            np.savez_compressed(Path(args.out).parent / "params_with_geometry.npz",
                                **{f"p{i}": np.asarray(v) for i, v in
                                   enumerate(jax.tree.flatten(params)[0])})

    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    a, b = out["no_geometry"], out["with_geometry"]
    print()
    print(f"unseen plant_b:  {a['val_b']:.4f} -> {b['val_b']:.4f}  "
          f"({(1 - b['val_b']/a['val_b'])*100:+.1f}%)")
    print(f"unseen plant_c:  {a['val_c']:.4f} -> {b['val_c']:.4f}  "
          f"({(1 - b['val_c']/a['val_c'])*100:+.1f}%)")
    return 0


if __name__ == "__main__":
    import jax
    raise SystemExit(main())
