#!/usr/bin/env python
"""Train the ensemble of controlled-variable models that MPC plans inside.

An ensemble, not a single model, because the controller needs two things from
it that one model cannot give: a worst case to be pessimistic about, and a
disagreement signal to use as a trust region.
"""
from __future__ import annotations

import argparse, pickle, time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from wmf.control import model as cvm
from wmf.control.data import load_split, norm_stats, windows


def adam_init(p):
    z = lambda: jax.tree.map(jnp.zeros_like, p)
    return ((z(), z()), 0)


def adam_update(p, g, st, lr, b1=.9, b2=.999, eps=1e-8):
    (m, v), t = st; t += 1
    m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
    v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
    mh = jax.tree.map(lambda a: a / (1 - b1 ** t), m)
    vh = jax.tree.map(lambda a: a / (1 - b2 ** t), v)
    p = jax.tree.map(lambda a, b, c: a - lr * b / (jnp.sqrt(c) + eps), p, mh, vh)
    return p, ((m, v), t)


def train_one(seed, d, stats, horizon, warmup, steps, batch, lr, hidden):
    rng = np.random.default_rng(1000 + seed)
    gen = windows(d, stats, horizon, batch, rng, warmup)
    p = cvm.init_params(jax.random.PRNGKey(seed), hidden)
    opt = adam_init(p)

    @jax.jit
    def step(p, opt, yw, uw, uf, yf):
        loss, g = jax.value_and_grad(cvm.rollout_loss)(p, yw, uw, uf, yf)
        p, opt = adam_update(p, g, opt, lr)
        return p, opt, loss

    run = None
    for i in range(steps):
        yw, uw, uf, yf = next(gen)
        p, opt, l = step(p, opt, jnp.asarray(yw), jnp.asarray(uw),
                         jnp.asarray(uf), jnp.asarray(yf))
        run = float(l) if run is None else .98 * run + .02 * float(l)
    return p, run


def evaluate(ps, d, stats, horizon, warmup, n_win=64):
    """Open-loop error over the MPC horizon, in physical units."""
    rng = np.random.default_rng(7)
    gen = windows(d, stats, horizon, n_win, rng, warmup)
    yw, uw, uf, yf = next(gen)
    preds = []
    for p in ps:
        h = cvm.warm_up(p, jnp.asarray(yw), jnp.asarray(uw))
        preds.append(cvm.predict(p, h, jnp.asarray(yw)[:, -1], jnp.asarray(uf)))
    pred = jnp.stack(preds).mean(0)
    err = np.abs(np.asarray(pred) - yf) * stats["y_std"]
    base = np.abs(yf - yw[:, -1:, :]) * stats["y_std"]     # "nothing changes"
    return err, base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_ctrl_v2")
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="figs/cv_model.pkl")
    a = ap.parse_args()

    tr = load_split("train", a.data)
    va = load_split("val", a.data)
    stats = norm_stats(tr)
    print(f"train {tr['y'].shape}  val {va['y'].shape}")

    ps = []
    for k in range(a.ensemble):
        t0 = time.time()
        p, loss = train_one(k, tr, stats, a.horizon, a.warmup, a.steps,
                            a.batch, a.lr, a.hidden)
        ps.append(p)
        print(f"  member {k}: loss {loss:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    names = ["t_exit [K]", "o2_exit [-]", "yf_exit [-]"]
    for split, dd in (("train", tr), ("val", va)):
        err, base = evaluate(ps, dd, stats, a.horizon, a.warmup)
        print(f"\n{split}: {a.horizon}ステップ({a.horizon*0.1:.1f}秒)先までの開ループ誤差")
        for i, n in enumerate(names):
            e1, eH = err[:, 0, i].mean(), err[:, -1, i].mean()
            b1, bH = base[:, 0, i].mean(), base[:, -1, i].mean()
            print(f"  {n:12s} 0.1s: {e1:8.4f} (無変化予測 {b1:8.4f})   "
                  f"{a.horizon*0.1:.1f}s: {eH:8.4f} (無変化予測 {bH:8.4f})")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"params": [jax.tree.map(np.asarray, p) for p in ps],
                     "stats": {k: np.asarray(v) for k, v in stats.items()},
                     "horizon": a.horizon, "warmup": a.warmup}, f)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
