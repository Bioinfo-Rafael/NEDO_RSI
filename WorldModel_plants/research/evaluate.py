"""Fixed evaluation: physical-unit errors with immutable baseline scales."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np

from research.data import CV, MV


def forecast_inputs(data, stats, warmup, origin=300, horizon=900):
    if not 1 <= warmup <= origin:
        raise ValueError(f"WARMUP must be in [1,{origin}]")
    if data["y"].shape[1] < origin + horizon:
        raise ValueError("episode too short for the fixed prediction origin/horizon")
    y = (data["y"] - stats["y_mean"]) / stats["y_std"]
    u = (data["u"] - stats["u_mean"]) / stats["u_std"]
    # u[k] produced y[k]. Thus u[origin] advances y[origin-1] to target y[origin].
    return (y[:, origin-warmup:origin], u[:, origin-warmup:origin],
            u[:, origin:origin+horizon], data["y"][:, origin:origin+horizon])


def forecast_metrics(pred, target, scale, horizons=(30, 300, 900)):
    pred, target, scale = np.asarray(pred, float), np.asarray(target, float), np.asarray(scale, float)
    if pred.shape != target.shape or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("invalid prediction shape or non-finite prediction")
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("invalid fixed metric scale")
    per, means = {}, []
    for h in horizons:
        if h > pred.shape[1]:
            raise ValueError("requested horizon exceeds prediction length")
        rmse = np.sqrt(np.mean((pred[:, :h] - target[:, :h]) ** 2, axis=(0, 1)))
        nr = rmse / scale
        per[str(h)] = {"rmse": dict(zip(CV, rmse.tolist())),
                       "nrmse": dict(zip(CV, nr.tolist())), "mean_nrmse": float(nr.mean())}
        means.append(float(nr.mean()))
    return {"forecast_nrmse": float(np.mean(means)), "horizons": per}


def forecast(wm, params, stats, data, metric_scale, output):
    import jax
    import jax.numpy as jnp
    start = time.monotonic()
    yw, uw, uf, target = forecast_inputs(data, stats, wm.WARMUP)
    yw, uw, uf = map(jnp.asarray, (yw, uw, uf))
    predictions = []
    for p in params:
        fn = jax.jit(lambda p: wm.predict(p, wm.warm_up(p, yw, uw), yw[:, -1], uf, bias=None))
        predictions.append(np.asarray(fn(p)))
    pn = np.mean(predictions, axis=0)
    pred = pn * stats["y_std"] + stats["y_mean"]
    out = forecast_metrics(pred, target, metric_scale)
    out["seconds"] = time.monotonic() - start
    # Aggregate diagnostics by episode and action level, without giving future y to the model.
    err = np.abs((pred - target) / metric_scale)
    out["episode_mean_normalized_absolute_error"] = err.mean(axis=(1, 2)).tolist()
    out["action_condition_error"] = {}
    u = data["u"][:, 300:1200]
    for k, name in enumerate(MV):
        edges = np.quantile(u[..., k], [0, 1/3, 2/3, 1])
        groups = []
        for j in range(3):
            mask = (u[..., k] >= edges[j]) & (u[..., k] <= edges[j+1])
            groups.append({"range": edges[j:j+2].tolist(), "count": int(mask.sum()),
                           "mean_normalized_absolute_error": float(err[mask].mean()) if mask.any() else None})
        out["action_condition_error"][name] = groups
    np.savez_compressed(Path(output) / "forecast.npz", prediction=pred, target=target,
                        origin_step=300, metric_scale=metric_scale)
    return out


def control_metrics(y, u, r, scenario, scale, du_max, settle=40):
    y, u, r, scale = map(lambda x: np.asarray(x, float), (y, u, r, scale))
    if not all(np.isfinite(a).all() for a in (y, u, r, scale)) or np.any(scale <= 0):
        raise ValueError("non-finite control rollout/invalid fixed metric scale")
    t = scenario["t_step"]
    mask = np.ones(len(y), bool)
    mask[:settle] = False
    mask[t:t+settle] = False
    if not mask.any() or t >= len(y):
        raise ValueError("control scenario has no scored samples")
    iae = (np.abs(y-r)[mask] / scale).mean(0)
    tracking = float(iae @ np.array([1.0, 1.0, 0.0, 0.7]))
    r0, r1 = np.asarray(scenario["r0"]), np.asarray(scenario["r1"])
    ch = int(np.argmax(np.abs(r1-r0) / scale))
    seg = y[t:, ch]
    overshoot = float(max(0.0, seg.max()-r1[ch] if r1[ch] > r0[ch] else r1[ch]-seg.min()) / scale[ch])
    move = float((np.abs(np.diff(u, axis=0)) / np.asarray(du_max)).mean())
    return {"tracking": tracking, "overshoot": overshoot, "move": move,
            "iae_per_channel": dict(zip(CV, iae.tolist())),
            "control_cost": tracking + 0.5*overshoot + 0.1*move}


def control(wm, params, stats, scale, scenarios, steps, output, *, capture=False, smoke=False):
    from run_control import Plant
    from wmf.control.mpc import MPC, MPCConfig
    from wmf.control.model import Scaler
    plant = Plant("a")
    out = {}
    start = time.monotonic()
    for scenario in scenarios:
        plant.reset()
        u = plant.u_init.copy()
        cfg = MPCConfig()
        if smoke:
            cfg.population, cfg.elites, cfg.iterations = 32, 8, 1
        ctl = MPC(params, Scaler(**stats), plant.u_lo, plant.u_hi, cfg, wm=wm)
        ctl.reset(u)
        ys, us, rs, fields, field_steps = [], [], [], [], []
        for k in range(steps):
            r = np.array(scenario["r0"] if k < scenario["t_step"] else scenario["r1"], np.float32)
            y = plant.step(u)
            if not np.isfinite(y).all() or not np.isfinite(u).all():
                raise ValueError(f"non-finite closed-loop state at step {k}")
            ys.append(y.copy()); us.append(u.copy()); rs.append(r.copy())
            if capture and scenario is scenarios[0] and k % 8 == 0:
                fields.append(np.asarray(plant.q[..., 0])); field_steps.append(k)
            if len(ys) >= wm.WARMUP:
                ctl.update_bias(y)
                u = ctl.act(np.array(ys[-wm.WARMUP:]), np.array(us[-wm.WARMUP:]), r, u)
        y, u, r = map(np.asarray, (ys, us, rs))
        out[scenario["name"]] = control_metrics(y, u, r, scenario, scale, cfg.du_max, 5 if smoke else 40)
        np.savez_compressed(Path(output) / f"control_{scenario['name']}.npz",
                            y=y, u=u, r=r, heat=np.asarray(fields), heat_steps=np.asarray(field_steps),
                            control_interval=plant.save_every*plant.prm.dt)
    out["control_cost"] = float(np.mean([x["control_cost"] for x in out.values()]))
    out["seconds"] = time.monotonic() - start
    return out
