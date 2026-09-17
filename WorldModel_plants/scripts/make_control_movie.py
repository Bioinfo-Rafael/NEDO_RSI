#!/usr/bin/env python
"""Closed-loop control, recorded: the furnace on the left, the tracking on the right.

The point of the animation is that both halves are the same run. The field is
what the simulator is actually doing; the traces are what the controller is
holding to setpoints while it does it.
"""
from __future__ import annotations

import argparse, json, pickle, sys, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))

from run_control import Plant, load_wm
from wmf.control import model as cvm
from wmf.control.mpc import MPC, MPCConfig

INK, INK2, MUTED = "#17131c", "#5a5262", "#b8b7b0"
plt.rcParams.update({"font.family": ["Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "Yu Gothic", "Helvetica Neue", "sans-serif"],
                     "axes.unicode_minus": False, "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": .7,
                     "axes.axisbelow": True})


def run(steps, target, model_path, target1=None, t_step=150):
    """``target`` until ``t_step``, then ``target1`` - the same setpoint step
    evaluate.py scores, so the movie shows the plant being *moved*."""
    b = pickle.load(open(model_path, "rb"))
    sc = cvm.Scaler(**{k: b["stats"][k] for k in ("y_mean", "y_std", "u_mean", "u_std")})
    p = Plant("a")
    y = p.reset()
    u = p.u_init.copy()
    ctl = MPC(b["params"], sc, p.u_lo, p.u_hi, MPCConfig(), wm=load_wm(b)); ctl.reset(u)
    warm = b["warmup"]

    Q, Y, U, R = [], [], [], []
    r = np.asarray(target, np.float32)
    t0 = time.time()
    for k in range(steps):
        if target1 is not None and k == t_step:
            r = np.asarray(target1, np.float32)
        y = p.step(u)
        Q.append(np.asarray(p.q)[..., 0].copy())     # temperature field
        Y.append(y.copy()); U.append(u.copy()); R.append(r.copy())
        if len(Y) >= warm:
            ctl.update_bias(y)
            u = ctl.act(np.array(Y[-warm:]), np.array(U[-warm:]), r, u)
    print(f"  {steps} steps in {time.time()-t0:.0f}s")
    return np.array(Q), np.array(Y), np.array(U), np.array(R), p


def render(Q, Y, U, R, plant, out, warm, stride=2, fps=12):
    mask = plant.geom.mask > 0
    t = np.arange(len(Y)) * 1.0          # control interval 1 s
    vmin, vmax = np.percentile(Q[:, mask], [2, 98])

    # The colourbar goes underneath the field, not beside it: a vertical bar
    # between the two halves collides with the traces' y-axis labels.
    fig = plt.figure(figsize=(11.4, 5.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.45], hspace=.42, wspace=.30,
                          left=.035, right=.975, top=.90, bottom=.11)
    axf = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, 1]); ax3 = fig.add_subplot(gs[2, 1])

    im = axf.imshow(np.where(mask, Q[0], np.nan), origin="lower", cmap="inferno",
                    vmin=vmin, vmax=vmax, interpolation="bilinear")
    axf.set_facecolor("#dbdbd9"); axf.set_xticks([]); axf.set_yticks([]); axf.grid(False)
    cb = fig.colorbar(im, ax=axf, orientation="horizontal", fraction=.055, pad=.04)
    cb.set_label("炉内温度 [K]", fontsize=8.5, color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=7.5)

    specs = [(ax1, 0, "出口温度 [K]", "#c43d28"),
             (ax2, 1, "出口 O2 [-]", "#c43d28"),
             (ax3, 3, "灰の未燃分", "#c43d28")]
    lines = []
    for ax, ch, lab, c in specs:
        ax.step(t, R[:, ch], where="post", color=c, lw=1.8, ls=(0, (5, 3)))
        ax.text(.995, R[-1, ch], f" 目標 {R[-1, ch]:.3g}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color=c, fontweight="bold")
        ln, = ax.plot([], [], color="#2a78d6", lw=2)
        lines.append(ln)
        ax.set_ylabel(lab, fontsize=8.5)
        ax.set_xlim(0, t[-1])
        lo = min(Y[:, ch].min(), R[:, ch].min()); hi = max(Y[:, ch].max(), R[:, ch].max())
        pad = .12 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
    ax3.set_xlabel("時間 [秒]", fontsize=8.5)
    ax1.set_title("World Model の中で計画した操作で、シミュレータを制御している",
                  fontsize=11, loc="left", color=INK, pad=8)

    warm_note = axf.text(.03, .965, "", transform=axf.transAxes, fontsize=9.5,
                         color="white", va="top", fontweight="bold")

    w = PillowWriter(fps=fps)
    with w.saving(fig, str(out), dpi=88):
        for k in range(0, len(Y), stride):
            im.set_data(np.where(mask, Q[k], np.nan))
            for ln, (ax, ch, *_rest) in zip(lines, specs):
                ln.set_data(t[:k + 1], Y[:k + 1, ch])
            warm_note.set_text("履歴を蓄積中（制御開始前）" if k < warm else "MPC 制御中")
            axf.set_title(f"t = {t[k]:5.1f} s", fontsize=10, color=INK)
            w.grab_frame()
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--model", default="figs/wm_bundle.pkl")
    ap.add_argument("--out", default="figs/control_movie.gif")
    a = ap.parse_args()
    sp = json.loads((ROOT / "figs" / "setpoints.json").read_text())["temp_step"]
    Q, Y, U, R, plant = run(a.steps, np.asarray(sp["r0"], np.float32), a.model,
                            target1=np.asarray(sp["r1"], np.float32), t_step=150)
    b = pickle.load(open(a.model, "rb"))
    print(render(Q, Y, U, R, plant, Path(a.out), b["warmup"]))
