#!/usr/bin/env python
"""Figures for the control work: setpoint tracking, and why it is hard."""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, MUTED = "#17131c", "#5a5262", "#b8b7b0"
C = {"hold": MUTED, "pi": "#eb6834", "mpc": "#2a78d6"}
LAB = {"hold": "制御なし", "pi": "PI制御", "mpc": "MPC（World Model 内で計画）"}
plt.rcParams.update({
    "font.family": ["Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "Yu Gothic", "Helvetica Neue", "sans-serif"],
    "axes.unicode_minus": False, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": .8,
    "axes.axisbelow": True})


def fig_tracking(js: Path, out: Path):
    runs = json.loads(js.read_text())
    tgt = np.asarray(next(iter(runs.values()))["target"])
    rows = [(0, "炉出口温度 [K]", tgt[0]), (1, "炉出口 O2 [-]", tgt[1]),
            (3, "灰の未燃分 [kg/m2/s]", tgt[3])]
    fig, axes = plt.subplots(len(rows), 1, figsize=(9.0, 7.6), sharex=True)
    first = next(iter(runs.values()))["log"]
    for ax, (ch, lab, r) in zip(axes, rows):
        if "r" in first[0]:
            tr = np.array([s["t"] for s in first]); rr = np.array([s["r"][ch] for s in first])
            ax.step(tr, rr, where="post", color="#c43d28", lw=2, ls=(0, (5, 3)), zorder=1)
            r = rr[-1]
        else:
            ax.axhline(r, color="#c43d28", lw=2, ls=(0, (5, 3)), zorder=1)
        lab_r = f"{r:.0f}" if abs(r) >= 100 else f"{r:.4g}"
        ax.annotate(f"目標 {lab_r}", (0.995, r), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=9, color="#c43d28", fontweight="bold")
        for name, run in runs.items():
            y = np.array([s["y"] for s in run["log"]])
            t = np.array([s["t"] for s in run["log"]])
            ax.plot(t, y[:, ch], color=C[name], lw=2, label=LAB[name], zorder=2)
        ax.set_ylabel(lab, fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=9.5, ncol=3, loc="upper left")
    axes[-1].set_xlabel("時間 [秒]", fontsize=9.5)
    fig.suptitle("同じ目標に対する3方式の追従  （プラント = 物理シミュレータ）",
                 fontsize=12, y=0.985)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


def fig_actions(js: Path, out: Path):
    runs = json.loads(js.read_text())
    names = ["火格子速度 [m/s]", "ごみ供給 [kg/s]", "一次空気 [Nm3/s]", "二次空気 [Nm3/s]"]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 4.8), sharex=True)
    for k, (ax, lab) in enumerate(zip(axes.ravel(), names)):
        for name, run in runs.items():
            u = np.array([s["u"] for s in run["log"]])
            t = np.array([s["t"] for s in run["log"]])
            ax.plot(t, u[:, k], color=C[name], lw=1.8, label=LAB[name])
        ax.set_ylabel(lab, fontsize=9)
    for ax in axes[-1]: ax.set_xlabel("時間 [秒]", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle("AIが実際に出している物理操作指令", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


def fig_authority(out: Path):
    """The measured input-output authority that reshaped the CV set."""
    mv = ["火格子速度", "ごみ供給", "一次空気", "二次空気"]
    cv = ["出口温度", "出口O2", "ガス未燃分", "灰の未燃分"]
    G = np.array([[0.34, 0.04, 0.03, 1.45],
                  [0.18, 0.00, 0.29, 0.42],
                  [0.83, 0.13, 0.41, 0.06],
                  [1.88, 1.21, 1.01, 0.27]])
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    im = ax.imshow(G, cmap="YlOrRd", vmin=0, vmax=2.0)
    ax.set_xticks(range(4), cv, fontsize=10)
    ax.set_yticks(range(4), mv, fontsize=10)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{G[i,j]:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if G[i, j] > 1.1 else INK,
                    fontweight="bold" if G[i, j] > 1.0 else "normal")
    ax.grid(False)
    cb = fig.colorbar(im, fraction=.046); cb.set_label("操作権限（制御量のσ単位）", fontsize=9, color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title("どの操作端が、どの制御量を動かせるか（実測）", fontsize=12, loc="left", pad=12)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


def fig_deadtime(out: Path):
    """Real plant dead times, from cross-correlation of the August 2020 log."""
    pairs = [("ストーカ速度 → 燃切点位置", 11.5), ("ストーカ速度 → 炉内ガス温度", 4.0),
             ("ストーカ速度 → 炉出口O2", 2.0), ("二次空気 → 主蒸気流量", 1.0)]
    labs = [p[0] for p in pairs][::-1]; vals = [p[1] for p in pairs][::-1]
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.barh(labs, vals, color="#2a78d6", height=.55)
    for i, v in enumerate(vals):
        ax.text(v + .2, i, f"{v:.0f} 分", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.axvline(16.7/60, color="#c43d28", lw=2, ls=(0, (4, 3)))
    ax.annotate("我々のシミュレータ\n(16.7秒)", (16.7/60, 3.3), fontsize=9,
                color="#c43d28", ha="left", va="top")
    ax.set_xlabel("むだ時間 [分]（相互相関のピーク遅れ・実機2020年8月ログ）", fontsize=9.5)
    ax.set_xlim(0, 13.5); ax.grid(axis="y", visible=False)
    ax.set_title("実機のむだ時間は、我々の想定より2桁大きい", fontsize=12, loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    F = Path("figs"); F.mkdir(exist_ok=True)
    if (F / "control_run.json").exists():
        print(fig_tracking(F / "control_run.json", F / "tracking.png"))
        print(fig_actions(F / "control_run.json", F / "actions.png"))
    print(fig_authority(F / "authority.png"))
    print(fig_deadtime(F / "deadtime.png"))


def fig_cost(out: Path):
    """The measurement that decided the loop was worth running."""
    labs = ["未学習\n(乱数初期値)", "5秒学習\nh=4", "40秒学習\nh=16", "150秒学習\nh=48"]
    vals = [3.4828, 0.7309, 0.9005, 0.9728]
    hold = 1.4359
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    cols = ["#8e2f33" if v > hold else "#1b6f53" for v in vals]
    ax.bar(labs, vals, color=cols, width=.58)
    ax.axhline(hold, color=INK2, lw=2, ls=(0, (5, 3)))
    ax.annotate("何もしない = 1.44", (3.52, hold), ha="right", va="bottom",
                fontsize=10, color=INK2, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + .08, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold",
                color=INK)
    ax.set_ylabel("control_cost（目標との差・低いほど良い）", fontsize=10)
    ax.grid(axis="x", visible=False); ax.margins(y=.14)
    ax.set_title("World Model を大きく・長く学習させるほど、制御は悪くなった",
                 fontsize=12.5, loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out
