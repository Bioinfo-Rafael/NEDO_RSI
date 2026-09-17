#!/usr/bin/env python
"""Figures for the report. One message per figure, nothing decorative."""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
FIGS = ROOT / "figs"

INK, INK2, MUTED = "#17131c", "#5a5262", "#b8b7b0"
HOLD, PI, MPC = "#b8b7b0", "#eb6834", "#2a78d6"
OK, BAD = "#1b6f53", "#8e2f33"
plt.rcParams.update({"font.family": ["Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "Yu Gothic", "Helvetica Neue", "sans-serif"],
    "axes.unicode_minus": False, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.grid": True, "grid.color": "#ececea",
    "grid.linewidth": .8, "axes.axisbelow": True})

LAB = {"hold": "制御なし", "pi": "PI制御（3ループ）", "mpc": "MPC（World Model 内で計画）"}
COL = {"hold": HOLD, "pi": PI, "mpc": MPC}


# ---------------------------------------------------------------- 1. 制約
def fig_constraints(out: Path):
    """Where PID actually loses: it asks for moves the hardware cannot make."""
    runs = json.loads((FIGS / "control_run.json").read_text())
    from wmf.control.mpc import MPCConfig
    dumax = np.asarray(MPCConfig().du_max)
    names = ["火格子速度", "ごみ供給", "一次空気", "二次空気"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.3),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    for m in ("pi", "mpc"):
        U = np.array([s["u"] for s in runs[m]["log"]])
        t = np.array([s["t"] for s in runs[m]["log"]])[1:]
        du = np.abs(np.diff(U, axis=0)) / dumax
        a1.plot(t, du.max(axis=1), color=COL[m], lw=1.9, label=LAB[m])
    a1.axhline(1.0, color=BAD, lw=2.2, ls=(0, (5, 3)))
    a1.text(.995, 1.06, "設備が追従できる限界", transform=a1.get_yaxis_transform(),
            ha="right", fontsize=9.5, color=BAD, fontweight="bold")
    a1.set_xlabel("時間 [秒]", fontsize=9.5)
    a1.set_ylabel("要求した操作変化 ÷ 設備の限界", fontsize=9.5)
    a1.legend(frameon=False, fontsize=9.5, loc="upper right")
    a1.set_title("要求した操作変化と、設備が追従できる限界", fontsize=12, loc="left")

    viol = {}
    for m in ("pi", "mpc"):
        U = np.array([s["u"] for s in runs[m]["log"]])
        du = np.abs(np.diff(U, axis=0)) / dumax
        viol[m] = (du > 1.0001).sum(axis=0)
    x = np.arange(4); w = .36
    a2.bar(x - w/2, viol["pi"], w, color=PI, label="PI制御")
    a2.bar(x + w/2, viol["mpc"], w, color=MPC, label="MPC")
    for xi, v in zip(x - w/2, viol["pi"]):
        if v: a2.text(xi, v + .8, str(int(v)), ha="center", fontsize=10, fontweight="bold", color=PI)
    for xi, v in zip(x + w/2, viol["mpc"]):
        a2.text(xi, v + .8, str(int(v)), ha="center", fontsize=10, fontweight="bold", color=MPC)
    a2.set_xticks(x, names, fontsize=9)
    a2.set_ylabel("制約を超えた回数", fontsize=9.5)
    a2.grid(axis="x", visible=False); a2.legend(frameon=False, fontsize=9.5)
    a2.set_title(f"合計 {int(viol['pi'].sum())} 回 対 {int(viol['mpc'].sum())} 回", fontsize=12, loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ---------------------------------------------------------------- 2. 追従
def fig_tracking(out: Path):
    runs = json.loads((FIGS / "control_run.json").read_text())
    tgt = np.asarray(runs["mpc"]["target"])
    rows = [(0, "炉出口温度 [K]", tgt[0]), (1, "炉出口 O2 [-]", tgt[1]),
            (3, "灰の未燃分 [kg/m2/s]", tgt[3])]
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 7.0), sharex=True)
    for ax, (ch, lab, r) in zip(axes, rows):
        ax.axhline(r, color=BAD, lw=2, ls=(0, (5, 3)))
        ax.text(.995, r, f" 目標 {r:g}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9, color=BAD, fontweight="bold")
        for m, run in runs.items():
            y = np.array([s["y"] for s in run["log"]])
            t = np.array([s["t"] for s in run["log"]])
            ax.plot(t, y[:, ch], color=COL[m], lw=2, label=LAB[m])
        ax.set_ylabel(lab, fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=9.5, ncol=3, loc="upper left")
    axes[-1].set_xlabel("時間 [秒]", fontsize=9.5)
    fig.suptitle("追従性能だけを見ると、差は大きくない（MPC は PI より 11% 良いだけ）",
                 fontsize=12, y=.985)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ---------------------------------------------------------------- 3. 二つのループ
def fig_two_loops(out: Path):
    h = json.loads((FIGS / "codex_history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    for ax, key, ylab in ((axes[0], "prediction", "予測誤差 val_nrmse"),
                          (axes[1], "control", "制御コスト control_cost")):
        d = h[key]; it = d["items"]
        v = [x["metric"] for x in it]; st = [x["status"] for x in it]
        x = np.arange(1, len(v) + 1)
        ax.plot(x, np.minimum.accumulate(v), color=OK, lw=2.3, zorder=2)
        for xi, vi, s in zip(x, v, st):
            ax.scatter(xi, vi, s=110, zorder=3, edgecolor="white", linewidth=1.5,
                       color=OK if s == "keep" else MUTED)
        ax.set_xticks(x); ax.set_xlabel("Codex が自分で行った実験", fontsize=9.5)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.margins(y=.42)          # headroom so the annotations clear the title
        imp = (v[0] - min(v)) / v[0] * 100
        ax.set_title(f"{d['label']}   改善 {imp:.0f}%", fontsize=11.5, loc="left", pad=14)
    # highlight the discarded geometry experiment
    it = h["prediction"]["items"][1]
    axes[0].annotate("設備の形状を入力に追加\n→ 棄却された", (2, it["metric"]),
                     xytext=(2.6, it["metric"] - .002), fontsize=9.5, color=BAD,
                     fontweight="bold", ha="left", va="top",
                     arrowprops=dict(arrowstyle="->", color=BAD, lw=1.6))
    it2 = h["control"]["items"][3]
    axes[1].annotate("Codex が我々のバグを修正", (4, it2["metric"]),
                     xytext=(2.2, it2["metric"] - .045), fontsize=9.5, color=MPC,
                     fontweight="bold", ha="left",
                     arrowprops=dict(arrowstyle="->", color=MPC, lw=1.6))
    fig.suptitle("同じ仕組みでも、何を採点するかで結果が変わる", fontsize=13, y=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ---------------------------------------------------------------- 4. 予測≠制御
def fig_pred_vs_control(out: Path):
    labs = ["未学習\n(乱数)", "5秒学習\nh=4", "40秒学習\nh=16", "150秒学習\nh=48"]
    vals = [3.4828, 0.7309, 0.9005, 0.9728]
    hold = 1.4359
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(labs, vals, color=[BAD if v > hold else OK for v in vals], width=.56)
    ax.axhline(hold, color=INK2, lw=2, ls=(0, (5, 3)))
    ax.annotate("制御しない場合 = 1.44", (3.5, hold), ha="right", va="bottom",
                fontsize=10, color=INK2, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + .08, f"{v:.2f}", ha="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.annotate("", xy=(3.30, 1.05), xytext=(1.10, 1.05),
                arrowprops=dict(arrowstyle="->", color=BAD, lw=2.2))
    ax.text(2.2, 1.13, "モデルを大きく・長く学習させるほど悪化", ha="center",
            fontsize=10.5, color=BAD, fontweight="bold")
    ax.set_ylabel("制御コスト（低いほど良い）", fontsize=10)
    ax.grid(axis="x", visible=False); ax.margins(y=.16)
    ax.set_title("予測精度を上げても、制御は良くならなかった", fontsize=12.5, loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    for f in (fig_constraints(FIGS / "constraints.png"),
              fig_tracking(FIGS / "tracking.png"),
              fig_two_loops(FIGS / "two_loops.png"),
              fig_pred_vs_control(FIGS / "pred_vs_control.png")):
        print(f)
