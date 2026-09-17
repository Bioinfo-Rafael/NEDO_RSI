#!/usr/bin/env python
"""Figures for the demo. Written for readers who do not read nrmse.

Everything is stated in units a reader already has intuition for: degrees for
temperature error, milliseconds for speed, per cent for improvement.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, MUTED = "#17131c", "#5a5262", "#b8b7b0"
TRUTH, PRED, ACC = "#2a78d6", "#c43d28", "#1b6f53"
# Hiragino Sans is present on macOS; without it every Japanese label renders as
# a box, which would make these figures worse than useless for their audience.
plt.rcParams.update({
    "font.family": ["Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "Yu Gothic", "Helvetica Neue", "sans-serif"],
    "axes.unicode_minus": False,
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": .8,
    "axes.axisbelow": True})


def fig_prediction(npz: Path, out: Path, title: str, steps=(0, 5, 15, 30, 40)):
    z = np.load(npz)
    pred, truth, g = z["pred"], z["truth"], z["g"]
    m = g[..., 0] > 0
    steps = [s for s in steps if s < len(pred)]
    vmin, vmax = np.percentile(truth[..., 0][:, m], [2, 98])

    fig, axes = plt.subplots(3, len(steps), figsize=(2.15 * len(steps), 6.4))
    for c, s in enumerate(steps):
        for r, (field, lab, cmap, vr) in enumerate([
                (truth[s, ..., 0], "シミュレータ（正解）", "inferno", (vmin, vmax)),
                (pred[s, ..., 0], "AIの予測", "inferno", (vmin, vmax)),
                (np.abs(pred[s, ..., 0] - truth[s, ..., 0]), "差", "Reds", (0, 200))]):
            ax = axes[r, c]
            im = ax.imshow(np.where(m, field, np.nan), origin="lower", cmap=cmap,
                           vmin=vr[0], vmax=vr[1], interpolation="nearest")
            ax.set_facecolor("#dbdbd9"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("開始時点" if s == 0 else f"{s}ステップ先", fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(lab, fontsize=10.5, color=INK)
            if c == len(steps) - 1:
                cb = fig.colorbar(im, ax=ax, fraction=.046)
                cb.ax.tick_params(labelsize=7.5, colors=INK2)
                cb.set_label("K" if r < 2 else "誤差 K", fontsize=8, color=INK2)
    fig.suptitle(title, fontsize=13, y=.98)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def fig_error_growth(npzs: dict, out: Path):
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for (label, path), color in zip(npzs.items(), (TRUTH, PRED)):
        z = np.load(path)
        m = z["g"][..., 0] > 0
        err = np.array([np.abs(z["pred"][k, ..., 0] - z["truth"][k, ..., 0])[m].mean()
                        for k in range(len(z["pred"]))])
        t = np.arange(len(err)) * 0.1
        ax.plot(t, err, color=color, lw=2.2, label=label)
        ax.annotate(f"{err[-1]:.0f} K", (t[-1], err[-1]), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=10,
                    fontweight="bold", va="center")
    ax.set_xlabel("予測した先の時間 [秒]（AIは自分の出力を入力に戻し続ける）", fontsize=9.5)
    ax.set_ylabel("炉内温度の平均誤差 [K]", fontsize=9.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title("自分の予測を入力に戻し続けても、誤差がどこまで育つか", fontsize=11.5, loc="left")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# The agent writes its descriptions in English; this page is read in Japanese.
# Matched on a substring so a reworded description still finds its label.
JA = [("unmodified baseline", "ベースライン"),
      ("geometry", "設備の形状を入力に追加"),
      ("dilate", "受容野を拡大"),
      ("two-step", "2ステップ先まで学習"),
      ("cosine", "学習率を徐々に下げる"),
      ("width", "モデルを大きく"),
      ("depth", "層を深く")]


def _ja(desc: str) -> str:
    low = desc.lower()
    for key, lab in JA:
        if key in low:
            return lab
    return desc.split(": ")[-1][:20]


def fig_loop(tsv: Path, out: Path):
    rows = [l.rstrip("\n").split("\t") for l in tsv.read_text().splitlines()[1:] if l.strip()]
    if not rows: return None
    v = [float(r[1]) for r in rows]
    status = [r[3] for r in rows]
    desc = [r[4] for r in rows]
    x = np.arange(1, len(v) + 1)
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    best = np.minimum.accumulate(v)
    ax.plot(x, best, color=ACC, lw=2.4, zorder=2, label="採用された最良値")
    for xi, vi, st in zip(x, v, status):
        ax.scatter(xi, vi, s=90, zorder=3, edgecolor="white", linewidth=1.4,
                   color=ACC if st == "keep" else MUTED)
    for xi, vi, dsc, st in zip(x, v, desc, status):
        ax.annotate(_ja(dsc), (xi, vi), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=9, color=INK if st == "keep" else INK2,
                    fontweight="bold" if st == "keep" else "normal")
    ax.set_xticks(x); ax.set_xlabel("AIが自分で行った実験", fontsize=9.5)
    ax.set_ylabel("予測誤差（低いほど良い）", fontsize=9.5)
    h = [plt.Line2D([0],[0],marker="o",ls="",mfc=ACC,mec="w",ms=9),
         plt.Line2D([0],[0],marker="o",ls="",mfc=MUTED,mec="w",ms=9)]
    ax.legend(h, ["採用", "棄却"], frameon=False, fontsize=9, loc="upper right")
    ax.set_title("AIが自分でモデルを書き換え、良くなった案だけ残した記録", fontsize=11.5, loc="left")
    ax.margins(x=.10, y=.30)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


def fig_ab(js: Path, out: Path):
    d = json.loads(js.read_text())
    labels = ["学習した設備\n(plant A)", "未知の設備\n(plant B)", "未知の設備\n(plant C)"]
    keys = ["val_iid", "val_b", "val_c"]
    a = [d["no_geometry"][k] for k in keys]
    b = [d["with_geometry"][k] for k in keys]
    x = np.arange(3); w = .36
    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    ax.bar(x - w/2, a, w, color=MUTED, label="設備の形状を見せない")
    ax.bar(x + w/2, b, w, color=ACC, label="設備の形状を条件として与える")
    # Neutral emphasis: two of the three bars get worse with geometry, so
    # bolding that series would be the figure arguing for a conclusion the
    # data does not support.
    for xi, v in zip(x - w/2, a): ax.text(xi, v + .012, f"{v:.2f}", ha="center", fontsize=9.5, color=INK2)
    for xi, v in zip(x + w/2, b): ax.text(xi, v + .012, f"{v:.2f}", ha="center", fontsize=9.5, color=INK2)
    for xi, va, vb in zip(x, a, b):
        better = vb < va
        ax.text(xi, max(va, vb) + .055, ("改善 " if better else "悪化 ") + f"{abs(1-vb/va)*100:.0f}%",
                ha="center", fontsize=9, color=(ACC if better else "#8e2f33"), fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("予測誤差（低いほど良い）", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9.5)
    ax.margins(y=.18)
    ax.set_title("設備の図面を与えると、見たことのない炉でどうなるか", fontsize=11.5, loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--figs", default="figs")
    a = ap.parse_args(); F = Path(a.figs)
    if (F/"rollout_val_iid.npz").exists():
        print(fig_prediction(F/"rollout_val_iid.npz", F/"predict_a.png",
                             "学習した設備での予測 — 上が正解、中がAI、下が差"))
    if (F/"rollout_val_b.npz").exists():
        print(fig_prediction(F/"rollout_val_b.npz", F/"predict_b.png",
                             "一度も学習していない設備での予測"))
        print(fig_error_growth({"学習した設備": F/"rollout_val_iid.npz",
                                "未知の設備": F/"rollout_val_b.npz"}, F/"error_growth.png"))
    if Path("integration/results.tsv").exists():
        print(fig_loop(Path("integration/results.tsv"), F/"loop.png"))
    if (F/"ab_geometry.json").exists():
        print(fig_ab(F/"ab_geometry.json", F/"ab.png"))
