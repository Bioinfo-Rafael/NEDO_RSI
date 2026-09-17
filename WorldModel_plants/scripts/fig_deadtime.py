#!/usr/bin/env python
"""Dead time, shown rather than asserted: one identification episode, the grate
speed the excitation commanded, and the unburnt solids leaving the grate. The
lag between them is the grate residence - about ten minutes - and it is the
reason a controller with no memory of what it did ten minutes ago cannot hold
the ash quality."""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wmf.control.data import load_split, CV_NAMES, MV_NAMES  # noqa: E402

INK, INK2, MUTED, ACC, DATA = "#17131c", "#5a5262", "#b8b7b0", "#c4442b", "#2a78d6"
plt.rcParams.update({"font.family": ["Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "Yu Gothic", "Helvetica Neue", "sans-serif"],
                     "axes.unicode_minus": False, "font.size": 10,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "axes.grid": True,
                     "grid.color": "#ececea", "grid.linewidth": .8, "axes.axisbelow": True})


def main(out=ROOT / "figs" / "deadtime.png", episode=0):
    d = load_split("train", ROOT / "data_ctrl_v2")
    g = json.loads((ROOT / "figs" / "pi_gains.json").read_text())
    L = g["dead_time_s"]
    y, u = d["y"][episode], d["u"][episode]
    t = np.arange(len(y))
    iu_s, iy_b = MV_NAMES.index("waste_feed"), CV_NAMES.index("unburnt_bed")
    iu_p, iy_t = MV_NAMES.index("primary_level"), CV_NAMES.index("t_exit")

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)
    for ax, (iu, iy, lab_u, lab_y, key) in zip(axes, [
            (iu_p, iy_t, "一次空気 [Nm³/s/ゾーン]", "炉出口温度 [K]", "primary_level->t_exit"),
            (iu_s, iy_b, "ごみ投入 [kg/s]", "灰の未燃分 [kg/m²/s]", "waste_feed->unburnt_bed")]):
        ax.step(t, u[:, iu], where="post", color=INK2, lw=1.4, label=lab_u)
        ax.set_ylabel(lab_u, fontsize=9.5, color=INK2)
        ax2 = ax.twinx(); ax2.grid(False)
        ax2.plot(t, y[:, iy], color=DATA, lw=1.8, label=lab_y)
        ax2.set_ylabel(lab_y, fontsize=9.5, color=DATA); ax2.tick_params(axis="y", colors=DATA)
        for s in ax2.spines.values(): s.set_visible(False)
        lag = L.get(key, None)
        if lag:
            ax.annotate(f"むだ時間 ≈ {lag} 秒" + ("（約 %.0f 分）" % (lag / 60) if lag >= 120 else ""),
                        (0.99, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                        fontsize=10.5, color=ACC, fontweight="bold")
    axes[0].set_title("速いループと遅いループ — 同じ 1 エピソードの同定データ", fontsize=12, loc="left")
    axes[-1].set_xlabel("時間 [秒]", fontsize=9.5)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(out)
    return out


if __name__ == "__main__":
    main()
