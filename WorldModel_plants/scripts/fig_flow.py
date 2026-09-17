#!/usr/bin/env python
"""The pipeline as one picture.

Two loops, drawn so they cannot be confused: the fast control loop runs along
the top and returns underneath it; the slow research loop hangs below and only
touches M. Keeping the return arrow clear of the Codex box is the whole reason
for this layout.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, INK2 = "#17131c", "#5a5262"
SIM, WM, CT, AG = "#c43d28", "#2a78d6", "#1b6f53", "#7b2258"
plt.rcParams.update({"font.family": ["Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "Yu Gothic", "Helvetica Neue", "sans-serif"],
                     "axes.unicode_minus": False})


def box(ax, x, y, w, h, n, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
                                lw=2.2, ec=color, fc="white", zorder=3))
    ax.text(x + w/2, y + h - .048, f"{n}  {title}", ha="center", va="top",
            fontsize=12.5, fontweight="bold", color=color, zorder=4)
    for i, ln in enumerate(lines):
        ax.text(x + w/2, y + h - .105 - i*.044, ln, ha="center", va="top",
                fontsize=9.2, color=INK2, zorder=4)


def arrow(ax, p0, p1, color, rad=0.0, lw=2.2):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=18, lw=lw,
                                 color=color, zorder=2))


def tag(ax, x, y, text, color, fs=9.8):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))


def _speed():
    """Measured timings (scripts/measure_speed.py); placeholders if not yet run."""
    import json
    f = Path(__file__).resolve().parent.parent / "figs" / "speed.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    return {"plant": f"{d['plant_step_ms']:.0f} ms / 制御ステップ" if "plant_step_ms" in d else "実測待ち",
            "mpc": f"{d['mpc_act_ms']:.0f} ms / 制御ステップ" if "mpc_act_ms" in d else "実測待ち",
            "futures": f"M の中で {d['futures_per_decision']/1000:.0f} 千本の未来を試す"
                       if "futures_per_decision" in d else "M の中で多数の未来を試す",
            "horizon": f"出力: {d.get('horizon_s', 30):.0f} 秒先までの4計測"}


def main(out: Path):
    sp = _speed()
    fig, ax = plt.subplots(figsize=(12.6, 7.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(.5, .975, "M は「何が起こるか」、C は「何をすべきか」、Codex は「M をどう作るか」",
            ha="center", fontsize=14, color=INK, fontweight="bold")

    TOP, H = .615, .275
    box(ax, .035, TOP, .28, H, "①", "Simulator（プラント）",
        ["2D 気相 + 1D ストーカ bed", "dt = 3 ms × 333 = 制御周期 1 s",
         "出力: 炉内場・出口4計測", sp["plant"]], SIM)
    box(ax, .360, TOP, .28, H, "②", "World Model  M",
        ["入力: 出口4計測 + 操作 の履歴", sp["horizon"],
         "構造は Codex が決める（初期: 線形状態空間）", "見えない bed をどう持つかが課題"], WM)
    box(ax, .685, TOP, .28, H, "③", "Controller  C",
        [sp["futures"], "CEM-MPC・後退地平",
         "悲観評価 + 信頼領域 + レート制約", sp["mpc"]], CT)

    # fast loop: along the top, returning just below the boxes
    arrow(ax, (.315, TOP + H/2), (.360, TOP + H/2), SIM)
    tag(ax, .3375, TOP + H/2 + .055, "同定データ", SIM, 9.2)
    arrow(ax, (.640, TOP + H/2), (.685, TOP + H/2), WM)
    tag(ax, .6625, TOP + H/2 + .055, "未来予測", WM, 9.2)

    RET = .545
    arrow(ax, (.825, TOP), (.825, RET), CT)
    arrow(ax, (.825, RET), (.175, RET), CT)
    arrow(ax, (.175, RET), (.175, TOP), CT)
    tag(ax, .50, RET, "操作 u  ← 計画した中の最初の 1 手だけを適用", CT, 10.2)

    # slow loop: hangs below, touching only M
    BOT, BH = .085, .245
    box(ax, .360, BOT, .28, BH, "④", "Codex（上位エージェント）",
        ["M の構造そのものを書き換える", "control_cost で採否を判断",
         "1 実験 ≈ 8 分 / C でも M でもない"], AG)

    arrow(ax, (.150, TOP - .01), (.360, BOT + BH*.62), SIM, rad=-.16)
    tag(ax, .225, .40, "計測 y と\n目標との差", SIM, 9.4)
    arrow(ax, (.640, BOT + BH*.62), (.850, TOP - .01), AG, rad=-.16)
    tag(ax, .782, .40, "M を改訂", AG, 9.4)
    arrow(ax, (.500, BOT + BH), (.500, TOP), AG, lw=2.6)
    tag(ax, .500, .47, "control_cost", AG, 10)

    # time scales, two rows so nothing collides
    ax.text(.035, .045, "時間スケール", fontsize=10.5, color=INK, fontweight="bold")
    items = [("C  操作を決める", "1 秒", CT), ("M  再学習", "数分", WM),
             ("Codex  構造を改訂", "数十分", AG), ("人間  目的と安全を決める", "節目ごと", INK2)]
    for i, (lab, scale, col) in enumerate(items):
        x = .175 + i*.205
        ax.text(x, .045, f"● {lab}", fontsize=9.6, color=col, fontweight="bold")
        ax.text(x + .012, .012, scale, fontsize=9.2, color=INK2)

    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main(Path("figs/flow.png")))
