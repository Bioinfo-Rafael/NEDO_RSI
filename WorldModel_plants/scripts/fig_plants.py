#!/usr/bin/env python
"""The plant population: one procedurally generated furnace per family, drawn
from its rasterised geometry, with the three hand-written plants for scale.
No simulation - this is what the world model is asked to generalise across."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wmf.plants.encode import build_geometry  # noqa: E402
from wmf.plants.generator import generate  # noqa: E402
from wmf.plants.schema import PlantConfig  # noqa: E402
from wmf.actions import envelope  # noqa: E402

INK, INK2, MUTED = "#17131c", "#5a5262", "#b8b7b0"
plt.rcParams.update({"font.family": ["Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "Yu Gothic", "Helvetica Neue", "sans-serif"],
                     "axes.unicode_minus": False, "font.size": 9, "text.color": INK,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def draw(ax, cfg, title):
    g = build_geometry(cfg)
    img = np.full(g.mask.shape + (3,), 0.93)
    img[g.mask > 0] = (1.0, 1.0, 1.0)
    img[g.mask == 0] = (0.55, 0.55, 0.58)
    lay = g.layout
    img[(lay > 0) & (lay <= 0.5)] = (0.93, 0.55, 0.30)      # primary zones (grate)
    img[lay > 0.5] = (0.25, 0.50, 0.85)                       # secondary nozzles
    for i in g.flue_cols:
        img[-2, i] = (0.85, 0.30, 0.25)
    ax.imshow(img, origin="lower", extent=(0, cfg.furnace.width, 0, cfg.furnace.height),
              interpolation="nearest", aspect="equal")
    e = envelope(cfg)
    ax.set_title(title, fontsize=8.5, loc="left", pad=3)
    ax.text(0.02, 0.97, f"{cfg.furnace.width:.0f}×{cfg.furnace.height:.0f} m  "
            f"{cfg.n_zones}z/{cfg.n_nozzles}n  LHV {cfg.waste.lhv:.1f}\n"
            f"投入 {e['waste_feed'][0]:.2f}–{e['waste_feed'][1]:.2f} kg/s",
            transform=ax.transAxes, va="top", fontsize=7, color=INK2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def main(out=ROOT / "figs" / "plants.png"):
    fams = generate(1, seed=0)
    hand = [("plant_a（基準・学習）", PlantConfig.from_yaml(ROOT / "configs/plants/plant_a.yaml")),
            ("plant_b（長い・湿った）", PlantConfig.from_yaml(ROOT / "configs/plants/plant_b.yaml")),
            ("plant_c（小型・急傾斜）", PlantConfig.from_yaml(ROOT / "configs/plants/plant_c.yaml"))]
    items = hand + [(f"族 {k}", v[0]) for k, v in fams.items()]
    n = len(items); cols = 5; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 2.5 * rows))
    for ax, (t, c) in zip(axes.ravel(), items):
        draw(ax, c, t)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("World Model が賄うべきプラントの母集団 — 3 つの手書き + 12 族から 1 つずつ  "
                 "（橙: 一次空気ゾーン、青: 二次ノズル、赤: 煙道）", fontsize=10.5, y=0.995)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(out, f"{n} plants")
    return out


if __name__ == "__main__":
    main()
