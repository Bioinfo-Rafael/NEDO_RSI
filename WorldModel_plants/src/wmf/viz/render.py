"""Render an episode: contact sheet (always) and an animated GIF (no ffmpeg).

    python -m wmf.viz.render data/train/plant_a_00000.npz --out out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

CHANNELS = {"T": (0, "inferno", "Temperature [K]"),
            "Y_F": (4, "viridis", "Fuel mass fraction"),
            "Y_O2": (5, "cividis", "O2 mass fraction")}


def _panel(ax, field, mask, cmap, vmin, vmax, title):
    shown = np.where(mask > 0, field, np.nan)
    im = ax.imshow(shown, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_facecolor("0.85")            # walls read as grey, not as cold gas
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def contact_sheet(ep, out: Path, n: int = 6):
    q, g = ep["q"].astype(np.float32), ep["g"]
    mask = g[..., 0]
    idx = np.linspace(0, q.shape[0] - 1, n).astype(int)
    fig, axes = plt.subplots(len(CHANNELS), n, figsize=(2.0 * n, 2.2 * len(CHANNELS)))
    axes = np.atleast_2d(axes)
    for r, (name, (ch, cmap, label)) in enumerate(CHANNELS.items()):
        vmin, vmax = np.nanpercentile(q[..., ch][:, mask > 0], [1, 99])
        for c, t in enumerate(idx):
            im = _panel(axes[r, c], q[t, ..., ch], mask, cmap, vmin, vmax,
                        f"{name}  t={t}" if r == 0 else f"t={t}")
        fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.02, label=label)
    meta = ep["meta"]
    fig.suptitle(f"{meta['plant_id']}  seed={meta['seed']}  "
                 f"dx={meta['dx']:.3f} m  dt={meta['dt']} s", fontsize=10)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def animate(ep, out: Path, channel: str = "T", fps: int = 12):
    ch, cmap, label = CHANNELS[channel]
    q, mask = ep["q"].astype(np.float32), ep["g"][..., 0]
    bed = ep["bed"]
    vmin, vmax = np.nanpercentile(q[..., ch][:, mask > 0], [1, 99])

    fig, (ax, axb) = plt.subplots(2, 1, figsize=(5.5, 5.4),
                                  gridspec_kw={"height_ratios": [4, 1]})
    im = _panel(ax, q[0, ..., ch], mask, cmap, vmin, vmax, "")
    fig.colorbar(im, ax=ax, fraction=0.035, label=label)
    lines = [axb.plot(bed[0, :, k], label=n)[0]
             for k, n in enumerate(("moisture", "volatile", "char"))]
    axb.set_ylim(0, max(bed.max() * 1.1, 1e-6)); axb.set_xlim(0, bed.shape[1] - 1)
    axb.set_xlabel("grate position"); axb.legend(fontsize=7, ncol=3)

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(out), dpi=90):
        for t in range(q.shape[0]):
            im.set_data(np.where(mask > 0, q[t, ..., ch], np.nan))
            ax.set_title(f"{ep['meta']['plant_id']}  {channel}  "
                         f"t = {t * ep['meta']['control_interval']:.1f} s", fontsize=10)
            for k, ln in enumerate(lines):
                ln.set_ydata(bed[t, :, k])
            writer.grab_frame()
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode")
    ap.add_argument("--out", default="out")
    ap.add_argument("--channel", default="T", choices=list(CHANNELS))
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args(argv)

    from ..sim.episode import load_episode
    ep = load_episode(args.episode)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.episode).stem

    print(contact_sheet(ep, out / f"{stem}_sheet.png"))
    if not args.no_gif:
        print(animate(ep, out / f"{stem}_{args.channel}.gif", args.channel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
