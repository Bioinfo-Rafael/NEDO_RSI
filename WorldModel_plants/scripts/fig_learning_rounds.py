#!/usr/bin/env python
"""Round 0, 1, 2: the world model learns from its own interaction with the plant,
and the plant tracks the target better each time.

    round 0   M trained on the identification data only
    round k   collect INTERACT_STEPS/rounds steps of closed-loop experience with
              C inside the current M (setpoints wandering, some exploration),
              append them to the training set, retrain M on the same budget
    after every round: the scored setpoint step, run to the end, recorded

Same controller, same plant, same target, same training budget - only what M
has experienced changes. Uses whatever worldmodel.py is in control_research/,
so it works for the baseline and for anything Codex leaves behind.
"""
from __future__ import annotations

import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "control_research"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import prepare  # noqa: E402
import worldmodel as wm  # noqa: E402
from interact import BudgetedPlant, collect, append  # noqa: E402
from evaluate import run_scenario, SCENARIOS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=prepare.TRAIN_SECONDS)
    ap.add_argument("--steps-per-round", type=int, default=None,
                    help="plant steps of experience per round (default: INTERACT_STEPS / rounds)")
    ap.add_argument("--explore", type=float, default=0.5)
    ap.add_argument("--scenario", default="temp_step")
    ap.add_argument("--out", default="figs/learning_rounds.json")
    a = ap.parse_args()
    per_round = a.steps_per_round or prepare.INTERACT_STEPS // max(a.rounds, 1)
    sc_def = next(s for s in SCENARIOS if s["name"] == a.scenario)

    data = prepare.load_split("train")
    stats = prepare.norm_stats(data)          # fixed: the same scaler for every round
    plant = BudgetedPlant(10 ** 9)            # metering is the loop's job here, not the plant's
    out = {"rounds": [], "scenario": sc_def, "steps_per_round": per_round}

    for r in range(a.rounds + 1):
        t0 = time.time()
        params = wm.train(data, stats, a.seconds, seed=r)      # plant not handed in: experience comes from below
        train_s = time.time() - t0
        ys, us, rs = run_scenario(params, stats, wm, sc_def, wm.WARMUP)
        sd = np.asarray(stats["y_std"])
        m = np.ones(len(ys), bool); m[:40] = False; m[sc_def["t_step"]:sc_def["t_step"] + 40] = False
        iae = (np.abs(ys - rs)[m] / sd).mean(axis=0)
        score = float(iae[0] + iae[1] + 0.7 * iae[3])
        out["rounds"].append({"round": r, "episodes": int(data["y"].shape[0]), "train_seconds": train_s,
                              "iae_sigma": iae.tolist(), "score": score,
                              "y": ys.tolist(), "u": us.tolist(), "r": rs.tolist()})
        print(f"round {r}: trained on {data['y'].shape[0]} episodes in {train_s:.0f}s  "
              f"tracking T {iae[0]:.2f}σ  O2 {iae[1]:.2f}σ  burnout {iae[3]:.2f}σ  score {score:.3f}", flush=True)
        if r == a.rounds:
            break
        # experience: C inside this M drives the plant, and M gets to see the result
        n_ep, got = 0, 0
        while got < per_round:
            y, u = collect(params, stats, wm, plant, steps=min(400, per_round - got),
                           explore=a.explore, seed=100 * r + n_ep)
            data = append(data, y, u); got += len(y); n_ep += 1
        print(f"         +{got} steps of closed-loop experience in {n_ep} episodes", flush=True)

    Path(a.out).write_text(json.dumps(out))
    print(f"-> {a.out}")

    # ---- figure ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED = "#17131c", "#5a5262", "#b8b7b0"
    plt.rcParams.update({"font.family": ["Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "sans-serif"],
                         "axes.unicode_minus": False, "font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "axes.grid": True, "grid.color": "#ececea",
                         "grid.linewidth": .8, "axes.axisbelow": True})
    cols = ["#b8b7b0", "#6fa3e8", "#1f4fa3", "#0b2a5e"]
    rows = [(0, "炉出口温度 [K]"), (1, "炉出口 O2 [-]"), (3, "灰の未燃分 [kg/m²/s]")]
    fig, axes = plt.subplots(len(rows), 1, figsize=(9.2, 7.6), sharex=True)
    R = np.asarray(out["rounds"][0]["r"]); t = np.arange(len(R))
    for ax, (ch, lab) in zip(axes, rows):
        ax.step(t, R[:, ch], where="post", color="#c43d28", lw=2, ls=(0, (5, 3)), zorder=1)
        for k, rd in enumerate(out["rounds"]):
            Y = np.asarray(rd["y"])
            ax.plot(t, Y[:, ch], color=cols[min(k, len(cols) - 1)], lw=2 if k else 1.6,
                    label=f"学習 {k} 回目（追従誤差 {rd['score']:.2f}）", zorder=2 + k)
        ax.set_ylabel(lab, fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", ncol=min(3, len(out["rounds"])))
    axes[0].annotate("目標", (0.995, R[-1, 0]), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                     fontsize=9, color="#c43d28", fontweight="bold")
    axes[-1].set_xlabel("時間 [秒]", fontsize=9.5)
    fig.suptitle("World Model が自分の行動の結果から学ぶたびに、目標への追従が変わる  "
                 f"（毎回 +{per_round} 秒の閉ループ経験、同じ学習予算・同じ制御器）", fontsize=11.5, y=0.985)
    fig.tight_layout()
    png = Path(a.out).with_suffix(".png"); fig.savefig(png, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"-> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
