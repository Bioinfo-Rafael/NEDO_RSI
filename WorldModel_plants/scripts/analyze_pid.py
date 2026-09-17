#!/usr/bin/env python
"""Why a well-tuned PID loop is not enough on this plant.

Not an argument from authority - three measurements:

1. The relative gain array. Bristol's RGA says which single-input-single-output
   pairings can be closed independently. Entries near 1 are clean; entries far
   from 1, and especially negative ones, mean the loops fight each other and no
   amount of tuning fixes it because the coupling is in the plant.

2. The dead time against the closed-loop bandwidth a PID can achieve. A PI loop
   on a process with dead time L is limited to roughly 1/L in bandwidth; beyond
   that it is unstable regardless of gains. MPC can plan through dead time
   because it predicts.

3. Constraints. PID has no representation of "the grate cannot change speed
   faster than this" or "keep the furnace above 850 degC" - it can only be
   clamped after the fact, which is not the same as planning within limits.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wmf.control.data import load_split, CV_NAMES, MV_NAMES


def steady_gain_matrix(d):
    """K[i,j] = d(CV_j)/d(MV_i), from the identification data."""
    y = d["y"].reshape(-1, d["y"].shape[-1])
    u = d["u"].reshape(-1, d["u"].shape[-1])
    un = (u - u.mean(0)) / u.std(0)
    A = np.c_[un, np.ones(len(un))]
    K = np.zeros((u.shape[1], y.shape[1]))
    for j in range(y.shape[1]):
        beta = np.linalg.lstsq(A, y[:, j], rcond=None)[0][:u.shape[1]]
        K[:, j] = beta / u.std(0) * u.std(0)      # keep in normalised-input units
    return K, y.std(0), u.std(0)


def rga(K):
    """Bristol RGA on the square sub-system. K is [n_mv, n_cv]."""
    Ki = np.linalg.pinv(K)
    return K * Ki.T


def main() -> int:
    d = load_split("train", ROOT / "data_ctrl_v2")
    K, y_sd, u_sd = steady_gain_matrix(d)

    # square it up on the two regulated variables and their two best handles
    cv_pick = [0, 1, 3]                       # t_exit, o2_exit, unburnt_bed
    mv_pick = [2, 3, 0]                       # primary, secondary, stoker
    # Scale to sigma units before anything that is not scale-invariant.
    # RGA is scale-invariant; the condition number is not, and computing it on
    # a matrix mixing kelvin with mass fractions gives a meaningless number.
    Ks_raw = K[np.ix_(mv_pick, cv_pick)]
    Ks = Ks_raw / y_sd[cv_pick][None, :]
    L = rga(Ks)

    cvn = [CV_NAMES[i] for i in cv_pick]
    mvn = [MV_NAMES[i] for i in mv_pick]

    print("=== 定常ゲイン（制御量のσ単位）===")
    print(f"{'':18s}" + "".join(f"{c:>14s}" for c in cvn))
    for i, m in enumerate(mvn):
        print(f"{m:18s}" + "".join(f"{Ks[i,j]:14.3f}" for j in range(3)))

    print("\n=== 相対ゲイン行列 RGA ===")
    print(f"{'':18s}" + "".join(f"{c:>14s}" for c in cvn))
    for i, m in enumerate(mvn):
        print(f"{m:18s}" + "".join(f"{L[i,j]:14.3f}" for j in range(3)))

    diag = np.array([L[i, i] for i in range(3)])
    worst = float(np.max(np.abs(L - np.eye(3))))
    neg = int((L < 0).sum())
    print(f"\n  対角成分: {np.round(diag,2)}   （1 に近いほど独立に制御できる）")
    print(f"  単位行列からの最大乖離: {worst:.2f}")
    print(f"  負の成分の数: {neg}   （負があると、そのペアは単独ループで不安定化しうる）")
    cond = float(np.linalg.cond(Ks))
    sv = np.linalg.svd(Ks, compute_uv=False)
    print(f"  条件数（σ単位で正規化後）: {cond:.1f}")
    print(f"  特異値: {np.round(sv,4)}   最小/最大 = {sv.min()/sv.max():.4f}")
    print(f"  → 最弱方向は最強方向の {sv.max()/sv.min():.0f} 倍の操作量を要する")

    out = {"cv": cvn, "mv": mvn, "gain": Ks.tolist(), "rga": L.tolist(),
           "diag": diag.tolist(), "worst_dev": worst, "n_negative": neg,
           "cond": cond, "svals": sv.tolist()}
    (ROOT / "figs" / "pid_analysis.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> figs/pid_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
