#!/usr/bin/env python
"""Identify the PI baseline and the input-authority table from the dataset.

The PI gains used to be hard-coded from an earlier dataset. When the plant's
time scales changed by a factor of forty they went stale silently, and the
baseline was being compared against MPC with gains for a different machine.
Everything here is derived from data_ctrl_v2/ and written to figs/, and both
run_control.py and program.md read it from there.

    steady gains   K[i,j] = dCV_j / dMV_i        least squares on the whole set
    dead time      L      first lag at which the cross-correlation of dMV
                          with dCV reaches half its peak (per pairing)
    SIMC           Kc = 0.5 / K,  Ti = min(tau, 4 L)  with tau ~ 4 L here
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wmf.control.data import load_split, CV_NAMES, MV_NAMES  # noqa: E402
from wmf.actions import envelope  # noqa: E402
from wmf.plants.schema import PlantConfig  # noqa: E402

# The grate loop is fixed; the two air loops are paired by the relative gain
# array measured on the data, because on this plant the secondary air has more
# authority over the exit temperature than the primary air does, and the
# obvious pairing (primary -> T, secondary -> O2) is the wrong one.
GRATE_PAIR = ("stoker_speed", "unburnt_bed")


def air_pairing(K):
    """RGA on the 2x2 air sub-system; return the pairing whose diagonal is
    closer to 1, and the RGA element it rests on."""
    i_p, i_s = MV_NAMES.index("primary_level"), MV_NAMES.index("secondary_level")
    j_t, j_o = CV_NAMES.index("t_exit"), CV_NAMES.index("o2_exit")
    G = np.array([[K[i_p, j_t], K[i_p, j_o]], [K[i_s, j_t], K[i_s, j_o]]])
    lam = G * np.linalg.pinv(G).T           # Bristol RGA
    straight = lam[0, 0]                    # primary->T, secondary->O2
    if straight >= 0.5:
        return [("primary_level", "t_exit"), ("secondary_level", "o2_exit")], float(straight), lam
    return [("secondary_level", "t_exit"), ("primary_level", "o2_exit")], float(lam[0, 1]), lam
# Measured but not closed as a PI loop: the feeder's effect on the ash arrives
# only once that material has crossed the grate, so this is the pair that shows
# the transport dead time cleanly. (Grate speed also changes the flux leaving
# the grate instantly - that is v * m at the last column - so its own lag is
# short even though the composition it drags behind it is not.)
EXTRA_LAGS = [("waste_feed", "unburnt_bed")]


def steady_gains(d):
    y = d["y"].reshape(-1, d["y"].shape[-1]); u = d["u"].reshape(-1, d["u"].shape[-1])
    A = np.c_[u - u.mean(0), np.ones(len(u))]
    K = np.zeros((u.shape[1], y.shape[1]))
    for j in range(y.shape[1]):
        K[:, j] = np.linalg.lstsq(A, y[:, j], rcond=None)[0][:u.shape[1]]
    return K, y.std(0), u.std(0)


def dead_time(d, mv, cv, max_lag=900):
    """Lag [steps] at which the |cross-correlation| of the input increments with
    the output increments first reaches half of its maximum."""
    i, j = MV_NAMES.index(mv), CV_NAMES.index(cv)
    acc = np.zeros(max_lag)
    for e in range(d["u"].shape[0]):
        du = np.diff(d["u"][e, :, i]); dy = np.diff(d["y"][e, :, j])
        du = (du - du.mean()) / (du.std() + 1e-9); dy = (dy - dy.mean()) / (dy.std() + 1e-9)
        n = len(du)
        for L in range(max_lag):
            if L + 10 < n:
                acc[L] += np.abs(np.dot(du[:n - L], dy[L:])) / (n - L)
    acc /= acc.max() + 1e-12
    return int(np.argmax(acc >= 0.5)), acc


def main() -> int:
    d = load_split("train", ROOT / "data_ctrl_v2")
    K, y_sd, u_sd = steady_gains(d)
    # authority in sigma units: full-range move of each MV, in output sigmas
    env = envelope(PlantConfig.from_yaml(ROOT / "configs" / "plants" / "plant_a.yaml"))
    span = np.array([env[k][1] - env[k][0] for k in
                     ("stoker_speed", "waste_feed", "primary_air", "secondary_air")])
    authority = np.abs(K) * span[:, None] / y_sd[None, :]

    print("=== 入力の権限（各MVをフルレンジ動かしたときの出力変化、出力σ単位）===")
    print(f"{'':18s}" + "".join(f"{c:>14s}" for c in CV_NAMES))
    for i, m in enumerate(MV_NAMES):
        print(f"{m:18s}" + "".join(f"{authority[i,j]:14.2f}" for j in range(len(CV_NAMES))))

    air_pairs, rga_diag, lam = air_pairing(K)
    PAIRS = air_pairs + [GRATE_PAIR]
    print(f"\n=== 空気2ループの相対ゲイン行列 (行: primary, secondary / 列: t_exit, o2_exit) ===")
    print(np.round(lam, 3))
    print(f"  採用する対: {PAIRS[0][0]} -> t_exit, {PAIRS[1][0]} -> o2_exit   (RGA 対角 {rga_diag:.2f})")
    kp, ti, bias, L_all = [], [], [], {}
    for mv, cv in PAIRS:
        i, j = MV_NAMES.index(mv), CV_NAMES.index(cv)
        L, _ = dead_time(d, mv, cv)
        L = max(L, 1)
        Kproc = K[i, j]
        kc = 0.5 / Kproc if abs(Kproc) > 1e-12 else 0.0
        tau = 4.0 * L
        kp.append(kc); ti.append(min(tau, 4.0 * L)); L_all[f"{mv}->{cv}"] = L
        bias.append(float(d["u"][..., i].mean()))
        print(f"  {mv:16s} -> {cv:12s}  K={Kproc:+.4g}  L={L:4d}s  Kc={kc:+.4g}  Ti={min(tau,4*L):.0f}s")

    for mv, cv in EXTRA_LAGS:
        L, _ = dead_time(d, mv, cv, max_lag=1100)
        L_all[f"{mv}->{cv}"] = int(L)
        print(f"  {mv:16s} -> {cv:12s}  L={L:4d}s   (transport dead time, not a PI loop)")
    out = {"kp": kp, "ti": ti, "bias": bias, "dead_time_s": L_all, "rga_air": lam.tolist(),
           "rga_diag": rga_diag,
           "pairs": PAIRS, "gain_matrix": K.tolist(), "authority_sigma": authority.tolist(),
           "cv": CV_NAMES, "mv": MV_NAMES, "y_std": y_sd.tolist()}
    (ROOT / "figs").mkdir(exist_ok=True)
    (ROOT / "figs" / "pi_gains.json").write_text(json.dumps(out, indent=1))
    print("-> figs/pi_gains.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
