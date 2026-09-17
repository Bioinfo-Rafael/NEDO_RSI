"""Fixed CSV preparation. All timestamps are interpreted in the CSV's local time."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_samples(raw, cfg):
    """Use exact minute readings, past-only lags, and real future timestamps."""
    time_col, target = cfg["time_column"], cfg["target"]
    features = cfg["features"]
    lags = cfg["lags_minutes"]
    horizon = cfg["horizon_minutes"]
    if (not isinstance(horizon, int) or horizon <= 0 or not lags
            or any(not isinstance(lag, int) or lag < 0 for lag in lags)
            or 0 not in lags or len(set(lags)) != len(lags)):
        raise ValueError("Use a positive integer horizon and unique nonnegative integer lags including 0.")
    if target not in features or len(set(features)) != len(features):
        raise ValueError("Features must be unique and include the target for the persistence baseline.")
    frame = raw.copy()
    frame[time_col] = pd.to_datetime(frame[time_col], errors="raise")
    if frame[time_col].isna().any():
        raise ValueError("Missing timestamp")
    profile = {
        "raw_rows": len(frame), "raw_columns": len(frame.columns),
        "raw_start": str(frame[time_col].min()), "raw_end": str(frame[time_col].max()),
        "raw_time_sorted": bool(frame[time_col].is_monotonic_increasing),
        "duplicate_timestamp_excess_rows": int(frame[time_col].duplicated().sum()),
        "missing_by_column": {k: int(v) for k, v in frame.isna().sum().items() if v},
    }
    # Exact copies can be collapsed. Conflicting timestamps have no documented
    # precedence, so omit ALL alternatives instead of choosing one arbitrarily.
    dedup = frame.drop_duplicates()
    conflicts = dedup[time_col].duplicated(keep=False)
    profile.update({
        "identical_duplicate_rows": len(frame) - len(dedup),
        "conflicting_rows_excluded": int(conflicts.sum()),
        "conflicting_timestamps_excluded": int(dedup.loc[conflicts, time_col].nunique()),
    })
    clean = dedup.loc[~conflicts].sort_values(time_col).set_index(time_col)
    # Reindex onto actual clock minutes. No averaging, interpolation, or filling.
    grid = pd.date_range(frame[time_col].min().ceil("min"), frame[time_col].max().floor("min"), freq="min")
    minute = clean.reindex(grid)
    numeric = minute[features].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    valid = numeric.notna().all(axis=1) & minute["稼働中"].eq(1)
    # Both the history and forecast interval must contain observed operating
    # minute readings. This defines an offline, continuously operating cohort.
    past_ok = valid.rolling(max(lags) + 1, min_periods=max(lags) + 1).sum().eq(max(lags) + 1)
    future_ok = valid.rolling(horizon + 1, min_periods=horizon + 1).sum().shift(-horizon).eq(horizon + 1)
    X = pd.concat([numeric.shift(lag).add_suffix(f"__lag_{lag}m") for lag in lags], axis=1)
    y = numeric[target].shift(-horizon)
    keep = past_ok & future_ok & X.notna().all(axis=1) & y.notna()
    samples = {"X": X.loc[keep], "y": y.loc[keep], "persistence": numeric.loc[keep, target]}
    profile.update({"minute_grid_rows": len(grid), "valid_operating_minute_rows": int(valid.sum()),
                    "eligible_samples": int(keep.sum()), "feature_count": X.shape[1]})
    return samples, profile


def split_masks(index, cfg):
    target_time = index + pd.Timedelta(minutes=cfg["horizon_minutes"])
    train_end, val_end = pd.Timestamp(cfg["train_end"]), pd.Timestamp(cfg["validation_end"])
    if train_end >= val_end:
        raise ValueError("train_end must precede validation_end")
    return {
        "train": target_time < train_end,
        "validation": (index >= train_end) & (target_time < val_end),
        "test": index >= val_end,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--force", action="store_true", help="Replace prepared data after changing the task")
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text())
    source = (cfg_path.parent / cfg["csv"]).resolve()
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        raise SystemExit("Prepared data already exists. Use --force only when intentionally redefining the task.")
    raw = pd.read_csv(source, encoding=cfg["encoding"])
    samples, profile = build_samples(raw, cfg)
    masks = split_masks(samples["X"].index, cfg)
    if any(int(mask.sum()) < 100 for mask in masks.values()):
        raise ValueError("Each split needs at least 100 samples. Check time boundaries and data quality.")
    args.output.mkdir(parents=True, exist_ok=True)
    profile["splits"] = {}
    for name, mask in masks.items():
        index = samples["X"].index[mask]
        np.savez_compressed(args.output / f"{name}.npz",
                            X=samples["X"].loc[mask].to_numpy(dtype=np.float32),
                            y=samples["y"].loc[mask].to_numpy(dtype=np.float64),
                            persistence=samples["persistence"].loc[mask].to_numpy(dtype=np.float64),
                            timestamps=index.to_numpy(dtype="datetime64[s]"))
        profile["splits"][name] = {"samples": len(index), "origin_start": str(index.min()),
                                   "origin_end": str(index.max()),
                                   "target_end": str(index.max() + pd.Timedelta(minutes=cfg["horizon_minutes"]))}
    profile.update({"config": cfg, "source_path": str(source), "source_sha256": sha256(source),
                    "prepare_sha256": sha256(__file__), "feature_names": list(samples["X"].columns),
                    "split_sha256": {name: sha256(args.output / f"{name}.npz") for name in masks}})
    (args.output / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in profile.items() if k not in {"feature_names", "config", "missing_by_column"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
