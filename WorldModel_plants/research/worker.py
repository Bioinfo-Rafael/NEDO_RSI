"""One trusted phase per subprocess; no Codex calls from numerical workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time
import traceback

# When invoked from a run's frozen reference, all imports use that snapshot.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from research.common import load_module, read_json, write_json, sha256
from research.data import generate, load_episodes, action_coverage


def training(job):
    import jax
    import numpy as np
    source, output = Path(job["source"]), Path(job["output"])
    prepare = load_module("prepare", ROOT / "control_research/prepare.py")
    original_windows = prepare.windows
    batches = [0]

    def counted_windows(*args, **kwargs):
        for batch in original_windows(*args, **kwargs):
            batches[0] += 1
            yield batch

    prepare.windows = counted_windows
    wm = load_module("worldmodel", source / "worldmodel.py")
    pipeline = load_module("pipeline", source / "pipeline.py")
    if wm.N_CV != 4 or not 1 <= wm.WARMUP <= 300 or not 1 <= wm.ENSEMBLE <= 8:
        raise ValueError("contract: N_CV=4, 1<=WARMUP<=300, 1<=ENSEMBLE<=8")
    stats = {k: np.asarray(v, np.float32) for k, v in read_json(job["stats"]).items()}
    data = load_episodes(job["data_files"])
    from wmf.actions import bounds
    from wmf.plants.schema import PlantConfig
    cfg = PlantConfig.from_yaml(ROOT / "configs/plants/plant_a.yaml")
    coverage = action_coverage(data, *bounds(cfg))
    for a in list(data.values()) + list(stats.values()):
        a.flags.writeable = False
    start = time.monotonic()
    params = pipeline.train(wm, data, stats, job["seconds"], job["seed"])
    jax.block_until_ready(params)
    elapsed = time.monotonic() - start
    if elapsed > job["seconds"] + 15:
        raise TimeoutError("training exceeded its requested budget plus 15s completion allowance")
    if not isinstance(params, list) or len(params) != wm.ENSEMBLE:
        raise ValueError("train must return ENSEMBLE parameter pytrees as a list")
    leaves = [np.asarray(x) for x in jax.tree.leaves(params)]
    if not leaves or not all(np.isfinite(v).all() for v in leaves):
        raise ValueError("non-finite/empty trained parameters")
    original_stats = read_json(job["stats"])
    if any(not np.array_equal(stats[k], np.asarray(original_stats[k], np.float32)) for k in stats):
        raise ValueError("pipeline changed the fixed normalization")
    updates = getattr(wm, "TRAINING_UPDATES", None)
    if updates is not None:
        updates = int(updates)
        if updates < 0:
            raise ValueError("negative update count")
    with (output / "model.pkl").open("wb") as f:
        pickle.dump({"params": jax.tree.map(np.asarray, params), "stats": stats,
                     "warmup": wm.WARMUP, "model_source_hash": sha256(source / "worldmodel.py")}, f)
    return {"training_seconds": elapsed, "training_budget_seconds": job["seconds"],
            "training_updates_self_reported": updates,
            "training_window_batches_observed": batches[0],
            "ensemble": len(params), "warmup": wm.WARMUP,
            "num_parameters": sum(int(v.size) for v in leaves), "episodes": len(data["y"]),
            "action_coverage": coverage, "devices": [str(d) for d in jax.devices()]}


def evaluating(job):
    import numpy as np
    from research.evaluate import forecast, control
    source, output = Path(job["source"]), Path(job["output"])
    load_module("prepare", ROOT / "control_research/prepare.py")
    wm = load_module("worldmodel", source / "worldmodel.py")
    # Only load our locally produced bundle, after importing its saved model module.
    with Path(job["bundle"]).open("rb") as f:
        bundle = pickle.load(f)
    if bundle["model_source_hash"] != sha256(source / "worldmodel.py"):
        raise ValueError("model bundle/source hash mismatch")
    stats = {k: np.asarray(v, np.float32) for k, v in bundle["stats"].items()}
    fixed = read_json(job["stats"])
    if any(not np.array_equal(stats[k], np.asarray(fixed[k], np.float32)) for k in stats):
        raise ValueError("bundle did not use the fixed controller normalization")
    scale = np.asarray(fixed["y_std"], np.float64)
    fr = forecast(wm, bundle["params"], stats, load_episodes(job["data_files"]), scale, output)
    cr = control(wm, bundle["params"], stats, scale, job["scenarios"], job["steps"], output,
                 capture=job.get("capture", False), smoke=job.get("smoke", False))
    return {"forecast_nrmse": fr["forecast_nrmse"], "control_cost": cr["control_cost"],
            "forecast": fr, "control": cr, "evaluation_split": job["split"],
            "scale_hash": sha256(job["stats"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    args = ap.parse_args()
    job = read_json(args.job)
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    try:
        if job["phase"] == "generate":
            result = {"episodes": generate(ROOT, output / "episodes", job["seeds"], job["policy"])}
        elif job["phase"] == "train":
            result = training(job)
        elif job["phase"] == "evaluate":
            result = evaluating(job)
        else:
            raise ValueError("unknown worker phase")
        write_json(output / "result.json", result)
        print(json.dumps(result, allow_nan=False), flush=True)
    except Exception as exc:
        write_json(output / "error.json", {"error": str(exc), "type": type(exc).__name__})
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
