"""Run one bounded experiment. Test scoring is explicit and never used by default."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent


def worker(args):
    import joblib
    import numpy as np
    import sklearn
    from threadpoolctl import threadpool_limits
    from prepare import sha256
    from train import fit_model

    start = time.perf_counter()
    profile_path = ROOT / "data/profile.json"
    profile = json.loads(profile_path.read_text())
    if sha256(ROOT / "prepare.py") != profile["prepare_sha256"]:
        raise ValueError("prepare.py changed since preparation; create a new dataset intentionally.")
    for name in ("train", args.split):
        if sha256(ROOT / "data" / f"{name}.npz") != profile["split_sha256"][name]:
            raise ValueError(f"Prepared {name} data changed")
    with np.load(ROOT / "data/train.npz", allow_pickle=False) as data:
        X_train, y_train = data["X"], data["y"]
    with threadpool_limits(limits=2):
        fit_start = time.perf_counter()
        model = fit_model(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        with np.load(ROOT / "data" / f"{args.split}.npz", allow_pickle=False) as data:
            prediction = np.asarray(model.predict(data["X"]), dtype=float)
            y, baseline = data["y"], data["persistence"]
        if prediction.shape != y.shape or not np.isfinite(prediction).all():
            raise ValueError("Predictions must be a finite 1-D vector with the expected length")
        rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
        persistence_rmse = float(np.sqrt(np.mean((baseline - y) ** 2)))
        metrics = {"split": args.split, "rmse": rmse, "mae": float(np.mean(np.abs(prediction-y))),
                   "persistence_rmse": persistence_rmse,
                   "improvement_vs_persistence_percent": 100 * (1-rmse/persistence_rmse) if persistence_rmse else None,
                   "train_samples": len(y_train), "evaluation_samples": len(y),
                   "fit_seconds": fit_seconds, "train_sha256": sha256(ROOT / "train.py"),
                   "runner_sha256": sha256(__file__), "dataset_sha256": sha256(profile_path),
                   "python": sys.version.split()[0], "sklearn": sklearn.__version__, "numpy": np.__version__}
    joblib.dump(model, args.run_dir / "model.joblib")
    metrics["worker_seconds"] = time.perf_counter() - start
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--timeout", type=int, default=300, help="Whole worker time limit in seconds (default 300)")
    parser.add_argument("--description", default="baseline")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.worker:
        worker(args)
        return
    if not (ROOT / "data/profile.json").exists():
        raise SystemExit("Run .venv/bin/python prepare.py first.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    run_dir = ROOT / "results" / run_id
    run_dir.mkdir(parents=True)
    for name in ("train.py", "run.py", "prepare.py", "config.json", "requirements.txt"):
        shutil.copy2(ROOT / name, run_dir / name)
    shutil.copy2(ROOT / "data/profile.json", run_dir / "profile.json")
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}
    record = {"run_id": run_id, "description": args.description, "split": args.split, "timeout_seconds": args.timeout}
    started = time.perf_counter()
    try:
        with (run_dir / "run.log").open("w") as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--split", args.split,
                                     "--run-dir", str(run_dir)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=False)
        if result.returncode:
            record.update(status="crash", returncode=result.returncode)
        else:
            record.update(json.loads((run_dir / "metrics.json").read_text()), status="ok")
    except subprocess.TimeoutExpired:
        record["status"] = "timeout"
    record["total_seconds"] = time.perf_counter() - started
    (run_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    with (ROOT / "results/results.jsonl").open("a") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Artifacts: {run_dir}")
    if record["status"] != "ok":
        print((run_dir / "run.log").read_text()[-4000:])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
