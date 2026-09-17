from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def append_json(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_hashes(root):
    root = Path(root)
    out = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"symlink not permitted: {rel}")
        if path.is_file():
            out[str(rel)] = sha256(path)
    return out


def check_edits(before, after, allowed):
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    illegal = sorted(set(changed) - set(allowed))
    if illegal:
        raise ValueError(f"files outside edit manifest changed: {illegal}")
    return changed


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_process(argv, cwd, directory, timeout, *, stdin=None, env=None, event_meta=None):
    """Capture stdout separately from stderr; kill the whole process group on timeout.

    Stream receipt timestamps are local observations, not API request latency.
    Never use shell=True: paths/prompts are data, including quotes and newlines.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    start = time.monotonic()
    effective_env = dict(os.environ)
    effective_env.update(env or {})
    effective_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if stdin is not None:
        (directory / "prompt.txt").write_text(stdin)
    with (directory / "stdout.jsonl").open("w") as out, (directory / "stderr.log").open("w") as err:
        p = subprocess.Popen(argv, cwd=cwd, env=effective_env, text=True,
                             stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=err, start_new_session=True)

        def consume():
            with (directory / "stream_events.jsonl").open("w") as events:
                for line in p.stdout:
                    out.write(line)
                    out.flush()
                    try:
                        payload = json.loads(line)
                    except ValueError:
                        continue
                    events.write(json.dumps({"received_at": utc_now(),
                                             "elapsed_seconds": time.monotonic() - start,
                                             "event": payload}) + "\n")
                    events.flush()

        reader = threading.Thread(target=consume, daemon=True)
        reader.start()
        if stdin is not None:
            try:
                p.stdin.write(stdin)
                p.stdin.close()
            except BrokenPipeError:
                pass
        status = "ok"
        try:
            p.wait(timeout=max(0.1, timeout))
            if p.returncode:
                status = "crash"
        except subprocess.TimeoutExpired:
            status = "timeout"
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        except BaseException:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise
        finally:
            # A child may outlive its parent and keep the stdout pipe open.
            reader.join(timeout=2)
            if reader.is_alive():
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                reader.join(timeout=2)
    result = {"status": status, "returncode": p.returncode,
              "started_at": started, "completed_at": utc_now(),
              "wall_seconds": time.monotonic() - start, "timeout_seconds": timeout,
              "argv": argv, "cwd": str(cwd), **(event_meta or {})}
    write_json(directory / "process.json", result)
    return result


def objective(forecast, control, baseline):
    import math
    vals = [forecast, control, baseline["forecast_nrmse"], baseline["control_cost"]]
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in vals):
        raise ValueError("non-finite/negative objective component")
    if min(vals[2:]) <= 1e-12:
        raise ValueError("baseline objective is zero; ratio is undefined")
    return 0.5 * (forecast / vals[2] + control / vals[3])
