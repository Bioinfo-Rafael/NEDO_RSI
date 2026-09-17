#!/usr/bin/env python
"""Bounded A/B research runner. Execute from any working directory."""
from __future__ import annotations

import argparse
import difflib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from research.common import (append_json, check_edits, objective, read_json, run_process,
                             sha256, tree_hashes, utc_now, write_json)
from research.data import DEFAULT_POLICY, load_episodes, validate_policy
from research.telemetry import collect

SOURCE_FILES = ("worldmodel.py", "pipeline.py", "sampling.json", "hypothesis.json", "ResearchState.md")


class StageFailure(RuntimeError):
    def __init__(self, status, stage, detail):
        super().__init__(f"{stage}: {detail}")
        self.status, self.stage = status, stage


def scenarios(reference, final=False, smoke=False):
    sp = read_json(Path(reference) / "figs/setpoints.json")
    result = []
    for name in ("temp_step", "o2_step"):
        r0, r1 = sp[name]["r0"], sp[name]["r1"]
        # Unseen direction and switch time; use the same already reachable endpoints.
        if final:
            r0, r1 = r1, r0
        result.append({"name": name, "r0": r0, "r1": r1,
                       "t_step": (30 if smoke else 300) if final else (30 if smoke else 150)})
    return result


def codex_argv(model, effort, workspace):
    executable = shutil.which("codex")
    if not executable:
        raise FileNotFoundError("codex CLI not found on PATH")
    return [executable, "exec", "--ignore-user-config", "--sandbox", "workspace-write",
            "--skip-git-repo-check", "--model", model,
            "-c", f'model_reasoning_effort="{effort}"',
            "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
            "-c", 'features.multi_agent=false', "--json", "--color", "never",
            "--cd", str(workspace), "-"]


class Experiment:
    def __init__(self, args):
        self.args = args
        self.smoke = args.mode == "smoke"
        self.start = time.monotonic()
        self.deadline = self.start + args.minutes * 60
        self.dev_seconds = min(args.dev_minutes * 60, args.minutes * 60 * .75)
        self.dev_deadline = self.start + self.dev_seconds
        tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:6]
        self.root = (args.run_dir or REPO / "research/runs" / tag).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.reference = self.root / "reference"
        self.rows = []
        self.events = self.root / "events.jsonl"
        self.arm_seconds = {"A": 0., "B": 0.}
        self.arm_data = {"A": [], "B": []}
        self.incumbent = {}
        self.memory = {"A": "No experiments yet.\n", "B": "No experiments yet.\n"}
        self.baseline = None
        self.frozen_hashes = {}

    def say(self, message):
        print(f"[{utc_now()}] {message}", flush=True)

    def event(self, **record):
        append_json(self.events, {"timestamp": utc_now(), **record})

    def verify_reference(self):
        if self.frozen_hashes and tree_hashes(self.reference) != self.frozen_hashes:
            raise RuntimeError("Frozen reference was modified; experiment invalidated")

    def prepare(self):
        self.say(f"run directory: {self.root}")
        for path in ("src", "configs", "control_research"):
            shutil.copytree(REPO / path, self.reference / path,
                            ignore=shutil.ignore_patterns("__pycache__", "*.log", ".git", "results.tsv"))
        (self.reference / "scripts").mkdir()
        shutil.copy2(REPO / "scripts/run_control.py", self.reference / "scripts/run_control.py")
        (self.reference / "figs").mkdir()
        shutil.copy2(REPO / "figs/setpoints.json", self.reference / "figs/setpoints.json")
        shutil.copytree(REPO / "data_ctrl_v2", self.reference / "data_ctrl_v2")
        shutil.copytree(REPO / "research", self.reference / "research",
                        ignore=shutil.ignore_patterns("runs", "__pycache__", ".pytest_cache"))
        self.base_files = sorted((self.reference / "data_ctrl_v2/train").glob("*.npz"))
        self.val_files = sorted((self.reference / "data_ctrl_v2/val").glob("*.npz"))
        if not self.base_files or not self.val_files:
            raise FileNotFoundError("data_ctrl_v2 needs train and validation episodes")
        import numpy as np
        d = load_episodes(self.base_files)
        # Identical normalization for data/model/MPC in all candidates; fixed scoring scales.
        self.stats_path = self.reference / "baseline_stats.json"
        stats = {}
        for name in ("y", "u"):
            x = d[name].reshape(-1, 4)
            stats[name+"_mean"] = x.mean(0).tolist()
            stats[name+"_std"] = (x.std(0) + 1e-8).tolist()
        write_json(self.stats_path, stats)
        self.frozen_hashes = tree_hashes(self.reference)
        write_json(self.root / "reference_hashes.json", self.frozen_hashes)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        (self.root / "preexisting.diff").write_text(subprocess.check_output(["git", "diff"], cwd=REPO, text=True))
        config = vars(self.args).copy()
        config["run_dir"] = str(self.root)
        self.manifest = {"schema_version": 1, "started_at": utc_now(), "config": config,
                         "repo_commit": commit, "python": sys.version, "platform": platform.platform(),
                         "codex_version": subprocess.check_output(["codex", "--version"], text=True).strip()
                         if shutil.which("codex") else None,
                         "smoke_only": self.smoke, "fixed_scale_hash": sha256(self.stats_path),
                         "phase": "prepared", "arms": {"A": "random APRBS", "B": "agent-directed APRBS"},
                         "development_budget_seconds": self.dev_seconds,
                         "test_protocol": {"episode_seeds": [900000+i for i in range(1 if self.smoke else 4)],
                                           "steps": 100 if self.smoke else 1600,
                                           "scenarios": scenarios(self.reference, True, self.smoke)}}
        versions = {}
        for package in ("jax", "jaxlib", "numpy", "matplotlib", "PyYAML", "pillow", "pytest"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        write_json(self.root / "environment.json", {"python": sys.version, "platform": platform.platform(),
                                                   "packages": versions})
        write_json(self.root / "manifest.json", self.manifest)
        self.arm_data = {arm: self.base_files.copy() for arm in ("A", "B")}

    def stage(self, experiment_id, phase, output, *, deadline=None, cap=180, **kwargs):
        self.verify_reference()
        deadline = min(deadline or self.deadline, self.deadline)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise StageFailure("timeout", phase, "budget exhausted before launch")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        job = {"phase": phase, "output": str(output), **kwargs}
        write_json(output / "job.json", job)
        call_id = f"{experiment_id}:{phase}:{uuid.uuid4().hex[:8]}"
        self.event(experiment_id=experiment_id, call_id=call_id, actor="harness", phase=phase, state="started")
        self.say(f"{experiment_id}: {phase}")
        proc = run_process([sys.executable, str(self.reference / "research/worker.py"), str(output / "job.json")],
                           self.reference, output / "process", min(remaining, cap),
                           env={"JAX_PLATFORMS": "cpu"}, event_meta={"experiment_id": experiment_id, "call_id": call_id})
        self.event(experiment_id=experiment_id, call_id=call_id, actor="harness", phase=phase,
                   state="completed", status=proc["status"], seconds=proc["wall_seconds"])
        self.verify_reference()
        if proc["status"] != "ok":
            detail = read_json(output / "error.json") if (output / "error.json").exists() else proc["status"]
            raise StageFailure(proc["status"], phase, str(detail))
        if not (output / "result.json").exists():
            raise StageFailure("crash", phase, "worker produced no result")
        return read_json(output / "result.json")

    def probe(self):
        if self.smoke:
            return
        directory = self.root / "preflight"
        directory.mkdir()
        proc = run_process(codex_argv(self.args.model, self.args.effort, directory), directory,
                           directory / "agent", min(90, self.dev_deadline-time.monotonic()),
                           stdin="Reply READY only. Do not use tools, read files, or change files.")
        telemetry = collect(directory / "agent", "preflight")
        write_json(directory / "usage.json", telemetry)
        if proc["status"] != "ok" or not telemetry["thread_ids"]:
            raise StageFailure(proc["status"], "codex_preflight", "see preflight/agent/stderr.log and stdout.jsonl")

    def seed_source(self, target):
        target.mkdir(parents=True)
        for name in ("worldmodel.py", "pipeline.py"):
            shutil.copy2(self.reference / "research/templates" / name, target / name)
        write_json(target / "sampling.json", DEFAULT_POLICY)
        write_json(target / "hypothesis.json", {"hypothesis": "Initial synchronous linear baseline",
                                               "change": "None", "expected_effect": "Comparison reference"})
        (target / "ResearchState.md").write_text("No experiments yet.\n")

    def fit_and_evaluate(self, eid, source, files, directory, deadline):
        data_manifest = [{"path": str(p), "sha256": sha256(p)} for p in files]
        write_json(directory / "training_data.json", data_manifest)
        training = self.stage(eid, "train", directory / "train", deadline=deadline,
                              cap=self.args.train_seconds+35, source=str(source), data_files=[str(p) for p in files],
                              stats=str(self.stats_path), seconds=self.args.train_seconds, seed=self.args.seed)
        for record in data_manifest:
            if sha256(record["path"]) != record["sha256"]:
                raise StageFailure("invalid", "train", "training input changed")
        metric = self.stage(eid, "evaluate", directory / "validation", deadline=deadline, cap=240,
                            source=str(source), bundle=str(directory / "train/model.pkl"),
                            stats=str(self.stats_path), data_files=[str(p) for p in self.val_files],
                            scenarios=scenarios(self.reference, smoke=self.smoke),
                            steps=80 if self.smoke else 400, split="validation", smoke=self.smoke)
        return training, metric

    def baseline_run(self):
        directory = self.root / "experiments/baseline"
        self.seed_source(directory / "source")
        training, metric = self.fit_and_evaluate("baseline", directory / "source", self.base_files,
                                                 directory, self.dev_deadline)
        self.baseline = {"id": "baseline", "arm": "baseline", "round": 0, "status": "ok", "decision": "keep",
                         "objective": 1.0, "forecast_nrmse": metric["forecast_nrmse"],
                         "control_cost": metric["control_cost"], "training": training,
                         "metrics": metric, "directory": str(directory), "source": str(directory / "source")}
        self.record(self.baseline)
        self.incumbent = {"A": self.baseline, "B": self.baseline}

    def record(self, row):
        self.rows.append(row)
        directory = self.root / "experiments" / row["id"]
        write_json(directory / "experiment.json", row)
        append_json(self.root / "results.jsonl", row)
        self.event(experiment_id=row["id"], actor="harness", phase="decision", state="completed",
                   status=row["status"], decision=row.get("decision"), objective=row.get("objective"))
        self.say(f"{row['id']}: {row['status']} / {row.get('decision')} / objective={row.get('objective')}")

    def agent_edit(self, arm, number, directory, deadline):
        workspace = self.root / "workspaces" / arm / f"round_{number:02d}"
        workspace.mkdir(parents=True)
        parent_source = Path(self.incumbent[arm]["source"])
        for name in SOURCE_FILES:
            shutil.copy2(parent_source / name, workspace / name)
        (workspace / "ResearchState.md").write_text(self.memory[arm])
        # A must use the canonical sampler. B's current policy is part of its proposal.
        if arm == "A":
            write_json(workspace / "sampling.json", DEFAULT_POLICY)
        write_json(workspace / "hypothesis.json", {"hypothesis": "", "change": "", "expected_effect": ""})
        for name in ("Prompt.md", "INTERFACES.md"):
            shutil.copy2(self.reference / "research" / name, workspace / name)
        feedback = {"arm": arm, "round": number, "training_seconds": self.args.train_seconds,
                    "seed": self.args.seed, "incumbent": self.incumbent[arm]["id"],
                    "baseline": self.baseline["metrics"], "current": self.incumbent[arm]["metrics"],
                    "history": [{k: r.get(k) for k in ("id", "round", "status", "decision", "objective",
                                 "forecast_nrmse", "control_cost", "hypothesis", "error", "training")}
                                for r in self.rows if r["arm"] == arm]}
        write_json(workspace / "feedback.json", feedback)
        allowed = list(SOURCE_FILES) if arm == "B" else [x for x in SOURCE_FILES if x != "sampling.json"]
        write_json(workspace / "file_manifest.json", {"editable": allowed,
                   "read_only": ["Prompt.md", "INTERFACES.md", "feedback.json", "file_manifest.json"] +
                                (["sampling.json"] if arm == "A" else []),
                   "scope": "Only this workspace. No external results, test files, source checkouts or sessions."})
        before = tree_hashes(workspace)
        write_json(directory / "workspace_before.json", before)
        if not self.smoke:
            prompt = (f"Read Prompt.md and follow it for condition {arm}, round {number}. "
                      "Make exactly one experiment proposal in the editable files. "
                      "The harness will generate data, train and evaluate after this turn. "
                      "Use only this workspace. Do not run training or simulation yourself. "
                      "Finish after writing hypothesis.json; report your hypothesis concisely.")
            proc = run_process(codex_argv(self.args.model, self.args.effort, workspace), workspace,
                               directory / "agent", max(.1, min(self.args.agent_seconds, deadline-time.monotonic())),
                               stdin=prompt, event_meta={"experiment_id": directory.name})
            telemetry = collect(directory / "agent", directory.name)
            self.event(experiment_id=directory.name, actor="codex", phase="proposal", state="completed",
                       status=proc["status"], seconds=proc["wall_seconds"], usage=telemetry["usage"],
                       usage_granularity=telemetry["usage_granularity"])
            if proc["status"] != "ok":
                raise StageFailure(proc["status"], "agent", "see agent/stdout.jsonl and stderr.log")
            mismatch = [m for m in telemetry["models"] if m["model"] != self.args.model or m["effort"] != self.args.effort]
            if mismatch:
                raise StageFailure("invalid", "agent", f"actual model/config differs: {mismatch}")
        else:
            write_json(workspace / "hypothesis.json", {"hypothesis": "Smoke: pipeline transport only",
                       "change": "No model edits; add default simulator episode", "expected_effect": "No scientific claim"})
        after = tree_hashes(workspace)
        write_json(directory / "workspace_after.json", after)
        changed = check_edits(before, after, allowed)
        hypothesis = read_json(workspace / "hypothesis.json")
        if any(not isinstance(hypothesis.get(k), str) or not hypothesis[k].strip()
               for k in ("hypothesis", "change", "expected_effect")):
            raise StageFailure("invalid", "proposal", "hypothesis/change/expected_effect are required")
        validate_policy(read_json(workspace / "sampling.json"))
        source = directory / "source"
        source.mkdir()
        diff = []
        for name in SOURCE_FILES:
            shutil.copy2(workspace / name, source / name)
            diff.extend(difflib.unified_diff((parent_source / name).read_text().splitlines(True),
                                           (source / name).read_text().splitlines(True),
                                           fromfile=f"parent/{name}", tofile=f"candidate/{name}"))
        (directory / "changes.diff").write_text("".join(diff))
        write_json(directory / "source_hashes.json", tree_hashes(source))
        self.memory[arm] = (source / "ResearchState.md").read_text()
        self.event(experiment_id=directory.name, actor="codex" if not self.smoke else "smoke",
                   phase="hypothesis", state="completed", hypothesis=hypothesis, changed=changed)
        return source, hypothesis

    def round(self, arm, number):
        eid = f"{arm}_{number:02d}"
        directory = self.root / "experiments" / eid
        directory.mkdir(parents=True)
        start = time.monotonic()
        deadline = min(self.dev_deadline, start + self.dev_seconds/2 - self.arm_seconds[arm])
        row = {"id": eid, "arm": arm, "round": number, "status": "pending", "decision": "discard",
               "objective": None, "forecast_nrmse": None, "control_cost": None,
               "directory": str(directory), "parent": self.incumbent[arm]["id"]}
        try:
            source, hypothesis = self.agent_edit(arm, number, directory, deadline)
            self.verify_reference()
            row.update(hypothesis=hypothesis, source=str(source))
            policy = DEFAULT_POLICY if arm == "A" else read_json(source / "sampling.json")
            row["sampling_policy"] = policy
            gen = self.stage(eid, "generate", directory / "generation", deadline=deadline, cap=240,
                             seeds=[100000+number*10+i for i in range(self.args.episodes)], policy=policy)
            files = [Path(x["file"]) for x in gen["episodes"]]
            self.arm_data[arm].extend(files)
            row["generation"] = gen
            training, metric = self.fit_and_evaluate(eid, source, self.arm_data[arm], directory, deadline)
            if tree_hashes(source) != read_json(directory / "source_hashes.json"):
                raise StageFailure("invalid", "worker", "candidate source changed during execution")
            score = objective(metric["forecast_nrmse"], metric["control_cost"], self.baseline)
            row.update(status="ok", objective=score, forecast_nrmse=metric["forecast_nrmse"],
                       control_cost=metric["control_cost"], training=training, metrics=metric)
            if score < self.incumbent[arm]["objective"]:
                row["decision"] = "keep"
                self.incumbent[arm] = row
        except Exception as exc:
            row.update(status=getattr(exc, "status", "invalid"), error=str(exc),
                       failed_stage=getattr(exc, "stage", "proposal_or_integrity"))
            (directory / "exception.txt").write_text(traceback.format_exc())
        row["wall_seconds"] = time.monotonic()-start
        self.arm_seconds[arm] += row["wall_seconds"]
        self.record(row)
        self.verify_reference()

    def select(self):
        completed = {arm: max([r["round"] for r in self.rows if r["arm"] == arm] or [0]) for arm in ("A", "B")}
        common = min(completed.values())
        selected = {"baseline": self.baseline}
        for arm in ("A", "B"):
            candidates = [self.baseline] + [r for r in self.rows if r["arm"] == arm and
                          r["round"] <= common and r["status"] == "ok"]
            selected[arm] = min(candidates, key=lambda r: r["objective"])
        return common, selected

    def finalize(self):
        common, selected = self.select()
        # Freeze selections BEFORE generating or observing held-out targets.
        selection = {"common_rounds": common, "selected": {k: v["id"] for k, v in selected.items()},
                     "frozen_at": utc_now()}
        write_json(self.root / "selection.json", selection)
        self.manifest["phase"] = "final_evaluation"
        write_json(self.root / "manifest.json", self.manifest)
        final = {**selection, "evaluations": {}, "joint_improvement": {}, "B_dominates_A": None}
        try:
            generated = self.stage("heldout", "generate", self.root / "final/data", cap=420,
                                   seeds=self.manifest["test_protocol"]["episode_seeds"], policy=DEFAULT_POLICY)
            test_files = [x["file"] for x in generated["episodes"]]
            cache = {}
            for label, row in selected.items():
                if row["id"] not in cache:
                    directory = self.root / "final" / row["id"]
                    try:
                        metric = self.stage("final_"+row["id"], "evaluate", directory, cap=600,
                                            source=row["source"], bundle=str(Path(row["directory"])/"train/model.pkl"),
                                            stats=str(self.stats_path), data_files=test_files,
                                            scenarios=self.manifest["test_protocol"]["scenarios"],
                                            steps=self.manifest["test_protocol"]["steps"], split="heldout",
                                            capture=True, smoke=self.smoke)
                        cache[row["id"]] = {"status": "ok", "metrics": metric, "directory": str(directory)}
                    except StageFailure as exc:
                        cache[row["id"]] = {"status": exc.status, "error": str(exc)}
                final["evaluations"][label] = cache[row["id"]]
            base = final["evaluations"]["baseline"]
            if base["status"] == "ok":
                for label in ("A", "B"):
                    e = final["evaluations"][label]
                    final["joint_improvement"][label] = (
                        all(e["metrics"][k] < base["metrics"][k] for k in ("forecast_nrmse", "control_cost"))
                        if e["status"] == "ok" else None)
            a, b = final["evaluations"]["A"], final["evaluations"]["B"]
            if a["status"] == b["status"] == "ok":
                final["B_dominates_A"] = all(b["metrics"][k] < a["metrics"][k] for k in ("forecast_nrmse", "control_cost"))
        except StageFailure as exc:
            final["error"] = str(exc)
        write_json(self.root / "final.json", final)
        self.manifest.update(phase="complete", finished_at=utc_now(),
                             wall_seconds=time.monotonic()-self.start, arm_seconds=self.arm_seconds)
        write_json(self.root / "manifest.json", self.manifest)

    def run(self):
        try:
            self.prepare()
            self.probe()
            self.baseline_run()
            self.manifest["phase"] = "development"
            write_json(self.root / "manifest.json", self.manifest)
            for number in range(1, self.args.rounds+1):
                if time.monotonic() >= self.dev_deadline or min(self.dev_seconds/2-x for x in self.arm_seconds.values()) <= 0:
                    break
                # Alternate order, isolated contexts; no cross-arm result sharing.
                for arm in (("A", "B") if number % 2 else ("B", "A")):
                    if time.monotonic() >= self.dev_deadline:
                        break
                    self.round(arm, number)
            self.finalize()
        except BaseException as exc:
            write_json(self.root / "run_error.json", {"error": str(exc), "traceback": traceback.format_exc()})
            if hasattr(self, "manifest"):
                self.manifest.update(phase="failed", error=str(exc), finished_at=utc_now())
                write_json(self.root / "manifest.json", self.manifest)
            raise
        finally:
            try:
                remaining = self.deadline - time.monotonic()
                if remaining > 2 and self.reference.exists():
                    result = run_process([sys.executable, str(self.reference / "research/report.py"), str(self.root)],
                                         self.root, self.root / "report_process", remaining)
                    if result["status"] != "ok":
                        from research.report import build_report
                        build_report(self.root, videos=False)
                else:
                    from research.report import build_report
                    build_report(self.root, videos=False)
                if hasattr(self, "manifest"):
                    self.manifest["wall_seconds_including_report"] = time.monotonic() - self.start
                    write_json(self.root / "manifest.json", self.manifest)
            except Exception:
                (self.root / "report_error.txt").write_text(traceback.format_exc())
        self.say(f"finished: {self.root / 'report.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["compare", "smoke", "report"])
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh"], default="low")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--train-seconds", type=float, default=150.)
    ap.add_argument("--agent-seconds", type=float, default=300.)
    ap.add_argument("--minutes", type=float, default=120.)
    ap.add_argument("--dev-minutes", type=float, default=90.)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "report":
        if not args.run_dir:
            ap.error("report requires --run-dir")
        from research.report import build_report
        build_report(args.run_dir.resolve())
        return
    if args.mode == "smoke":
        args.rounds, args.episodes, args.train_seconds = 1, 1, 5.
        args.minutes, args.dev_minutes = 10., 7.
    if any(x <= 0 for x in [args.rounds, args.episodes, args.train_seconds, args.minutes, args.agent_seconds, args.dev_minutes]):
        ap.error("budgets and counts must be positive")
    Experiment(args).run()


if __name__ == "__main__":
    main()
