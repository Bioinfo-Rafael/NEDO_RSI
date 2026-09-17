from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.common import check_edits, objective, run_process, tree_hashes, write_json
from research.data import DEFAULT_POLICY, actions, validate_policy
from research.evaluate import forecast_inputs, forecast_metrics, control_metrics
from research.run import Experiment, StageFailure, codex_argv
from research.telemetry import parse_sessions, parse_cli, read_jsonl, collect


def test_post_action_alignment_and_no_future_observations():
    u = np.arange(1200*4, dtype=np.float32).reshape(1, 1200, 4)
    y = np.cumsum(u, axis=1)
    stats = {"y_mean": np.zeros(4), "y_std": np.ones(4),
             "u_mean": np.zeros(4), "u_std": np.ones(4)}
    hist, uh, uf, target = forecast_inputs({"y": y, "u": u}, stats, 20)
    assert hist.shape == (1, 20, 4)
    np.testing.assert_array_equal(hist[:, -1], y[:, 299])
    np.testing.assert_array_equal(uf[:, 0], u[:, 300])
    np.testing.assert_array_equal(hist[:, -1] + uf[:, 0], target[:, 0])
    changed = y.copy(); changed[:, 300:] = -1e6
    h2, uh2, uf2, target2 = forecast_inputs({"y": changed, "u": u}, stats, 20)
    for x, z in zip((hist, uh, uf), (h2, uh2, uf2)):
        np.testing.assert_array_equal(x, z)
    assert not np.array_equal(target, target2)
    with pytest.raises(ValueError):
        forecast_inputs({"y": y, "u": u}, stats, 301)


def test_fixed_forecast_scale_and_horizon_aggregation():
    target = np.zeros((2, 900, 4))
    scale = np.array([10., 2., .1, 5.])
    pred = np.broadcast_to(scale, target.shape).copy()
    pred[:, 300:] *= 2
    result = forecast_metrics(pred, target, scale)
    assert result["horizons"]["30"]["mean_nrmse"] == pytest.approx(1)
    assert result["horizons"]["300"]["mean_nrmse"] == pytest.approx(1)
    assert result["horizons"]["900"]["mean_nrmse"] == pytest.approx(np.sqrt(3.))
    assert result["forecast_nrmse"] == pytest.approx((2+np.sqrt(3.))/3)
    pred[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        forecast_metrics(pred, target, scale)


def test_control_scale_matches_original_formula_without_candidate_stats():
    y = np.full((400, 4), 2.)
    u = np.zeros((400, 4))
    r = np.zeros((400, 4)); r[150:, 0] = 1
    scenario = {"r0": [0, 0, 0, 0], "r1": [1, 0, 0, 0], "t_step": 150}
    scale = np.array([2., 4., 1., 2.])
    result = control_metrics(y, u, r, scenario, scale, np.ones(4))
    m = np.ones(400, bool); m[:40] = False; m[150:190] = False
    expected = ((np.abs(y-r)[m]/scale).mean(0)*[1, 1, 0, .7]).sum()+.5*(1/2)
    assert result["control_cost"] == pytest.approx(expected)
    # Adding an extreme training episode cannot enter the score function.
    huge_training_spread = np.ones(4)*1000
    assert not np.array_equal(scale, huge_training_spread)
    assert control_metrics(y, u, r, scenario, scale, np.ones(4)) == result


def test_sampler_default_exactly_reproduces_reference_and_bounds():
    from wmf.actions import sample_identification_sequence, envelope
    from wmf.plants.schema import PlantConfig
    root = Path(__file__).resolve().parents[2]
    cfg = PlantConfig.from_yaml(root / "configs/plants/plant_a.yaml")
    assert actions(cfg, 12, 120, DEFAULT_POLICY) == sample_identification_sequence(cfg, np.random.default_rng(12), 120)
    p = copy.deepcopy(DEFAULT_POLICY)
    p["bounds_fraction"]["stoker_speed"] = [.8, 1.]
    p["slow_probability"]["stoker_speed"] = 1.
    seq = actions(cfg, 12, 120, p)
    env = envelope(cfg)
    assert all(.026-1e-8 <= a["stoker_speed"] <= .030 for a in seq)
    for a in seq:
        for k, value in a.items():
            assert np.all(np.asarray(value) >= env[k][0])
            assert np.all(np.asarray(value) <= env[k][1])


@pytest.mark.parametrize("mutate", [
    lambda p: p["bounds_fraction"].update(waste_feed=[-.1, 1]),
    lambda p: p["bounds_fraction"].update(waste_feed=[.8, .2]),
    lambda p: p["slow_probability"].update(primary_level=float("nan")),
    lambda p: p.update(physical_dt=1.),
])
def test_sampler_rejects_invalid_or_physics_edits(mutate):
    p = copy.deepcopy(DEFAULT_POLICY); mutate(p)
    with pytest.raises(ValueError):
        validate_policy(p)


def test_edit_manifest_rejects_additions_deletions_and_symlinks(tmp_path):
    (tmp_path / "worldmodel.py").write_text("old")
    (tmp_path / "feedback.json").write_text("{}"); before = tree_hashes(tmp_path)
    (tmp_path / "worldmodel.py").write_text("new")
    assert check_edits(before, tree_hashes(tmp_path), ["worldmodel.py"]) == ["worldmodel.py"]
    (tmp_path / "feedback.json").unlink()
    with pytest.raises(ValueError):
        check_edits(before, tree_hashes(tmp_path), ["worldmodel.py"])
    (tmp_path / "outside").symlink_to(tmp_path / "worldmodel.py")
    with pytest.raises(ValueError):
        tree_hashes(tmp_path)


def test_timeout_then_next_process_can_complete(tmp_path):
    failed = run_process([sys.executable, "-c", "import time; time.sleep(20)"], tmp_path,
                         tmp_path / "failed", .1)
    assert failed["status"] == "timeout"
    assert failed["wall_seconds"] < 5
    success = run_process([sys.executable, "-c", "print('recovered')"], tmp_path,
                          tmp_path / "next", 5)
    assert success["status"] == "ok"
    assert (tmp_path / "next/stdout.jsonl").read_text().strip() == "recovered"


def test_response_usage_deduplicated_and_cumulative_counts_ignored():
    def usage(rid, input_n, output_n):
        return {"type": "token_usage_record", "payload": {"response_id": rid, "session_id": "s", "turn_id": "t",
                "usage": {"input_tokens": input_n, "cached_input_tokens": input_n-2,
                          "output_tokens": output_n, "reasoning_output_tokens": 1},
                "turn_token_usage": {"total_tokens": 9000}}}
    rows = [{"type": "session_meta", "payload": {"id": "s"}}, usage("r1", 10, 3), usage("r1", 10, 3),
            {"type": "event_msg", "payload": {"type": "token_count", "info": {"total_tokens": 9000}}},
            usage("r2", 20, 4)]
    r = parse_sessions(rows)
    assert len(r["responses"]) == 2
    assert r["usage"]["total_tokens"] == 37
    assert r["usage"]["cached_input_tokens"] == 26
    assert r["usage"]["reasoning_output_tokens"] == 2
    assert r["responses"][0]["api_duration_seconds"] is None


def test_partial_jsonl_and_tool_duration(tmp_path):
    rows = [
        {"type": "session_meta", "payload": {"id": "s"}},
        {"type": "turn_context", "payload": {"turn_id": "t", "model": "gpt-6-astra", "effort": "low"}},
        {"type": "response_item", "timestamp": "2026-09-15T00:00:00Z", "payload":
         {"type": "custom_tool_call", "call_id": "c", "name": "exec", "input": "read"}},
        {"type": "response_item", "timestamp": "2026-09-15T00:00:02Z", "payload":
         {"type": "custom_tool_call_output", "call_id": "c", "output": "done"}},
        {"type": "response_item", "payload": {"type": "reasoning", "encrypted_content": "opaque"}},
    ]
    path = tmp_path / "partial.jsonl"
    path.write_text("\n".join(map(json.dumps, rows))+ '\n{"incomplete"')
    parsed, invalid = read_jsonl(path)
    result = parse_sessions(parsed)
    assert invalid == 1
    assert result["tool_calls"][0]["duration_seconds"] == 2
    assert result["tool_calls"][0]["turn_id"] == "t"
    assert "opaque" not in json.dumps(result)


def test_cli_fallback_does_not_pretend_response_granularity():
    r = parse_cli([{"type": "thread.started", "thread_id": "s"},
                   {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}}])
    assert r["thread_ids"] == ["s"]
    assert r["usage"]["total_tokens"] == 12
    assert r["usage_records"] == 1


def test_response_tool_links_mark_inference_and_do_not_cross_missing_usage():
    def item(kind, cid):
        return {"type": "response_item", "payload": {"type": kind, "call_id": cid, "name": "exec"}}

    def usage(rid):
        return {"type": "token_usage_record", "payload": {"session_id": "s", "turn_id": "t",
                "response_id": rid, "usage": {"input_tokens": 10, "output_tokens": 2}}}

    rows = [{"type": "session_meta", "payload": {"id": "s"}},
            {"type": "turn_context", "payload": {"turn_id": "t"}},
            item("function_call", "parallel1"), item("function_call", "parallel2"), usage("r1"),
            item("function_call_output", "parallel1"), item("function_call_output", "parallel2"),
            item("function_call", "missing"), usage("r1"),  # duplicate cannot claim a new call
            item("function_call_output", "missing"),
            item("function_call", "next"), usage("r2")]
    calls = {c["call_id"]: c for c in parse_sessions(rows)["tool_calls"]}
    assert calls["parallel1"]["response_id"] == calls["parallel2"]["response_id"] == "r1"
    assert calls["parallel1"]["response_link_method"] == "inferred_next_usage_before_tool_result"
    assert calls["missing"]["response_id"] is None
    assert calls["next"]["response_id"] == "r2"


def test_report_timeline_contains_proposal_and_acceptance(tmp_path):
    import csv
    from research.report import build_report
    write_json(tmp_path / "manifest.json", {"phase": "failed"})
    agent = tmp_path / "experiments/A_01/agent"
    agent.mkdir(parents=True)
    write_json(agent / "process.json", {"started_at": "2026-09-15T00:00:00Z"})
    events = [{"timestamp": "2026-09-15T00:00:05Z", "experiment_id": "A_01", "actor": "codex",
               "state": "completed", "phase": "proposal", "status": "ok", "seconds": 5.},
              {"timestamp": "2026-09-15T00:03:00Z", "experiment_id": "A_01", "actor": "harness",
               "state": "completed", "phase": "decision", "status": "ok", "decision": "keep", "objective": .8}]
    (tmp_path / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
    build_report(tmp_path, videos=False)
    with (tmp_path / "timeline.csv").open() as f:
        timeline = list(csv.DictReader(f))
    assert [r["name"] for r in timeline] == ["proposal", "decision"]
    assert timeline[0]["started_at"] == "2026-09-15T00:00:00Z"
    assert timeline[1]["decision"] == "keep" and timeline[1]["objective"] == "0.8"


def test_missing_usage_is_null_not_zero(tmp_path):
    assert all(v is None for v in parse_sessions([])["usage"].values())
    agent = tmp_path / "agent"
    agent.mkdir()
    result = collect(agent, "failed_agent", sessions_root=tmp_path / "sessions")
    assert result["usage_granularity"] == "unavailable"
    assert all(v is None for v in result["usage"].values())


def test_resource_export_retains_training_before_evaluation_failure(tmp_path):
    from research.report import resource_records
    directory = tmp_path / "experiments/A_01"
    write_json(directory / "train/result.json", {"episodes": 34, "training_seconds": 150.2,
               "training_updates_self_reported": 123, "num_parameters": 840})
    write_json(directory / "generation/result.json", {"episodes": [{"wall_seconds": 47.}, {"wall_seconds": 46.}]})
    rows = [{"id": "A_01", "arm": "A", "round": 1, "status": "timeout"}]
    calls = [{"experiment_id": "A_01", "source": "harness", "name": "evaluate", "duration_seconds": 240.}]
    r = resource_records(tmp_path, rows, [], calls)[0]
    assert r["training_episodes"] == 34 and r["optimizer_updates"] == 123
    assert r["simulator_generation_seconds"] == 93.
    assert r["evaluate_process_seconds"] == 240.
    assert r["input_tokens"] is None and r["proposal_process_seconds"] is None


def test_best_candidate_selection_uses_common_prefix_not_last_edit():
    e = Experiment.__new__(Experiment)
    e.baseline = {"id": "baseline", "objective": 1.}
    e.rows = [{"arm": "A", "id": "A_01", "round": 1, "status": "ok", "objective": .9},
              {"arm": "B", "id": "B_01", "round": 1, "status": "ok", "objective": .8},
              {"arm": "A", "id": "A_02", "round": 2, "status": "ok", "objective": .2}]
    n, selected = e.select()
    assert n == 1 and selected["A"]["id"] == "A_01"
    e.rows.append({"arm": "B", "id": "B_02", "round": 2, "status": "crash", "objective": None})
    n, selected = e.select()
    assert n == 2 and selected["B"]["id"] == "B_01"


def test_objective_never_selects_failure_as_zero():
    base = {"forecast_nrmse": 2., "control_cost": 4.}
    assert objective(1., 4., base) == .75
    for invalid in [None, float("nan"), float("inf"), -1]:
        with pytest.raises(ValueError):
            objective(invalid, 1., base)
    with pytest.raises(ValueError):
        objective(1., 1., {"forecast_nrmse": 0., "control_cost": 1.})


def test_failed_candidate_recovery_preserves_incumbent_and_generated_data(tmp_path):
    from types import SimpleNamespace
    e = Experiment.__new__(Experiment)
    e.root = tmp_path
    e.dev_deadline = time.monotonic()+100
    e.dev_seconds = 100
    e.arm_seconds = {"A": 0., "B": 0.}
    e.arm_data = {"A": [tmp_path / "base.npz"]}
    e.args = SimpleNamespace(episodes=1)
    e.baseline = {"id": "baseline", "objective": 1., "forecast_nrmse": 1., "control_cost": 1.}
    e.incumbent = {"A": e.baseline}
    seen, recorded = [], []
    e.verify_reference = lambda: None
    e.record = lambda r: recorded.append(r.copy())

    def edit(arm, number, directory, deadline):
        seen.append(e.incumbent[arm]["id"])
        source = directory / "source"; source.mkdir()
        (source / "worldmodel.py").write_text(f"candidate={number}")
        write_json(directory / "source_hashes.json", tree_hashes(source))
        return source, {"hypothesis": "test"}

    e.agent_edit = edit
    e.stage = lambda eid, phase, output, **kwargs: {"episodes": [{"file": str(output / "added.npz")}]}

    def fit(eid, source, files, directory, deadline):
        if eid == "A_01":
            raise StageFailure("timeout", "train", "injected timeout")
        return {"episodes": len(files)}, {"forecast_nrmse": .8, "control_cost": .8}

    e.fit_and_evaluate = fit
    e.round("A", 1)
    assert e.incumbent["A"]["id"] == "baseline"
    assert recorded[0]["objective"] is None
    e.round("A", 2)
    assert seen == ["baseline", "baseline"]
    assert e.incumbent["A"]["id"] == "A_02"
    assert len(e.arm_data["A"]) == 3
    assert recorded[1]["training"]["episodes"] == 3


def test_codex_command_pins_model_and_sandbox(tmp_path):
    args = codex_argv("gpt-6-astra", "low", tmp_path)
    assert args[args.index("--model")+1] == "gpt-6-astra"
    assert args[args.index("--sandbox")+1] == "workspace-write"
    assert 'model_reasoning_effort="low"' in args
    assert "--ephemeral" not in args  # sessions must be saved
    assert "--dangerously-bypass-approvals-and-sandbox" not in args
