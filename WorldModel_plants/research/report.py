"""Offline scientific plots and a local report from immutable run artifacts."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.common import read_json, write_json, sha256, utc_now
from research.telemetry import TOKEN_KEYS, read_jsonl, parse_sessions
from research.data import DEFAULT_POLICY

COLORS = {"baseline": "#64748b", "A": "#3173b9", "B": "#ce7140"}


def fmt(value):
    return "—" if value is None else f"{value:.6f}"


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def resource_records(root, rows, tokens, calls):
    """Join measured costs by experiment, including work done before failures."""
    records = []
    usage_by_id = {r["experiment_id"]: r for r in tokens}
    for r in rows:
        directory = Path(root) / "experiments" / r["id"]
        training = r.get("training", {})
        if not training and (directory / "train/result.json").exists():
            training = read_json(directory / "train/result.json")
        generated = r.get("generation", {})
        if not generated and (directory / "generation/result.json").exists():
            generated = read_json(directory / "generation/result.json")
        usage = usage_by_id.get(r["id"], {})
        phases = [c for c in calls if c.get("experiment_id") == r["id"] and
                  c.get("source") == "harness" and c.get("name") != "hypothesis"]
        record = {"id": r["id"], "arm": r["arm"], "round": r["round"], "status": r["status"],
                  "training_episodes": training.get("episodes"),
                  "training_seconds": training.get("training_seconds"),
                  "optimizer_updates": training.get("training_updates_self_reported"),
                  "window_batches_observed": training.get("training_window_batches_observed"),
                  "num_parameters": training.get("num_parameters"),
                  "added_episodes": len(generated["episodes"]) if "episodes" in generated else None,
                  "simulator_generation_seconds": sum(x["wall_seconds"] for x in generated["episodes"])
                  if "episodes" in generated else None,
                  "input_tokens": usage.get("input_tokens"), "cached_input_tokens": usage.get("cached_input_tokens"),
                  "output_tokens": usage.get("output_tokens"),
                  "action_coverage_json": json.dumps(training.get("action_coverage")) if training else None,
                  "sampling_policy_json": json.dumps(r.get("sampling_policy")) if r.get("sampling_policy") else None}
        for name in ("proposal", "generate", "train", "evaluate"):
            durations = [c["duration_seconds"] for c in phases if c.get("name") == name and
                         c.get("duration_seconds") is not None]
            record[name+"_process_seconds"] = sum(durations) if durations else None
        records.append(record)
    return records


def make_video(trace_path, output, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    with np.load(trace_path, allow_pickle=False) as z:
        heat, indices, y, r = z["heat"], z["heat_steps"], z["y"], z["r"]
        dt = float(z["control_interval"])
    if not len(heat):
        return
    selected = np.unique(np.linspace(0, len(heat)-1, min(100, len(heat))).astype(int))
    fig = plt.figure(figsize=(10, 6), dpi=90, constrained_layout=True)
    grid = fig.add_gridspec(4, 2, width_ratios=[1.1, 1.5])
    ax = fig.add_subplot(grid[:, 0])
    im = ax.imshow(heat[0], origin="lower", cmap="inferno", vmin=300, vmax=1800)
    ax.set(xticks=[], yticks=[], title="Simulator temperature [K]")
    fig.colorbar(im, ax=ax, shrink=.65)
    lines = []
    labels = ["Exit T [K]", "O2 [-]", "Flue unburnt [-]", "Bed unburnt"]
    tt = (np.arange(len(y))+1)*dt
    for ch in range(4):
        a = fig.add_subplot(grid[ch, 1])
        a.plot(tt, r[:, ch], "--", color="#cc4b3d", label="Target")
        ln, = a.plot([], [], color="#176d9c", label="Simulator")
        lower, upper = min(y[:, ch].min(), r[:, ch].min()), max(y[:, ch].max(), r[:, ch].max())
        margin = max((upper-lower)*.1, 1e-7)
        a.set(xlim=(0, tt[-1]), ylim=(lower-margin, upper+margin), ylabel=labels[ch])
        a.grid(alpha=.2)
        if ch == 0:
            a.legend(loc="best", fontsize=7)
        if ch == 3:
            a.set_xlabel("Simulated seconds")
        lines.append(ln)
    frames = []
    for i in selected:
        k = int(indices[i])
        im.set_data(heat[i])
        for ch, ln in enumerate(lines):
            ln.set_data(tt[:k+1], y[:k+1, ch])
        fig.suptitle(f"{label} | t = {tt[k]:.1f} s | simulated plant, fixed MPC")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB"))
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=100, loop=0)
    plt.close(fig)


def make_control_plot(root, evaluations, scenario, filename, title):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for label, e in evaluations:
        path = Path(e["directory"]) / f"control_{scenario}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            tt = (np.arange(len(z["y"]))+1) * float(z["control_interval"])
            for ch, ax in enumerate(axes):
                ax.plot(tt, z["y"][:, ch], label=label, color=COLORS[label], alpha=.85)
                if label == evaluations[0][0]:
                    ax.plot(tt, z["r"][:, ch], "k--", label="Target")
    for ch, ax in enumerate(axes):
        ax.set_ylabel(["Exit T [K]", "O2 [-]", "Flue unburnt [-]", "Bed unburnt"][ch])
        ax.grid(alpha=.2)
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel("Simulated seconds")
    fig.suptitle(title)
    fig.savefig(Path(root) / filename)
    plt.close(fig)


def build_report(root, videos=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    rows, _ = read_jsonl(root / "results.jsonl")
    final = read_json(root / "final.json") if (root / "final.json").exists() else {}
    events, _ = read_jsonl(root / "events.jsonl")
    telemetry, token_rows, response_rows, call_rows = [], [], [], []
    # Only this run's agent logs. Never scan reference snapshots or unrelated sessions.
    paths = sorted((root / "experiments").glob("*/agent/telemetry.json"))
    if (root / "preflight/agent/telemetry.json").exists():
        paths.insert(0, root / "preflight/agent/telemetry.json")
    for path in paths:
        t = read_json(path)
        telemetry.append(t)
        token_rows.append({"experiment_id": t["experiment_id"], "granularity": t["usage_granularity"],
                           "responses": len(t["responses"]), **t["usage"]})
        for r in t["responses"]:
            response_rows.append({k: v for k, v in r.items() if k != "usage"} | r["usage"])
        # Reprocess only raw logs already saved by this experiment. This allows
        # richer analysis without changing frozen evaluations or original logs.
        saved = []
        for raw in sorted((path.parent / "raw_sessions").glob("*.jsonl")):
            records, _ = read_jsonl(raw)
            saved.extend(records)
        tool_calls = parse_sessions(saved)["tool_calls"] if saved else t["tool_calls"]
        for c in tool_calls:
            c = {**c, "experiment_id": t["experiment_id"]}
            call_rows.append({k: c.get(k) for k in ("experiment_id", "session_id", "turn_id", "call_id", "name",
                             "response_id", "response_link_method", "started_at", "completed_at", "duration_seconds",
                             "source", "actor", "status", "exit_code")})
    tool_starts = {}
    for e in events:
        key = e.get("call_id")
        if e.get("state") == "started" and key:
            tool_starts[key] = e
        elif e.get("state") == "completed" and key:
            call_rows.append({"experiment_id": e.get("experiment_id"), "call_id": key,
                              "name": e.get("phase"), "actor": e.get("actor"), "status": e.get("status"),
                              "started_at": tool_starts.get(key, {}).get("timestamp"),
                              "completed_at": e["timestamp"], "duration_seconds": e.get("seconds"),
                              "source": "harness"})
        elif e.get("state") == "completed" and e.get("phase") in ("proposal", "hypothesis", "decision"):
            proc_path = root / "experiments" / e["experiment_id"] / "agent/process.json"
            proc = read_json(proc_path) if e["phase"] == "proposal" and proc_path.exists() else {}
            call_rows.append({"experiment_id": e["experiment_id"], "name": e["phase"],
                              "actor": e.get("actor"), "status": e.get("status"),
                              "decision": e.get("decision"), "objective": e.get("objective"),
                              "changed_files": json.dumps(e["changed"]) if "changed" in e else None,
                              "started_at": proc.get("started_at", e["timestamp"]),
                              "completed_at": e["timestamp"], "duration_seconds": e.get("seconds", 0.),
                              "source": "harness"})
    call_rows.sort(key=lambda row: row.get("started_at") or "")
    failures, recovered = 0, 0
    for arm in ("A", "B"):
        rr = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["round"])
        for previous, following in zip(rr, rr[1:]):
            if previous["status"] != "ok":
                failures += 1
                recovered += following["status"] == "ok"
    directed = [r["id"] for r in rows if r["arm"] == "B" and r.get("sampling_policy") and
                r["sampling_policy"] != DEFAULT_POLICY and r.get("generation", {}).get("episodes")]
    analysis = {"usage": {k: sum(r[k] for r in token_rows if r[k] is not None) for k in TOKEN_KEYS},
                "usage_missing_experiments": [r["experiment_id"] for r in token_rows if r["granularity"] == "unavailable"],
                "recovery": {"failed_attempts_with_successor": failures, "next_attempt_ok": recovered,
                             "rate": recovered/failures if failures else None},
                "response_records": len(response_rows), "tool_calls": call_rows,
                "model_records": [m for t in telemetry for m in t["models"]],
                "B_generated_nondefault_policy_trials": directed,
                "postprocessing": {"at": utc_now(), "report_sha256": sha256(__file__),
                                   "telemetry_sha256": sha256(Path(__file__).with_name("telemetry.py"))},
                "limitations": ["API request/retry latency unavailable", "No hidden reasoning reconstruction",
                                "Tool-response links inferred from event order are labeled; unavailable links remain null",
                                "Tool-selection accuracy requires an external labeling rubric",
                                "Single A/B run; no confidence intervals or scaling-law fit"]}
    write_json(root / "analysis.json", analysis)
    write_csv(root / "tokens.csv", token_rows, ["experiment_id", "granularity", "responses", *TOKEN_KEYS])
    write_csv(root / "responses.csv", response_rows, ["experiment_id", "session_id", "turn_id", "response_id",
                                                     "timestamp", *TOKEN_KEYS, "api_duration_seconds"])
    write_csv(root / "timeline.csv", call_rows, ["experiment_id", "actor", "name", "call_id", "session_id", "turn_id",
                                               "response_id", "response_link_method", "started_at", "completed_at",
                                               "duration_seconds", "status", "decision", "objective", "changed_files", "source"])
    write_csv(root / "comparison.csv", rows, ["id", "arm", "round", "status", "decision", "objective",
                                             "forecast_nrmse", "control_cost", "wall_seconds", "error"])
    resources = resource_records(root, rows, token_rows, call_rows)
    write_csv(root / "resources.csv", resources, list(resources[0]) if resources else ["id", "status"])
    measured = [("validation", r["arm"], r["id"], r.get("metrics", {})) for r in rows if r["status"] == "ok"]
    measured += [("heldout", arm, final["selected"][arm], e["metrics"])
                 for arm, e in final.get("evaluations", {}).items() if e["status"] == "ok"]
    forecasts, controls = [], []
    for split, arm, eid, metric in measured:
        for horizon, values in metric.get("forecast", {}).get("horizons", {}).items():
            for channel, rmse in values["rmse"].items():
                forecasts.append({"split": split, "arm": arm, "id": eid, "horizon_steps": horizon,
                                  "channel": channel, "rmse": rmse, "nrmse": values["nrmse"][channel]})
        for scenario, values in metric.get("control", {}).items():
            if not isinstance(values, dict):
                continue
            controls.append({"split": split, "arm": arm, "id": eid, "scenario": scenario,
                             **{k: values[k] for k in ("control_cost", "tracking", "overshoot", "move")},
                             **{f"iae_{k}": v for k, v in values["iae_per_channel"].items()}})
    write_csv(root / "forecast.csv", forecasts, ["split", "arm", "id", "horizon_steps", "channel", "rmse", "nrmse"])
    write_csv(root / "control.csv", controls, list(controls[0]) if controls else ["split", "arm", "id", "scenario"])
    plots = []
    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
        baseline = next((r for r in rows if r["arm"] == "baseline"), None)
        for ax, key, label in zip(axes, ["forecast_nrmse", "control_cost", "objective"],
                                   ["Forecast NRMSE", "Control cost", "Joint validation objective"]):
            for arm, color in [("A", "#3173b9"), ("B", "#ce7140")]:
                values = ([baseline] if baseline else []) + [r for r in rows if r["arm"] == arm]
                ax.plot([r["round"] for r in values],
                        [r.get(key) if r.get(key) is not None else np.nan for r in values],
                        "o-", label=arm, color=color)
            ax.set(xlabel="Proposal round (failed trials retained)", ylabel=label)
            ax.grid(alpha=.2); ax.legend()
        fig.suptitle("Validation candidates — used for selection; not held-out evidence")
        for suffix in ("png", "svg"):
            fig.savefig(root / f"validation.{suffix}")
        plt.close(fig)
        plots.append("validation.png")
        es = final.get("evaluations", {})
        good = [(k, e) for k, e in es.items() if e["status"] == "ok"]
        if good:
            fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
            for label, e in good:
                m = e["metrics"]
                ax.scatter(m["forecast_nrmse"], m["control_cost"], s=65, color=COLORS[label])
                ax.annotate(label, (m["forecast_nrmse"], m["control_cost"]), xytext=(6, 6), textcoords="offset points")
            ax.set(xlabel="Held-out forecast NRMSE (lower is better)", ylabel="Held-out control cost (lower is better)",
                   title="Final evaluation — selected before test data generation")
            ax.grid(alpha=.2); fig.savefig(root / "heldout.png"); plt.close(fig)
            plots.append("heldout.png")
            for scenario, filename, title in [
                ("temp_step", "control_comparison.png", "Held-out reverse temperature step | fixed MPC"),
                ("o2_step", "control_o2_comparison.png", "Held-out reverse oxygen step | fixed MPC"),
            ]:
                make_control_plot(root, good, scenario, filename, title)
                plots.append(filename)
    heading = "Smoke test — 科学的な性能比較には使用しない" if manifest.get("smoke_only") else "固定LLM World Model 比較実験"
    text = [f"# {heading}", "", f"実行状態: {manifest.get('phase', 'unknown')}",
            f"モデル指定: `{manifest.get('config', {}).get('model')}` / `{manifest.get('config', {}).get('effort')}`。",
            f"比較対象の共通ラウンド数: {final.get('common_rounds', '未確定')}。", "",
            "## 開発中の結果", "", "この表は候補選択に使ったvalidation。失敗した候補のスコアは欠損値。", "",
            "| 実験 | 状態 | 採否 | 予測NRMSE | 制御cost | 選択目的関数 |", "|---|---|---|---:|---:|---:|"]
    for r in rows:
        text.append(f"| {r['id']} | {r['status']} | {r.get('decision')} | {fmt(r.get('forecast_nrmse'))} | {fmt(r.get('control_cost'))} | {fmt(r.get('objective'))} |")
    text += ["", "## 未使用条件での最終評価", "", "| 条件 | 選ばれた候補 | 予測NRMSE | 制御cost | 初期モデルから両方改善 |", "|---|---|---:|---:|---|"]
    for label, e in final.get("evaluations", {}).items():
        m = e.get("metrics", {})
        joint = final.get("joint_improvement", {}).get(label)
        text.append(f"| {label} | {final['selected'][label]} | {fmt(m.get('forecast_nrmse'))} | {fmt(m.get('control_cost'))} | {joint if joint is not None else '—'} |")
    if final.get("evaluations"):
        text += ["", "制御costのシナリオ別内訳（小さいほどよい）。成功判定は上表の平均に対するもので、各条件・各出力の改善を保証しない。", "",
                 "| シナリオ | 初期モデル | A | B |", "|---|---:|---:|---:|"]
        for scenario in ("temp_step", "o2_step"):
            values = [final["evaluations"].get(arm, {}).get("metrics", {}).get("control", {}).get(scenario, {}).get("control_cost")
                      for arm in ("baseline", "A", "B")]
            text.append(f"| {scenario} | " + " | ".join(map(fmt, values)) + " |")
    text += ["", f"BがAを両指標で上回ったか: **{final.get('B_dominates_A', '未評価')}**。",
             "数値は1回の比較の結果。改善の再現性・統計的有意性・スケーリング則は未検証。", "",
             f"Bで標準と異なる生成条件を実際に使った試行: {directed or 'なし'}。",
             "該当試行がない場合、このrunから診断に基づくデータ生成の優位性は判断できない。モデルや学習方法の改善とは分けて解釈する。", "",
             "## Agentと計算の記録", "", f"応答単位のusage記録: {analysis['response_records']}件。",
             "集計範囲はrunnerが呼んだpreflightと各提案。実装・監視の親チャットや他のチャットは含まない。",
             f"入力token: {analysis['usage']['input_tokens']:,}（うちcache: {analysis['usage']['cached_input_tokens']:,}）。出力token: {analysis['usage']['output_tokens']:,}。",
             f"上記は取得できたusageの合計。usage未取得の実験: {analysis['usage_missing_experiments']}。",
             "cacheは入力の内数、reasoningは出力の内数。再加算しない。API通信時間・全再試行回数はこの記録からは取得できない。",
             "toolと応答IDの対応は、ログのイベント順から推定した場合にresponse_link_methodへ明記する。対応不明は空欄。",
             f"失敗後の次回試行成功: {recovered}/{failures}（次回が存在する失敗のみ）。", "",
             "[実験比較CSV](comparison.csv) / [計算時間・データ・学習更新数](resources.csv) / [実験別token](tokens.csv) / [応答別token](responses.csv) / [処理タイムライン](timeline.csv) / [集計JSON](analysis.json)", "",
             "[出力・予測区間別RMSE/NRMSE](forecast.csv) / [シナリオ別の追従・overshoot・操作変化](control.csv)", "",
             "## 仮説と変更", ""]
    for r in rows:
        if r["arm"] == "baseline":
            continue
        h = r.get("hypothesis", {})
        text += [f"### {r['id']}", "", f"仮説: {h.get('hypothesis', '取得できず')}",
                 f"変更: {h.get('change', '—')}", f"結果: {r.get('error', r['decision'])}",
                 f"[差分](experiments/{r['id']}/changes.diff) / [実験JSON](experiments/{r['id']}/experiment.json)", ""]
    if (root / "run_error.json").exists():
        text += ["## 実行エラー", "", read_json(root / "run_error.json")["error"], ""]
    if final.get("error"):
        text += ["最終評価エラー: " + final["error"], ""]
    video_links = []
    seen = set()
    for label, e in final.get("evaluations", {}).items():
        if e.get("status") != "ok":
            continue
        selected = final["selected"][label]
        name = f"control_{selected}.gif"
        if selected not in seen and videos and not (root / name).exists():
            make_video(Path(e["directory"]) / "control_temp_step.npz", root / name, selected)
        seen.add(selected)
        if (root / name).exists():
            video_links.append((label, name))
    text += ["## グラフと動画", ""]
    text.extend(f"![{p}]({p})\n" for p in plots)
    text.extend(f"- [{label}の制御動画]({p})" for label, p in video_links)
    (root / "report.md").write_text("\n".join(text)+"\n")
    # A self-contained local index; markdown remains the primary readable report.
    links = " ".join(f'<a href="{html.escape(name)}">{html.escape(label)} GIF</a>' for label, name in video_links)
    document = ('<!doctype html><meta charset="utf-8"><title>World Model experiment</title>'
                '<style>body{font:15px system-ui;max-width:1150px;margin:40px auto;padding:0 24px;background:#fafbfc;color:#1e293b}'
                'pre{white-space:pre-wrap;line-height:1.6;background:white;padding:24px;border:1px solid #dde3eb}'
                'img{max-width:100%;background:white}a{color:#176d9c}</style>'
                f'<h1>{html.escape(heading)}</h1><p>{links}</p>' +
                ''.join(f'<img src="{html.escape(p)}" alt="{html.escape(p)}">' for p in plots) +
                '<pre>' + html.escape("\n".join(text[:text.index("## グラフと動画")])) + '</pre>')
    (root / "report.html").write_text(document)
    return root / "report.md"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--no-videos", action="store_true")
    args = ap.parse_args()
    print(build_report(args.root, videos=not args.no_videos))
