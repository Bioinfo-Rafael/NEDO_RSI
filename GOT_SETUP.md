# Graph of Trace — NEDO_RSI local setup

## Start

```bash
cd /Users/cls-lab/Git/NEDO_RSI
./codex-got.sh login  # first use: this isolated Codex home is not logged in
./codex-got.sh
# In another terminal:
./got-viewer.sh
```

Open http://127.0.0.1:4500 . In Codex, `/mcp` shows `graph_of_trace` with `build_trace`, `append_trace_node`, and `get_trace`.
Codex starts and stops its stdio MCP subprocess automatically. No separate server is required.
The wrapper selects `.codex-got`; it does not copy credentials or change `~/.codex`.
MCP configuration is in `.codex-got/config.toml`. This dedicated home avoids modifying global project trust/configuration.

## API-free recording (current mode)

Use `append_trace_node` and `get_trace`; Codex authors the nodes and GoT only validates and stores them.
No extraction API key is required. Legacy `build_trace` remains exposed for compatibility and still needs an extraction provider; do not call it in API-free mode.
The dedicated `.codex-got/config.toml` loads `GOT_CODEX_INSTRUCTIONS.md` through its developer instructions.
The rules keep run/project IDs stable, record meaningful completed steps, and choose baseline/sibling/multiple-experiment parents correctly.

## Runs and viewer

Outputs: `Graph-of-Trace/runs/<project>/<session>/got.json`.
The server wrapper pins this path, and local.yaml also sets it for direct server launches.

```bash
./got-viewer.sh my-project my-run
# Optional standalone HTTP MCP server (Codex wrapper uses stdio instead):
./got-server.sh --transport streamable-http --host 127.0.0.1 --port 8000
```

Viewer binds only to 127.0.0.1:4500, serves the built frontend and runs, and stops with Ctrl+C.
The default viewer displays `nedo-rsi-test/smoke-test`.
Use one stable project/session pair per run. `GOT_CODEX_INSTRUCTIONS.md` is loaded by the dedicated Codex configuration to guide recording with `append_trace_node`. Recording covers completed, verifiable steps, not every raw command. No production experiments have been started.

## Validation and limitations

- Python 3.14.5; Node 26.7.0; npm 11.19.0; Codex 0.154.0.
- Editable graph-of-trace 0.1.0 in `Graph-of-Trace/.venv`.
- MCP SDK pinned to 1.30.0: upstream `mcp>=1.0` otherwise installs incompatible MCP 2.x.
- Viewer `npm run build` passed; Vite 5.4.21 requires Node ^18 or >=20.
- Full Python suite: 36 passed (includes original tests and 17 direct-mode cases).
- Actual Codex app-server discovered all three MCP tools; evidence in `.cache/codex-direct-mcp-status.json`.
- New direct MCP smoke uses real caller-authored nodes and the real writer: no LLM stub, API keys, or network access. Test harness forbids internet sockets and extraction-module imports.
- `runs/nedo-rsi-direct/smoke-test/got.json` contains Baseline (N001), Ablation A/B (N002/N003), Compare A/B (N004 with both parents).
- Actual frontend converter and Dagre layout passed: 4 nodes with sibling branches and comparison merge. Browser visual inspection remains unavailable due to computer/browser permissions.
- The original `runs/nedo-rsi-test/smoke-test/got.json` is retained as the earlier legacy smoke fixture.
- GoT frontend, global settings and existing research code were not modified. Only direct-mode backend additions, tests, docs and dedicated Codex configuration changed.
- NEDO_RSI itself currently has no `.git`. If it is later initialized, exclude `.codex-got/` in its `.git/info/exclude` before staging files.
- Upstream `scripts/start.sh` writes into frontend/public by default and uses `wait -n`, unsupported by macOS stock Bash 3.2. The local wrappers avoid both issues.

## Reinstall dependencies

```bash
cd Graph-of-Trace
.venv/bin/python -m pip install --cache-dir .cache/pip -c local-setup/constraints.txt -e . pytest
cd frontend
npm ci --cache ../.cache/npm --no-audit --no-fund
npm run build
```

View the new direct DAG:

```bash
./got-viewer.sh nedo-rsi-direct smoke-test
```

URL: http://127.0.0.1:4500/?src=/runs/nedo-rsi-direct/smoke-test/got.json

Test evidence: `Graph-of-Trace/.cache/codex-direct-mcp-status.json` and `Graph-of-Trace/runs/nedo-rsi-direct/smoke-test/got.json`.
Tool JSON schemas: `Graph-of-Trace/.cache/direct-tool-schemas.json`.
The smoke script refuses to overwrite an existing test trace.

References: https://github.com/NeuroAIHub/Graph-of-Trace and https://learn.chatgpt.com/docs/extend/mcp?surface=cli .
