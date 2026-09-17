# Codex-authored Graph of Trace (no extraction API)

Use `graph_of_trace.append_trace_node` to record completed work and `get_trace` to
inspect the graph. Do not call legacy `build_trace`: it uses an external extraction
LLM. Codex determines the node content and dependencies itself. No API key is needed.

- Do not start experiments merely because these instructions are loaded. Follow the user's task scope.
- At the start of an authorized autoresearch run choose a fixed `project.name` and
  `session.id`. Reuse both for the entire run. A different run gets a new session.
- After a completed experiment, substantive implementation, figure, comparison or
  conclusion, append one node promptly. Do not record planning, chain-of-thought,
  code typo fixes, minor debugging or trivial reruns.
- Record actual results only. Include principal metrics, configuration/seed and
  artifact paths in `description`; also list files in `artifacts` using paths
  relative to NEDO_RSI (or absolute paths when necessary). Never include secrets.
- Inspect `get_trace` when entering/resuming a session or whenever parent IDs are
  uncertain. IDs are allocated by GoT; never guess the next ID or supply `node.id`.
- For the first node send `parents: []`; GoT makes it N001 with the schema's root
  self-reference. Every subsequent node must name at least one existing parent.
- Ablations derived from a baseline use that baseline as parent. Parallel variants
  are siblings sharing the baseline; never chain A → B merely because B ran later.
- Comparison and conclusion nodes name all experiments/analyses logically required
  for the result. E.g. Compare A/B has both Ablation A and Ablation B as parents.
- Use relation `necessitated_by`, with a short explanation for each dependency.
- Save each returned ID. Record each deliverable once. After a timeout or uncertain
  response, call `get_trace` and check whether the node exists before retrying.
- Report failed recording honestly; never treat an MCP error as successful storage.

Both tools take a top-level `payload` argument. Example root call:

```json
{"payload":{"project":{"name":"nedo-rsi"},"session":{"id":"run-001"},"node":{"title":"Baseline","description":"Describe the completed run, measured metrics, seed and artifact paths.","status":"completed","parents":[],"artifacts":[]}}}
```

Read existing nodes:

```json
{"payload":{"project":{"name":"nedo-rsi"},"session":{"id":"run-001"}}}
```
