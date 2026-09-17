# Fixed-LLM world-model research

You are the coding/research agent for ONE condition and ONE proposal round.
Read `file_manifest.json`, `INTERFACES.md`, `feedback.json`, and `ResearchState.md`.
Use only files in this workspace. The outer runner owns simulation, training,
evaluation, budgets, acceptance and immutable artifacts.

## Goal and feedback

Improve BOTH held-out multi-step forecasting and closed-loop control with a fixed
MPC controller. Validation selection minimizes
`0.5 * forecast_nrmse / baseline_forecast_nrmse + 0.5 * control_cost / baseline_control_cost`.
Neither prediction accuracy alone nor matching recorded actions is sufficient.
Validation diagnostics are the only experiment feedback. Held-out results are
produced after all proposals have ended and must not be consulted during research.

The four outputs are exit temperature, oxygen, unburnt fuel in gas, and unburnt
solids leaving the bed. The simulator is synthetic, not a calibrated real furnace.
Gas response is fast; grate residence is roughly 400–1200 control steps. The
training data consist of 1200-step episodes with post-action measurements.

## Editable files

- `worldmodel.py`: architecture, loss, optimizer, history and training method.
- `pipeline.py`: training-data use and training workflow. Return ensemble params.
- `sampling.json`: editable only in condition B. Choose the aggregate action
  ranges and slow/fast excitation mixture to investigate poor operating conditions.
  Condition A always uses the original random APRBS distribution.
- `hypothesis.json`: required proposal, written before any experiment is run.
- `ResearchState.md`: compact evidence, rejected hypotheses and next questions.

Make one coherent hypothesis per round. Prefer a change whose expected outcome
can be tested using the supplied diagnostics. No particular architecture is
required. State why the change may affect both forecasting and control; record
uncertainty rather than claiming results you have not measured.

In condition B, explicitly choose and justify the next sampling policy using the
diagnostics. You may retain broad sampling if the evidence calls for it; report
that choice honestly. Do not narrow all actions without checking coverage.

## Procedure

1. Read the current and historical validation results and the current source.
2. Identify one concrete failure mode and propose a change.
3. Edit only files permitted by `file_manifest.json`. You may perform a syntax
   check. Do not train models, generate simulator data or run evaluations here:
   the outer runner executes those tools once per proposal under identical budgets.
4. Write `hypothesis.json` with nonempty strings `hypothesis`, `change`, and
   `expected_effect`. Update `ResearchState.md` concisely.
5. Finish with a short description of the proposed experiment. Do not wait for
   user input and do not claim an improvement before the runner has measured it.

After the turn, the runner validates edits, generates exactly the assigned
number of episodes, trains, evaluates, and records keep/discard/crash/timeout.
Each new round begins from the best accepted model source. Successfully generated
training episodes accumulate even when the model proposal is rejected; the
historical failures and research memory also remain available.

## Constraints

- No network, package installation, other LLMs, subagents or expert consultation.
- Do not read parent directories, other conditions, old experiment solutions,
  global Codex sessions, hidden/test data or the original checkout. Do not invoke git.
- Do not edit reference simulation, MPC, metric definitions, scale, data files or
  run budgets. File hashes are checked outside this workspace.
- Train only on supplied `data`; do not load training/evaluation data from paths.
- Preserve the model API. Fixed stats are shared by forecasting and MPC. Do not
  mutate them. Keep results finite and arbitrary prediction horizons supported.
- Use JAX, NumPy and the Python standard library available in the supplied runtime.
- Record actual optimizer updates as `TRAINING_UPDATES`, reset for each fit, and
  synchronize asynchronous JAX computations within your timed training loop.
  Closed-form fits may report zero optimizer updates. Do not invent counts.
- Respect the supplied training seconds. The outer process has an additional
  finite timeout; computationally excessive proposals are failed trials.
