#!/usr/bin/env bash
# Launch the autonomous world-model loop in control_research/ (scored by control
# performance, not prediction error).
#
# Safety, deliberately not configurable from here:
#   * sandbox = workspace-write  - Codex may write inside this repo and nowhere
#     else, and has no network. NOT run with --dangerously-bypass-*.
#   * a dedicated git branch     - main is never the working branch.
#   * no API key                 - Codex authenticates with the ChatGPT
#     subscription (~/.codex/auth.json, outside this repo). No credential in the
#     working tree, no per-token billing.
#
# program.md tells the agent to edit only control_research/worldmodel.py. The
# branch is the backstop if it does anyway.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-$(date +%b%d | tr 'A-Z' 'a-z')}"
MAX_EXPERIMENTS="${2:-8}"
BRANCH="control/${TAG}"
LOG="autoresearch_control_${TAG}.jsonl"

if [ -z "$(ls data_ctrl_v2/train/*.npz 2>/dev/null)" ]; then
  echo "no identification data - run scripts/gen_control_data.py first" >&2; exit 1
fi
for f in figs/setpoints.json figs/pi_gains.json; do
  [ -f "$f" ] || { echo "missing $f - run scripts/pick_setpoints.py / identify_pi.py" >&2; exit 1; }
done

if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout -q "$BRANCH"
else
  git checkout -q -b "$BRANCH"
fi
echo "branch:  $BRANCH"
echo "log:     $LOG"
echo "budget:  $(grep -o 'TRAIN_SECONDS = [0-9.]*' control_research/prepare.py) s training per experiment"
echo "stop:    after ${MAX_EXPERIMENTS} experiments"
echo

codex exec \
  --sandbox workspace-write \
  --cd "$PWD" \
  --json \
  "Read control_research/program.md and run the experiment loop it describes. \
The run tag is ${TAG}; the branch ${BRANCH} is already checked out, so skip the \
branch-creation step. Prefix every commit subject and every results.tsv \
description with '${TAG} k/${MAX_EXPERIMENTS}:' where k is the experiment number. \
Stop after ${MAX_EXPERIMENTS} experiments and write a short summary of what you \
tried, what moved control_cost and what did not, to control_research/SUMMARY.md. \
Do not modify anything outside control_research/." \
  < /dev/null 2>&1 | tee "$LOG"
