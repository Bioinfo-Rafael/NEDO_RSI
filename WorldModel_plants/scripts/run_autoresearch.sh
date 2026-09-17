#!/usr/bin/env bash
# Launch the autonomous experiment loop.
#
# Safety, deliberately not configurable from here:
#   * sandbox = workspace-write  - Codex may write inside this repo and nowhere
#     else, and has no network. It is NOT run with --dangerously-bypass-*.
#   * a dedicated git branch     - main is never the working branch, so every
#     change the agent makes is reviewable and revertible.
#   * no API key                 - Codex authenticates with the ChatGPT
#     subscription (~/.codex/auth.json, outside this repo). There is no
#     credential in the working tree for it to read or leak, and no per-token
#     billing to run up.
#
# The agent is told in program.md not to touch src/wmf or data/. The branch is
# the backstop if it does anyway.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-$(date +%b%d | tr 'A-Z' 'a-z')}"
BRANCH="autoresearch/${TAG}"
LOG="autoresearch_${TAG}.jsonl"

if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  echo "branch $BRANCH already exists - pick another tag" >&2; exit 1
fi
if [ ! -f data/prepared.npz ] && [ -z "$(ls data/train/*.npz 2>/dev/null)" ]; then
  echo "no dataset - run scripts/gen_dataset.py first" >&2; exit 1
fi

git checkout -q -b "$BRANCH"
echo "branch:  $BRANCH"
echo "log:     $LOG"
echo "budget:  $(grep -o 'TRAIN_SECONDS = [0-9.]*' integration/prepare.py) s per experiment"
echo

codex exec \
  --sandbox workspace-write \
  --cd "$PWD" \
  --json \
  "Read integration/program.md and run the experiment loop it describes. \
The run tag is ${TAG} and the branch ${BRANCH} is already checked out, so skip \
the branch-creation step. Do not modify anything outside integration/." \
  < /dev/null 2>&1 | tee "$LOG"      # close stdin so exec does not wait on it
