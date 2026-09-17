#!/bin/sh
# ORIGIN: ORIGINAL
# IMPLEMENTATION_NOTE: Bounded real-data demo; all artifacts under outputs/.
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
DEMO_RUN='openevolve::MyOpenEvolve::21d972d0-0b7d-49dd-8f88-f98a26dbc6ee'
python3 -m meta_search_rsi.cli tree --run-id "$DEMO_RUN"
python3 -m meta_search_rsi.cli build-memory --db ../experience_store/experience.db \
  --run-id run_4a1b8ad7c89c78d032bf0310 \
  --run-id run_56709ea154983586c368ff29 \
  --run-id run_660e45cf1cdc2d1538b777eb
python3 -m meta_search_rsi.cli retrieve --query 'evaluate model optimization failure loss' --structure tree
python3 -m meta_search_rsi.cli replay --run-id "$DEMO_RUN" --policy meta_memory
# Each demo invocation creates an independent fixed study. No earlier trial is overwritten.
DEMO_STUDY="outputs/autoresearch/demo-$(date +%Y%m%dT%H%M%S)-$$"
python3 -m meta_search_rsi.cli evaluate --study "$DEMO_STUDY" --label v1 \
  --run-id "$DEMO_RUN" \
  --run-id openevolve::MyOpenEvolve::710ca863-c3ea-4d3a-88bb-e3444cec5c4a \
  --run-id openevolve::MyOpenEvolve::8b752b40-098d-4f6f-a2a1-1eed187e3cf2 \
  --run-id openevolve::MyOpenEvolve::8bcb31d9-fdd0-428a-825b-234ac66f0204 \
  --run-id openevolve::MyOpenEvolve::abf1c182-d6c4-4873-b57f-c79d968ce587 \
  --run-id sweagent::SWE-smith::Cog-Creators__Red-DiscordBot.33e0eac7.combine_file__1tzqbvbn.2ocrdqst
printf 'Study: %s\n' "$DEMO_STUDY"
