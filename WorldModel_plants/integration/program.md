# autoresearch — incinerator world model

An autonomous experiment loop. You improve a learned world model of a stoker
incinerator, scored against a physics simulator that defines ground truth.

## The task

The simulator in `src/wmf/` produces episodes of a 2D reacting gas coupled to a
1D grate. Your job is a world model that predicts the next state:

    q_{t+1} = F(q_t, a_t, g, dx)

`train.py` trains it. `evaluate.py` scores it. **Get `val_nrmse` as low as you
can.** Lower is better.

## Files

| File | |
|---|---|
| `train.py` | **the only file you edit.** Model, optimiser, training loop. |
| `prepare.py` | READ-ONLY. Data loading and the normalisation statistics. |
| `evaluate.py` | READ-ONLY. The metric. This is the ground truth of the experiment. |

**Do not touch anything outside `integration/`.** `src/wmf/` is the simulator —
it produces the data and is not part of the experiment. `data/` is the dataset.
Changing either invalidates every previous result.

**Do not install packages.** Only `jax`, `numpy` and what is already imported.
**Do not use the network.** Everything you need is on disk.

## The data

Per episode, 250 frames at a 0.1 s control interval (25 s of furnace time).

| Array | Shape | |
|---|---|---|
| `q` | `[T, 64, 64, 7]` | `T, u, v, p, Y_F, Y_O2, Y_P` — the state |
| `a` | `[T, 64, 64, 4]` | `primary_air, secondary_air, fuel_feed, stoker_speed` |
| `g` | `[64, 64, 4]` | `mask, sdf, actuator_layout, sensor_mask` — **the geometry** |
| `dx` | scalar per episode | metres per cell — **differs between plants (0.14–0.23)** |

`prepare.batches()` yields `((q_t normalised, a_t, g, dx), Δq normalised)`.

### Splits — this is the point of the project

| Split | Plant | |
|---|---|---|
| `train` | plant_a | what you train on |
| `val_iid` | plant_a, unseen seeds | **`val_nrmse` is measured here** |
| `val_b` | plant_b | an **unseen plant**, inside the design range |
| `val_c` | plant_c | an **unseen plant**, outside it |

The three plants have different furnace dimensions, different numbers of air
zones and nozzles, and the flue in a different place. A model that memorises
plant_a scores well on `val_iid` and badly on `val_b`/`val_c`. `gen_gap_b` in
the output is that ratio.

## The metric

`val_nrmse` is the error of a **16-step open-loop rollout** — the model is fed
its own prediction 16 times — normalised per channel by that channel's standard
deviation, then averaged. One-step accuracy is not what is scored.

`persistence_iid` is printed alongside: the score you get by predicting that
nothing changes. **A model above that line is worse than a constant.**

## Where the baseline is weak

It is deliberately weak. Four things are wrong with it on purpose:

1. `build_input` accepts `g` and `dx` and **throws them away**. The model cannot
   see the walls, the flue, the air inlets, or how large a cell is in metres.
2. Two conv layers, 16 channels — a receptive field of 5 cells, smaller than the
   plume it has to predict.
3. It trains on one step but is scored on sixteen.
4. Plain L2 over all seven channels equally, so temperature (~1000 K) dominates
   the mass fractions (~0.1) even after normalisation.

You are not limited to these. Architecture, optimiser, loss, schedule, batch
size, data augmentation — all fair game inside `train.py`.

## Running an experiment

    cd integration && ../.venv/bin/python train.py > run.log 2>&1

Redirect — do not let the output flood your context. Then read the result:

    grep "^val_nrmse:\|^gen_gap_b:\|^num_steps:" run.log

Training is capped at `prepare.TRAIN_SECONDS` (180 s) of wall clock, so every
experiment costs about the same and you are comparing ideas, not machines. A run
that takes more than 8 minutes total has hung — kill it and treat it as a crash.

If the grep comes back empty the run crashed. `tail -n 40 run.log` for the
traceback. Fix it if it is a typo or a shape error; abandon the idea if it is
fundamentally broken.

## Logging

Append one line per experiment to `results.tsv` (tab-separated). Leave it
untracked by git.

    commit	val_nrmse	gen_gap_b	status	description

`status` is `keep`, `discard` or `crash`. Use `0.000000` for a crash.

## The loop

Run on a dedicated branch, `autoresearch/<tag>`.

1. Look at the git state.
2. Change `train.py` with one idea.
3. `git commit`.
4. Run it, redirected to `run.log`.
5. Read `val_nrmse`.
6. Record the line in `results.tsv`.
7. If `val_nrmse` improved, keep the commit and continue from it.
8. If it did not, `git reset --hard` back to where you started.

One idea per experiment. If you change three things at once and the number
moves, you have learned nothing about which of them did it.

**Simpler is better, all else equal.** An improvement that comes from deleting
code is the best kind. A 0.001 gain that adds 30 lines of special-casing is not
worth keeping.

**Do not stop to ask whether to continue.** The loop runs until you are
interrupted. If you run out of ideas, re-read the data description above — the
geometry channels are sitting there unused.
