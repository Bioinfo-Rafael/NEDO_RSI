# autoresearch — world model for control

You are improving a **world model** `M` of a stoker incinerator. A separate
**controller** `C` searches inside `M` to decide what the plant should do. You
do not touch `C`. You are scored by how well the plant it steers tracks its
setpoints.

    M : (state, action) -> next state        "what will happen?"     <- you
    C : searches M for a plan                "what should I do?"     <- fixed

**Lower `control_cost` is better.** That is the only number that counts.

## The thing to understand before you start

**Prediction error is not the objective.** A model can be excellent at
prediction and make a poor controller. `C` is an optimiser: it searches for the
action plan that `M` says is best. Wherever `M` is confidently wrong, `C` will
be drawn there and the plant will not do what `M` promised. So:

- being accurate in the region `C` actually operates matters more than average
  accuracy over the dataset;
- being *honestly uncertain* where the data was thin is worth more than being
  confidently wrong there — `C` uses the ensemble's disagreement as a trust
  region and will avoid plans the members argue about;
- near-term accuracy closes the loop (`C` applies only the first move of each
  plan) but the far end of the horizon still shapes which plan gets chosen.

## Files

| File | |
|---|---|
| `worldmodel.py` | **the only file you edit.** There is no intended architecture. |
| `prepare.py` | READ-ONLY. Identification data and the training budget. |
| `evaluate.py` | READ-ONLY. The metric. Runs `C` against the simulator. |
| `run.py` | READ-ONLY. Runs one experiment. |

Do not touch anything outside `control_research/`. `src/wmf/` is the simulator
and the controller. `data_ctrl_v2/` is the dataset. Changing either invalidates
every previous result. Do not install packages. Do not use the network.

## What is open

**The architecture is not decided, and nothing in the repository should be read
as deciding it.** The file you inherit holds a linear state-space model. That is
the classical system-identification answer, put there because it is not a neural
architecture and so does not point at one. It is a floor to beat. Deleting all
of it is a normal move, not a drastic one.

Recurrent, convolutional, attentional, operator-based, an ensemble of small
linear models fitted to different operating points, a Gaussian process, a coarse
physical model with a learned residual, something with no accepted name - all of
these are on the table and none of them is favoured. Whatever the last
experiment left behind carries no authority either: if the evidence points away
from it, discard it, including work that took several experiments to build.

Two habits are worth more than any particular architecture here:

- **Change one thing per experiment.** Three changes and a moving number teach
  you nothing about which one moved it.
- **Be willing to lose ground.** An experiment that is worse but isolates *why*
  is more useful than one that is better for reasons you cannot name. Record
  what you expected before you run it, and say in the description whether that
  expectation held.

## The plant you are modelling

Its time scales are the hard part, and they are the real plant's, not a
convenient approximation:

| | |
|---|---|
| control interval | 1 s |
| gas response | seconds |
| **grate residence** | **400-1200 s** depending on grate speed |
| effect of a feed change on burnout | delayed by roughly that residence |

Nothing measures what is lying on the grate. A model with no memory cannot know
it, and the delay is long enough that a short memory will not reach it either.
How to represent that is one of the open questions, not a solved one.

## The interface you must keep

`C` calls these. Change anything inside them; do not change their signatures.

    init_params(key)                        -> params for one ensemble member
    warm_up(params, y_hist, u_hist)         -> state  [B, ...]
    predict(params, state, y_last, u, bias) -> [B, H, 4]
    train(data, stats, seconds, seed)       -> list of params (the ensemble)

Also keep the module constants `N_CV = 4`, `WARMUP`, and `ENSEMBLE`.

## The plant

Controlled variables `y` (what `C` holds to setpoints):

| | | why it is there |
|---|---|---|
| `t_exit` | flue gas temperature [K] | regulated; a real furnace must stay above 850 degC |
| `o2_exit` | flue oxygen [-] | regulated |
| `yf_exit` | unburnt fuel in the flue [-] | quality |
| `unburnt_bed` | unburnt solids leaving the grate [kg/m2/s] | **what the grate speed is for** — ignition loss of the ash |

Manipulated variables `u` (what `C` commands):
`stoker_speed`, `waste_feed`, `primary_level`, `secondary_level`.

**The hard part of this plant is its time scales**, and they are the real
plant's: the gas responds in about a second; material takes 400-1200 s to cross
the grate (about 600 s at the nominal grate speed), so a change in waste feed or
grate speed does not reach `unburnt_bed` until long after it was made. Nothing
measures what is lying on the grate. A model with no memory cannot know it;
how a model should carry information across ten minutes of 1 s samples is one
of the open questions, not a solved one.

Measured input authority, in units of each output's spread (from the
identification data; regenerated by `scripts/identify_pi.py`):

|            | t_exit | o2_exit | yf_exit | unburnt_bed |
|---|---|---|---|---|
| stoker_speed    | 0.25 | 0.65 | 0.43 | **1.36** |
| waste_feed      | 0.80 | 0.11 | 0.33 | 0.11 |
| primary_level   | 0.57 | **1.39** | 1.13 | 0.12 |
| secondary_level | 1.37 | **2.44** | 0.71 | 0.04 |

**Measured dead times** (cross-correlation of increments): secondary_level->t_exit 1 s, primary_level->o2_exit 1 s, stoker_speed->unburnt_bed 1 s, waste_feed->unburnt_bed 1084 s





Secondary air dominates both temperature and oxygen, so those two setpoints
fight each other. That coupling is real, and it is why a controller that plans
is worth having.

## Running an experiment

    cd control_research && ../.venv/bin/python run.py > run.log 2>&1
    grep "^control_cost:\|^temp_step_tracking:\|^o2_step_tracking:" run.log

Training is capped at `prepare.TRAIN_SECONDS`. Evaluation runs `C` against the
simulator over two 400 s setpoint steps, which takes a few minutes and cannot
be shortened — it is the measurement. Budget about 8 minutes per experiment; a
run past 20 minutes has hung, so kill it and treat it as a crash.

An empty grep means it crashed. `tail -n 40 run.log` for the traceback.

## Logging

Append one tab-separated line per experiment to `results.tsv`, untracked by git:

    commit	control_cost	temp_track	o2_track	status	description

`status` is `keep`, `discard` or `crash`; use `0.000000` for a crash.

## The loop

Work on a branch `control/<tag>`.

1. Look at the git state.
2. Change `worldmodel.py` with **one** idea.
3. `git commit`.
4. Run it, redirected to `run.log`.
5. Read `control_cost`.
6. Record the line in `results.tsv`.
7. Improved? Keep the commit and continue from it. Otherwise `git reset --hard`.

One idea per experiment. Change three things at once and a moving number tells
you nothing about which one moved it.

**Simpler is better, all else equal.** A gain that comes from deleting code is
the best kind.

**Do not stop to ask whether to continue.** The loop runs until interrupted.
