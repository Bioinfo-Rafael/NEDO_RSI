# Numerical interfaces and immutable evaluation

`data = {"y": [episodes, 1200, 4], "u": [episodes, 1200, 4]}` in physical units.
Only training episodes are supplied to the training function. `stats` contains
`y_mean`, `y_std`, `u_mean`, `u_std`, frozen from the original 32 training episodes.
All candidates and MPC use those same stats. They are not recalculated on added data.

Outputs: `t_exit [K]`, `o2_exit [-]`, `yf_exit [-]`, `unburnt_bed [kg/m2/s]`.
Actions: `stoker_speed [m/s]`, `waste_feed [kg/s]`, `primary_level`, `secondary_level`
(air in Nm3/s per zone/nozzle, aggregated by averaging).

## Time alignment

`u[k]` is applied during a control interval and produces the recorded `y[k]`.
After observing `y[k-1]`, predicting `y[k]` uses future `u[k]`.
333 solver steps of 0.003 seconds make one control interval (0.999 seconds).
Future actions are provided during open-loop forecasting; this is action-conditioned
prediction, not a forecast with unknown future controls.

## World-model API

```python
N_CV = 4
WARMUP = 20       # editable integer, 1..300
ENSEMBLE = 5      # editable integer, 1..8
TRAINING_UPDATES = 0  # actual optimizer updates in most recent fit; not a budget

init_params(key) -> parameter_pytree
warm_up(params, y_hist, u_hist) -> state    # [B, D], common coordinates
predict(params, state, y_last, u_future, bias=None) -> [B, H, 4]
train(data, stats, seconds, seed) -> list[parameter_pytree]
```

Model inputs/outputs are normalized. `warm_up` receives `[B,WARMUP,4]`
observations/actions. `predict` must work for H=1,30,300,900 without future
observations, and respect additive `bias` in normalized output coordinates.
MPC averages the warm-up states across ensemble members, so raw/shared-coordinate
states work; unrelated latent coordinate systems do not. State must be a 2D array
`[batch,D]` compatible with the unchanged MPC.

`pipeline.train(wm, data, stats, seconds, seed)` calls model training and returns
ensemble parameters. A trusted `prepare.windows` generator is available using
the original signature. Its yielded batch count is monitored independently.

## Sampling policy

`sampling.json` contains `bounds_fraction` and `slow_probability`, each mapping
all four action names to values. Fractional ranges are `[low,high]` inside `[0,1]`
with `low < high`. They select part of the existing physical action envelope.
Slow probability lies in `[0,1]`; fast holds last 10–39 steps and slow holds
200–599 steps. Air zones/nozzles retain the original 10% spatial jitter and clipping.
Physical bounds remain stoker 0.010–0.030, feed 1.0–1.6, primary 1.0–1.5,
secondary 0.2–0.6. Full ranges with slow probability 0.5 reproduce the original
sampler exactly. A/B use paired generation seeds and the same episode counts.

## Scoring

Forecasts always start at index 300, with at most the preceding 300 observations
available. Predict 900 future steps without replacing predicted states by targets.
For each horizon 30,300,900, compute physical RMSE per channel and divide by the
fixed original-training standard deviation. Average over channels and horizons
to obtain `forecast_nrmse`. This number is not a percentage of an operating range.

Control validation uses two 400-step reachable setpoint scenarios. It reports
mean normalized absolute tracking error, overshoot and action movement, with the
original cost weights. MPC configuration, dynamics and normalization are fixed.
Smoke tests use shorter control scenarios and a smaller MPC population and are
explicitly excluded from scientific comparisons.

The runner keeps a candidate only if the fixed joint validation objective improves.
Held-out testing is separate and evaluates the selected candidate, not the last
edited source. Increasing the best-so-far validation score does not by itself
establish generalization or a scaling law.
