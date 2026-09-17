"""Editable training pipeline. Data contain training episodes only.

The supplied stats are fixed baseline normalization (also used by MPC).
Do not mutate data or stats. Preprocessing may be represented inside the model.
Return a nonempty list of ensemble parameter pytrees; synchronize training.
"""


def train(wm, data, stats, seconds, seed):
    return wm.train(data, stats, seconds, seed)
