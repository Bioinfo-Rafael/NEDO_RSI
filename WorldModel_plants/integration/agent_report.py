"""Training report for the agent, and the schema of what it must reply with.

The agent does not read the simulator and does not read the raw fields; it
reads this report and answers with structured JSON.  Free-form "try something
around here" replies are not actionable and cannot be validated, so the reply
is schema-constrained and every proposed action is projected back onto the
operating envelope before it reaches the simulator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wmf.actions import RANGES

# JSON Schema for the agent's reply (OpenAI Structured Outputs / Codex SDK)
REPLY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["experiment_id", "hypothesis", "simulation_parameters",
                 "expected_information_gain", "requested_code_change"],
    "properties": {
        "experiment_id": {"type": "integer"},
        "hypothesis": {"type": "string"},
        "simulation_parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["plant", "stoker_speed", "waste_feed",
                         "primary_air_level", "secondary_air_level"],
            "properties": {
                "plant": {"type": "string", "enum": ["plant_a", "plant_b", "plant_c"]},
                "stoker_speed": {"type": "number", **dict(zip(("minimum", "maximum"), RANGES["stoker_speed"]))},
                "waste_feed": {"type": "number", **dict(zip(("minimum", "maximum"), RANGES["waste_feed"]))},
                "primary_air_level": {"type": "number", **dict(zip(("minimum", "maximum"), RANGES["primary_air"]))},
                "secondary_air_level": {"type": "number", **dict(zip(("minimum", "maximum"), RANGES["secondary_air"]))},
            },
        },
        "expected_information_gain": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "requested_code_change": {"type": ["string", "null"]},
    },
}


def build_report(per_channel: dict, history: list | None = None) -> dict:
    """``per_channel`` maps split name -> per-channel nrmse array."""
    names = ["T", "u", "v", "p", "Y_F", "Y_O2", "Y_P"]
    splits = {s: {"nrmse": float(np.mean(v)),
                  "per_channel": {n: float(x) for n, x in zip(names, v)}}
              for s, v in per_channel.items()}
    iid = splits.get("val_iid", {}).get("nrmse", float("nan"))
    return {
        "metric": "val_nrmse (16-step open-loop rollout, per-channel normalised; lower is better)",
        "val_nrmse": iid,
        "splits": splits,
        "generalisation_gap": {
            s: (splits[s]["nrmse"] / iid if iid else None)
            for s in splits if s != "val_iid"
        },
        "action_envelope": {k: list(v) for k, v in RANGES.items()},
        "notes": [
            "val_b is an unseen plant inside the design range; val_c is outside it.",
            "A gap much greater than 1 means the model has memorised plant_a rather "
            "than learned to read the geometry it is given.",
            "The geometry tensor `g` and the grid spacing `dx` are in the dataset "
            "and the baseline ignores both.",
        ],
        "history": history or [],
        "reply_schema": REPLY_SCHEMA,
    }


def write_report(per_channel: dict, path: str | Path, history=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_report(per_channel, history), indent=2))
    return path


def validate_reply(reply: dict) -> dict:
    """Project the agent's proposal onto the envelope. A model that asks for an
    out-of-range condition gets it clipped, not passed through."""
    sp = dict(reply["simulation_parameters"])
    for key, rng_key in (("stoker_speed", "stoker_speed"), ("waste_feed", "waste_feed"),
                         ("primary_air_level", "primary_air"),
                         ("secondary_air_level", "secondary_air")):
        lo, hi = RANGES[rng_key]
        sp[key] = float(np.clip(sp[key], lo, hi))
    out = dict(reply)
    out["simulation_parameters"] = sp
    return out


if __name__ == "__main__":
    demo = {"val_iid": np.array([.55, .80, .66, .63, .52, .40, .57]),
            "val_b": np.array([.49, .98, .75, .85, .43, .37, .49]),
            "val_c": np.array([.39, .56, .49, .79, .51, .38, .44])}
    print(json.dumps(build_report(demo), indent=2)[:900])
