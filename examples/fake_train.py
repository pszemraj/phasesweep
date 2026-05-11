#!/usr/bin/env python3
"""Toy training script used by phasesweep examples and tests.

Reads hyperparameters either from Hydra-style key=value args on argv or from a
JSON file (--overrides-path). Computes a deterministic synthetic eval_loss and
param_bytes, writes both to result.json, and also logs them to stdout in a form
the log_regex extractor can parse.

The synthetic objective rewards moderate depth (8 layers) and small lr (~3e-4)
and weight_decay (~0.05). param_bytes scales with n_layers so the 16 MB
constraint actually bites at high depth.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _parse_kv(tokens: list[str]) -> dict[str, Any]:
    """Parse Hydra-style ``key=value`` tokens into a dict with light type inference.

    Args:
        tokens: Strings like ``"lr=0.001"``, ``"depth=8"``, ``"use_amp=true"``.

    Returns:
        Mapping from key to a coerced Python value (``int``/``float``/``bool``/``str``).

    """
    out: dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            if "." in v or "e" in v or "E" in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            if v.lower() == "true":
                out[k] = True
            elif v.lower() == "false":
                out[k] = False
            else:
                out[k] = v
    return out


def main() -> None:
    """Run the toy trainer end-to-end: parse args, compute synthetic metrics, write result.json."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--overrides-path", default=None)
    p.add_argument("--fail", action="store_true", help="simulate a crash")
    args, rest = p.parse_known_args()

    overrides: dict[str, Any] = {}
    if args.overrides_path:
        overrides = json.loads(Path(args.overrides_path).read_text())

        # Flatten nested if needed
        def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
            out: dict[str, Any] = {}
            for k, v in d.items():
                key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                if isinstance(v, dict):
                    out.update(flatten(v, key))
                else:
                    out[key] = v
            return out

        overrides = flatten(overrides)

    overrides.update(_parse_kv(rest))

    if args.fail:
        print("simulated failure", file=sys.stderr)
        sys.exit(3)

    n_layers = int(overrides.get("n_layers", 8))
    lr = float(overrides.get("lr", 3e-4))
    weight_decay = float(overrides.get("weight_decay", 0.05))
    dropout = float(overrides.get("dropout", 0.1))

    # Synthetic loss: convex bowl in (n_layers, log10 lr, weight_decay, dropout).
    loss = (
        0.05 * (n_layers - 8) ** 2
        + 0.5 * (math.log10(max(lr, 1e-12)) - math.log10(3e-4)) ** 2
        + 2.0 * (weight_decay - 0.05) ** 2
        + 1.5 * (dropout - 0.10) ** 2
        + 0.30  # noise floor
    )

    # Pretend each layer is 1 MiB of params; cap so 16-layer just exceeds 16 MB budget.
    param_bytes = n_layers * 1_100_000

    result = {
        "eval_loss": loss,
        "param_bytes": param_bytes,
        "config": {
            "n_layers": n_layers,
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2))

    # Stdout for log_regex extractor demonstrations
    print(f"step=100 eval_loss={loss:.6f}")
    print(f"final param_bytes={param_bytes}")


if __name__ == "__main__":
    main()
