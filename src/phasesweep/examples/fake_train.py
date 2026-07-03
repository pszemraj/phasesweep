"""Toy training script used by phasesweep examples and tests."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


def _parse_kv(tokens: list[str]) -> dict[str, Any]:
    """Parse Hydra-style ``key=value`` tokens into a dict with light type inference.

    :param list[str] tokens: Extra CLI tokens such as ``lr=0.001`` or ``use_amp=true``.
    :return dict[str, Any]: Parsed override values keyed by override name.
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


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested override JSON into dotted keys.

    :param dict[str, Any] d: Nested override mapping loaded from JSON.
    :param str prefix: Existing dotted prefix used during recursion.
    :return dict[str, Any]: Flat mapping where nested keys are joined with dots.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def main() -> None:
    """Run the toy trainer end-to-end: parse args, compute metrics, write result.json."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--overrides-path", default=None)
    p.add_argument("--n_layers", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--dropout", type=float)
    p.add_argument("--fail", action="store_true", help="simulate a crash")
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="seconds to sleep before writing result (simulates a long trial; used by tests)",
    )
    args, rest = p.parse_known_args()

    overrides: dict[str, Any] = {}
    if args.overrides_path:
        overrides = _flatten(json.loads(Path(args.overrides_path).read_text()))

    overrides.update(
        {
            "n_layers": args.n_layers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
        }
    )
    overrides = {k: v for k, v in overrides.items() if v is not None}
    overrides.update(_parse_kv(rest))

    if args.fail:
        print("simulated failure", file=sys.stderr)
        sys.exit(3)

    if args.sleep > 0:
        time.sleep(args.sleep)

    n_layers = int(overrides.get("n_layers", 8))
    lr = float(overrides.get("lr", 3e-4))
    weight_decay = float(overrides.get("weight_decay", 0.05))
    dropout = float(overrides.get("dropout", 0.1))

    loss = (
        0.05 * (n_layers - 8) ** 2
        + 0.5 * (math.log10(max(lr, 1e-12)) - math.log10(3e-4)) ** 2
        + 2.0 * (weight_decay - 0.05) ** 2
        + 1.5 * (dropout - 0.10) ** 2
        + 0.30
    )

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

    print(f"step=100 eval_loss={loss:.6f}")
    print(f"final param_bytes={param_bytes}")


if __name__ == "__main__":
    main()
