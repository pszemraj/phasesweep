"""Toy training script used by phasesweep examples and tests."""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

from phasesweep import report_objective


def _parse_kv(tokens: list[str]) -> dict[str, Any]:
    """Parse optional ``key=value`` compatibility tokens with light type inference.

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


def main() -> None:
    """Run the toy trainer end-to-end: parse args, compute metrics, write result.json."""
    p = argparse.ArgumentParser()
    p.add_argument("--n_layers", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--dropout", type=float)
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="seconds to sleep before writing result (simulates a long trial; used by tests)",
    )
    args, rest = p.parse_known_args()

    overrides: dict[str, Any] = {
        "n_layers": args.n_layers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    overrides.update(_parse_kv(rest))

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

    report_objective(
        loss,
        name="eval_loss",
        split="validation",
        policy="synthetic",
        checkpoint="toy_formula",
        step=100,
        extra={
            "param_bytes": param_bytes,
            "config": {
                "n_layers": n_layers,
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
            },
        },
    )

    print(f"step=100 eval_loss={loss:.6f}")
    print(f"final param_bytes={param_bytes}")


if __name__ == "__main__":
    main()
