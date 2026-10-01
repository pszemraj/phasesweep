"""Fit one weight to ``y = 2x`` and report the held-out loss only through W&B.

Run as a trial subprocess by ``tests/test_wandb_live.py``. W&B takes the target
project, the immutable run ID, and the never-resume policy from the
environment PhaseSweep composes, so the trainer sets none of them itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

TRAIN_X = [-1.0 + 2.0 * index / 63 for index in range(64)]
EVAL_X = [-0.95 + 1.9 * index / 31 for index in range(32)]


def _mse(weight: float, xs: list[float]) -> float:
    """Return the mean squared error of ``weight * x`` against ``2 * x``.

    :param float weight: The single model weight.
    :param list[float] xs: Inputs to evaluate.
    :return float: Mean squared error.
    """
    return sum((weight * x - 2.0 * x) ** 2 for x in xs) / len(xs)


def main(argv: list[str] | None = None) -> int:
    """Train with full-batch SGD for 20 steps, logging held-out loss every 5.

    Eval metrics use their own ``eval/step`` x-axis and a ``min`` summary, so
    the finished summary reports ``eval/loss.min`` and history records the step
    each evaluation ran at.

    :param list[str] | None argv: Command-line arguments.
    :return int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    parameters = {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay}
    args.receipt.write_text(json.dumps(parameters) + "\n", encoding="utf-8")
    mean_square = sum(x * x for x in TRAIN_X) / len(TRAIN_X)
    weight = 0.0
    with wandb.init(
        config=parameters, dir=str(args.receipt.parent), settings=wandb.Settings(silent=True)
    ) as run:
        run.define_metric("eval/step", hidden=True)
        run.define_metric("eval/*", step_metric="eval/step")
        run.define_metric("eval/loss", summary="min")
        for step in range(1, 21):
            gradient = 2.0 * (weight - 2.0) * mean_square + args.weight_decay * weight
            weight -= args.learning_rate * gradient
            if step % 5 == 0:
                run.log({"eval/loss": _mse(weight, EVAL_X), "eval/step": step})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
