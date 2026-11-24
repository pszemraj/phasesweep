"""Fit one weight to ``y = 2x``, or log a resumable curve, reporting only through W&B.

Run as a trial subprocess by ``tests/test_wandb_live.py``. W&B takes the target
project, the trial's run ID and the resume policy from the environment
PhaseSweep composes, so the trainer sets none of them itself. The SDK is imported where
it is used, so the test module can import the curve without it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

TRAIN_X = [-1.0 + 2.0 * index / 63 for index in range(64)]
EVAL_X = [-0.95 + 1.9 * index / 31 for index in range(32)]
# The resumable curve: its minimum comes early, and the first attempt stops after it.
CURVE_STEPS = 12
CURVE_MINIMUM_STEP = 3
INTERRUPT_STEP = 6


def curve_loss(step: int) -> float:
    """Return the resumable curve's held-out loss at ``step``.

    :param int step: One-based evaluation step.
    :return float: Loss, lowest at :data:`CURVE_MINIMUM_STEP`.
    """
    return (step - CURVE_MINIMUM_STEP) ** 2 / 10 + 0.1


def _mse(weight: float, xs: list[float]) -> float:
    """Return the mean squared error of ``weight * x`` against ``2 * x``.

    :param float weight: The single model weight.
    :param list[float] xs: Inputs to evaluate.
    :return float: Mean squared error.
    """
    return sum((weight * x - 2.0 * x) ** 2 for x in xs) / len(xs)


def _wait_until_uploaded(run_path: str, step: int) -> None:
    """Wait until W&B serves the history and summary logged through ``step``.

    The first attempt is killed right after, so whatever W&B has not received by then is
    lost; waiting makes the server hold exactly the rows the checkpoint covers.

    :param str run_path: ``entity/project/run_id`` of the run.
    :param int step: One-based evaluation step just logged, W&B ``_step`` ``step - 1``.
    :raises SystemExit: W&B did not serve the rows within two minutes.
    """
    import wandb

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            served = wandb.Api(timeout=30).run(run_path)
            if (
                served.lastHistoryStep >= step - 1
                and served.summary_metrics.get("_step") == step - 1
            ):
                return
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit(f"W&B never served step {step} of {run_path}")


def _resumable(interrupt_marker: Path) -> None:
    """Log :func:`curve_loss` once per step, resuming from the trial directory's checkpoint.

    The attempt that starts from step 0 stops at :data:`INTERRUPT_STEP`, once W&B holds its
    rows: it touches ``interrupt_marker`` and sleeps until the test kills its runner and
    the next run's recovery stops it. A rerun continues from the checkpoint.

    :param Path interrupt_marker: File the first attempt touches when it may be killed.
    """
    import wandb

    checkpoint = Path(os.environ["PHASESWEEP_TRIAL_DIR"]) / "checkpoint.json"
    start = json.loads(checkpoint.read_text())["step"] if checkpoint.exists() else 0
    with wandb.init() as run:
        run.define_metric("eval/loss", summary="min")
        for step in range(start + 1, CURVE_STEPS + 1):
            run.log({"eval/loss": curve_loss(step), "eval/step": step})
            checkpoint.write_text(json.dumps({"step": step}))
            if step == INTERRUPT_STEP and start == 0:
                _wait_until_uploaded(f"{run.entity}/{run.project}/{run.id}", step)
                interrupt_marker.touch()
                time.sleep(600)


def main(argv: list[str] | None = None) -> int:
    """Train with full-batch SGD for 20 steps, logging held-out loss every 5.

    Eval metrics use their own ``eval/step`` x-axis and a ``min`` summary, so
    the finished summary reports ``eval/loss.min`` and history records the step
    each evaluation ran at. With ``--interrupt_marker`` the trainer logs the resumable
    curve instead.

    :param list[str] | None argv: Command-line arguments.
    :return int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--interrupt_marker", type=Path, default=None)
    args = parser.parse_args(argv)
    parameters = {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay}
    args.receipt.write_text(json.dumps(parameters) + "\n", encoding="utf-8")
    if args.interrupt_marker is not None:
        _resumable(args.interrupt_marker)
        return 0
    import wandb

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
