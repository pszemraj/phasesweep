"""Deterministic CPU trainer using MLflow's native logging API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import yaml

from phasesweep.tracking import bind_mlflow_run


def main(argv: list[str] | None = None) -> int:
    """Log deterministic observations in either run ownership mode.

    :param list[str] | None argv: Optional command-line arguments.
    :return int: Zero after successful native logging.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    depth, rate = config["model"]["depth"], config["optimizer"]["rate"]
    loss = float((depth - 2) ** 2 + (rate - 3) ** 2)
    with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")) as run:
        if "MLFLOW_RUN_ID" not in os.environ:
            bind_mlflow_run(run.info.run_id)
        mlflow.log_params({"depth": depth, "rate": rate})
        mlflow.set_tag("trainer", "phasesweep-cpu-example")
        mlflow.log_metric("eval/loss", loss + 1, step=0)
        mlflow.log_metric("eval/loss", loss, step=1)
        mlflow.log_metric("cost", float(depth + rate), step=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
