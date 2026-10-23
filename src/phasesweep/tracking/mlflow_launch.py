"""Run a trainer script inside PhaseSweep's managed MLflow run.

Pass the script path followed by its normal arguments. The active MLflow run
stays open across the script's training and final evaluation.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run the supplied script with the managed run active.

    :param list[str] | None argv: Script and arguments, defaulting to the command line.
    :return int: Zero after successful script execution.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("script", type=Path, help="trainer script path")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="trainer arguments")
    args = parser.parse_args(argv)
    run_id = os.environ.pop("MLFLOW_RUN_ID", None)
    if not run_id:
        parser.error("MLFLOW_RUN_ID is missing; this launcher requires mlflow.run_mode: managed")

    import mlflow

    sys.argv = [str(args.script), *args.args]
    sys.path.insert(0, str(args.script.resolve().parent))
    with mlflow.start_run(run_id=run_id):
        try:
            runpy.run_path(str(args.script), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
