"""Run a trainer script or module inside PhaseSweep's managed MLflow run.

Pass SCRIPT followed by its normal arguments, or -m MODULE and its arguments.
The active run stays open across training and final evaluation. Install
phasesweep[mlflow] in the Python environment that executes this launcher.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


class _Arguments(argparse.Namespace):
    """Trainer target and its unchanged argument tail."""

    module: bool
    target: str
    args: list[str]


def main(argv: list[str] | None = None) -> int:
    """Run the supplied script or module with the managed run active.

    :param list[str] | None argv: Target and arguments, defaulting to the command line.
    :return int: Zero after successful trainer execution.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phasesweep.tracking.mlflow_launch",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            'trial_command: "python -m phasesweep.tracking.mlflow_launch '
            '-m my_project.train --report_to mlflow {overrides}" '
            "(override_format: argparse; mlflow.run_mode: managed)"
        ),
    )
    parser.add_argument("-m", dest="module", action="store_true", help="run TARGET as a module")
    parser.add_argument("target", metavar="TARGET", help="trainer script path, or module with -m")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="trainer arguments")
    args = parser.parse_args(argv, namespace=_Arguments())
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        parser.error("MLFLOW_RUN_ID is missing; this launcher requires mlflow.run_mode: managed")
    script = None if args.module else Path(args.target)
    if script is not None and not script.is_file():
        parser.error(f"trainer script does not exist or is not a file: {script}")

    try:
        import mlflow
    except ModuleNotFoundError as error:
        if error.name != "mlflow":
            raise
        parser.error('Install the MLflow SDK: pip install "phasesweep[mlflow]"')

    os.environ.pop("MLFLOW_RUN_ID")
    sys.argv = [args.target, *args.args]
    if script is not None:
        if not sys.flags.safe_path:
            del sys.path[0]
        sys.path.insert(0, str(script.resolve().parent))
    with mlflow.start_run(run_id=run_id):
        try:
            if args.module:
                runpy.run_module(args.target, run_name="__main__", alter_sys=True)
            else:
                runpy.run_path(str(script), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
