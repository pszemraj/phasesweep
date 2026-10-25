"""Run a console entry point or Python script inside a managed MLflow run."""

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Sequence
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path


class TrainerCommandError(ValueError):
    """The requested trainer or managed run cannot be resolved for this process."""


def _resolve_trainer(command: str) -> EntryPoint | Path:
    """Resolve a console script in this environment, or a local Python script.

    :param str command: First word of the trainer command.
    :raises TrainerCommandError: Neither accepted trainer form resolves.
    :return EntryPoint | Path: Installed entry point or existing script path.
    """
    installed = entry_points(group="console_scripts", name=command)
    if installed:
        return next(iter(installed))
    script = Path(command)
    if script.is_file():
        return script
    raise TrainerCommandError(
        f"Cannot resolve trainer {command!r}. Supply an installed console script from this "
        "Python environment or an existing Python script path, followed by its arguments; "
        "shell commands and module names are not supported."
    )


def run_managed(command: Sequence[str]) -> int:
    """Keep the supplied managed run active while executing a trainer in this process.

    Console entry points receive their ordinary argument tail. Python scripts receive
    their directory on the import path. Child processes cannot inherit the active run.

    :param Sequence[str] command: Console script or Python script path and trainer arguments.
    :raises TrainerCommandError: The trainer, managed identity or optional SDK is unavailable.
    :raises SystemExit: The trainer exits or returns a nonzero status.
    :return int: Zero when training and final evaluation finish successfully.
    """
    if not command:
        raise TrainerCommandError(
            "Supply a console script or Python script path and its arguments."
        )
    trainer = _resolve_trainer(command[0])
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        raise TrainerCommandError(
            "MLFLOW_RUN_ID is missing; mlflow-run requires mlflow.run_mode: managed."
        )
    try:
        import mlflow
    except ModuleNotFoundError as error:
        if error.name != "mlflow":
            raise
        raise TrainerCommandError(
            'Install the MLflow SDK: pip install "phasesweep[mlflow]"'
        ) from error

    original_argv, original_path = sys.argv, sys.path[:]
    sys.argv = list(command)
    if isinstance(trainer, Path):
        if not sys.flags.safe_path:
            del sys.path[0]
        sys.path.insert(0, str(trainer.resolve().parent))
    os.environ.pop("MLFLOW_RUN_ID")
    try:
        with mlflow.start_run(run_id=run_id):
            try:
                if isinstance(trainer, EntryPoint):
                    callback: object = trainer.load()
                    if not callable(callback):
                        raise TrainerCommandError(f"Console script {command[0]!r} is not callable.")
                    result: object = callback()
                    if result is not None and result != 0:
                        raise SystemExit(result)
                else:
                    runpy.run_path(str(trainer), run_name="__main__")
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise
        return 0
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path
        os.environ["MLFLOW_RUN_ID"] = run_id
