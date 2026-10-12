"""Native experiment-tracker integration without eager SDK imports."""

from __future__ import annotations


def bind_mlflow_run(run_id: str) -> None:
    """Bind a trainer-created MLflow run to the current PhaseSweep attempt.

    Call once inside the trainer's active run, before logging evaluations.

    :param str run_id: Real run ID returned by the trainer's MLflow logger.
    :raises ValueError: The run conflicts with the current attempt or experiment.
    """
    from phasesweep.tracking.session import bind_trainer_run

    bind_trainer_run(run_id)
