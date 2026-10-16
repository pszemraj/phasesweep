"""Native experiment-tracker integration without eager SDK imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phasesweep.tracking.mlflow import _SdkClient


def bind_mlflow_run(run_id: str) -> None:
    """Bind a trainer-created MLflow run to the current PhaseSweep attempt.

    Call once inside the trainer's active run, before logging evaluations.

    :param str run_id: Real run ID returned by the trainer's MLflow logger.
    :raises ValueError: The run conflicts with the current attempt or experiment.
    """
    import os
    from pathlib import Path

    from mlflow import MlflowClient

    from phasesweep.tracking.mlflow import identity_tags, verify_remote
    from phasesweep.tracking.records import validate_binding
    from phasesweep.tracking.session import BINDING_FILE, read_binding, read_record, write_record

    directory = Path(os.environ["PHASESWEEP_TRIAL_DIR"])
    expected = read_record(directory / "mlflow-expected.json")
    if (
        expected["run_mode"] != "trainer"
        or expected["attempt_id"] != os.environ["PHASESWEEP_ATTEMPT_ID"]
        or expected["generation_id"] != os.environ["PHASESWEEP_GENERATION_ID"]
        or expected["overrides_sha256"] != os.environ["PHASESWEEP_OVERRIDES_SHA256"]
    ):
        raise ValueError("MLflow binding helper requires this trainer-owned attempt.")
    binding = validate_binding({**expected, "run_id": run_id})
    if (directory / BINDING_FILE).exists() and read_binding(directory) != binding:
        raise ValueError("This attempt is already bound to another MLflow run.")
    client: _SdkClient = MlflowClient(tracking_uri=binding["tracking_uri"])
    run = client.get_run(run_id)
    if run.info.experiment_id != expected["experiment_id"] or run.info.lifecycle_stage != "active":
        raise ValueError("Trainer MLflow run belongs to a different experiment.")
    # An ended run predates this attempt, so its metrics cannot be its evidence.
    if run.info.status != "RUNNING":
        raise ValueError("Bind the trainer's MLflow run while it is still RUNNING.")
    tags = identity_tags(binding)
    if any(key in run.data.tags and run.data.tags[key] != value for key, value in tags.items()):
        raise ValueError("Trainer MLflow run already belongs to another attempt.")
    for key, value in tags.items():
        client.set_tag(run_id, key, value)
    verify_remote(client.get_run(run_id), binding)
    write_record(directory / BINDING_FILE, binding)
