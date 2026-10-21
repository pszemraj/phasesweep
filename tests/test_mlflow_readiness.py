"""MLflow failures identify the configured target and missing evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from phasesweep.config.models import Experiment
from phasesweep.engine.errors import TrackingTargetUnavailableError
from phasesweep.runtime.json import JsonObject
from phasesweep.tracking import session
from phasesweep.tracking.mlflow import OperationRejectedError
from tests.tracking_helpers import native_experiment


@pytest.mark.parametrize(
    ("selector", "value"), [("experiment_name", "missing-trainer"), ("experiment_id", "42")]
)
def test_tracking_target_refusal_names_configured_experiment_and_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selector: str, value: str
) -> None:
    """The operator can identify which configured MLflow target needs fixing."""
    original = native_experiment(tmp_path)
    assert original.mlflow is not None
    target = original.mlflow.model_dump()
    target.pop("experiment_name")
    target.pop("experiment_id")
    target.update(tracking_uri="http://127.0.0.1:54321", **{selector: value})
    experiment = Experiment.model_validate({**original.model_dump(), "mlflow": target})

    def unavailable(*_args: object, **_kwargs: object) -> JsonObject:
        raise OperationRejectedError("MLflow experiment is missing or deleted.")

    monkeypatch.setattr(session, "call_tracking", unavailable)
    with pytest.raises(TrackingTargetUnavailableError) as refused:
        session.require_tracking_target(
            experiment, generation_id="diagnostic", environment={}, deadline=None
        )
    message = str(refused.value)
    assert value in message
    assert "http://127.0.0.1:54321" in message
    assert "No trial was started" in message
