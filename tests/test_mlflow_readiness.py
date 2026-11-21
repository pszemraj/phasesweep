"""MLflow failures identify the configured target and missing evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from phasesweep.config.models import Experiment
from phasesweep.engine.errors import TrackingTargetUnavailableError
from phasesweep.errors import OperatorAction
from phasesweep.runtime.json import JsonObject
from phasesweep.tracking import session
from phasesweep.tracking.mlflow import (
    OperationOutcomeUnknownError,
    OperationRejectedError,
    OperationUnavailableError,
)
from phasesweep.tracking.records import TrackingAction
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


@pytest.mark.parametrize(
    ("failure", "action"),
    [
        ("refused", OperatorAction.FIX_CONFIG),
        ("timeout", OperatorAction.RETRY),
        ("unavailable", OperatorAction.RETRY),
        ("outcome-unknown", OperatorAction.RETRY),
        ("creation-unconfirmed", OperatorAction.RETRY),
    ],
)
def test_a_target_that_gives_no_answer_gets_the_retry_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, action: OperatorAction
) -> None:
    """A target check MLflow gives no answer to is retried; one MLflow answered with a refusal is fixed.

    No answer covers a timeout, a refused connection, a lost worker answer, and a summary-run
    creation whose answer was lost and whose run a search did not find yet. A missing
    experiment is an answer, which a retry would only meet again.
    """
    experiment = native_experiment(tmp_path)
    errors = {
        "refused": OperationRejectedError("MLflow experiment is missing or deleted."),
        "timeout": TimeoutError("MLflow operation exceeded its deadline."),
        "unavailable": OperationUnavailableError("the tracking server refused the connection"),
        "outcome-unknown": OperationOutcomeUnknownError("MLflow worker failed (exit 1)."),
    }

    def tracker(
        _experiment: object,
        _generation: str,
        request: TrackingAction,
        *_args: object,
        **_kwargs: object,
    ) -> JsonObject:
        if failure != "creation-unconfirmed":
            raise errors[failure]
        if request["action"] == "resolve":
            return {"experiment_id": "1"}
        if request["action"] == "create":
            raise TimeoutError("MLflow operation exceeded its deadline.")
        return {"found": False}

    monkeypatch.setattr(session, "call_tracking", tracker)
    with pytest.raises(TrackingTargetUnavailableError) as refused:
        session.require_tracking_target(
            experiment, generation_id="diagnostic", environment={}, deadline=None
        )
    assert refused.value.action is action
    assert refused.value.remediation is None
