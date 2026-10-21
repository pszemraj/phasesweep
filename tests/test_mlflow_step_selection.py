"""MLflow failures identify the configured target and missing evidence."""

from __future__ import annotations

import pytest

from phasesweep.evidence.mlflow import select_observation
from phasesweep.evidence.models import MlflowExtractor


def test_missing_mlflow_step_names_requested_and_available_steps() -> None:
    """A bad selected step reports enough of the observed history to correct it."""
    spec = MlflowExtractor(type="mlflow", metric_key="loss", select="step", step=5)
    history = [
        {"value": 1.0, "step": step, "timestamp": timestamp}
        for timestamp, step in enumerate([10, 2, 0, 2])
    ]
    with pytest.raises(ValueError) as missing:
        select_observation(history, spec)
    message = str(missing.value)
    assert "loss" in message
    assert "5" in message
    assert "[0, 2, 10]" in message
