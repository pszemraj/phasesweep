"""Run closeout distinguishes accepted incomplete results and pending MLflow delivery."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from phasesweep.config.models import Experiment
from phasesweep.engine.paths import _experiment_dir
from phasesweep.engine.read import read_status
from phasesweep.engine.run import TerminalReport, run_experiment
from phasesweep.runtime.json import JsonObject
from phasesweep.tracking import publication
from phasesweep.tracking.mlflow import OperationRejectedError
from tests.tracking_helpers import remote_mlflow_example


@pytest.mark.integration
def test_pending_mlflow_uploads_get_final_closeout_without_failing_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The final warning names pending delivery while the committed result remains successful."""
    config, _uploads = remote_mlflow_example(tmp_path, monkeypatch)
    experiment = Experiment.model_validate(config)

    def outage(*_args: object, **_kwargs: object) -> JsonObject:
        raise OperationRejectedError("MLflow server unavailable")

    monkeypatch.setattr(publication, "call_tracking", outage)
    caplog.set_level(logging.INFO)
    reports: list[TerminalReport] = []
    winners = run_experiment(experiment, terminal_callback=reports.append)
    [report] = reports
    assert set(winners) == {"depth", "rate"}
    assert report.primary_error is None
    assert read_status(experiment)["published_generation_id"] == report.generation_id
    assert publication.delivery_status(experiment, report.generation_id) == {
        "pending": 7,
        "delivered": 0,
    }

    records = [record for record in caplog.records if record.name == publication.__name__]
    final = records[-1]
    assert final.levelno == logging.WARNING
    message = final.getMessage()
    assert "pending=7" in message
    assert "published result" in message.lower()
    assert "sync-tracking" in message
    assert report.generation_id in message
    assert message in (_experiment_dir(experiment) / "run.log").read_text()
