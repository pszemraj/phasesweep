"""MLflow failures identify the configured target and missing evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from phasesweep.config.models import Experiment
from phasesweep.engine.paths import _experiment_dir, _generation_dir
from phasesweep.engine.read import read_status
from phasesweep.engine.run import TerminalReport, run_experiment
from phasesweep.runtime.json import JsonObject
from phasesweep.tracking import mlflow as worker
from phasesweep.tracking import session
from tests.tracking_helpers import remote_mlflow_example


def test_no_work_mlflow_replay_creates_no_empty_tracking_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay of the published winners writes no generation and leaves tracking alone."""
    config, _uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path / 'study.journal'}"
    experiment = Experiment.model_validate(config)
    reports: list[TerminalReport] = []
    winners = run_experiment(experiment, terminal_callback=reports.append)
    root = session.tracking_root(experiment)
    directories = {path.relative_to(root) for path in root.rglob("*") if path.is_dir()}

    def no_worker(*_args: object, **_kwargs: object) -> JsonObject:
        pytest.fail("A no-work replay must not call an MLflow worker")

    monkeypatch.setattr(session, "request_operation", no_worker)
    monkeypatch.setattr(worker, "request_operation", no_worker)
    replayed = run_experiment(experiment, terminal_callback=reports.append)

    assert {name: winner.metric for name, winner in replayed.items()} == {
        name: winner.metric for name, winner in winners.items()
    }
    assert reports[1].generation_id != reports[0].generation_id
    assert not _generation_dir(_experiment_dir(experiment), reports[1].generation_id).exists()
    assert read_status(experiment)["published_generation_id"] == reports[0].generation_id
    assert {path.relative_to(root) for path in root.rglob("*") if path.is_dir()} == directories
    assert all(any((root / directory).iterdir()) for directory in directories)
