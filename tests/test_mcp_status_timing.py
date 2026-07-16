"""Status timing surfaces: per-phase progress counts, elapsed time, and the
adaptive poll interval derived from completed-trial durations."""

from __future__ import annotations

import time
from pathlib import Path

import optuna
import pytest

from phasesweep.config import load_config
from phasesweep.engine.optuna import _phase_study_name, phase_completed_trial_durations
from phasesweep.mcp.runs import RunHandle, RunStore, write_status_file
from phasesweep.mcp.server import (
    POLL_DEFAULT_SECONDS,
    POLL_MAX_SECONDS,
    POLL_MIN_SECONDS,
    _poll_after_seconds,
    _run_elapsed_seconds,
)
from phasesweep.mcp.time import utc_now_iso
from tests.mcp_helpers import make_mcp_app, mcp_experiment_config_text, write_mcp_config_catalog


def _experiment(tmp_path: Path, *, storage: str | None = None):
    text = mcp_experiment_config_text(tmp_path)
    if storage is not None:
        text = text.replace(f"storage: sqlite:///{tmp_path}/srv.db", f"storage: {storage}")
    config = tmp_path / "exp.yaml"
    config.write_text(text)
    return load_config(config)


def _complete_trials(experiment, *, n: int, sleep: float = 0.0) -> None:
    study = optuna.create_study(
        study_name=_phase_study_name(experiment, experiment.phases[0]),
        storage=experiment.storage,
        direction="minimize",
    )
    for i in range(n):
        trial = study.ask()
        if sleep:
            time.sleep(sleep)
        study.tell(trial, float(i))


def test_completed_trial_durations_sqlite(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    phase = experiment.phases[0]
    assert phase_completed_trial_durations(experiment, phase) == []  # DB absent: no side effects

    _complete_trials(experiment, n=2, sleep=0.05)
    study = optuna.load_study(
        study_name=_phase_study_name(experiment, phase), storage=experiment.storage
    )
    study.ask()  # a RUNNING trial must not contribute a duration

    durations = phase_completed_trial_durations(experiment, phase)
    assert len(durations) == 2
    assert all(d >= 0.05 for d in durations)


def test_completed_trial_durations_journal(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path, storage=f"journal:///{tmp_path}/srv.journal")
    phase = experiment.phases[0]
    assert phase_completed_trial_durations(experiment, phase) == []

    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    storage = JournalStorage(JournalFileBackend(str(tmp_path / "srv.journal")))
    study = optuna.create_study(
        study_name=_phase_study_name(experiment, phase), storage=storage, direction="minimize"
    )
    trial = study.ask()
    time.sleep(0.05)
    study.tell(trial, 0.1)

    durations = phase_completed_trial_durations(experiment, phase)
    assert len(durations) == 1
    assert durations[0] >= 0.05


def test_poll_after_seconds_clamps_median() -> None:
    assert _poll_after_seconds(None) == POLL_DEFAULT_SECONDS
    assert _poll_after_seconds(2.0) == POLL_MIN_SECONDS
    assert _poll_after_seconds(90.4) == 90
    assert _poll_after_seconds(10_000.0) == POLL_MAX_SECONDS


def _handle(store: RunStore, run_id: str, *, started_at: str) -> RunHandle:
    return RunHandle(
        run_id=run_id,
        experiment_id="srv",
        config_sha256="0" * 64,
        pid=1,
        pgid=1,
        pid_starttime=None,
        started_at=started_at,
        log_path=str(store.log_path(run_id)),
        status_path=str(store.status_path(run_id)),
    )


def test_elapsed_seconds_running_counts_from_launch(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = _handle(store, "r1", started_at=utc_now_iso())
    elapsed = _run_elapsed_seconds(store, handle, "running")
    assert elapsed is not None
    assert 0 <= elapsed <= 5


def test_elapsed_seconds_terminal_prefers_runner_stamp(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = _handle(store, "r1", started_at="2026-07-16T00:00:00+00:00")
    write_status_file(
        store.status_path("r1"),
        {
            "run_id": "r1",
            "returncode": 0,
            "error_class": None,
            "cleanup_confirmed": True,
            "ended_at": "2026-07-16T00:02:05+00:00",
        },
    )
    assert _run_elapsed_seconds(store, handle, "succeeded") == 125


def test_elapsed_seconds_terminal_falls_back_to_status_mtime(tmp_path: Path) -> None:
    # Runs recorded before the ended_at stamp existed still report a duration.
    store = RunStore(tmp_path / "state")
    started = utc_now_iso()
    handle = _handle(store, "r1", started_at=started)
    write_status_file(
        store.status_path("r1"),
        {"run_id": "r1", "returncode": 0, "error_class": None, "cleanup_confirmed": True},
    )
    elapsed = _run_elapsed_seconds(store, handle, "succeeded")
    assert elapsed is not None
    assert 0 <= elapsed <= 5


def test_elapsed_seconds_none_without_status_or_valid_start(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    # Terminal with no status.json at all (e.g. SIGKILL before any write).
    assert _run_elapsed_seconds(
        store, _handle(store, "r1", started_at=utc_now_iso()), "failed"
    ) is (None)
    # Malformed launch timestamp.
    assert _run_elapsed_seconds(store, _handle(store, "r2", started_at="bogus"), "running") is None


def test_status_reports_progress_and_poll_fields(tmp_path: Path) -> None:
    config_text = mcp_experiment_config_text(tmp_path)
    catalog = write_mcp_config_catalog(tmp_path, {"srv": config_text})
    app, _registry, _store = make_mcp_app(catalog)

    status = app.status(experiment_id="srv")
    assert status["run"] is None
    assert status["elapsed_seconds"] is None
    assert status["poll_after_seconds"] == POLL_DEFAULT_SECONDS
    (phase,) = status["phases"]
    assert phase["n_trials"] == 1
    assert phase["completed"] == 0

    # Completed trials feed both the per-phase count and the poll suggestion.
    experiment = _registry.get("srv").experiment
    _complete_trials(experiment, n=3, sleep=0.0)
    status = app.status(experiment_id="srv")
    (phase,) = status["phases"]
    assert phase["completed"] == 3
    assert POLL_MIN_SECONDS <= status["poll_after_seconds"] <= POLL_MAX_SECONDS


@pytest.mark.parametrize("median", [None, 0.1, 3600.0])
def test_poll_bounds_hold_for_all_medians(median: float | None) -> None:
    assert POLL_MIN_SECONDS <= _poll_after_seconds(median) <= POLL_MAX_SECONDS
