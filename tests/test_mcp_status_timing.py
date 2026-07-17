"""Status timing surfaces: per-phase progress counts, elapsed time, the
adaptive poll interval derived from completed-trial durations, and the
await_run wait loop built on top of them."""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path

import optuna
import pytest
import yaml

from phasesweep.config import load_config
from phasesweep.engine.optuna import _phase_study_name, phase_completed_trial_durations
from phasesweep.engine.state import _winner_path
from phasesweep.mcp.runs import RunHandle, RunStore, write_status_file
from phasesweep.mcp.server import (
    AWAIT_MAX_TIMEOUT_SECONDS,
    AWAIT_MIN_TIMEOUT_SECONDS,
    AWAIT_RECHECK_SECONDS,
    POLL_DEFAULT_SECONDS,
    POLL_MAX_SECONDS,
    POLL_MIN_SECONDS,
    PhaseSweepMCP,
    _poll_after_seconds,
    _run_elapsed_seconds,
)
from phasesweep.mcp.time import utc_now_iso
from tests.mcp_helpers import (
    make_mcp_app,
    make_run_handle,
    mcp_experiment_config_text,
    write_mcp_config_catalog,
    write_run_status,
)


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


def _handle(run_id: str, *, started_at: str) -> RunHandle:
    return RunHandle(
        run_id=run_id,
        experiment_id="srv",
        config_sha256="0" * 64,
        pid=1,
        pgid=1,
        pid_starttime=None,
        started_at=started_at,
    )


def test_elapsed_seconds_running_counts_from_launch(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = _handle("r1", started_at=utc_now_iso())
    elapsed = _run_elapsed_seconds(store, handle, "running")
    assert elapsed is not None
    assert 0 <= elapsed <= 5


def test_elapsed_seconds_terminal_prefers_runner_stamp(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = _handle("r1", started_at="2026-07-16T00:00:00+00:00")
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
    handle = _handle("r1", started_at=started)
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
    assert _run_elapsed_seconds(store, _handle("r1", started_at=utc_now_iso()), "failed") is None
    # Malformed launch timestamp.
    assert _run_elapsed_seconds(store, _handle("r2", started_at="bogus"), "running") is None


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


def test_terminal_run_reads_do_not_drift_with_shared_study_state(tmp_path: Path) -> None:
    app, registry, store = _app_with_run(tmp_path)
    experiment = registry.get("srv").experiment
    winner_path = _winner_path(experiment, "p")
    winner_path.parent.mkdir(parents=True, exist_ok=True)
    winner_path.write_text(
        yaml.safe_dump(
            {
                "trial_number": 9,
                "metric": {"loss": 9.9},
                "params": {"lr": 0.009},
                "effective_overrides": {"lr": 0.009},
            }
        )
    )
    write_run_status(
        store,
        "r1",
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot={
            "status": {
                "metric": {"name": "loss", "goal": "minimize"},
                "phases": [
                    {
                        "phase": "p",
                        "trials": {"COMPLETE": 2, "FAIL": 1},
                        "running": 0,
                        "n_trials": 3,
                        "completed": 2,
                        "winner_present": True,
                    }
                ],
                "summary_present": False,
                "median_trial_seconds": 17.0,
            },
            "winners": [
                {
                    "phase": "p",
                    "trial_number": 1,
                    "metric": 0.25,
                    "params": {"lr": 0.00025},
                    "gates_passed": None,
                    "incomplete": False,
                }
            ],
        },
    )

    run_status = app.status(run_id="r1")
    assert run_status["phases"][0]["trials"] == {"COMPLETE": 2, "FAIL": 1}
    assert run_status["poll_after_seconds"] == 17
    assert app.winners(run_id="r1")["phases"][0]["metric"] == 0.25

    # Experiment-id reads remain the current shared-storage view.
    assert app.winners(experiment_id="srv")["phases"][0]["metric"] == 9.9


def _app_with_run(tmp_path: Path, run_id: str = "r1"):
    """App plus a fabricated live run resolvable by run_id (snapshot + handle)."""
    config_text = mcp_experiment_config_text(tmp_path)
    catalog = write_mcp_config_catalog(tmp_path, {"srv": config_text})
    app, registry, store = make_mcp_app(catalog)
    data = (tmp_path / "srv.yaml").read_bytes()
    handle = make_run_handle(
        store,
        run_id=run_id,
        experiment_id="srv",
        config_sha256=hashlib.sha256(data).hexdigest(),
    )
    store.save(handle)
    store.config_snapshot_path(run_id).write_bytes(data)
    return app, registry, store


def _fake_clock(monkeypatch: pytest.MonkeyPatch, app: PhaseSweepMCP) -> dict[str, float]:
    """Replace await_run's deadline clock and recheck sleep with a manual clock."""
    clock = {"now": 0.0, "sleeps": 0.0}

    async def advance(seconds: float) -> None:
        clock["now"] += seconds
        clock["sleeps"] += seconds

    monkeypatch.setattr("phasesweep.mcp.server.time.monotonic", lambda: clock["now"])
    app._sleep = advance
    return clock


def test_await_run_returns_immediately_on_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _registry, store = _app_with_run(tmp_path)
    write_run_status(store, "r1", returncode=0, error_class=None, cleanup_confirmed=True)
    clock = _fake_clock(monkeypatch, app)

    result = asyncio.run(app.await_run("r1"))
    assert result["reason"] == "terminal"
    assert result["changed"] is False  # already terminal when the wait began
    assert result["run"]["state"] == "succeeded"
    assert clock["sleeps"] == 0.0  # no recheck pause was needed
    assert result["poll_after_seconds"] == POLL_DEFAULT_SECONDS
    assert isinstance(result["elapsed_seconds"], int)


def test_await_run_times_out_with_unchanged_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _registry, _store = _app_with_run(tmp_path)
    clock = _fake_clock(monkeypatch, app)

    result = asyncio.run(app.await_run("r1", timeout_seconds=AWAIT_MIN_TIMEOUT_SECONDS))
    assert result["reason"] == "timeout"
    assert result["changed"] is False
    assert result["run"]["state"] == "running"
    assert clock["sleeps"] == pytest.approx(AWAIT_MIN_TIMEOUT_SECONDS)


def test_await_run_clamps_timeout_to_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _registry, _store = _app_with_run(tmp_path)
    clock = _fake_clock(monkeypatch, app)

    result = asyncio.run(app.await_run("r1", timeout_seconds=1))

    assert result["reason"] == "timeout"
    assert clock["sleeps"] == pytest.approx(AWAIT_MIN_TIMEOUT_SECONDS)


def test_await_run_clamps_timeout_to_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _registry, _store = _app_with_run(tmp_path)
    clock = _fake_clock(monkeypatch, app)

    result = asyncio.run(app.await_run("r1", timeout_seconds=10_000))
    assert result["reason"] == "timeout"
    assert clock["sleeps"] == pytest.approx(AWAIT_MAX_TIMEOUT_SECONDS)


def test_await_run_returns_when_phase_gains_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, registry, _store = _app_with_run(tmp_path)
    experiment = registry.get("srv").experiment
    clock = {"now": 0.0}

    async def sleep_then_write_winner(seconds: float) -> None:
        clock["now"] += seconds
        winner = _winner_path(experiment, experiment.phases[0].name)
        winner.parent.mkdir(parents=True, exist_ok=True)
        winner.write_text("{}\n")

    monkeypatch.setattr("phasesweep.mcp.server.time.monotonic", lambda: clock["now"])
    app._sleep = sleep_then_write_winner

    result = asyncio.run(app.await_run("r1", timeout_seconds=AWAIT_MAX_TIMEOUT_SECONDS))
    assert result["reason"] == "phase_completed"
    assert result["changed"] is True
    assert result["run"]["state"] == "running"
    assert result["phases"][0]["winner_present"] is True
    # The winner appeared after one recheck pause, well before the timeout.
    assert clock["now"] == pytest.approx(AWAIT_RECHECK_SECONDS)


def test_await_run_returns_when_run_fails_mid_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _registry, store = _app_with_run(tmp_path)
    clock = {"now": 0.0}

    async def sleep_then_fail(seconds: float) -> None:
        clock["now"] += seconds
        write_run_status(
            store,
            "r1",
            returncode=1,
            error_class="RuntimeError",
            cleanup_confirmed=True,
        )

    monkeypatch.setattr("phasesweep.mcp.server.time.monotonic", lambda: clock["now"])
    app._sleep = sleep_then_fail

    result = asyncio.run(app.await_run("r1", timeout_seconds=AWAIT_MAX_TIMEOUT_SECONDS))

    assert result["reason"] == "terminal"
    assert result["changed"] is True
    assert result["run"]["state"] == "failed"
    assert clock["now"] == pytest.approx(AWAIT_RECHECK_SECONDS)


def test_await_run_unknown_run_id(tmp_path: Path) -> None:
    config_text = mcp_experiment_config_text(tmp_path)
    catalog = write_mcp_config_catalog(tmp_path, {"srv": config_text})
    app, _registry, _store = make_mcp_app(catalog)
    with pytest.raises(Exception, match="unknown run id"):
        asyncio.run(app.await_run("missing"))


def test_await_run_is_cancellable_during_recheck_pause(tmp_path: Path) -> None:
    app, _registry, _store = _app_with_run(tmp_path)

    async def exercise() -> None:
        entered_sleep = asyncio.Event()

        async def wait_forever(_seconds: float) -> None:
            entered_sleep.set()
            await asyncio.Event().wait()

        app._sleep = wait_forever
        task = asyncio.create_task(app.await_run("r1"))
        await entered_sleep.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
