"""Status progress, elapsed time, and the await_run wait loop."""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from pathlib import Path

import optuna
import pytest
import yaml

from phasesweep.engine.guards import _validate_artifact_root_binding
from phasesweep.engine.optuna import _phase_study_name
from phasesweep.engine.state import _generation_winner_path, _winner_path
from phasesweep.mcp.redaction import status_payload
from phasesweep.mcp.runs import RunHandle, RunStore, write_status_file
from phasesweep.mcp.server import (
    AWAIT_DEFAULT_TIMEOUT_SECONDS,
    AWAIT_MAX_TIMEOUT_SECONDS,
    AWAIT_MIN_TIMEOUT_SECONDS,
    AWAIT_RECHECK_SECONDS,
    _run_elapsed_seconds,
)
from phasesweep.mcp.snapshots import capture_result_snapshot
from phasesweep.runtime.time import utc_now_iso
from tests.mcp_helpers import (
    make_mcp_app,
    make_run_handle,
    mcp_experiment_config_text,
    write_mcp_config_catalog,
    write_run_status,
)


def _complete_trials(experiment, *, n: int) -> None:
    study = optuna.create_study(
        study_name=_phase_study_name(experiment, experiment.phases[0]),
        storage=experiment.storage,
        direction="minimize",
    )
    for i in range(n):
        trial = study.ask()
        study.tell(trial, float(i))


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


def test_elapsed_seconds_none_without_terminal_timestamp(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = _handle("r1", started_at=utc_now_iso())
    write_status_file(
        store.status_path("r1"),
        {"run_id": "r1", "returncode": 0, "error_class": None, "cleanup_confirmed": True},
    )
    assert _run_elapsed_seconds(store, handle, "succeeded") is None


def test_elapsed_seconds_none_without_status(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    # Terminal with no status.json at all (e.g. SIGKILL before any write).
    assert _run_elapsed_seconds(store, _handle("r1", started_at=utc_now_iso()), "failed") is None


def test_status_reports_progress_fields(tmp_path: Path) -> None:
    config_text = mcp_experiment_config_text(tmp_path)
    catalog = write_mcp_config_catalog(tmp_path, {"srv": config_text})
    app, _registry, _store = make_mcp_app(catalog)

    status = app.status(experiment_id="srv")
    assert status["run"] is None
    assert status["elapsed_seconds"] is None
    (phase,) = status["phases"]
    assert phase["target_terminal_trials"] == 1
    assert phase["completed_trials_total"] == 0
    assert phase["trials"] == {
        "WAITING": 0,
        "RUNNING": 0,
        "COMPLETE": 0,
        "PRUNED": 0,
        "FAIL": 0,
    }
    assert phase["terminal_trials_total"] == 0
    assert phase["terminal_trials_before_run"] == 0
    assert phase["attempts_launched_this_run"] == 0
    assert phase["terminal_trials_this_run"] == 0
    assert phase["target_already_satisfied"] is False
    assert phase["remaining_trials"] == 1
    assert phase["trial_data_available"] is True
    assert status["result_source"] == "current_shared_study"

    # Completed trials feed the per-phase progress counts.
    experiment = _registry.get("srv").experiment
    _complete_trials(experiment, n=3)
    status = app.status(experiment_id="srv")
    (phase,) = status["phases"]
    assert phase["completed_trials_total"] == 3
    assert phase["terminal_trials_total"] == 3
    assert phase["terminal_trials_before_run"] == 3
    assert phase["attempts_launched_this_run"] == 0
    assert phase["terminal_trials_this_run"] == 0
    assert phase["target_already_satisfied"] is True
    assert phase["remaining_trials"] == 0
    assert phase["trial_data_available"] is True


def test_status_floors_inconsistent_historical_terminal_count() -> None:
    """A partial snapshot cannot expose a negative pre-run trial count."""
    status = {
        "current_generation_id": "generation-current",
        "published_generation_id": None,
        "represented_generation_id": "generation-current",
        "is_published": False,
        "publication_integrity": "absent",
        "result_context": "current_config",
        "published_config_matches_current": None,
        "result_phase_plan": ["p"],
        "metric": {"name": "loss", "goal": "minimize"},
        "summary_present": False,
        "phases": [
            {
                "phase": "p",
                "n_trials": 3,
                "trials": {"COMPLETE": 1},
                "generation_trials": {"COMPLETE": 2},
                "winner_present": False,
                "trial_data_available": True,
            }
        ],
    }

    payload = status_payload(
        "srv",
        status,
        None,
        result_source="current_shared_study",
        elapsed_seconds=None,
    )

    assert payload["phases"][0]["terminal_trials_before_run"] == 0


def test_terminal_run_reads_do_not_drift_with_shared_study_state(tmp_path: Path) -> None:
    app, registry, store = _app_with_run(tmp_path)
    experiment = registry.get("srv").experiment
    _complete_trials(experiment, n=2)
    study = optuna.load_study(
        study_name=_phase_study_name(experiment, experiment.phases[0]),
        storage=experiment.storage,
    )
    study.tell(study.ask(), state=optuna.trial.TrialState.FAIL)
    _validate_artifact_root_binding(experiment, claim_fresh=True)
    winner_path = _winner_path(experiment, "p")
    winner_path.parent.mkdir(parents=True, exist_ok=True)
    winner_path.write_text(
        yaml.safe_dump(
            {
                "trial_number": 1,
                "metric": {"loss": 0.25},
                "params": {"lr": 0.00025},
                "effective_overrides": {"lr": 0.00025},
                "winner_source": {
                    "kind": "phase_trial",
                    "phase": "p",
                    "trial_number": 1,
                    "generation_id": "prior-generation",
                    "attempt_id": "attempt-1",
                    "study": None,
                },
            }
        )
    )
    write_run_status(
        store,
        "r1",
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot=capture_result_snapshot(experiment),
    )
    winner_path.write_text(
        yaml.safe_dump(
            {
                "trial_number": 9,
                "metric": {"loss": 9.9},
                "params": {"lr": 0.009},
                "effective_overrides": {"lr": 0.009},
                "winner_source": {
                    "kind": "phase_trial",
                    "phase": "p",
                    "trial_number": 9,
                    "generation_id": "later-generation",
                    "attempt_id": "attempt-9",
                    "study": None,
                },
            }
        )
    )

    run_status = app.status(run_id="r1")
    assert run_status["phases"][0]["trials"] == {
        "WAITING": 0,
        "RUNNING": 0,
        "COMPLETE": 2,
        "PRUNED": 0,
        "FAIL": 1,
    }
    assert run_status["phases"][0]["terminal_trials_total"] == 3
    assert run_status["phases"][0]["terminal_trials_before_run"] == 3
    assert run_status["phases"][0]["attempts_launched_this_run"] == 0
    assert run_status["phases"][0]["terminal_trials_this_run"] == 0
    assert run_status["phases"][0]["target_already_satisfied"] is True
    assert run_status["phases"][0]["remaining_trials"] == 0
    assert run_status["result_source"] == "frozen_run_snapshot"
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
        run_id=run_id,
        experiment_id="srv",
        config_sha256=hashlib.sha256(data).hexdigest(),
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(data)
    return app, registry, store


def _fake_clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """Replace await_run's deadline clock and recheck sleep with a manual clock."""
    clock = {"now": 0.0, "sleeps": 0.0, "pauses": 0.0}

    async def advance(seconds: float) -> None:
        clock["now"] += seconds
        clock["sleeps"] += seconds
        clock["pauses"] += 1

    monkeypatch.setattr("phasesweep.mcp.server.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("phasesweep.mcp.server.asyncio.sleep", advance)
    return clock


def test_await_run_returns_immediately_on_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, registry, store = _app_with_run(tmp_path)
    write_run_status(
        store,
        "r1",
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        ended_at=utc_now_iso(),
        result_snapshot=capture_result_snapshot(
            registry.get("srv").experiment,
        ),
    )
    clock = _fake_clock(monkeypatch)

    result = asyncio.run(app.await_run("r1"))
    assert result["reason"] == "terminal"
    assert result["changed"] is False  # already terminal when the wait began
    assert result["run"]["state"] == "succeeded"
    assert clock["sleeps"] == 0.0  # no recheck pause was needed
    assert isinstance(result["elapsed_seconds"], int)


def test_await_run_times_out_with_unchanged_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _registry, _store = _app_with_run(tmp_path)
    clock = _fake_clock(monkeypatch)

    result = asyncio.run(app.await_run("r1", timeout_seconds=AWAIT_MIN_TIMEOUT_SECONDS))
    assert result["reason"] == "timeout"
    assert result["changed"] is False
    assert result["run"]["state"] == "running"
    assert clock["sleeps"] == pytest.approx(AWAIT_MIN_TIMEOUT_SECONDS)


def test_await_run_rechecks_mid_wait_at_the_default_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _registry, _store = _app_with_run(tmp_path)
    clock = _fake_clock(monkeypatch)

    result = asyncio.run(app.await_run("r1"))

    assert result["reason"] == "timeout"
    assert clock["sleeps"] == pytest.approx(AWAIT_DEFAULT_TIMEOUT_SECONDS)
    # The recheck cadence must divide the default wait into more than one pause,
    # otherwise status is only read at entry and at the deadline.
    assert AWAIT_RECHECK_SECONDS < AWAIT_DEFAULT_TIMEOUT_SECONDS
    assert clock["pauses"] == AWAIT_DEFAULT_TIMEOUT_SECONDS / AWAIT_RECHECK_SECONDS
    assert clock["pauses"] > 1


def test_await_run_reports_failed_trial_progress_at_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, registry, _store = _app_with_run(tmp_path)
    experiment = registry.get("srv").experiment
    clock = {"now": 0.0}

    async def sleep_then_fail_trial(seconds: float) -> None:
        clock["now"] += seconds
        study = optuna.create_study(
            study_name=_phase_study_name(experiment, experiment.phases[0]),
            storage=experiment.storage,
            direction="minimize",
        )
        study.tell(study.ask(), state=optuna.trial.TrialState.FAIL)

    monkeypatch.setattr("phasesweep.mcp.server.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("phasesweep.mcp.server.asyncio.sleep", sleep_then_fail_trial)

    result = asyncio.run(app.await_run("r1", timeout_seconds=AWAIT_MIN_TIMEOUT_SECONDS))

    assert result["reason"] == "timeout"
    assert result["changed"] is True
    assert result["phases"][0]["trials"]["FAIL"] == 1


def test_await_run_returns_immediately_when_recovery_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _registry, store = _app_with_run(tmp_path)
    handle = store.get("r1")
    assert handle is not None
    store.mark_cleanup_uncertain(handle)
    clock = _fake_clock(monkeypatch)

    result = asyncio.run(app.await_run("r1"))

    assert result["reason"] == "recovery_required"
    assert result["changed"] is False
    assert result["run"]["state"] == "running"
    assert result["run"]["recovery_required"] is True
    assert clock["sleeps"] == 0.0


@pytest.mark.parametrize(
    ("requested_timeout", "effective_timeout"),
    [
        pytest.param(1, AWAIT_MIN_TIMEOUT_SECONDS, id="floor"),
        pytest.param(10_000, AWAIT_MAX_TIMEOUT_SECONDS, id="cap"),
    ],
)
def test_await_run_clamps_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested_timeout: float,
    effective_timeout: float,
) -> None:
    app, _registry, _store = _app_with_run(tmp_path)
    clock = _fake_clock(monkeypatch)

    result = asyncio.run(app.await_run("r1", timeout_seconds=requested_timeout))

    assert result["reason"] == "timeout"
    assert clock["sleeps"] == pytest.approx(effective_timeout)


def _await_with_timed_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    read_seconds: float,
) -> tuple[dict[str, object], list[float], float]:
    """Run one minimum-timeout wait against a manually timed status reader."""
    app, _registry, _store = _app_with_run(tmp_path)
    clock = {"now": 0.0}
    read_starts: list[float] = []
    real_read = app._read_status_target

    def timed_read(**kwargs):
        read_starts.append(clock["now"])
        result = real_read(**kwargs)
        clock["now"] += read_seconds
        return result

    async def advance(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr("phasesweep.mcp.server.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("phasesweep.mcp.server.asyncio.sleep", advance)
    monkeypatch.setattr(app, "_read_status_target", timed_read)

    result = asyncio.run(app.await_run("r1", timeout_seconds=AWAIT_MIN_TIMEOUT_SECONDS))
    return result, read_starts, clock["now"]


@pytest.mark.parametrize(
    ("read_seconds", "expected_elapsed", "expect_single_read"),
    [
        pytest.param(0.2, AWAIT_MIN_TIMEOUT_SECONDS, False, id="reserve-final-status-read"),
        pytest.param(2.6, AWAIT_MIN_TIMEOUT_SECONDS, True, id="wait-remaining-budget"),
        pytest.param(6.1, 6.1, True, id="in-progress-read-crosses-deadline"),
    ],
)
def test_await_run_read_timeout_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_seconds: float,
    expected_elapsed: float,
    expect_single_read: bool,
) -> None:
    """Do not return early, but do not claim a blocking read can be preempted."""
    result, read_starts, elapsed = _await_with_timed_reads(
        tmp_path,
        monkeypatch,
        read_seconds=read_seconds,
    )

    assert result["reason"] == "timeout"
    if expect_single_read:
        assert read_starts == [0.0]
    else:
        assert len(read_starts) > 1
        assert max(read_starts) < AWAIT_MIN_TIMEOUT_SECONDS
    assert elapsed == pytest.approx(expected_elapsed)


def test_await_run_returns_when_phase_gains_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, registry, _store = _app_with_run(tmp_path)
    experiment = registry.get("srv").experiment
    clock = {"now": 0.0}

    async def sleep_then_write_winner(seconds: float) -> None:
        clock["now"] += seconds
        _validate_artifact_root_binding(experiment, claim_fresh=True)
        winner = _generation_winner_path(experiment, "r1", experiment.phases[0].name)
        winner.parent.mkdir(parents=True, exist_ok=True)
        winner.write_text("{}\n")

    monkeypatch.setattr("phasesweep.mcp.server.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("phasesweep.mcp.server.asyncio.sleep", sleep_then_write_winner)

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
    app, registry, store = _app_with_run(tmp_path)
    clock = {"now": 0.0}

    async def sleep_then_fail(seconds: float) -> None:
        clock["now"] += seconds
        write_run_status(
            store,
            "r1",
            returncode=1,
            error_class="RuntimeError",
            cleanup_confirmed=True,
            result_snapshot=capture_result_snapshot(
                registry.get("srv").experiment,
            ),
        )

    monkeypatch.setattr("phasesweep.mcp.server.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("phasesweep.mcp.server.asyncio.sleep", sleep_then_fail)

    result = asyncio.run(app.await_run("r1", timeout_seconds=AWAIT_MAX_TIMEOUT_SECONDS))

    assert result["reason"] == "terminal"
    assert result["changed"] is True
    assert result["run"]["state"] == "failed"
    assert clock["now"] == pytest.approx(AWAIT_RECHECK_SECONDS)


def test_terminal_run_snapshot_failure_returns_structured_unavailable_results(
    tmp_path: Path,
) -> None:
    app, _registry, store = _app_with_run(tmp_path)
    write_run_status(store, "r1", returncode=0, error_class=None, cleanup_confirmed=True)

    status = app.status(run_id="r1")
    winners = app.winners(run_id="r1")
    awaited = asyncio.run(app.await_run("r1"))

    assert status["result_source"] == "terminal_snapshot_unavailable"
    assert status["publication_integrity"] == "unknown"
    assert status["run"]["state"] == "succeeded"
    assert status["run"]["failure"]["code"] == "result_snapshot_unavailable"
    assert status["run"]["failure"]["actor"] == "operator"
    assert status["phases"][0]["trial_data_available"] is False
    assert winners["result_source"] == "terminal_snapshot_unavailable"
    assert winners["publication_integrity"] == "unknown"
    assert winners["winner_count"] == 0
    assert winners["failure"]["code"] == "result_snapshot_unavailable"
    assert awaited["reason"] == "terminal"
    assert awaited["changed"] is False
    assert awaited["result_source"] == "terminal_snapshot_unavailable"
    assert awaited["publication_integrity"] == "unknown"
    assert awaited["run"]["failure"]["code"] == "result_snapshot_unavailable"


def test_await_run_unknown_run_id(tmp_path: Path) -> None:
    config_text = mcp_experiment_config_text(tmp_path)
    catalog = write_mcp_config_catalog(tmp_path, {"srv": config_text})
    app, _registry, _store = make_mcp_app(catalog)
    with pytest.raises(Exception, match="unknown run id"):
        asyncio.run(app.await_run("missing"))


def test_await_run_storage_read_does_not_block_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _registry, _store = _app_with_run(tmp_path)
    target_id, status, run, handle, result_source = app._read_status_target(
        experiment_id=None,
        run_id="r1",
    )
    assert run is not None and handle is not None
    terminal_run = {**run, "state": "failed"}
    entered = threading.Event()
    release = threading.Event()

    def blocked_read(**_kwargs: object):
        entered.set()
        release.wait(timeout=2.0)
        return target_id, status, terminal_run, handle, result_source

    monkeypatch.setattr(app, "_read_status_target", blocked_read)

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, release.set)
        started = time.monotonic()

        result = await app.await_run("r1")

        assert time.monotonic() - started < 1.0
        assert entered.is_set()
        assert result["run"]["state"] == "failed"

    asyncio.run(exercise())


def test_await_run_is_cancellable_during_recheck_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _registry, _store = _app_with_run(tmp_path)

    async def exercise() -> None:
        entered_sleep = asyncio.Event()

        async def wait_forever(_seconds: float) -> None:
            entered_sleep.set()
            await asyncio.Event().wait()

        monkeypatch.setattr("phasesweep.mcp.server.asyncio.sleep", wait_forever)
        task = asyncio.create_task(app.await_run("r1"))
        await entered_sleep.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
