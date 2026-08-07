"""Detached runner: exercises the real subprocess, the engine's signal teardown,
and the status.json written on the cancel path.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna
import pytest
import yaml

from phasesweep.config import Experiment, Phase, Sampler, load_config
from phasesweep.engine import (
    ExperimentLockBusyError,
    NoFeasibleTrialError,
    ProcessCleanupUncertainError,
    SamplerContinuationUnsupportedError,
    StudyStorageUnavailableError,
    TerminalReport,
    TrialTargetRegressionError,
    read_status,
    run_experiment,
)
from phasesweep.engine.guards import _experiment_lock
from phasesweep.engine.state import (
    Winner,
    _generation_path,
    _generations_dir,
    _last_successful_generation_id,
    _summary_path,
    _trial_dir_for,
    _winner_path,
)
from phasesweep.mcp import runner as mcp_runner
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.mcp.time import utc_now_iso
from phasesweep.runtime.files import open_private_text
from phasesweep.runtime.process import (
    PROCESS_IDENTITY_FILE,
    PhaseSweepShutdown,
    ShutdownCleanupReport,
    _process_group_alive,
    read_boot_id,
)
from tests.conftest import REPO, make_experiment, write_constant_trainer, write_trainer
from tests.mcp_helpers import (
    claim_runner_handle,
    make_run_handle,
    runner_main,
    slow_mcp_config_text,
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="cancel path relies on POSIX process groups + /proc liveness",
)

# Persistent storage rejects an unseeded stochastic sampler; seeded random is
# reproducible and resumable, so it needs no non-resumable acknowledgement.
SEEDED_RANDOM = Sampler(type="random", seed=0)


def _slow_config(tmp_path: Path, *, sleep: float = 30.0) -> Path:
    config = tmp_path / "exp.yaml"
    config.write_text(
        slow_mcp_config_text(
            tmp_path,
            trainer=REPO / "src" / "phasesweep" / "examples" / "fake_train.py",
            name="cancel_me",
            sleep=sleep,
        )
    )
    return config


def _constant_trial_config(tmp_path: Path, name: str) -> tuple[Path, str]:
    """Write a one-trial experiment the detached runner can complete in-process.

    :param Path tmp_path: Directory receiving the trainer, config, and workdir.
    :param str name: Experiment name, also the catalog id used by the runner.
    :return tuple[Path, str]: Config path and its SHA-256, as the server pins it.
    """
    trainer = write_constant_trainer(tmp_path)
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(
        f"""
experiment: {name}
workdir: {tmp_path}/runs
trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
metric:
  name: x
  goal: minimize
  extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
phases:
  - name: p
    n_trials: 1
    search_space: {{}}
"""
    )
    return config_path, hashlib.sha256(config_path.read_bytes()).hexdigest()


def _wait_for_running_trial(config: Path, proc: subprocess.Popen, log_path: Path) -> Path:
    experiment = load_config(config)
    deadline = time.time() + 25
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"runner exited early ({proc.returncode}); log:\n{log_path.read_text()}"
            )
        status = read_status(experiment)
        if status["phases"][0]["running"] >= 1:
            phase_dir = _trial_dir_for(experiment, "p", 0).parent
            for trial_dir in phase_dir.glob("trial_00000__*"):
                if (trial_dir / PROCESS_IDENTITY_FILE).is_file():
                    return trial_dir
        time.sleep(0.2)
    raise AssertionError(f"trial never reached RUNNING; log:\n{log_path.read_text()}")


def test_runner_persists_terminal_evidence_before_snapshot_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_path = tmp_path / "status.json"
    observed: dict[str, object] = {}

    def fail_finalization(*_args: object, **_kwargs: object) -> dict[str, object]:
        observed.update(json.loads(status_path.read_text()))
        raise RuntimeError("snapshot serialization failed")

    monkeypatch.setattr(mcp_runner, "finalize_result_snapshot", fail_finalization)
    mcp_runner._write_status(
        status_path,
        {
            "run_id": "r0",
            "returncode": 143,
            "error_class": "cancelled",
            "cleanup_confirmed": True,
        },
        result_snapshot={},
        result_snapshot_error=None,
    )

    assert observed["error_class"] == "cancelled"
    assert observed["cleanup_confirmed"] is True
    assert "ended_at" in observed
    assert observed["result_snapshot_state"] == "pending"
    assert observed["result_snapshot"] == {}
    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "failed"
    assert final["result_snapshot_error"] == "RuntimeError"
    assert "result_snapshot" not in final


def test_runner_defers_shutdown_until_snapshot_finalization_is_durable(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    child_code = """
import sys
import time
from pathlib import Path

import phasesweep.mcp.runner as runner

runner.install_signal_handlers()

def slow_finalization(snapshot):
    time.sleep(0.5)
    return snapshot

runner.finalize_result_snapshot = slow_finalization
runner._write_status(
    Path(sys.argv[1]),
    {
        "run_id": "r-signal",
        "returncode": 0,
        "error_class": None,
        "cleanup_confirmed": True,
    },
    result_snapshot={"captured": True},
    result_snapshot_error=None,
)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code, str(status_path)],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if status_path.is_file():
                status = json.loads(status_path.read_text())
                if status.get("result_snapshot_state") == "pending":
                    break
            if proc.poll() is not None:
                raise AssertionError(f"finalizer exited early with {proc.returncode}")
            time.sleep(0.01)
        else:
            raise AssertionError("runner did not persist the pending snapshot state")

        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)

    assert proc.returncode == 143
    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "complete"
    assert final["result_snapshot"] == {"captured": True}


def test_runner_finalizes_pre_captured_terminal_snapshot(tmp_path: Path) -> None:
    config = _slow_config(tmp_path)
    status_path = tmp_path / "status.json"
    experiment = load_config(config)
    assert isinstance(experiment, Experiment)
    snapshot = mcp_runner.capture_result_snapshot(experiment)

    mcp_runner._write_status(
        status_path,
        {
            "run_id": "r0",
            "returncode": 0,
            "error_class": None,
            "cleanup_confirmed": True,
        },
        result_snapshot=snapshot,
        result_snapshot_error=None,
    )

    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "complete"
    assert final["result_snapshot"]["status"]["phases"][0]["phase"] == "p"


def test_terminal_snapshot_is_captured_before_experiment_lock_release(tmp_path: Path) -> None:
    trainer = write_constant_trainer(tmp_path)
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        n_trials=1,
        sampler=Sampler(type="random", seed=0),
    )
    captured: dict[str, object] = {}

    def capture_locked(report: TerminalReport) -> None:
        assert report.primary_error is None
        with (
            pytest.raises(RuntimeError, match="Another phasesweep process"),
            _experiment_lock(experiment),
        ):
            pass
        captured.update(
            mcp_runner.capture_result_snapshot(
                experiment,
                generation_id=report.generation_id,
            )
        )

    run_experiment(experiment, terminal_callback=capture_locked)

    top_up = experiment.model_copy(
        update={"phases": [experiment.phases[0].model_copy(update={"n_trials": 2})]}
    )
    run_experiment(top_up)

    captured_phase = captured["status"]["phases"][0]  # type: ignore[index]
    assert captured_phase["n_trials"] == 1
    assert captured_phase["completed"] == 1
    assert captured_phase["generation_trials"] == {"COMPLETE": 1}
    current_phase = mcp_runner.capture_result_snapshot(top_up)["status"]["phases"][0]
    assert current_phase["n_trials"] == 2
    assert current_phase["completed"] == 2
    assert current_phase["generation_trials"] == {"COMPLETE": 1}


@pytest.mark.parametrize(
    ("error", "code"),
    [
        pytest.param(
            SamplerContinuationUnsupportedError("unsupported continuation"),
            "sampler_continuation_unsupported",
            id="sampler",
        ),
        pytest.param(
            TrialTargetRegressionError("target moved backward"),
            "trial_target_regression",
            id="target",
        ),
    ],
)
def test_continuation_preflight_failures_have_actionable_mcp_categories(
    error: RuntimeError,
    code: str,
) -> None:
    failure = mcp_runner._safe_failure_payload(error, stage="preflight")

    assert failure["code"] == code
    assert failure["stage"] == "preflight"
    assert failure["retryable"] is False
    assert failure["actor"] == "operator"


def test_external_engine_lock_is_retryable_and_freezes_pre_generation_snapshot(
    tmp_path: Path,
) -> None:
    config_path = _slow_config(tmp_path)
    experiment = load_config(config_path)
    assert isinstance(experiment, Experiment)
    store = RunStore(tmp_path / "state")
    run_id = "lock-busy"
    status_path = store.status_path(run_id)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    started_at = utc_now_iso()
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
        experiment_id="cancel_me",
    )

    with (
        _experiment_lock(experiment),
        pytest.raises(ExperimentLockBusyError),
    ):
        runner_main(
            [
                "--run-id",
                run_id,
                "--config",
                str(config_path),
                "--config-sha256",
                config_sha256,
                "--status-path",
                str(status_path),
                "--state-dir",
                str(tmp_path / "state"),
                "--experiment-id",
                "cancel_me",
                "--started-at",
                started_at,
            ],
            cwd=tmp_path,
        )

    status = json.loads(status_path.read_text())
    assert status["failure"]["code"] == "experiment_busy"
    assert status["failure"]["stage"] == "preflight"
    assert status["failure"]["retryable"] is True
    assert status["failure"]["actor"] == "agent"
    assert status["generation_unavailable_reason"] == "engine_generation_not_claimed"
    assert status["result_snapshot_state"] == "complete"
    snapshot = status["result_snapshot"]
    assert snapshot["status"]["current_generation_id"] is None
    assert snapshot["status"]["published_generation_id"] is None
    assert snapshot["winners"] == []
    assert all(phase["trial_data_available"] is False for phase in snapshot["status"]["phases"])


def test_terminal_snapshot_reads_partial_winners_from_failed_generation(tmp_path: Path) -> None:
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import os
        import sys

        if os.environ["PHASESWEEP_PHASE"] == "a":
            print("x=0.5")
        else:
            sys.exit(2)
        """,
    )
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        trial_command=f"{sys.executable} {trainer} {{overrides}}",
        phases=[
            Phase(name="a", n_trials=1, sampler=SEEDED_RANDOM, search_space={}),
            Phase(
                name="b",
                n_trials=1,
                max_consecutive_failures=1,
                sampler=SEEDED_RANDOM,
                search_space={},
            ),
        ],
    )
    captured: dict[str, object] = {}

    def capture_failed(report: TerminalReport) -> None:
        assert isinstance(report.primary_error, NoFeasibleTrialError)
        captured.update(
            mcp_runner.capture_result_snapshot(
                experiment,
                generation_id=report.generation_id,
            )
        )

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment, terminal_callback=capture_failed)

    phases = captured["status"]["phases"]  # type: ignore[index]
    assert phases[0]["winner_present"] is True
    assert phases[1]["winner_present"] is False
    (winner,) = captured["winners"]  # type: ignore[index]
    assert winner["phase"] == "a"
    assert winner["source"] == {
        "kind": "phase_trial",
        "phase": "a",
        "trial_number": winner["trial_number"],
        "generation_id": winner["generation_id"],
        "attempt_id": winner["attempt_id"],
        "study": None,
    }


def test_terminal_snapshot_tolerates_missing_lifecycle_record(tmp_path: Path) -> None:
    """A missing per-generation record must not fail the snapshot capture.

    The engine writes that record as an optional post-commit diagnostic; a
    capture that hard-required it contradicted engine-defined success
    whenever the best-effort write had failed (review v0.5.16 / blocker 2).
    """
    experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1)
    generation_path = _generation_path(experiment)
    generation_path.parent.mkdir(parents=True)
    generation_path.write_text("generation_id: prior-generation\n")

    snapshot = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id="pinned-generation",
    )

    assert snapshot["status"]["represented_generation_id"] == "pinned-generation"
    assert snapshot["status"]["is_published"] is False


def test_snapshot_finalization_keeps_prior_attempt_out_of_generation_counts(
    tmp_path: Path,
) -> None:
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        phases=[Phase(name="p", n_trials=1, sampler=SEEDED_RANDOM, search_space={})],
    )
    study = optuna.create_study(study_name="t::p", storage=experiment.storage, direction="minimize")
    trial = study.ask()
    trial.set_user_attr("phasesweep_generation_id", "old-generation")
    trial.set_user_attr("phasesweep_attempt_id", "old-attempt")

    captured = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id="current-generation",
    )
    unowned = mcp_runner.finalize_result_snapshot(captured)

    assert unowned == captured
    assert unowned["status"]["phases"][0]["trials"]["RUNNING"] == 1

    finalized = mcp_runner.finalize_result_snapshot(
        captured,
        confirmed_attempt_ids={"old-attempt"},
    )

    phase = finalized["status"]["phases"][0]
    assert phase["trials"]["RUNNING"] == 0
    assert phase["trials"]["FAIL"] == 1
    assert phase["generation_trials"] == {}
    assert study.get_trials(deepcopy=False)[0].state == optuna.trial.TrialState.RUNNING


@pytest.mark.parametrize("storage_kind", ["none", "missing-sqlite"])
def test_terminal_snapshot_freezes_unavailable_trial_data_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage_kind: str,
) -> None:
    """Unavailable trial storage freezes explicit flags without a redundant study read.

    ``running_attempts`` is ``None`` rather than ``[]`` here: nothing was read,
    so an empty list would claim there are no RUNNING rows (PR #5 review /
    reviewer 2, blocker 6).
    """
    storage = None if storage_kind == "none" else f"sqlite:///{tmp_path / 'missing' / 'studies.db'}"
    experiment = make_experiment(workdir=tmp_path / "runs", storage=storage, n_trials=1)

    def fail_redundant_read(*args: object, **kwargs: object) -> None:
        raise OSError("storage remains unavailable")

    # Patched on Optuna itself, so the guard holds no matter how PhaseSweep
    # imports its own study helpers: any study load at all fails the capture.
    monkeypatch.setattr(optuna, "load_study", fail_redundant_read)

    snapshot = mcp_runner.capture_result_snapshot(experiment)

    phase = snapshot["status"]["phases"][0]
    assert phase["trial_data_available"] is False
    assert phase["running_attempts"] is None


def test_terminal_snapshot_survives_a_post_engine_study_load_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful run's result is frozen even if studies became unloadable.

    PR #5 review / reviewer 2, blocker 6: the capture used to reread every
    phase study after ``read_status`` to collect RUNNING rows, with no
    tolerance for failure. One transient lock in that millisecond-wide window
    failed the capture, and a terminal snapshot that was never captured is
    unrecoverable by design -- so a completed run's frozen result was lost to
    storage flakiness that the tolerant status read itself shrugs off. The
    fatal read simply no longer happens.
    """
    trainer = write_constant_trainer(tmp_path)
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        n_trials=1,
        sampler=SEEDED_RANDOM,
    )
    reports: list[TerminalReport] = []

    run_experiment(experiment, terminal_callback=reports.append)

    (report,) = reports
    assert report.primary_error is None
    assert report.winners

    loads = 0

    def refuse_to_load(*args: object, **kwargs: object) -> None:
        nonlocal loads
        loads += 1
        raise StudyStorageUnavailableError("database is locked")

    # Patched on Optuna itself rather than on a PhaseSweep helper, so no
    # import binding can hide a study reload from this test. Every remaining
    # storage read must be the tolerant one read_status performs.
    monkeypatch.setattr(optuna, "load_study", refuse_to_load)

    snapshot = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id=report.generation_id,
        engine_winners=report.winners,
    )

    assert loads == 0
    phase = snapshot["status"]["phases"][0]
    assert phase["trial_data_available"] is True
    assert phase["running_attempts"] == []
    assert phase["completed"] == 1
    (frozen,) = snapshot["winners"]
    assert frozen["phase"] == "p"
    assert frozen["metric"] == report.winners["p"].metric
    assert frozen["trial_number"] == report.winners["p"].trial_number

    status_path = tmp_path / "status.json"
    mcp_runner._write_status(
        status_path,
        {"run_id": "r0", "returncode": 0, "error_class": None, "cleanup_confirmed": True},
        result_snapshot=snapshot,
        result_snapshot_error=None,
    )

    assert json.loads(status_path.read_text())["result_snapshot_state"] == "complete"


def test_terminal_snapshot_reports_running_attempts_from_the_status_read(
    tmp_path: Path,
) -> None:
    """RUNNING identities come from the one tolerant read, unchanged in shape."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        phases=[Phase(name="p", n_trials=2, sampler=SEEDED_RANDOM, search_space={})],
    )
    study = optuna.create_study(study_name="t::p", storage=experiment.storage, direction="minimize")
    identified = study.ask()
    identified.set_user_attr("phasesweep_generation_id", "current-generation")
    identified.set_user_attr("phasesweep_attempt_id", "current-attempt")
    # A RUNNING row written before attempt identity existed: reported with
    # null identity, never dropped, so the counts and the list agree.
    study.ask()

    snapshot = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id="current-generation",
    )

    phase = snapshot["status"]["phases"][0]
    assert phase["trials"]["RUNNING"] == 2
    assert phase["trial_data_available"] is True
    assert phase["running_attempts"] == [
        {
            "trial_number": 0,
            "generation_id": "current-generation",
            "attempt_id": "current-attempt",
        },
        {"trial_number": 1, "generation_id": None, "attempt_id": None},
    ]

    finalized = mcp_runner.finalize_result_snapshot(
        snapshot,
        confirmed_attempt_ids={"current-attempt"},
    )

    finalized_phase = finalized["status"]["phases"][0]
    assert finalized_phase["trials"] == {"RUNNING": 1, "FAIL": 1}
    assert finalized_phase["generation_trials"] == {"RUNNING": 0, "FAIL": 1}
    assert finalized_phase["running_attempts"] == [
        {"trial_number": 1, "generation_id": None, "attempt_id": None}
    ]


def test_snapshot_freezes_engine_winners_without_rereading_files(tmp_path: Path) -> None:
    """Engine-supplied winners are frozen verbatim, with no second file read."""
    experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1)
    winner = Winner(
        trial_number=4,
        params={"x": 3},
        effective_overrides={"x": 3},
        metric=0.25,
        gates=[{"type": "g", "passed": True}],
        completion={"incomplete": False},
        generation_id="engine-generation",
        attempt_id="engine-attempt",
    )

    snapshot = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id="engine-generation",
        engine_winners={"p": winner},
    )

    # No winner file exists anywhere on disk; the frozen winner is exactly
    # the engine's own in-memory outcome.
    (frozen,) = snapshot["winners"]
    assert frozen["phase"] == "p"
    assert frozen["trial_number"] == 4
    assert frozen["metric"] == 0.25
    assert frozen["gates_passed"] is True
    assert frozen["incomplete"] is False
    assert frozen["generation_id"] == "engine-generation"
    assert frozen["source"] == {
        "kind": "phase_trial",
        "phase": "p",
        "trial_number": 4,
        "generation_id": "engine-generation",
        "attempt_id": "engine-attempt",
        "study": None,
    }


def test_record_write_failure_still_yields_succeeded_run_with_complete_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end (review v0.5.16 / blocker 2): the engine and MCP agree on success.

    The per-generation lifecycle record write is injected to fail after the
    pointer commit — a failure the engine's own transaction contract accepts.
    The detached runner must still record returncode 0 with a complete frozen
    snapshot, and the run store must derive ``succeeded``.
    """
    import phasesweep.engine.run as engine_run

    trainer = write_constant_trainer(tmp_path)
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        f"""
experiment: record_fail
workdir: {tmp_path}/runs
trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
metric:
  name: x
  goal: minimize
  extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
phases:
  - name: p
    n_trials: 1
    search_space: {{}}
"""
    )
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    def fail_record_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated record write failure")

    monkeypatch.setattr(engine_run, "_write_generation_record_once", fail_record_write)

    store = RunStore(tmp_path / "state")
    run_id = "record-fail-run"
    started_at = utc_now_iso()
    store.create(
        RunHandle(
            run_id=run_id,
            experiment_id="record_fail",
            config_sha256=config_sha256,
            pid=None,
            pgid=None,
            pid_starttime=None,
            started_at=started_at,
            launch_state="launching",
        )
    )
    status_path = store.status_path(run_id)

    assert (
        runner_main(
            [
                "--run-id",
                run_id,
                "--config",
                str(config_path),
                "--config-sha256",
                config_sha256,
                "--status-path",
                str(status_path),
                "--state-dir",
                str(tmp_path / "state"),
                "--experiment-id",
                "record_fail",
                "--started-at",
                started_at,
            ],
            cwd=tmp_path,
        )
        == 0
    )

    terminal = json.loads(status_path.read_text())
    assert terminal["returncode"] == 0
    assert terminal["result_snapshot_state"] == "complete"
    assert [w["phase"] for w in terminal["result_snapshot"]["winners"]] == ["p"]
    handle = store.get(run_id)
    assert handle is not None
    assert store.state(handle) == "succeeded"


def test_shutdown_during_terminal_snapshot_capture_keeps_the_published_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end (PR #5 review / reviewer 2 pass 2, blocker 3): cancel loses the race.

    A real SIGTERM is delivered to this process while the terminal snapshot of
    an already-published generation is being captured. The engine invokes that
    callback inside a diagnostic boundary that swallows ``BaseException``, so
    an unabsorbed :class:`PhaseSweepShutdown` would be consumed there and the
    frozen result destroyed for good: the run would end ``returncode`` 0 with
    ``result_snapshot_state="failed"``, no snapshot, and no trace of the
    cancellation - and ``recover-run`` refuses to rebuild a missing snapshot.

    The absorbed window makes the outcome deterministic instead: the snapshot
    is captured and durably ``complete``, the engine's published outcome is
    untouched, and only then does the process exit with the POSIX signalled
    code for the signal it held.
    """
    import phasesweep.runtime.process as runtime_process

    # Restores the module's pending-shutdown marker to ``None`` at teardown, so
    # a failure before the runner services the signal cannot leak an absorbed
    # shutdown into an unrelated later test.
    monkeypatch.setattr(runtime_process, "_deferred_shutdown_signum", None)

    config_path, config_sha256 = _constant_trial_config(tmp_path, "cancel_at_capture")
    real_capture = mcp_runner.capture_result_snapshot

    def capture_under_shutdown(*args: object, **kwargs: object) -> dict:
        # A real signal through the real installed handler: the absorb window
        # is the only thing that can keep it from raising through the capture.
        os.kill(os.getpid(), signal.SIGTERM)
        return real_capture(*args, **kwargs)

    monkeypatch.setattr(mcp_runner, "capture_result_snapshot", capture_under_shutdown)

    store = RunStore(tmp_path / "state")
    run_id = "capture-cancel"
    started_at = utc_now_iso()
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
        experiment_id="cancel_at_capture",
    )

    with pytest.raises(PhaseSweepShutdown) as exc_info:
        runner_main(
            [
                "--run-id",
                run_id,
                "--config",
                str(config_path),
                "--config-sha256",
                config_sha256,
                "--status-path",
                str(store.status_path(run_id)),
                "--state-dir",
                str(tmp_path / "state"),
                "--experiment-id",
                "cancel_at_capture",
                "--started-at",
                started_at,
            ],
            cwd=tmp_path,
        )

    # The shutdown is honored, but only after terminal evidence is durable:
    # it surfaces out of the terminal status write's own defer window.
    assert exc_info.value.signum == signal.SIGTERM
    assert exc_info.value.code == 128 + signal.SIGTERM
    assert runtime_process._deferred_shutdown_signum is None

    terminal = json.loads(store.status_path(run_id).read_text())
    assert terminal["result_snapshot_state"] == "complete"
    assert [w["phase"] for w in terminal["result_snapshot"]["winners"]] == ["p"]
    # The engine published before the signal arrived, so its own outcome - and
    # the terminal status recording it - stay a success.
    assert terminal["returncode"] == 0
    assert terminal["error_class"] is None
    assert terminal["failure"] is None
    assert _last_successful_generation_id(load_config(config_path)) == run_id


@pytest.mark.parametrize("from_phase", [None, "b"])
def test_failed_fingerprint_preflight_preserves_published_results(
    tmp_path: Path,
    from_phase: str | None,
) -> None:
    """A failed preflight advances the current pointer but never the published one.

    The current-generation pointer legitimately moves to this new (failed)
    invocation -- a new invocation always overwrites it starting from
    "preflighting", and every outcome path must drive it to a terminal state
    (review v0.5.15 / blocker 3) -- but the legacy compatibility caches and,
    critically, the last-success pointer stay exactly as the prior successful
    publication left them.
    """
    trainer = write_constant_trainer(tmp_path)
    phases = [
        Phase(
            name="a", n_trials=1, fixed_overrides={"k": 1}, sampler=SEEDED_RANDOM, search_space={}
        ),
        Phase(name="b", n_trials=1, inherits=["a"], sampler=SEEDED_RANDOM, search_space={}),
    ]
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        phases=phases,
    )
    run_experiment(experiment)
    first_generation = _last_successful_generation_id(experiment)
    assert first_generation is not None

    # The legacy compatibility caches must stay untouched by a failed resume;
    # the current-generation pointer is deliberately excluded here since it
    # legitimately advances even on a preflight failure (see docstring).
    protected_paths = [
        _summary_path(experiment),
        *(_winner_path(experiment, phase.name) for phase in phases),
    ]
    before = {path: path.read_bytes() for path in protected_paths}
    generations_before = set(_generations_dir(experiment).iterdir())
    changed = experiment.model_copy(
        update={
            "phases": [
                phases[0].model_copy(update={"fixed_overrides": {"k": 2}}),
                phases[1],
            ]
        }
    )
    callback_generations: list[str] = []
    captured: list[dict[str, object]] = []

    def capture_failed_resume(report: TerminalReport) -> None:
        assert isinstance(report.primary_error, RuntimeError)
        callback_generations.append(report.generation_id)
        captured.append(
            mcp_runner.capture_result_snapshot(
                changed,
                generation_id=report.generation_id,
            )
        )

    with pytest.raises(RuntimeError, match="different phase config"):
        run_experiment(
            changed,
            from_phase=from_phase,
            terminal_callback=capture_failed_resume,
        )

    assert len(callback_generations) == 1
    assert len(captured) == 1
    # current_generation_id is always the actual mutable pointer, which this
    # failed preflight attempt does legitimately claim (a new invocation
    # overwrites it with "preflighting" before it knows whether preflight
    # will succeed). published_generation_id is always the actual validated
    # last-success pointer -- the *first* (successful) run here -- never
    # forced to equal the pinned generation_id, even though this pinned
    # capture's own represented_generation_id is that failed generation
    # (review v0.5.15 / blocker 3, defect 2: "pinned reads lie").
    assert captured[0]["status"]["current_generation_id"] == callback_generations[0]  # type: ignore[index]
    assert captured[0]["status"]["published_generation_id"] == first_generation  # type: ignore[index]
    assert captured[0]["status"]["represented_generation_id"] == callback_generations[0]  # type: ignore[index]
    assert captured[0]["status"]["is_published"] is False  # type: ignore[index]
    assert captured[0]["status"]["summary_present"] is False  # type: ignore[index]
    assert all(
        phase["generation_trials"] == {}
        for phase in captured[0]["status"]["phases"]  # type: ignore[index]
    )
    assert captured[0]["winners"] == []
    # Legacy compatibility caches are untouched, and the published pointer
    # still resolves to the prior successful generation.
    assert {path: path.read_bytes() for path in protected_paths} == before
    assert _last_successful_generation_id(experiment) == first_generation
    failed_generations = set(_generations_dir(experiment).iterdir()) - generations_before
    assert len(failed_generations) == 1
    assert "state: failed" in (failed_generations.pop() / "generation.yaml").read_text()
    # The current pointer legitimately advanced to this failed attempt.
    current_pointer = yaml.safe_load(_generation_path(experiment).read_text())
    assert current_pointer["generation_id"] == callback_generations[0]
    assert current_pointer["state"] == "failed"


def test_terminal_report_preserves_secondary_cleanup_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A primary run failure cannot hide a later cleanup-uncertain result."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    calls = 0
    cleanup_error = ProcessCleanupUncertainError("cleanup could not be proven")

    def preflight(_experiment: Experiment, *, cleanup_report, from_phase) -> dict:
        nonlocal calls
        del from_phase
        calls += 1
        if calls == 2:
            cleanup_report.mark_uncertain(cleanup_error)
            raise cleanup_error
        return {}

    def fail_run(*args: object, **kwargs: object) -> None:
        raise NoFeasibleTrialError("trainer failed")

    captured: list[TerminalReport] = []
    monkeypatch.setattr("phasesweep.engine.run._preflight_existing_studies", preflight)
    monkeypatch.setattr("phasesweep.engine.run._run_experiment_inner", fail_run)

    with pytest.raises(ProcessCleanupUncertainError) as exc_info:
        run_experiment(experiment, terminal_callback=captured.append)

    assert isinstance(exc_info.value.__cause__, NoFeasibleTrialError)
    assert calls == 2
    assert len(captured) == 1
    report = captured[0]
    assert isinstance(report.primary_error, NoFeasibleTrialError)
    assert report.cleanup_confirmed is False
    assert report.cleanup_error is cleanup_error


def test_terminal_report_preserves_shutdown_cleanup_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Storage reconciliation cannot override shutdown process evidence."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    shutdown = PhaseSweepShutdown(
        signal.SIGTERM,
        ShutdownCleanupReport(
            signum=signal.SIGTERM,
            cleanup_confirmed=False,
            child_pgids=(1234,),
        ),
    )
    preflight_calls = 0

    def preflight(_experiment: Experiment, *, cleanup_report, from_phase) -> dict:
        nonlocal preflight_calls
        del cleanup_report, from_phase
        preflight_calls += 1
        return {}

    def cancel_run(*args: object, **kwargs: object) -> None:
        raise shutdown

    captured: list[TerminalReport] = []
    monkeypatch.setattr("phasesweep.engine.run._preflight_existing_studies", preflight)
    monkeypatch.setattr("phasesweep.engine.run._run_experiment_inner", cancel_run)

    with pytest.raises(PhaseSweepShutdown) as exc_info:
        run_experiment(experiment, terminal_callback=captured.append)

    assert exc_info.value is shutdown
    assert preflight_calls == 2
    assert len(captured) == 1
    report = captured[0]
    assert report.primary_error is shutdown
    assert report.cleanup_confirmed is False
    assert report.cleanup_error is shutdown


def test_shutdown_during_post_error_reconciliation_remains_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = make_experiment(workdir=tmp_path / "runs")
    shutdown = PhaseSweepShutdown(
        signal.SIGTERM,
        ShutdownCleanupReport(
            signum=signal.SIGTERM,
            cleanup_confirmed=True,
            child_pgids=(),
        ),
    )
    preflight_calls = 0

    def preflight(_experiment: Experiment, *, cleanup_report, from_phase) -> dict:
        nonlocal preflight_calls
        del cleanup_report, from_phase
        preflight_calls += 1
        if preflight_calls == 2:
            raise shutdown
        return {}

    def fail_run(*args: object, **kwargs: object) -> None:
        raise NoFeasibleTrialError("trainer failed")

    captured: list[TerminalReport] = []
    monkeypatch.setattr("phasesweep.engine.run._preflight_existing_studies", preflight)
    monkeypatch.setattr("phasesweep.engine.run._run_experiment_inner", fail_run)

    with pytest.raises(PhaseSweepShutdown) as exc_info:
        run_experiment(experiment, terminal_callback=captured.append)

    assert exc_info.value is shutdown
    assert len(captured) == 1
    assert captured[0].primary_error is shutdown
    assert captured[0].cleanup_confirmed is True


def test_runner_records_snapshot_serialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(
        mcp_runner,
        "finalize_result_snapshot",
        lambda *_args, **_kwargs: {"not_json": object()},
    )

    mcp_runner._write_status(
        status_path,
        {
            "run_id": "r0",
            "returncode": 0,
            "error_class": None,
            "cleanup_confirmed": True,
        },
        result_snapshot={},
        result_snapshot_error=None,
    )

    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "failed"
    assert final["result_snapshot_error"] == "TypeError"
    assert "result_snapshot" not in final


_TERMINAL_PAYLOAD: dict[str, object] = {
    "run_id": "r0",
    "returncode": 0,
    "error_class": None,
    "cleanup_confirmed": True,
}


def test_transient_terminal_status_write_failure_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-shot OS failure must not cost a finished run its terminal evidence.

    Injected at ``os.replace`` (PR #5 review / reviewer 2 pass 2, blocker 6) so
    the retry runs through the real private atomic writer - temp file, fsync,
    replace - rather than a stubbed one.
    """
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(mcp_runner, "finalize_result_snapshot", lambda snapshot: snapshot)
    real_replace = os.replace
    replacements: list[object] = []

    def replace_failing_once(src: object, dst: object, **kwargs: object) -> None:
        replacements.append(dst)
        if len(replacements) == 1:
            raise OSError("simulated transient atomic replace failure")
        real_replace(src, dst, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("phasesweep.runtime.files.os.replace", replace_failing_once)

    assert (
        mcp_runner._write_status(
            status_path,
            dict(_TERMINAL_PAYLOAD),
            result_snapshot={"captured": True},
            result_snapshot_error=None,
        )
        is True
    )

    # Failed pending write, its retry, then the complete write.
    assert len(replacements) == 3
    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "complete"
    assert final["result_snapshot"] == {"captured": True}
    assert "result_snapshot_error" not in final


def test_transient_snapshot_state_write_failure_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The complete transition retries too; a blip must not downgrade the record."""
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(mcp_runner, "finalize_result_snapshot", lambda snapshot: snapshot)
    real_write = mcp_runner.write_status_file
    states: list[object] = []

    def write_failing_on_the_complete_transition(path: Path, payload: dict) -> None:
        states.append(payload["result_snapshot_state"])
        if len(states) == 2:
            raise OSError("simulated transient status write failure")
        real_write(path, payload)

    monkeypatch.setattr(mcp_runner, "write_status_file", write_failing_on_the_complete_transition)

    assert (
        mcp_runner._write_status(
            status_path,
            dict(_TERMINAL_PAYLOAD),
            result_snapshot={"captured": True},
            result_snapshot_error=None,
        )
        is True
    )

    assert states == ["pending", "complete", "complete"]
    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "complete"
    assert final["result_snapshot"] == {"captured": True}


def test_exhausted_status_write_retries_report_missing_terminal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """When nothing can be persisted, say so instead of returning as if it were."""
    status_path = tmp_path / "status.json"
    attempts: list[Path] = []

    def refuse_every_write(path: Path, _payload: dict) -> None:
        attempts.append(path)
        raise OSError("simulated persistent status write failure")

    monkeypatch.setattr(mcp_runner, "write_status_file", refuse_every_write)

    with caplog.at_level(logging.ERROR, logger="phasesweep.mcp.runner"):
        assert (
            mcp_runner._write_status(
                status_path,
                dict(_TERMINAL_PAYLOAD),
                result_snapshot={"captured": True},
                result_snapshot_error=None,
            )
            is False
        )

    assert len(attempts) == 3
    assert not status_path.exists()
    assert any("no terminal MCP evidence" in record.getMessage() for record in caplog.records)


def test_unpersistable_complete_transition_keeps_the_recoverable_pending_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable ``pending`` record with its snapshot outranks a degraded ``failed`` one.

    ``pending`` plus a dead runner is exactly what ``recover-run`` finalizes,
    so the runner must not trade it for the permanent ``failed`` state when the
    complete transition cannot be persisted (PR #5 review / reviewer 2 pass 2,
    blocker 6).
    """
    store = RunStore(tmp_path / "state")
    run_id = "pending-run"
    handle = make_run_handle(run_id=run_id, experiment_id="exp", pid=999999, starttime=111)
    store.create(handle)
    status_path = store.status_path(run_id)
    monkeypatch.setattr(mcp_runner, "finalize_result_snapshot", lambda snapshot: snapshot)
    real_write = mcp_runner.write_status_file
    states: list[object] = []

    def write_only_the_pending_record(path: Path, payload: dict) -> None:
        states.append(payload["result_snapshot_state"])
        if len(states) > 1:
            raise OSError("simulated persistent status write failure")
        real_write(path, payload)

    monkeypatch.setattr(mcp_runner, "write_status_file", write_only_the_pending_record)

    assert (
        mcp_runner._write_status(
            status_path,
            {**_TERMINAL_PAYLOAD, "run_id": run_id},
            result_snapshot={"captured": True},
            result_snapshot_error=None,
        )
        is True
    )

    # Three exhausted complete attempts, and no downgrade write after them.
    assert states == ["pending", "complete", "complete", "complete"]
    durable = json.loads(status_path.read_text())
    assert durable["result_snapshot_state"] == "pending"
    assert durable["result_snapshot"] == {"captured": True}
    assert "result_snapshot_error" not in durable
    assert store.state(handle) == "running"
    assert store.snapshot_recovery_required(handle)


def test_unserializable_captured_snapshot_still_records_terminal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot that cannot be serialized is a capture defect, not a write failure.

    It is never retried, and it must not leave the run with no status file at
    all - that would derive ``running`` forever and hold a concurrency slot.
    """
    status_path = tmp_path / "status.json"
    real_write = mcp_runner.write_status_file
    writes: list[object] = []

    def counting_write(path: Path, payload: dict) -> None:
        writes.append(payload["result_snapshot_state"])
        real_write(path, payload)

    monkeypatch.setattr(mcp_runner, "write_status_file", counting_write)

    assert (
        mcp_runner._write_status(
            status_path,
            dict(_TERMINAL_PAYLOAD),
            result_snapshot={"not_json": object()},
            result_snapshot_error=None,
        )
        is True
    )

    assert writes == ["pending", "failed"]
    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "failed"
    assert final["result_snapshot_error"] == "TypeError"
    assert "result_snapshot" not in final


def test_runner_exits_nonzero_when_terminal_evidence_cannot_be_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A sweep with no MCP-visible outcome is not a clean exit.

    The engine's own result is untouched, but with no status.json the run
    derives ``running`` forever and keeps its launch concurrency slot, so the
    runner reports the missing evidence in its exit code instead of claiming an
    outcome it cannot show (PR #5 review / reviewer 2 pass 2, blocker 6).
    """
    config_path, config_sha256 = _constant_trial_config(tmp_path, "no_status")

    def refuse_every_write(_path: Path, _payload: dict) -> None:
        raise OSError("simulated persistent status write failure")

    monkeypatch.setattr(mcp_runner, "write_status_file", refuse_every_write)

    store = RunStore(tmp_path / "state")
    run_id = "no-status-run"
    started_at = utc_now_iso()
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
        experiment_id="no_status",
    )

    with caplog.at_level(logging.ERROR, logger="phasesweep.mcp.runner"):
        assert (
            runner_main(
                [
                    "--run-id",
                    run_id,
                    "--config",
                    str(config_path),
                    "--config-sha256",
                    config_sha256,
                    "--status-path",
                    str(store.status_path(run_id)),
                    "--state-dir",
                    str(tmp_path / "state"),
                    "--experiment-id",
                    "no_status",
                    "--started-at",
                    started_at,
                ],
                cwd=tmp_path,
            )
            == 1
        )

    assert not store.status_path(run_id).exists()
    assert any("no terminal MCP evidence" in record.getMessage() for record in caplog.records)
    assert _last_successful_generation_id(load_config(config_path)) == run_id
    # The consequence the exit code has to carry: this identity is exactly what
    # the runner persisted for itself (spawned, this live process), and with no
    # status.json the run stays ``running`` and holds its concurrency slot.
    assert store.state(make_run_handle(run_id=run_id, experiment_id="no_status")) == "running"


def test_runner_refuses_to_persist_handle_without_linux_process_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_runner, "read_proc_starttime", lambda _pid: None)

    with pytest.raises(RuntimeError, match="/proc start time is unavailable"):
        mcp_runner._persist_spawned_handle(
            state_dir=tmp_path / "state",
            run_id="r0",
            experiment_id="exp",
            config_sha256="a" * 64,
            started_at=utc_now_iso(),
            allow_cancel=True,
        )

    assert RunStore(tmp_path / "state").get("r0") is None


def test_runner_binds_its_persisted_identity_to_the_current_boot(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    started_at = utc_now_iso()
    claim_runner_handle(
        store,
        run_id="r-boot",
        config_sha256="a" * 64,
        started_at=started_at,
        experiment_id="cancel_me",
        visible_params_at_launch=["lr"],
    )

    mcp_runner._persist_spawned_handle(
        state_dir=tmp_path / "state",
        run_id="r-boot",
        experiment_id="cancel_me",
        config_sha256="a" * 64,
        started_at=started_at,
        allow_cancel=False,
    )

    handle = store.get("r-boot")
    assert handle is not None
    assert handle.boot_id is not None
    assert handle.boot_id == read_boot_id()
    assert handle.visible_params_at_launch == ["lr"]


def _runner_argv(store: RunStore, *, run_id: str, config: Path, started_at: str) -> list[str]:
    """Build the runner argv the server constructs, minus ``--cwd``."""
    return [
        "--run-id",
        run_id,
        "--config",
        str(config),
        "--config-sha256",
        "a" * 64,
        "--status-path",
        str(store.status_path(run_id)),
        "--state-dir",
        str(store.log_path(run_id).parent.parent),
        "--experiment-id",
        "cancel_me",
        "--started-at",
        started_at,
    ]


def test_runner_enters_the_project_directory_after_its_identity_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--cwd`` replaces the ambient cwd the server used to hand the child.

    The chdir happens after ``_persist_spawned_handle``, so the first thing
    that can observe the project directory already has a durable identity the
    server can find and terminate.
    """
    store = RunStore(tmp_path / "state")
    project = tmp_path / "project"
    project.mkdir()
    started_at = utc_now_iso()
    claim_runner_handle(
        store,
        run_id="r-cwd",
        config_sha256="a" * 64,
        started_at=started_at,
        experiment_id="cancel_me",
    )
    observed: dict[str, object] = {}

    def record_cwd(*_args: object, **_kwargs: object) -> None:
        observed["cwd"] = Path.cwd()
        observed["handle"] = store.get("r-cwd")
        raise ValueError("stop once the project directory is in effect")

    monkeypatch.setattr(mcp_runner, "load_experiment_snapshot", record_cwd)

    with pytest.raises(RuntimeError, match="stop once the project directory"):
        runner_main(
            _runner_argv(
                store,
                run_id="r-cwd",
                config=tmp_path / "unread.yaml",
                started_at=started_at,
            ),
            cwd=project,
        )

    assert observed["cwd"] == project.resolve()
    handle = observed["handle"]
    assert handle is not None and handle.launch_state == "spawned"


def test_runner_persists_identity_even_when_the_project_directory_is_gone(
    tmp_path: Path,
) -> None:
    """An unusable ``--cwd`` must not cost the run its durable identity.

    If the chdir ran before the handle write, this failure would leave a
    ``launching`` handle that only operator recovery could retire.
    """
    store = RunStore(tmp_path / "state")
    started_at = utc_now_iso()
    claim_runner_handle(
        store,
        run_id="r-nocwd",
        config_sha256="a" * 64,
        started_at=started_at,
        experiment_id="cancel_me",
    )

    with pytest.raises(FileNotFoundError):
        runner_main(
            _runner_argv(
                store,
                run_id="r-nocwd",
                config=tmp_path / "unread.yaml",
                started_at=started_at,
            ),
            cwd=tmp_path / "does-not-exist",
        )

    handle = store.get("r-nocwd")
    assert handle is not None
    assert handle.launch_state == "spawned"
    status = json.loads(store.status_path("r-nocwd").read_text())
    assert status["returncode"] == 1
    assert status["error_class"] == "FileNotFoundError"


def test_runner_cancel_records_cancelled(tmp_path: Path) -> None:
    config = _slow_config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r1"
    status_path = store.status_path(run_id)
    log_path = store.log_path(run_id)
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    started_at = utc_now_iso()
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
        experiment_id="cancel_me",
    )
    cmd = [
        sys.executable,
        "-m",
        "phasesweep.mcp.runner",
        "--run-id",
        run_id,
        "--config",
        str(config),
        "--config-sha256",
        config_sha256,
        "--status-path",
        str(status_path),
        "--state-dir",
        str(tmp_path / "state"),
        "--experiment-id",
        "cancel_me",
        "--started-at",
        started_at,
        "--cwd",
        str(tmp_path),
    ]
    with open_private_text(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # runner is its own session/group leader -> pgid == pid
        )
    try:
        trial_dir = _wait_for_running_trial(config, proc, log_path)
        trial_pgid = json.loads((trial_dir / PROCESS_IDENTITY_FILE).read_text())["pgid"]

        # SIGTERM the runner's process group. The trial runs in its OWN session,
        # so this does not reach it directly; the runner's installed shutdown
        # handler tears the trial group down, then exits 128+SIGTERM = 143.
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)

    assert proc.returncode == 143, (
        f"expected 143, got {proc.returncode}; log:\n{log_path.read_text()}"
    )
    status = json.loads(status_path.read_text())
    assert status["returncode"] == 143
    assert status["error_class"] == "cancelled"
    assert status["cleanup_confirmed"] is True
    assert status["result_snapshot_state"] == "complete"
    assert status["result_snapshot"]["status"]["phases"][0]["trials"]["FAIL"] == 1
    assert status["result_snapshot"]["winners"] == []
    assert not _process_group_alive(trial_pgid)


def test_runner_cancelled_before_first_trial_still_records_cancelled(tmp_path: Path) -> None:
    """A cancel arriving before any trial starts must still write status.json.

    The server creates a launching handle before spawn. The runner installs
    shutdown handlers before updating that handle to spawned, so the transition
    proves SIGTERM is caught even while the config is still loading.
    """
    config = _slow_config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r2"
    status_path = store.status_path(run_id)
    log_path = store.log_path(run_id)
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    started_at = utc_now_iso()
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
        experiment_id="cancel_me",
    )
    cmd = [
        sys.executable,
        "-m",
        "phasesweep.mcp.runner",
        "--run-id",
        run_id,
        "--config",
        str(config),
        "--config-sha256",
        config_sha256,
        "--status-path",
        str(status_path),
        "--state-dir",
        str(tmp_path / "state"),
        "--experiment-id",
        "cancel_me",
        "--started-at",
        started_at,
        "--cwd",
        str(tmp_path),
    ]
    with open_private_text(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        # The spawned transition occurs after handler installation, so it is
        # the synchronization point for this pre-trial cancellation.
        deadline = time.time() + 25
        while True:
            handle = store.get(run_id)
            if handle is not None and handle.launch_state == "spawned":
                break
            if proc.poll() is not None:
                raise AssertionError(
                    f"runner exited early ({proc.returncode}); log:\n{log_path.read_text()}"
                )
            if time.time() > deadline:
                raise AssertionError(f"handle never became spawned; log:\n{log_path.read_text()}")
            time.sleep(0.02)
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)

    assert proc.returncode == 143, (
        f"expected 143, got {proc.returncode}; log:\n{log_path.read_text()}"
    )
    status = json.loads(status_path.read_text())
    assert status["returncode"] == 143
    assert status["error_class"] == "cancelled"
    assert status["cleanup_confirmed"] is True


def test_artifact_root_conflict_is_not_reported_as_fingerprint_or_internal() -> None:
    """A workdir conflict must carry its own code with the rebind remediation.

    Mapping it to ``fingerprint_mismatch`` would steer the operator toward a
    new experiment name or archiving a healthy study; falling through to
    ``internal_error`` would report a plain misconfiguration as a bug.
    """
    from phasesweep.engine.errors import ArtifactRootConflictError

    payload = mcp_runner._base_failure_payload(
        ArtifactRootConflictError("bound to /a, offered /b"), stage="preflight"
    )

    assert payload["code"] == "artifact_root_conflict"
    assert payload["retryable"] is False
    assert payload["actor"] == "operator"
    assert "rebind-workdir" in str(payload["remediation"])


def test_legacy_artifact_root_migration_reports_the_conflict_category() -> None:
    """The pre-binding migration refusal must classify as its parent conflict.

    It is the same operator problem and the same remedy surface, so it must
    not fall through to ``internal_error``; the remediation text also has to
    fit a study that records no root at all (re-review v0.5.19 / blocker B1).
    """
    from phasesweep.engine.errors import LegacyArtifactRootMigrationRequiredError

    payload = mcp_runner._base_failure_payload(
        LegacyArtifactRootMigrationRequiredError("holds 2 trial(s) but records no artifact root"),
        stage="preflight",
    )

    assert payload["code"] == "artifact_root_conflict"
    assert payload["actor"] == "operator"
    assert "rebind-workdir" in str(payload["remediation"])
    assert "bound to" not in str(payload["remediation"])
