"""Detached runner: exercises the real subprocess, the engine's signal teardown,
and the status.json written on the cancel path.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import optuna
import pytest
import yaml

import phasesweep.engine.generation as generation_ops
import phasesweep.engine.ledger as engine_ledger
import phasesweep.engine.optuna as engine_optuna
from phasesweep.config import ExecutionContext, Experiment, Phase, Sampler, load_experiment
from phasesweep.engine import (
    ActiveAttemptPersistenceError,
    ArtifactRootConflictError,
    ExperimentLockBusyError,
    NoFeasibleTrialError,
    ProcessCleanupUncertainError,
    PublishedStudyMissingError,
    SamplerContinuationUnsupportedError,
    StudyStorageUnavailableError,
    TerminalReport,
    TrialTargetRegressionError,
    read_status,
    run_experiment,
)
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.locking import _experiment_lock
from phasesweep.engine.paths import (
    _experiment_dir,
    _generation_path,
    _generation_summary_path,
    _generations_dir,
    _last_successful_generation_path,
    _summary_path,
    _trial_dir_for,
    _winner_path,
)
from phasesweep.engine.publication import _last_successful_generation_id
from phasesweep.engine.state import Winner, WinnerSource
from phasesweep.errors import UnsafeProcessCleanupError
from phasesweep.mcp import runner as mcp_runner
from phasesweep.mcp.errors import ConcurrencyLimitError
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.runtime.files import open_private_text
from phasesweep.runtime.reaper import PROCESS_IDENTITY_FILE, _process_group_alive, read_boot_id
from phasesweep.runtime.shutdown import PhaseSweepShutdown, ShutdownCleanupReport
from phasesweep.runtime.time import utc_now_iso
from tests.conftest import (
    REPO,
    make_experiment,
    mark_current_format,
    reaped_pid,
    requires_nonroot,
    write_constant_trainer,
    write_trainer,
)
from tests.mcp_helpers import (
    claim_runner_handle,
    live_runs,
    make_mcp_app,
    make_run_handle,
    runner_argv,
    runner_main,
    slow_mcp_config_text,
    write_mcp_catalog,
)
from tests.recovery_helpers import recover_run_cli

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="cancel path relies on POSIX process groups + /proc liveness",
)

# Persistent storage rejects an unseeded stochastic sampler; seeded random is
# reproducible and resumable, so it needs no non-resumable acknowledgement.
SEEDED_RANDOM = Sampler(type="random", seed=0)


@pytest.mark.integration
def test_in_process_runner_helper_restores_host_signal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process-entry-point runner must not retain pytest's signal ownership."""
    import phasesweep.runtime.shutdown as runtime_shutdown

    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_shutdown._SHUTDOWN_SIGNALS}
    prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())

    def host_handler(_signum: int, _frame: object) -> None:
        return None

    def fake_main(_argv: list[str]) -> int:
        runtime_shutdown.install_signal_handlers()
        os.chdir(tmp_path)
        return 23

    try:
        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            signal.signal(sig, host_handler)
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        original_cwd = Path.cwd()
        monkeypatch.setattr(mcp_runner, "main", fake_main)

        assert runner_main([], cwd=tmp_path) == 23

        assert Path.cwd() == original_cwd
        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is host_handler
        assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == (prior_mask | {signal.SIGTERM})
        assert not runtime_shutdown._process_lifetime_owner
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


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
    experiment = make_experiment(
        experiment=name,
        workdir=tmp_path / "runs",
        trainer=write_constant_trainer(tmp_path),
        n_trials=1,
        search_space={},
    )
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=False))
    return config_path, hashlib.sha256(config_path.read_bytes()).hexdigest()


def _wait_for_running_trial(config: Path, proc: subprocess.Popen, log_path: Path) -> Path:
    experiment = load_experiment(config)
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


@pytest.mark.integration
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
    experiment = load_experiment(config)
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


@pytest.mark.integration
def test_terminal_snapshot_is_captured_before_experiment_lock_release(tmp_path: Path) -> None:
    experiment = make_experiment(
        persistent=tmp_path, trainer=write_constant_trainer(tmp_path), n_trials=1
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


def test_attempt_registry_failure_routes_the_workdir_repair() -> None:
    """The payload names the tree repair and leaves the phase-abort detail to the message.

    Only persistent storage keeps the abort this refusal records, so raising
    n_trials is not a step every such raise requires; the message says when.
    """
    failure = mcp_runner._safe_failure_payload(
        ActiveAttemptPersistenceError("attempt registry is unwritable"),
        stage="execution",
    )

    assert failure["code"] == "storage_unavailable"
    assert failure["retryable"] is False
    assert failure["actor"] == "operator"
    assert failure["remediation"] == (
        "Ask the operator to repair the experiment tree's files and permissions, deleting a "
        "file only if certain nothing is running. The error in the PhaseSweep run log gives "
        "the details."
    )


@pytest.mark.integration
def test_missing_published_study_is_an_operator_preflight_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnosed absent ledger needs restoration, not process recovery."""
    monkeypatch.chdir(tmp_path)
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage="auto",
        n_trials=1,
        trial_command="echo x=0.5 {overrides}",
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(experiment.model_dump(mode="json")))
    run_experiment(experiment)
    generation_before = _generation_path(experiment).read_bytes()
    (_experiment_dir(experiment) / "study.journal").unlink()

    snapshot = mcp_runner.capture_result_snapshot(experiment)
    assert snapshot["status"]["phases"][0]["published_study_unavailable"] is True
    assert read_status(experiment)["phases"][0]["published_study_unavailable"] is True

    store = RunStore(tmp_path / "state")
    run_id = "missing-published-study"
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    started_at = utc_now_iso()
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
        experiment_id="t",
    )
    with pytest.raises(PublishedStudyMissingError):
        runner_main(
            runner_argv(
                store,
                run_id=run_id,
                config=config_path,
                config_sha256=config_sha256,
                experiment_id="t",
                started_at=started_at,
            ),
            cwd=tmp_path,
        )

    status = json.loads(store.status_path(run_id).read_text())
    assert status["failure"] == {
        "code": "published_study_missing",
        "stage": "preflight",
        "retryable": False,
        "actor": "operator",
        "remediation": (
            "Ask the operator to restore or repair the storage ledger and access to it. The "
            "error in the PhaseSweep run log gives the details."
        ),
    }
    assert status["generation_unavailable_reason"] == "engine_generation_not_claimed"
    assert status["result_snapshot_state"] == "complete"
    handle = store.get(run_id)
    assert handle is not None
    assert store.recovery_required(handle) is False
    assert _generation_path(experiment).read_bytes() == generation_before
    assert not (_experiment_dir(experiment) / "study.journal").exists()


@pytest.mark.parametrize("history", ["fresh", "published", "resume"])
@pytest.mark.parametrize(
    "damage",
    ["corrupt-interior", pytest.param("permission-denied", marks=requires_nonroot)],
)
@pytest.mark.integration
def test_damaged_storage_recovery_restores_catalog_capacity(
    tmp_path: Path, damage: str, history: str
) -> None:
    """Repairing a pre-launch ledger failure must let operator recovery release its slot.

    A torn or garbage *final* record is now self-repaired before a write, so
    only damage a repair may never cut remains here: a malformed record
    another record follows (never repaired, an interior corruption cannot be
    an in-flight append) and a permission denial (the repair itself refused).
    """
    published = history != "fresh"
    from_phase = "q" if history == "resume" else None
    ledger = tmp_path / "study.journal"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{ledger}",
        n_trials=1,
        trial_command="echo x=0.5 {overrides}",
        execution=ExecutionContext(cwd=str(tmp_path), inherit_env="none"),
    )
    if from_phase is not None:
        experiment = experiment.model_copy(
            update={
                "phases": [
                    *experiment.phases,
                    Phase(name="q", n_trials=1, inherits=["p"], sampler=SEEDED_RANDOM),
                ]
            }
        )
    if published:
        run_experiment(experiment)
        if from_phase is not None:
            optuna.delete_study(
                study_name="t::p", storage=_resolve_storage(experiment.resolved_storage)
            )
        # A later configured phase legitimately has no study to restore yet.
        experiment = experiment.model_copy(
            update={
                "phases": [
                    *experiment.phases,
                    experiment.phases[0].model_copy(update={"name": "later"}),
                ]
            }
        )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(experiment.model_dump(mode="json")))
    other = experiment.model_copy(update={"experiment": "other", "storage": "auto"})
    other_path = tmp_path / "other.yaml"
    other_path.write_text(yaml.safe_dump(other.model_dump(mode="json")))
    app, _registry, store = make_mcp_app(
        write_mcp_catalog(
            tmp_path,
            {"t": config_path, "other": other_path},
            allow={"launch": True, "from_phase": True},
        )
    )
    healthy = ledger.read_bytes() if published else b""
    generation_before = _generation_path(experiment).read_bytes() if published else None
    # A valid record follows the malformed one, so no repair may ever cut it.
    trailer = healthy.rstrip(b"\n").split(b"\n")[-1] if healthy else b'{"op_code": 0}'
    damaged = healthy if damage == "permission-denied" else healthy + b"garbage\n" + trailer + b"\n"
    ledger.write_bytes(damaged)
    original_mode = ledger.stat().st_mode
    if damage == "permission-denied":
        ledger.chmod(0)
    resume_args = ["--from-phase", from_phase] if from_phase is not None else []
    try:
        refused_run = subprocess.run(
            [sys.executable, "-m", "phasesweep", "run", str(config_path), *resume_args],
            capture_output=True,
            text=True,
            start_new_session=True,
            timeout=30,
        )
        assert refused_run.returncode != 0
        assert "Restore the original complete storage ledger" in refused_run.stderr

        run_id = "damaged-storage"
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        started_at = utc_now_iso()
        claim_runner_handle(
            store, run_id=run_id, config_sha256=digest, started_at=started_at, experiment_id="t"
        )
        snapshot = store.config_snapshot_path(run_id)
        snapshot.write_bytes(config_path.read_bytes())
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "phasesweep.mcp.runner",
                *runner_argv(
                    store,
                    run_id=run_id,
                    config=snapshot,
                    config_sha256=digest,
                    experiment_id="t",
                    started_at=started_at,
                ),
                "--cwd",
                str(tmp_path),
                *resume_args,
            ],
            capture_output=True,
            text=True,
            start_new_session=True,
            timeout=30,
        )
        assert result.returncode == 1, result.stderr
        terminal_before = store.status_path(run_id).read_bytes()
        status = json.loads(terminal_before)
        assert status["from_phase"] == from_phase
        assert status["cleanup_confirmed"] is False
        assert status["failure"]["code"] == "cleanup_uncertain"
        assert status["failure"]["retryable"] is False
        assert status["failure"]["cause"]["code"] == "storage_unavailable"
        assert status["failure"]["cause"]["stage"] == "preflight"
        # The cause routes a ledger restore, which only the operator can do;
        # test_error_routing.py's routing sweep is the one owner of the
        # composed remediation text for cleanup_uncertain + storage_unavailable.
        assert status["failure"]["cause"]["retryable"] is False
        assert app.status(run_id=run_id)["run"]["failure"] == status["failure"]
        assert str(ledger) not in json.dumps(status["failure"])
        assert status["generation_unavailable_reason"] == "engine_generation_not_claimed"
        handle = store.get(run_id)
        assert handle is not None
        assert store.state(handle) == "running"
        assert store.recovery_required(handle)
        with pytest.raises(ConcurrencyLimitError):
            app.launch("other")

        state_dir = tmp_path / "state"
        for confirm in (False, True):
            blocked = recover_run_cli(state_dir, run_id, confirm=confirm)
            assert blocked.exit_code != 0
            assert "could not be" in blocked.output
            assert "Restore the original complete storage ledger" in blocked.output
            if damage == "permission-denied":
                assert ledger.stat().st_mode & 0o777 == 0
            else:
                assert ledger.read_bytes() == damaged
            assert store.status_path(run_id).read_bytes() == terminal_before
            assert not store.cleanup_recovery_path(run_id).exists()
    finally:
        if damage == "permission-denied":
            ledger.chmod(original_mode)
    if damage == "permission-denied":
        assert ledger.read_bytes() == damaged

    ledger.write_bytes(healthy)
    # A storage error after generation allocation, or an unrelated failure,
    # still requires cleanup evidence attributable to that run.
    for changed_fact in ("generation_claimed", "unrelated_failure", "execution_failure"):
        unrelated = json.loads(terminal_before)
        if changed_fact == "generation_claimed":
            unrelated.pop("generation_unavailable_reason")
        elif changed_fact == "unrelated_failure":
            unrelated["failure"]["cause"]["code"] = "trainer_failed"
        else:
            unrelated["failure"]["cause"]["stage"] = "execution"
        store.status_path(run_id).write_text(json.dumps(unrelated))
        refused_recovery = recover_run_cli(state_dir, run_id)
        assert refused_recovery.exit_code != 0
        assert "could not confirm any trial-level cleanup evidence" in refused_recovery.output
        assert not store.cleanup_recovery_path(run_id).exists()
    store.status_path(run_id).write_bytes(terminal_before)
    preflight = recover_run_cli(state_dir, run_id)
    assert preflight.exit_code == 0, preflight.output
    assert store.recovery_required(handle)
    confirmed = recover_run_cli(state_dir, run_id, confirm=True)
    assert confirmed.exit_code == 0, confirmed.output
    assert store.state(handle) == "failed"
    assert not store.recovery_required(handle)
    assert live_runs(store) == []
    recovered_status = app.status(run_id=run_id)
    assert recovered_status["run"]["state"] == "failed"
    assert recovered_status["run"]["recovery_required"] is False
    assert recovered_status["run"]["failure"] == status["failure"]
    assert store.status_path(run_id).read_bytes() == terminal_before
    assert (
        _generation_path(experiment).read_bytes() if _generation_path(experiment).exists() else None
    ) == generation_before
    relaunch = app.launch("t", from_phase=from_phase)
    relaunch_status = asyncio.run(app.await_run(relaunch["run_id"], timeout_seconds=30))
    assert relaunch_status["reason"] == "terminal"
    assert relaunch_status["run"]["state"] == "succeeded"
    other_relaunch = app.launch("other")
    other_status = asyncio.run(app.await_run(other_relaunch["run_id"], timeout_seconds=30))
    assert other_status["reason"] == "terminal"
    assert other_status["run"]["state"] == "succeeded"


@pytest.mark.integration
def test_external_engine_lock_is_retryable_and_freezes_pre_generation_snapshot(
    tmp_path: Path,
) -> None:
    config_path = _slow_config(tmp_path)
    experiment = load_experiment(config_path)
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
            runner_argv(
                store,
                run_id=run_id,
                config=config_path,
                config_sha256=config_sha256,
                experiment_id="cancel_me",
                started_at=started_at,
            ),
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
    assert all(
        phase["published_study_unavailable"] is None for phase in snapshot["status"]["phases"]
    )


@pytest.mark.integration
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
        storage=f"journal:///{tmp_path / 'studies.journal'}",
        trial_command=f"{sys.executable} {trainer} {{overrides}}",
        override_format="argparse",
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
    }


def test_terminal_snapshot_tolerates_missing_lifecycle_record(tmp_path: Path) -> None:
    """A missing per-generation record must not fail the snapshot capture.

    The engine writes that record as an optional post-commit diagnostic; a
    capture that hard-required it contradicted engine-defined success
    whenever the best-effort write had failed (review v0.5.16 / blocker 2).
    """
    experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1)
    mark_current_format(experiment)
    generation_path = _generation_path(experiment)
    generation_path.parent.mkdir(parents=True, exist_ok=True)
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
        storage=f"journal:///{tmp_path / 'studies.journal'}",
        phases=[Phase(name="p", n_trials=1, sampler=SEEDED_RANDOM, search_space={})],
    )
    journal = _resolve_storage(experiment.resolved_storage)
    study = optuna.create_study(study_name="t::p", storage=journal, direction="minimize")
    mark_current_format(experiment, study)
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


@pytest.mark.parametrize("storage_kind", ["none", "corrupt-journal"])
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
    storage = None
    if storage_kind == "corrupt-journal":
        ledger = tmp_path / "studies.journal"
        # A bad line that another line follows is corruption, not a final record.
        ledger.write_bytes(b"not a journal record\n" * 2)
        storage = f"journal:///{ledger}"
    experiment = make_experiment(workdir=tmp_path / "runs", storage=storage, n_trials=1)

    def fail_redundant_read(*args: object, **kwargs: object) -> None:
        raise OSError("storage remains unavailable")

    # Patched on Optuna itself, so the guard holds no matter how PhaseSweep
    # imports its own study helpers: any study load at all fails the capture.
    monkeypatch.setattr(optuna, "load_study", fail_redundant_read)

    if storage_kind == "corrupt-journal":
        with pytest.raises(StudyStorageUnavailableError, match="could not be completely read"):
            mcp_runner.capture_result_snapshot(experiment)
        assert not _experiment_dir(experiment).exists()
        return

    snapshot = mcp_runner.capture_result_snapshot(experiment)

    phase = snapshot["status"]["phases"][0]
    assert phase["trial_data_available"] is False
    assert phase["running_attempts"] is None


@pytest.mark.parametrize("pointer_commits", [True, False], ids=["commit", "abort"])
@pytest.mark.integration
def test_publication_snapshot_rebinds_unavailable_trial_data_only_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pointer_commits: bool,
) -> None:
    """A pre-commit storage failure describes the new publication only after commit."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'studies.journal'}",
        trial_command="echo x=0.5 {overrides}",
        n_trials=1,
    )
    generation_id = f"storage-unavailable-{'commit' if pointer_commits else 'abort'}"
    capture_reads = 0
    original_stats = engine_ledger._trial_stats

    def transient_capture_failure(
        ledger: engine_ledger.ValidatedLedger,
        published_trials: Mapping[str, engine_optuna._TrialRef | None],
    ) -> dict[str, engine_optuna._PhaseTrialStats]:
        nonlocal capture_reads
        if _generation_summary_path(ledger.experiment, generation_id).is_file():
            capture_reads += 1

            def locked_read(*_args: object, **_kwargs: object) -> None:
                raise OSError("journal storage is locked")

            with monkeypatch.context() as capture_patch:
                capture_patch.setattr(engine_ledger, "_journal_snapshot_storage", locked_read)
                return original_stats(ledger, published_trials)

        return original_stats(ledger, published_trials)

    monkeypatch.setattr(engine_ledger, "_trial_stats", transient_capture_failure)
    if not pointer_commits:
        pointer_path = _last_successful_generation_path(experiment)
        original_write = generation_ops.artifact_io._write_yaml_atomic

        def fail_pointer_write(path: Path, payload: object) -> None:
            if path == pointer_path:
                raise OSError("simulated publication abort")
            original_write(path, payload)

        monkeypatch.setattr(generation_ops.artifact_io, "_write_yaml_atomic", fail_pointer_write)

    hook = mcp_runner._RunnerPublicationHook(
        tmp_path / "status.json",
        {
            "run_id": generation_id,
            "returncode": 0,
            "error_class": None,
            "cleanup_confirmed": True,
            "failure": None,
        },
    )
    if pointer_commits:
        run_experiment(experiment, generation_id=generation_id, publication_hook=hook)
    else:
        with pytest.raises(OSError, match="simulated publication abort"):
            run_experiment(experiment, generation_id=generation_id, publication_hook=hook)

    assert capture_reads == 1
    assert hook.snapshot is not None
    phase = hook.snapshot["status"]["phases"][0]
    assert phase["trial_data_available"] is False
    assert phase["running_attempts"] is None
    assert phase["published_study_unavailable"] is pointer_commits
    assert hook.snapshot["status"]["is_published"] is pointer_commits


def test_published_snapshot_rebinds_selected_phase_flags_from_summary(tmp_path: Path) -> None:
    """Commit covers winners, carried flags, and omitted phases."""
    phase_names = ("carried", "new", "old-only")
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        phases=[
            Phase(name=name, n_trials=1, sampler=SEEDED_RANDOM, search_space={})
            for name in phase_names
        ],
    )
    generation_id = "candidate-generation"
    mark_current_format(experiment)
    published_summary = {
        "experiment": experiment.experiment,
        "generation_id": generation_id,
        "phases": [
            {
                "name": "carried",
                "trial_number": 2,
                "generation_id": "prior-generation",
                "attempt_id": "carried-attempt",
            },
            {
                "name": "new",
                "trial_number": 0,
                "generation_id": generation_id,
                "attempt_id": "new-attempt",
            },
        ],
    }
    summary_path = _generation_summary_path(experiment, generation_id)
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(yaml.safe_dump(published_summary))
    snapshot = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id=generation_id,
        engine_winners={},
    )
    phases = {phase["phase"]: phase for phase in snapshot["status"]["phases"]}
    phases["carried"]["trial_data_available"] = True
    phases["carried"]["running_attempts"] = []
    phases["carried"]["published_study_unavailable"] = True
    phases["old-only"]["published_study_unavailable"] = True
    committed = mcp_runner.mark_result_snapshot_published(
        snapshot,
        generation_id=generation_id,
        published_summary=published_summary,
    )

    committed_phases = {phase["phase"]: phase for phase in committed["status"]["phases"]}
    assert committed_phases["new"]["published_study_unavailable"] is True
    assert committed_phases["carried"]["published_study_unavailable"] is True
    assert committed_phases["old-only"]["published_study_unavailable"] is False


@pytest.mark.integration
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

    The one tolerant read replays a journal snapshot through ``optuna.load_study``,
    so exactly one load succeeds and any reload after it fails.
    """
    experiment = make_experiment(
        persistent=tmp_path, trainer=write_constant_trainer(tmp_path), n_trials=1
    )
    reports: list[TerminalReport] = []

    run_experiment(experiment, terminal_callback=reports.append)

    (report,) = reports
    assert report.primary_error is None
    assert report.winners

    loads = 0
    real_load_study = optuna.load_study

    def fail_on_reload(*args: object, **kwargs: object) -> optuna.Study:
        nonlocal loads
        loads += 1
        if loads > 1:
            raise StudyStorageUnavailableError("database is locked")
        return real_load_study(*args, **kwargs)

    # Patched on Optuna itself rather than on a PhaseSweep helper, so no
    # import binding can hide a study reload from this test. Every remaining
    # storage read must be the tolerant one read_status performs.
    monkeypatch.setattr(optuna, "load_study", fail_on_reload)

    snapshot = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id=report.generation_id,
        engine_winners=report.winners,
    )

    assert loads == 1
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
        storage=f"journal:///{tmp_path / 'studies.journal'}",
        phases=[Phase(name="p", n_trials=2, sampler=SEEDED_RANDOM, search_space={})],
    )
    journal = _resolve_storage(experiment.resolved_storage)
    study = optuna.create_study(study_name="t::p", storage=journal, direction="minimize")
    mark_current_format(experiment, study)
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

    fully_finalized = mcp_runner.finalize_result_snapshot(
        snapshot,
        confirmed_attempt_ids={"current-attempt", "allocated-attempt"},
        confirmed_attempt_locations={
            "allocated-attempt": ("p", 1, "prior-generation"),
        },
    )

    fully_finalized_phase = fully_finalized["status"]["phases"][0]
    assert fully_finalized_phase["trials"] == {"RUNNING": 0, "FAIL": 2}
    assert fully_finalized_phase["generation_trials"] == {"RUNNING": 0, "FAIL": 1}
    assert fully_finalized_phase["running_attempts"] == []


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
        source=WinnerSource(
            kind="phase_trial",
            phase="p",
            trial_number=4,
            generation_id="engine-generation",
            attempt_id="engine-attempt",
        ),
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
    }


@pytest.mark.integration
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
    import phasesweep.engine.generation as generation_ops

    config_path, config_sha256 = _constant_trial_config(tmp_path, "record_fail")

    def fail_record_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated record write failure")

    monkeypatch.setattr(generation_ops, "_write_generation_record_once", fail_record_write)

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
            runner_argv(
                store,
                run_id=run_id,
                config=config_path,
                config_sha256=config_sha256,
                experiment_id="record_fail",
                started_at=started_at,
            ),
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


@pytest.mark.integration
@pytest.mark.signals_own_pid
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
    import phasesweep.runtime.shutdown as runtime_shutdown

    # Resets the pending-shutdown marker at teardown, so a failure before the
    # runner services the signal cannot leak the absorbed shutdown into a later test.
    monkeypatch.setattr(runtime_shutdown, "_deferred_shutdown_signum", None)

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
            runner_argv(
                store,
                run_id=run_id,
                config=config_path,
                config_sha256=config_sha256,
                experiment_id="cancel_at_capture",
                started_at=started_at,
            ),
            cwd=tmp_path,
        )

    # The shutdown is honored, but only after terminal evidence is durable:
    # it surfaces out of the terminal status write's own defer window.
    assert exc_info.value.signum == signal.SIGTERM
    assert exc_info.value.code == 128 + signal.SIGTERM
    assert runtime_shutdown._deferred_shutdown_signum is None

    terminal = json.loads(store.status_path(run_id).read_text())
    assert terminal["result_snapshot_state"] == "complete"
    assert [w["phase"] for w in terminal["result_snapshot"]["winners"]] == ["p"]
    # The engine published before the signal arrived, so its own outcome - and
    # the terminal status recording it - stay a success.
    assert terminal["returncode"] == 0
    assert terminal["error_class"] is None
    assert terminal["failure"] is None
    assert _last_successful_generation_id(load_experiment(config_path)) == run_id


@pytest.mark.parametrize("from_phase", [None, "b"])
@pytest.mark.integration
def test_failed_fingerprint_preflight_preserves_published_results(
    tmp_path: Path,
    from_phase: str | None,
) -> None:
    """A failed preflight advances the current pointer but never the published one.

    The current-generation pointer legitimately moves to this new (failed)
    invocation -- a new invocation always overwrites it starting from
    "preflighting", and every outcome path must drive it to a terminal state
    (review v0.5.15 / blocker 3) -- but the convenience root projections and,
    critically, the last-success pointer stay exactly as the prior successful
    publication left them.
    """
    phases = [
        Phase(
            name="a", n_trials=1, fixed_overrides={"k": 1}, sampler=SEEDED_RANDOM, search_space={}
        ),
        Phase(name="b", n_trials=1, inherits=["a"], sampler=SEEDED_RANDOM, search_space={}),
    ]
    experiment = make_experiment(
        persistent=tmp_path, trainer=write_constant_trainer(tmp_path), phases=phases
    )
    run_experiment(experiment)
    first_generation = _last_successful_generation_id(experiment)
    assert first_generation is not None

    # The convenience root projections must stay untouched by a failed resume;
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
    # Convenience root projections are untouched, and the published pointer
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

    def preflight(_ledger: engine_ledger.ClaimedLedger, *, cleanup_report, from_phase) -> dict:
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
    monkeypatch.setattr("phasesweep.engine.guards._preflight_existing_studies", preflight)
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


@pytest.mark.parametrize(
    "error_type",
    [OSError, ArtifactRootConflictError],
)
def test_terminal_report_marks_failed_root_discovery_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    """Discovery failure during reconciliation cannot attest to unscanned attempts."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    cleanup_error = error_type("artifact root cannot be inspected")
    registry_scans = 0

    def fail_discovery(*args: object, **kwargs: object) -> None:
        raise cleanup_error

    def scan_registry(*args: object, **kwargs: object) -> None:
        nonlocal registry_scans
        registry_scans += 1

    def fail_run(*args: object, **kwargs: object) -> None:
        # Initial discovery and preflight succeeded. Only reconciliation loses
        # access to the root, before it can perform a second registry scan.
        monkeypatch.setattr("phasesweep.engine.guards.validate_ledger", fail_discovery)
        raise NoFeasibleTrialError("trainer failed")

    captured: list[TerminalReport] = []
    monkeypatch.setattr("phasesweep.engine.guards._preflight_active_attempts", scan_registry)
    monkeypatch.setattr("phasesweep.engine.run._run_experiment_inner", fail_run)

    with pytest.raises(ProcessCleanupUncertainError) as exc_info:
        run_experiment(experiment, terminal_callback=captured.append)

    assert isinstance(exc_info.value.__cause__, NoFeasibleTrialError)
    assert registry_scans == 1
    assert len(captured) == 1
    assert isinstance(captured[0].primary_error, NoFeasibleTrialError)
    assert captured[0].cleanup_confirmed is False
    assert captured[0].cleanup_error is cleanup_error


def test_cleanup_uncertainty_outer_failure_controls_a_cancelled_cause() -> None:
    shutdown = PhaseSweepShutdown(
        signal.SIGTERM,
        ShutdownCleanupReport(
            signum=signal.SIGTERM,
            cleanup_confirmed=False,
            child_pgids=(1234,),
        ),
    )

    failure = mcp_runner._cleanup_failure_payload(shutdown, cause_stage="execution")

    assert failure["code"] == "cleanup_uncertain"
    assert failure["actor"] == "operator"
    assert failure["retryable"] is False
    assert failure["cause"]["code"] == "cancelled"
    assert failure["cause"]["actor"] == "agent"
    assert failure["cause"]["retryable"] is True


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

    def preflight(_ledger: engine_ledger.ClaimedLedger, *, cleanup_report, from_phase) -> dict:
        nonlocal preflight_calls
        del cleanup_report, from_phase
        preflight_calls += 1
        return {}

    def cancel_run(*args: object, **kwargs: object) -> None:
        raise shutdown

    captured: list[TerminalReport] = []
    monkeypatch.setattr("phasesweep.engine.guards._preflight_existing_studies", preflight)
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


@pytest.mark.parametrize(
    ("initial_error", "cleanup_confirmed"),
    [
        pytest.param(NoFeasibleTrialError("trainer failed"), True, id="ordinary-failure"),
        pytest.param(
            UnsafeProcessCleanupError("trial cleanup could not be confirmed"),
            False,
            id="unsafe-process-cleanup",
        ),
        pytest.param(
            PhaseSweepShutdown(
                signal.SIGTERM,
                ShutdownCleanupReport(
                    signum=signal.SIGTERM,
                    cleanup_confirmed=False,
                    child_pgids=(1234,),
                ),
            ),
            False,
            id="earlier-uncertain-shutdown",
        ),
    ],
)
def test_shutdown_during_post_error_reconciliation_preserves_cleanup_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_error: BaseException,
    cleanup_confirmed: bool,
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

    def preflight(_ledger: engine_ledger.ClaimedLedger, *, cleanup_report, from_phase) -> dict:
        nonlocal preflight_calls
        del cleanup_report, from_phase
        preflight_calls += 1
        if preflight_calls == 2:
            raise shutdown
        return {}

    def fail_run(*args: object, **kwargs: object) -> None:
        raise initial_error

    captured: list[TerminalReport] = []
    monkeypatch.setattr("phasesweep.engine.guards._preflight_existing_studies", preflight)
    monkeypatch.setattr("phasesweep.engine.run._run_experiment_inner", fail_run)

    with pytest.raises(PhaseSweepShutdown) as exc_info:
        run_experiment(experiment, terminal_callback=captured.append)

    assert exc_info.value is shutdown
    assert exc_info.value.__cause__ is initial_error
    assert len(captured) == 1
    assert captured[0].primary_error is shutdown
    assert captured[0].cleanup_confirmed is cleanup_confirmed
    assert captured[0].cleanup_error is (None if cleanup_confirmed else initial_error)


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

    ``pending`` plus a dead runner is exactly what ``recover-run`` finalizes, so the runner
    must not trade it for the permanent ``failed`` state when the complete transition
    cannot be persisted (PR #5 review / reviewer 2 pass 2, blocker 6).
    """
    store = RunStore(tmp_path / "state")
    run_id = "pending-run"
    handle = make_run_handle(run_id=run_id, experiment_id="exp", pid=reaped_pid(), starttime=111)
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


@pytest.mark.integration
def test_runner_exits_nonzero_when_terminal_evidence_cannot_be_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A sweep with no MCP-visible outcome is not a clean exit.

    The required precommit snapshot write now fails the publication itself.
    The process entry point converts this propagated error to a nonzero exit;
    an in-process invocation sees the original error and no pointer advances.
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

    with (
        caplog.at_level(logging.ERROR, logger="phasesweep.mcp.runner"),
        pytest.raises(OSError, match="simulated persistent status write failure"),
    ):
        runner_main(
            runner_argv(
                store,
                run_id=run_id,
                config=config_path,
                config_sha256=config_sha256,
                experiment_id="no_status",
                started_at=started_at,
            ),
            cwd=tmp_path,
        )

    assert not store.status_path(run_id).exists()
    assert any("no terminal MCP evidence" in record.getMessage() for record in caplog.records)
    assert _last_successful_generation_id(load_experiment(config_path)) is None
    # The consequence the exit code has to carry: this identity is exactly what
    # the runner persisted for itself (spawned, this live process), and with no
    # status.json the run stays ``running`` and holds its concurrency slot.
    assert store.state(make_run_handle(run_id=run_id, experiment_id="no_status")) == "running"


@pytest.mark.parametrize(
    ("crash_boundary", "expected_exit", "damage_publication"),
    (
        ("before_pointer", 72, False),
        ("after_pointer", 73, False),
        pytest.param("after_pointer", 73, True, id="after-pointer-invalid-publication"),
    ),
)
@pytest.mark.integration
def test_recover_run_reconciles_hard_exit_around_publication_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_boundary: str,
    expected_exit: int,
    damage_publication: bool,
) -> None:
    """A hard exit cannot separate a publication from its frozen MCP result."""
    config_path, _config_sha256 = _constant_trial_config(tmp_path, crash_boundary)
    raw_config = yaml.safe_load(config_path.read_text())
    raw_config["storage"] = "auto"
    raw_config["provenance"] = {"revision": "test-fixture-v1"}
    raw_config["phases"][0]["sampler"] = {"type": "random", "seed": 0}
    config_path.write_text(yaml.safe_dump(raw_config))
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    store = RunStore(tmp_path / "state")
    run_id = f"hard-exit-{crash_boundary.replace('_', '-')}"
    started_at = utc_now_iso()
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
        experiment_id=crash_boundary,
    )
    with open_private_text(store.config_snapshot_path(run_id), "x") as output:
        output.write(config_path.read_text())
    experiment = load_experiment(config_path)
    original_stats = engine_ledger._trial_stats

    def unavailable_during_prepared_capture(
        ledger: engine_ledger.ValidatedLedger,
        published_trials: Mapping[str, engine_optuna._TrialRef | None],
    ) -> dict[str, engine_optuna._PhaseTrialStats]:
        if _generation_summary_path(ledger.experiment, run_id).is_file():

            def locked_read(*_args: object, **_kwargs: object) -> None:
                raise OSError("journal storage is locked")

            with monkeypatch.context() as capture_patch:
                capture_patch.setattr(engine_ledger, "_journal_snapshot_storage", locked_read)
                return original_stats(ledger, published_trials)

        return original_stats(ledger, published_trials)

    monkeypatch.setattr(engine_ledger, "_trial_stats", unavailable_during_prepared_capture)

    if crash_boundary == "before_pointer":
        original_prepare = mcp_runner._RunnerPublicationHook.prepare

        def exit_after_prepare(
            self: mcp_runner._RunnerPublicationHook,
            *,
            experiment: Experiment,
            generation_id: str,
            winners: dict[str, Winner],
            summary: Mapping[str, object],
        ) -> None:
            original_prepare(
                self,
                experiment=experiment,
                generation_id=generation_id,
                winners=winners,
                summary=summary,
            )
            os._exit(expected_exit)

        monkeypatch.setattr(
            mcp_runner._RunnerPublicationHook,
            "prepare",
            exit_after_prepare,
        )
    else:

        def exit_before_commit_receipt(
            _self: mcp_runner._RunnerPublicationHook,
            *,
            generation_id: str,
        ) -> None:
            assert generation_id == run_id
            os._exit(expected_exit)

        monkeypatch.setattr(
            mcp_runner._RunnerPublicationHook,
            "committed",
            exit_before_commit_receipt,
        )

    child = os.fork()
    if child == 0:
        mcp_runner.main(
            [
                *runner_argv(
                    store,
                    run_id=run_id,
                    config=config_path,
                    config_sha256=config_sha256,
                    experiment_id=crash_boundary,
                    started_at=started_at,
                ),
                "--cwd",
                str(tmp_path),
            ]
        )
        os._exit(0)
    _pid, wait_status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(wait_status) == expected_exit

    prepared_bytes = store.status_path(run_id).read_bytes()
    prepared = json.loads(prepared_bytes)
    assert prepared["result_snapshot_state"] == "pending"
    assert prepared["result_publication_state"] == "prepared"
    assert prepared["result_publication_generation_id"] == run_id
    prepared_phase = prepared["result_snapshot"]["status"]["phases"][0]
    assert prepared_phase["trial_data_available"] is False
    assert prepared_phase["published_study_unavailable"] is False
    if damage_publication:
        _generation_summary_path(experiment, run_id).unlink()
    assert _last_successful_generation_id(experiment) == (
        run_id if crash_boundary == "after_pointer" and not damage_publication else None
    )

    dry_run = recover_run_cli(tmp_path / "state", run_id)
    if damage_publication:
        assert read_status(experiment)["publication_integrity"] == "failed"
        assert dry_run.exit_code != 0
        assert "last-success publication is invalid or unreadable" in dry_run.output
        recovered = recover_run_cli(tmp_path / "state", run_id, confirm=True)
        assert recovered.exit_code != 0
        assert "last-success publication is invalid or unreadable" in recovered.output
        assert store.status_path(run_id).read_bytes() == prepared_bytes
        assert not store.cleanup_recovery_path(run_id).exists()
        return
    assert dry_run.exit_code == 0, dry_run.output
    assert "prepared" in dry_run.output
    recovered = recover_run_cli(tmp_path / "state", run_id, confirm=True)
    assert recovered.exit_code == 0, recovered.output

    terminal = json.loads(store.status_path(run_id).read_text())
    assert terminal["result_snapshot_state"] == "complete"
    snapshot = terminal["result_snapshot"]
    assert snapshot["status"]["represented_generation_id"] == run_id
    assert snapshot["winners"][0]["metric"] == 0.5
    assert snapshot["status"]["phases"][0]["published_study_unavailable"] is (
        crash_boundary == "after_pointer"
    )
    if crash_boundary == "after_pointer":
        assert terminal["returncode"] == 0
        assert terminal["result_publication_state"] == "committed"
        assert snapshot["status"]["is_published"] is True
        assert snapshot["status"]["published_generation_id"] == run_id
        failure = None
    else:
        assert terminal["returncode"] == 1
        assert terminal["error_class"] == "PublicationNotCommitted"
        assert "result_publication_state" not in terminal
        assert snapshot["status"]["is_published"] is False
        assert _last_successful_generation_id(experiment) is None
        failure = {
            "code": "publication_not_committed",
            "stage": "execution",
            "retryable": True,
            "actor": "agent",
            "remediation": (
                "Report that recovery could not confirm this run as the current published "
                "result. Start a new run only if the user still wants a published result."
            ),
        }

    assert terminal.get("failure") == failure
    app, _registry, _store = make_mcp_app(
        write_mcp_catalog(tmp_path, {crash_boundary: config_path})
    )
    monkeypatch.setattr("phasesweep.mcp.tools.AWAIT_MIN_TIMEOUT_SECONDS", 0)
    for payload in (
        app.status(run_id=run_id),
        asyncio.run(app.await_run(run_id, timeout_seconds=0)),
        app.latest_run(crash_boundary),
    ):
        assert payload["run"]["failure"] == failure
    assert app.winners(run_id=run_id)["failure"] == failure


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


# Binds the persisted handle to the host's real process group and boot id, so
# it needs an unconfined process namespace like the detached-runner tests.
@pytest.mark.integration
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


@pytest.mark.integration
def test_runner_enters_the_project_directory_after_its_identity_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--cwd`` replaces the ambient cwd the server used to hand the child.

    The inherited lease stays held through ``_persist_spawned_handle``. The
    subsequent chdir therefore exposes the project only after the server has
    a durable identity it can find and terminate.
    """
    store = RunStore(tmp_path / "state")
    project = tmp_path / "project"
    project.mkdir()
    pending = make_run_handle(
        run_id="r-cwd",
        config_sha256="a" * 64,
        experiment_id="cancel_me",
        launch_state="launching",
    )
    preparation = store.prepare_launch(pending, b"experiment: unread\n")
    inherited_lease_fd = os.dup(preparation.lease_fd)
    preparation.close()
    ready_read, ready_write = os.pipe()
    ack_read, ack_write = os.pipe()
    assert os.write(ack_write, b"A") == 1
    os.close(ack_write)
    ack_write = -1
    observed: dict[str, object] = {}
    real_persist = mcp_runner._persist_spawned_handle

    def persist_with_held_lease(*args: object, **kwargs: object) -> None:
        os.fstat(inherited_lease_fd)
        assert not store.is_pre_spawn_orphan("r-cwd")
        real_persist(*args, **kwargs)

    def record_cwd(*_args: object, **_kwargs: object) -> None:
        observed["cwd"] = Path.cwd()
        observed["handle"] = store.get("r-cwd")
        raise ValueError("stop once the project directory is in effect")

    monkeypatch.setattr(mcp_runner, "_persist_spawned_handle", persist_with_held_lease)
    monkeypatch.setattr(mcp_runner, "load_experiment_snapshot", record_cwd)

    try:
        with pytest.raises(RuntimeError, match="stop once the project directory"):
            runner_main(
                [
                    *runner_argv(
                        store,
                        run_id="r-cwd",
                        config=preparation.config_snapshot_path,
                        config_sha256="a" * 64,
                        experiment_id="cancel_me",
                        started_at=pending.started_at,
                    ),
                    "--launch-ready-fd",
                    str(ready_write),
                    "--launch-ack-fd",
                    str(ack_read),
                    "--launch-lease-fd",
                    str(inherited_lease_fd),
                ],
                cwd=project,
            )
        os.set_blocking(ready_read, False)
        assert os.read(ready_read, 1) == b"R"
    finally:
        for fd in (ready_read, ready_write, ack_read, ack_write, inherited_lease_fd):
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)

    assert observed["cwd"] == project.resolve()
    handle = observed["handle"]
    assert handle is not None and handle.launch_state == "spawned"


@pytest.mark.integration
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
            runner_argv(
                store,
                run_id="r-nocwd",
                config=tmp_path / "unread.yaml",
                config_sha256="a" * 64,
                experiment_id="cancel_me",
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


@pytest.mark.integration
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
        *runner_argv(
            store,
            run_id=run_id,
            config=config,
            config_sha256=config_sha256,
            experiment_id="cancel_me",
            started_at=started_at,
        ),
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


@pytest.mark.integration
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
        *runner_argv(
            store,
            run_id=run_id,
            config=config,
            config_sha256=config_sha256,
            experiment_id="cancel_me",
            started_at=started_at,
        ),
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
    """A workdir conflict must carry its own code and fresh-state remediation.

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
    assert "fresh artifact root and local storage" in str(payload["remediation"])
