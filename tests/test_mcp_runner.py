"""Detached runner: process identity, terminal status and failure routing, and cancellation."""

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
from collections.abc import Iterator, Mapping
from pathlib import Path
from unittest.mock import DEFAULT, Mock

import optuna
import pytest
import yaml

import phasesweep.engine.generation as generation_ops
import phasesweep.engine.ledger as engine_ledger
import phasesweep.engine.optuna as engine_optuna
from phasesweep.config.io import load_experiment
from phasesweep.config.models import ExecutionContext, Experiment, Phase
from phasesweep.engine.artifact_roots import _bind_study_artifact_root
from phasesweep.engine.attempts import _register_active_attempt
from phasesweep.engine.errors import (
    ActiveAttemptPersistenceError,
    ArtifactRootConflictError,
    ExperimentLockBusyError,
    PublishedStudyMissingError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    TrackingTargetUnavailableError,
    TrialTargetRefusedError,
)
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.locking import _experiment_lock
from phasesweep.engine.paths import (
    _experiment_dir,
    _generation_path,
    _generation_record_path,
    _generation_summary_path,
    _trial_dir_for,
)
from phasesweep.engine.read import read_status
from phasesweep.engine.run import TerminalReport, run_experiment
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    GENERATION_ID_ATTR,
    TRIAL_DIR_ATTR,
    Winner,
)
from phasesweep.errors import (
    OPERATOR_STEPS,
    OperatorAction,
    ProcessCleanupUncertainError,
    UnsafeProcessCleanupError,
)
from phasesweep.mcp import runner as mcp_runner
from phasesweep.mcp.errors import ConcurrencyLimitError
from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.mcp.snapshots import capture_result_snapshot
from phasesweep.mcp.tools import PhaseSweepMCP
from phasesweep.runtime.files import open_private_text
from phasesweep.runtime.process import write_attempt_lifecycle
from phasesweep.runtime.reaper import (
    PROCESS_IDENTITY_FILE,
    _process_group_alive,
    read_boot_id,
    read_proc_starttime,
)
from phasesweep.runtime.shutdown import PhaseSweepShutdown, ShutdownCleanupReport
from phasesweep.runtime.time import utc_now_iso
from tests.conftest import (
    make_experiment,
    mark_current_format,
    requires_nonroot,
    write_constant_trainer,
    write_trainer,
)
from tests.ledger_fixtures import published_generation_id
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    SEEDED_RANDOM,
    _catalog,
    _config,
    _write_experiment_config,
    claim_runner_handle,
    live_runs,
    make_dead_run_handle,
    make_mcp_app,
    make_run_handle,
    run_claimed_runner,
    runner_argv,
    runner_main,
    write_mcp_catalog,
    write_slow_mcp_config,
)
from tests.recovery_helpers import recover_run_cli


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
    _write_experiment_config(config_path, experiment)
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
        if status["phases"][0]["trials"].get("RUNNING", 0) >= 1:
            phase_dir = _trial_dir_for(experiment, "p", 0).parent
            for trial_dir in phase_dir.glob("trial_00000__*"):
                if (trial_dir / PROCESS_IDENTITY_FILE).is_file():
                    return trial_dir
        time.sleep(0.2)
    raise AssertionError(f"trial never reached RUNNING; log:\n{log_path.read_text()}")


@contextlib.contextmanager
def _spawned_runner(
    store: RunStore, run_id: str, config: Path, *, cwd: Path
) -> Iterator[subprocess.Popen]:
    """Claim ``run_id`` and run the detached runner on ``config`` in a real subprocess.

    The runner is its own session and group leader, so its pgid is its pid. Its
    group is killed on exit unless the body already reaped it.

    :param RunStore store: Run store receiving the reservation, status, and runner log.
    :param str run_id: Run identifier to claim.
    :param Path config: Configuration the runner loads as its snapshot.
    :param Path cwd: Runner working directory.
    :return Iterator[subprocess.Popen]: The running runner process.
    """
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    started_at = utc_now_iso()
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        config_dir=config.resolve().parent,
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
        str(cwd),
    ]
    with open_private_text(store.log_path(run_id), "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


@pytest.mark.parametrize(
    ("error", "code", "remediation_fragment"),
    [
        pytest.param(
            TrackingTargetUnavailableError("MLflow tracking target is not ready"),
            "tracking_unavailable",
            "correct the experiment configuration",
            id="tracking",
        ),
        pytest.param(
            TrialTargetRefusedError("target moved backward"),
            "trial_target_refused",
            "correct the experiment configuration",
            id="target",
        ),
        pytest.param(
            ArtifactRootConflictError("bound to /a, offered /b"),
            "artifact_root_conflict",
            "fresh namespace for the state identified in the error",
            id="artifact_root",
        ),
    ],
)
def test_continuation_preflight_failures_have_actionable_mcp_categories(
    error: RuntimeError,
    code: str,
    remediation_fragment: str,
) -> None:
    """Each preflight refusal carries its own code and the operator step that fits it.

    A workdir conflict mapped to ``fingerprint_mismatch`` would steer the operator
    toward a new experiment name or archiving a healthy study; falling through to
    ``internal_error`` would report a plain misconfiguration as a bug.
    """
    failure = mcp_runner._terminal_failure_payload(error, stage="preflight", cleanup_confirmed=True)

    assert failure["code"] == code
    assert failure["stage"] == "preflight"
    assert failure["retryable"] is False
    assert failure["actor"] == "operator"
    assert remediation_fragment in str(failure["remediation"])


def test_attempt_registry_failure_routes_the_workdir_repair() -> None:
    """The payload names the tree repair and leaves the phase-abort detail to the message.

    Only persistent storage keeps the abort this refusal records, so raising
    n_trials is not a step every such raise requires; the message says when.
    """
    failure = mcp_runner._terminal_failure_payload(
        ActiveAttemptPersistenceError("attempt registry is unwritable"),
        stage="execution",
        cleanup_confirmed=True,
    )

    assert failure["code"] == "storage_unavailable"
    assert failure["retryable"] is False
    assert failure["actor"] == "operator"
    assert failure["remediation"] == (
        "Ask the operator to restore the files and permissions identified in the error, deleting a "
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
    _write_experiment_config(config_path, experiment)
    run_experiment(experiment)
    generation_before = _generation_path(experiment).read_bytes()
    (_experiment_dir(experiment) / "study.journal").unlink()

    snapshot = mcp_runner.capture_result_snapshot(experiment)
    assert snapshot["status"]["phases"][0]["published_study_unavailable"] is True
    assert read_status(experiment)["phases"][0]["published_study_unavailable"] is True

    store = RunStore(tmp_path / "state")
    run_id = "missing-published-study"
    with pytest.raises(PublishedStudyMissingError):
        run_claimed_runner(store, run_id, config_path, experiment_id="t", cwd=tmp_path)

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
    assert store.observe(handle).recovery_required is False
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

    Torn or malformed final records are repaired before writes. Interior
    corruption cannot be an in-flight append, and permission denial prevents
    repair; both require operator restoration before recovery releases capacity.
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
    _write_experiment_config(config_path, experiment)
    other = experiment.model_copy(update={"experiment": "other", "storage": "auto"})
    other_path = tmp_path / "other.yaml"
    _write_experiment_config(other_path, other)
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
        assert "restore or repair the storage ledger and access to it" in refused_run.stderr

        run_id = "damaged-storage"
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        started_at = utc_now_iso()
        claim_runner_handle(
            store,
            run_id=run_id,
            config_sha256=digest,
            config_dir=config_path.resolve().parent,
            started_at=started_at,
            experiment_id="t",
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
        assert status["failure"]["cause"]["retryable"] is False
        assert app.status(run_id=run_id)["run"]["failure"] == status["failure"]
        assert str(ledger) not in json.dumps(status["failure"])
        assert status["generation_unavailable_reason"] == "engine_generation_not_claimed"
        handle = store.get(run_id)
        assert handle is not None
        observation = store.observe(handle)
        assert observation.state == "running"
        assert observation.recovery_required
        with pytest.raises(ConcurrencyLimitError):
            app.launch("other")

        state_dir = tmp_path / "state"
        for confirm in (False, True):
            blocked = recover_run_cli(state_dir, run_id, confirm=confirm)
            assert blocked.exit_code != 0
            assert "could not be" in blocked.output
            assert f"fix: {OPERATOR_STEPS[OperatorAction.RESTORE_LEDGER]}." in blocked.output
            assert "Retry phasesweep mcp recover-run after repairing" in blocked.output
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
    assert store.observe(handle).recovery_required
    confirmed = recover_run_cli(state_dir, run_id, confirm=True)
    assert confirmed.exit_code == 0, confirmed.output
    observation = store.observe(handle)
    assert observation.state == "failed"
    assert not observation.recovery_required
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
def test_prior_abort_refusal_is_a_refused_target_not_a_trainer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun refused for a recorded abort launches no trainer, so it reports like the other refusals."""
    monkeypatch.chdir(tmp_path)
    trainer = write_trainer(tmp_path / "trainer.py", "raise SystemExit(1)")
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'abort.journal'}",
        trainer=trainer,
        n_trials=2,
        max_consecutive_failures=1,
        sampler={"type": "random", "seed": 7},
    )
    config_path = tmp_path / "experiment.yaml"
    _write_experiment_config(config_path, experiment)
    store = RunStore(tmp_path / "state")

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_claimed_runner(store, "aborting", config_path, experiment_id="t", cwd=tmp_path)
    aborting = json.loads(store.status_path("aborting").read_text())["failure"]
    assert (aborting["code"], aborting["stage"]) == ("trainer_failed", "execution")

    with pytest.raises(NoFeasibleTrialError, match="previously aborted"):
        run_claimed_runner(store, "refused", config_path, experiment_id="t", cwd=tmp_path)
    refused = json.loads(store.status_path("refused").read_text())["failure"]
    assert (refused["code"], refused["stage"]) == ("trial_target_refused", "preflight")
    assert (refused["retryable"], refused["actor"]) == (False, "operator")
    assert "correct the experiment configuration" in refused["remediation"]


@pytest.mark.integration
def test_external_engine_lock_is_retryable_and_freezes_pre_generation_snapshot(
    tmp_path: Path,
) -> None:
    config_path = write_slow_mcp_config(tmp_path)
    experiment = load_experiment(config_path)
    assert isinstance(experiment, Experiment)
    store = RunStore(tmp_path / "state")
    run_id = "lock-busy"
    status_path = store.status_path(run_id)

    with (
        _experiment_lock(experiment),
        pytest.raises(ExperimentLockBusyError),
    ):
        run_claimed_runner(store, run_id, config_path, experiment_id="cancel_me", cwd=tmp_path)

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
def test_runner_rejects_config_snapshot_hash_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r1"
    status_path = store.status_path(run_id)

    with pytest.raises(RuntimeError, match="hash mismatch"):
        run_claimed_runner(store, run_id, config, config_sha256="0" * 64)

    status = json.loads(status_path.read_text())
    assert status["returncode"] == 1
    assert status["error_class"] == "RuntimeError"
    assert status["cleanup_confirmed"] is True


@pytest.mark.integration
def test_preflight_failure_is_actionable_through_run_reads(tmp_path: Path) -> None:
    trainer = write_constant_trainer(tmp_path)
    config = tmp_path / "srv.yaml"
    experiment = make_experiment(
        experiment="srv",
        persistent=tmp_path,
        trainer=trainer,
        phases=[
            Phase(
                name="p",
                n_trials=1,
                fixed_overrides={"k": 1},
                sampler=SEEDED_RANDOM,
                search_space={},
            )
        ],
    )
    run_experiment(experiment)
    changed = experiment.model_copy(
        update={"phases": [experiment.phases[0].model_copy(update={"fixed_overrides": {"k": 2}})]}
    )
    _write_experiment_config(config, changed)
    registry = Registry.load(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    store = RunStore(registry.state_dir)
    run_id = "srv-preflight-failure"
    snapshot_path = store.config_snapshot_path(run_id)
    snapshot_path.write_bytes(config.read_bytes())

    with pytest.raises(StudyFingerprintMismatchError, match="different phase config"):
        run_claimed_runner(store, run_id, snapshot_path)

    handle = store.get(run_id)
    assert handle is not None
    terminal = store.recorded_terminal_status(handle)
    assert terminal is not None
    assert terminal["result_snapshot_state"] == "complete"
    assert terminal["failure"]["code"] == "fingerprint_mismatch"
    app = PhaseSweepMCP(registry, store)
    latest = app.latest_run("srv")
    status = app.status(run_id=run_id)
    awaited = asyncio.run(app.await_run(run_id))
    winners = app.winners(run_id=run_id)

    for payload in (latest["run"], status["run"], awaited["run"]):
        assert payload["failure"]["code"] == "fingerprint_mismatch"
        assert payload["failure"]["retryable"] is False
        assert payload["failure"]["actor"] == "operator"
    assert status["result_source"] == "frozen_run_snapshot"
    assert status["summary_present"] is False
    assert status["phases"][0]["attempts_launched_this_run"] == 0
    assert awaited["reason"] == "terminal"
    assert winners["winner_count"] == 0
    assert winners["failure"]["code"] == "fingerprint_mismatch"


@pytest.mark.integration
def test_aggregated_schema_preflight_preserves_actionable_failure_category(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        phases="""\
  - name: a
    n_trials: 1
    sampler: { type: random, seed: 0 }
    search_space: {}
  - name: b
    n_trials: 1
    sampler: { type: random, seed: 1 }
    search_space: {}
""",
    )
    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)
    for phase in experiment.phases:
        study = optuna.create_study(
            study_name=f"{experiment.experiment}::{phase.name}",
            storage=engine_ledger._resolve_storage(experiment.resolved_storage),
            direction="minimize",
        )
        # Two populated, unmarked studies model an unsupported local ledger.
        study.set_user_attr(ARTIFACT_ROOT_ATTR, str(_experiment_dir(experiment)))
        study.add_trial(
            optuna.trial.create_trial(
                value=0.5,
                state=optuna.trial.TrialState.COMPLETE,
            )
        )

    store = RunStore(tmp_path / "state")
    run_id = "srv-schema-mismatch"

    with pytest.raises(StudySchemaMismatchError, match="unsupported PhaseSweep study state"):
        run_claimed_runner(store, run_id, config)

    terminal = json.loads(store.status_path(run_id).read_text())
    assert terminal["result_snapshot_state"] == "complete"
    assert terminal["failure"]["code"] == "study_schema_mismatch"
    assert terminal["failure"]["retryable"] is False
    # Unsupported state routes to starting fresh, rather than to archiving
    # this study and continuing under the current one.
    remediation = terminal["failure"]["remediation"]
    assert "fresh namespace for the state identified in the error" in remediation


@pytest.mark.integration
def test_record_write_failure_still_yields_succeeded_run_with_complete_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: the engine and MCP agree on success.

    The per-generation lifecycle record write is injected to fail after the
    pointer commit — a failure the engine's own transaction contract accepts.
    The detached runner must still record returncode 0 with a complete frozen
    snapshot, and the run store must derive ``succeeded``.
    """

    config_path, config_sha256 = _constant_trial_config(tmp_path, "record_fail")

    real_write = generation_ops.atomic_create_text

    def fail_record_write(path: Path, text: str) -> bool:
        if path.name == "generation.yaml":
            raise OSError("simulated record write failure")
        return real_write(path, text)

    monkeypatch.setattr(generation_ops, "atomic_create_text", fail_record_write)

    store = RunStore(tmp_path / "state")
    run_id = "record-fail-run"
    status_path = store.status_path(run_id)

    assert (
        run_claimed_runner(
            store,
            run_id,
            config_path,
            experiment_id="record_fail",
            config_sha256=config_sha256,
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
    assert store.observe(handle).state == "succeeded"


@pytest.mark.integration
@pytest.mark.signals_own_pid
def test_shutdown_during_terminal_snapshot_capture_keeps_the_published_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a cancel arriving during snapshot capture loses the race.

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

    with pytest.raises(PhaseSweepShutdown) as exc_info:
        run_claimed_runner(
            store,
            run_id,
            config_path,
            experiment_id="cancel_at_capture",
            config_sha256=config_sha256,
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
    assert published_generation_id(load_experiment(config_path)) == run_id


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

    def fail_run(
        *args: object, generation: generation_ops.InvocationGeneration, **kwargs: object
    ) -> None:
        generation.claim()
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
    scan_registry = Mock(return_value=None)
    fail_discovery = Mock(side_effect=cleanup_error)

    def fail_run(
        *args: object, generation: generation_ops.InvocationGeneration, **kwargs: object
    ) -> None:
        generation.claim()
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
    scan_registry.assert_called_once()
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
    preflight = Mock(return_value={})

    def cancel_run(
        *args: object, generation: generation_ops.InvocationGeneration, **kwargs: object
    ) -> None:
        generation.claim()
        raise shutdown

    captured: list[TerminalReport] = []
    monkeypatch.setattr("phasesweep.engine.guards._preflight_existing_studies", preflight)
    monkeypatch.setattr("phasesweep.engine.run._run_experiment_inner", cancel_run)

    with pytest.raises(PhaseSweepShutdown) as exc_info:
        run_experiment(experiment, terminal_callback=captured.append)

    assert exc_info.value is shutdown
    assert preflight.call_count == 2
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
    preflight = Mock(side_effect=[{}, shutdown])

    def fail_run(
        *args: object, generation: generation_ops.InvocationGeneration, **kwargs: object
    ) -> None:
        generation.claim()
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


@pytest.mark.integration
def test_runner_records_cleanup_uncertainty_for_cleanup_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r1"
    status_path = store.status_path(run_id)

    def raise_cleanup_uncertain(*args: object, **kwargs: object) -> None:
        raise UnsafeProcessCleanupError("trial cleanup uncertain")

    monkeypatch.setattr("phasesweep.mcp.runner.run_experiment", raise_cleanup_uncertain)

    with pytest.raises(UnsafeProcessCleanupError, match="trial cleanup uncertain"):
        run_claimed_runner(store, run_id, config)

    status = json.loads(status_path.read_text())
    assert status["returncode"] == 1
    assert status["error_class"] == "UnsafeProcessCleanupError"
    assert status["cleanup_confirmed"] is False


@pytest.mark.integration
def test_runner_makes_cleanup_uncertainty_actionable_and_preserves_primary_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r-secondary-cleanup"
    status_path = store.status_path(run_id)
    primary = NoFeasibleTrialError("trainer failed")

    def fail_with_uncertain_cleanup(
        config_obj: Experiment,
        *,
        from_phase: str | None,
        dry_run: bool,
        terminal_callback,
        generation_id: str,
        publication_hook: object,
    ) -> None:
        del config_obj, from_phase, dry_run, publication_hook
        terminal_callback(
            TerminalReport(
                generation_id=generation_id,
                primary_error=primary,
                cleanup_confirmed=False,
                recovered_attempt_ids=frozenset({"attempt-reconciled"}),
                uncertain_attempt_ids=frozenset({"attempt-1"}),
                recovered_attempt_generations={"attempt-reconciled": generation_id},
                cleanup_error=ProcessCleanupUncertainError("cleanup uncertain"),
            )
        )
        raise ProcessCleanupUncertainError("cleanup uncertain") from primary

    monkeypatch.setattr(
        "phasesweep.mcp.runner.run_experiment",
        fail_with_uncertain_cleanup,
    )

    with pytest.raises(ProcessCleanupUncertainError):
        run_claimed_runner(store, run_id, config)

    status = json.loads(status_path.read_text())
    assert status["returncode"] == 1
    assert status["error_class"] == "ProcessCleanupUncertainError"
    assert status["cleanup_confirmed"] is False
    assert status["recovered_attempt_ids"] == ["attempt-reconciled"]
    assert status["recovered_attempt_generations"] == {
        "attempt-reconciled": run_id,
    }
    assert status["uncertain_attempt_ids"] == ["attempt-1"]
    assert status["failure"]["code"] == "cleanup_uncertain"
    assert status["failure"]["stage"] == "cleanup"
    assert status["failure"]["retryable"] is False
    assert status["failure"]["actor"] == "operator"
    assert status["failure"]["cause"]["code"] == "trainer_failed"
    assert status["failure"]["cause"]["stage"] == "execution"


@pytest.mark.integration
def test_runner_persists_registered_terminal_identity_uncertainty(tmp_path: Path) -> None:
    """A terminal trial missing its attempt id remains attributable to recover-run."""
    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r-terminal-identity"
    status_path = store.status_path(run_id)

    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)
    phase = experiment.phases[0]
    study = optuna.create_study(
        study_name=f"{experiment.experiment}::{phase.name}",
        storage=_resolve_storage(experiment.resolved_storage),
        direction="minimize",
    )
    mark_current_format(experiment, study)
    _bind_study_artifact_root(study, experiment)
    trial = study.ask()
    attempt_id = "terminal-identity-attempt"
    generation_id = "earlier-generation"
    trial_dir = _trial_dir_for(
        experiment,
        phase.name,
        trial.number,
        generation_id=generation_id,
        attempt_id=attempt_id,
    )
    trial_dir.mkdir(parents=True)
    write_attempt_lifecycle(trial_dir, attempt_id=attempt_id, state="allocated")
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    trial.set_user_attr(GENERATION_ID_ATTR, generation_id)
    trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(trial, state=optuna.trial.TrialState.FAIL)
    _register_active_attempt(
        experiment,
        attempt_id=attempt_id,
        phase_name=phase.name,
        study_name=study.study_name,
        trial_number=trial.number,
        trial_dir=trial_dir,
        generation_id=generation_id,
    )

    with pytest.raises(ProcessCleanupUncertainError, match="identity is missing"):
        run_claimed_runner(store, run_id, config)

    status = json.loads(status_path.read_text())
    assert status["cleanup_confirmed"] is False
    assert status["uncertain_attempt_ids"] == [attempt_id]
    # The refusal names the ledger as the first repair, so the agent hears the same.
    assert status["failure"]["code"] == "cleanup_uncertain"
    assert status["failure"]["remediation"] == (
        "Ask the operator to restore or repair the storage ledger and access to it, then "
        "run phasesweep mcp recover-run before another launch. The error in the "
        "PhaseSweep run log gives the details."
    )


def _write_captured_terminal_status(
    status_path: Path, snapshot: dict[str, object], *, run_id: str = "r0"
) -> bool:
    """Persist a successful terminal outcome and its captured result snapshot."""
    return mcp_runner._write_status(
        status_path,
        {"run_id": run_id, "returncode": 0, "error_class": None, "cleanup_confirmed": True},
        result_snapshot=snapshot,
        result_snapshot_error=None,
    )


def test_runner_records_snapshot_serialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(
        mcp_runner,
        "finalize_result_snapshot",
        lambda *_args, **_kwargs: {"not_json": object()},
    )

    _write_captured_terminal_status(status_path, {})

    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "failed"
    assert final["result_snapshot_error"] == "TypeError"
    assert "result_snapshot" not in final


def test_transient_terminal_status_write_failure_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-shot OS failure must not cost a finished run its terminal evidence.

    Injected at ``os.replace`` so the retry runs through the real private
    atomic writer - temp file, fsync, replace - rather than a stubbed one.
    """
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(mcp_runner, "finalize_result_snapshot", lambda snapshot: snapshot)
    replace_file = Mock(
        wraps=os.replace,
        side_effect=[OSError("simulated transient atomic replace failure"), DEFAULT, DEFAULT],
    )
    monkeypatch.setattr("phasesweep.runtime.files.os.replace", replace_file)

    assert _write_captured_terminal_status(status_path, {"captured": True}) is True

    # Failed pending write, its retry, then the complete write.
    assert replace_file.call_count == 3
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

    assert _write_captured_terminal_status(status_path, {"captured": True}) is True

    assert states == ["pending", "complete", "complete"]
    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "complete"
    assert final["result_snapshot"] == {"captured": True}


def test_exhausted_status_write_retries_report_missing_terminal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """When nothing can be persisted, say so instead of returning as if it were."""
    status_path = tmp_path / "status.json"
    write_status = Mock(side_effect=OSError("simulated persistent status write failure"))
    monkeypatch.setattr(mcp_runner, "write_status_file", write_status)

    with caplog.at_level(logging.ERROR, logger="phasesweep.mcp.runner"):
        assert _write_captured_terminal_status(status_path, {"captured": True}) is False

    assert write_status.call_count == 3
    assert not status_path.exists()
    assert any("no terminal MCP evidence" in record.getMessage() for record in caplog.records)


def test_unpersistable_complete_transition_keeps_the_recoverable_pending_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable ``pending`` record with its snapshot outranks a degraded ``failed`` one.

    ``pending`` plus a dead runner is exactly what ``recover-run`` finalizes, so the runner
    must not trade it for the permanent ``failed`` state when the complete transition
    cannot be persisted.
    """
    store = RunStore(tmp_path / "state")
    run_id = "pending-run"
    handle = make_dead_run_handle(run_id=run_id, experiment_id="exp")
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

    assert _write_captured_terminal_status(status_path, {"captured": True}, run_id=run_id) is True

    # Three exhausted complete attempts, and no downgrade write after them.
    assert states == ["pending", "complete", "complete", "complete"]
    durable = json.loads(status_path.read_text())
    assert durable["result_snapshot_state"] == "pending"
    assert durable["result_snapshot"] == {"captured": True}
    assert "result_snapshot_error" not in durable
    observation = store.observe(handle)
    assert observation.state == "running"
    assert observation.snapshot_recovery_required


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

    assert _write_captured_terminal_status(status_path, {"not_json": object()}) is True

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

    A failed required precommit snapshot write prevents publication.
    The process entry point converts this propagated error to a nonzero exit;
    an in-process invocation sees the original error and no pointer advances.
    """
    config_path, config_sha256 = _constant_trial_config(tmp_path, "no_status")

    def refuse_every_write(_path: Path, _payload: dict) -> None:
        raise OSError("simulated persistent status write failure")

    monkeypatch.setattr(mcp_runner, "write_status_file", refuse_every_write)

    store = RunStore(tmp_path / "state")
    run_id = "no-status-run"

    with (
        caplog.at_level(logging.ERROR, logger="phasesweep.mcp.runner"),
        pytest.raises(OSError, match="simulated persistent status write failure"),
    ):
        run_claimed_runner(
            store,
            run_id,
            config_path,
            experiment_id="no_status",
            config_sha256=config_sha256,
            cwd=tmp_path,
        )

    assert not store.status_path(run_id).exists()
    assert any("no terminal MCP evidence" in record.getMessage() for record in caplog.records)
    assert published_generation_id(load_experiment(config_path)) is None
    # The consequence the exit code has to carry: this identity is exactly what
    # the runner persisted for itself (spawned, this live process), and with no
    # status.json the run stays ``running`` and holds its concurrency slot.
    assert (
        store.observe(make_run_handle(run_id=run_id, experiment_id="no_status")).state == "running"
    )


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
        config_dir=config_path.resolve().parent,
        started_at=started_at,
        experiment_id=crash_boundary,
    )
    with open_private_text(store.config_snapshot_path(run_id), "x") as output:
        output.write(config_path.read_text())
    experiment = load_experiment(config_path)
    original_stats = engine_ledger.read_trial_stats

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

    monkeypatch.setattr(
        "phasesweep.engine.read.read_trial_stats", unavailable_during_prepared_capture
    )

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
    assert published_generation_id(experiment) == (
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
        assert published_generation_id(experiment) is None
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
        config_dir=tmp_path,
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


@pytest.mark.parametrize("field", ["experiment_id", "config_sha256", "started_at", "allow_cancel"])
def test_runner_refuses_protocol_metadata_that_differs_from_launch_reservation(
    tmp_path: Path, field: str
) -> None:
    """Process identity cannot replace the immutable metadata reserved by the server."""
    store = RunStore(tmp_path / "state")
    run_id = "srv-identity-metadata"
    started_at = "2026-06-24T00:00:00Z"
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256="a" * 64,
        config_dir=tmp_path,
        started_at=started_at,
    )
    pending = store.get(run_id)
    assert pending is not None

    with pytest.raises(ValueError, match=field):
        mcp_runner._persist_spawned_handle(
            state_dir=tmp_path / "state",
            run_id=run_id,
            experiment_id="other" if field == "experiment_id" else "srv",
            config_sha256="b" * 64 if field == "config_sha256" else "a" * 64,
            started_at="2026-06-25T00:00:00Z" if field == "started_at" else started_at,
            allow_cancel=field == "allow_cancel",
        )

    assert store.get(run_id) == pending


@pytest.mark.integration
def test_runner_persists_spawned_handle_for_restart_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "srv-recover"
    started_at = "2026-06-24T00:00:00Z"
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    calls: list[tuple[str, str | None, bool]] = []

    def fake_run_experiment(
        config_obj: Experiment,
        *,
        from_phase: str | None,
        dry_run: bool,
        terminal_callback,
        generation_id: str,
        publication_hook: object,
    ) -> None:
        del publication_hook
        assert generation_id == run_id
        calls.append((config_obj.experiment, from_phase, dry_run))
        mark_current_format(config_obj)
        generation_path = _generation_record_path(config_obj, generation_id)
        generation_path.parent.mkdir(parents=True, exist_ok=True)
        generation_path.write_text(f"generation_id: {generation_id}\n")
        terminal_callback(
            TerminalReport(
                generation_id=generation_id,
                primary_error=None,
                cleanup_confirmed=True,
                recovered_attempt_ids=frozenset(),
                uncertain_attempt_ids=frozenset(),
            )
        )

    monkeypatch.setattr("phasesweep.mcp.runner.run_experiment", fake_run_experiment)

    def capture_without_fake_trial_storage(
        experiment: Experiment,
        *,
        generation_id: str,
        engine_winners: object = None,
    ) -> dict[str, object]:
        return capture_result_snapshot(
            experiment,
            generation_id=generation_id,
        )

    monkeypatch.setattr(
        "phasesweep.mcp.runner.capture_result_snapshot",
        capture_without_fake_trial_storage,
    )

    assert (
        run_claimed_runner(
            store, run_id, config, config_sha256=config_sha256, started_at=started_at
        )
        == 0
    )

    handle = store.get(run_id)
    assert handle is not None
    assert handle.launch_state == "spawned"
    assert handle.experiment_id == "srv"
    assert handle.config_sha256 == config_sha256
    assert handle.pid == os.getpid()
    assert handle.pgid == os.getpgrp()
    assert handle.pid_starttime == read_proc_starttime(os.getpid())
    assert handle.started_at == started_at
    assert store.observe(handle).state == "succeeded"
    assert calls == [("srv", None, False)]
    terminal = store.recorded_terminal_status(handle)
    assert terminal is not None
    assert terminal["result_snapshot_state"] == "complete"
    assert terminal["result_snapshot"]["status"]["phases"][0]["phase"] == "p"
    assert terminal["result_snapshot"]["winners"] == []


@pytest.mark.parametrize(
    ("reader", "match"),
    [
        pytest.param("read_proc_starttime", "/proc start time is unavailable", id="proc_starttime"),
        pytest.param("read_boot_id", "boot id", id="boot_id"),
    ],
)
def test_runner_does_not_persist_a_receipt_without_process_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: str,
    match: str,
) -> None:
    """A runner that cannot read its Linux process identity leaves its claimed handle launching."""
    store = RunStore(tmp_path / "state")
    run_id = "srv-no-identity"
    started_at = "2026-06-24T00:00:00Z"
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256="0" * 64,
        config_dir=tmp_path,
        started_at=started_at,
    )
    monkeypatch.setattr(mcp_runner, reader, lambda *_args: None)

    with pytest.raises(RuntimeError, match=match):
        mcp_runner._persist_spawned_handle(
            state_dir=tmp_path / "state",
            run_id=run_id,
            experiment_id="srv",
            config_sha256="0" * 64,
            started_at=started_at,
            allow_cancel=False,
        )

    pending = store.get(run_id)
    assert pending is not None
    assert pending.launch_state == "launching"


@pytest.mark.integration
def test_runner_enters_the_project_directory_after_its_identity_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--cwd`` supplies the project directory instead of an ambient cwd inherited from the server.

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

    def persist_with_held_lease(*args: object, **kwargs: object) -> RunHandle:
        os.fstat(inherited_lease_fd)
        assert not store.is_pre_spawn_orphan("r-cwd")
        return real_persist(*args, **kwargs)

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

    with pytest.raises(FileNotFoundError):
        run_claimed_runner(
            store,
            "r-nocwd",
            tmp_path / "unread.yaml",
            experiment_id="cancel_me",
            config_sha256="a" * 64,
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
    config = write_slow_mcp_config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r1"
    status_path = store.status_path(run_id)
    log_path = store.log_path(run_id)
    with _spawned_runner(store, run_id, config, cwd=tmp_path) as proc:
        trial_dir = _wait_for_running_trial(config, proc, log_path)
        trial_pgid = json.loads((trial_dir / PROCESS_IDENTITY_FILE).read_text())["pgid"]

        # SIGTERM the runner's process group. The trial runs in its OWN session,
        # so this does not reach it directly; the runner's installed shutdown
        # handler tears the trial group down, then exits 128+SIGTERM = 143.
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)

    assert proc.returncode == 143, (
        f"expected 143, got {proc.returncode}; log:\n{log_path.read_text()}"
    )
    status = json.loads(status_path.read_text())
    assert status["returncode"] == 143
    assert status["error_class"] == "cancelled"
    assert status["cleanup_confirmed"] is True
    assert status["result_snapshot_state"] == "complete"
    assert status["result_snapshot"]["status"]["phases"][0]["trials"]["INTERRUPTED"] == 1
    assert status["result_snapshot"]["winners"] == []
    assert not _process_group_alive(trial_pgid)


@pytest.mark.integration
def test_runner_cancelled_before_first_trial_still_records_cancelled(tmp_path: Path) -> None:
    """A cancel arriving before any trial starts must still write status.json.

    The server creates a launching handle before spawn. The runner installs
    shutdown handlers before updating that handle to spawned, so the transition
    proves SIGTERM is caught even while the config is still loading.
    """
    config = write_slow_mcp_config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r2"
    status_path = store.status_path(run_id)
    log_path = store.log_path(run_id)
    with _spawned_runner(store, run_id, config, cwd=tmp_path) as proc:
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

    assert proc.returncode == 143, (
        f"expected 143, got {proc.returncode}; log:\n{log_path.read_text()}"
    )
    status = json.loads(status_path.read_text())
    assert status["returncode"] == 143
    assert status["error_class"] == "cancelled"
    assert status["cleanup_confirmed"] is True
