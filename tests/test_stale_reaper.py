"""Stale trial reaper: detect RUNNING Optuna trials whose orchestrator died, kill any leaked process group (PID-reuse-safe via starttime check), and mark the trial FAIL. Fail-closed throughout — never silently advance the study while a leaked process may be live."""

from __future__ import annotations

import contextlib
import errno
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import optuna
import pytest

import phasesweep.engine.ledger as engine_ledger
from phasesweep.config import (
    Experiment,
    IntParam,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
)
from phasesweep.engine import (
    ActiveAttemptPersistenceError,
    ArtifactRootConflictError,
    NoFeasibleTrialError,
    PhaseSweepError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    run_experiment,
)
from phasesweep.engine.attempts import (
    _inspect_active_attempts,
    _preflight_active_attempts,
    _PreflightCleanupReport,
    _record_stale_trial_failure,
    _register_active_attempt,
)
from phasesweep.engine.cleanup import _reap_stale_trials
from phasesweep.engine.fingerprints import _phase_fingerprint
from phasesweep.engine.guards import _preflight_existing_studies, _reconcile_existing_studies
from phasesweep.engine.paths import _attempts_dir, _experiment_dir, _trial_dir_for
from phasesweep.engine.phase import _run_phase
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    GENERATION_ID_ATTR,
    PHASE_ABORT_ATTR,
    PHASE_FINGERPRINT_ATTR,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_ENV_NAMES_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_TARGET_ATTR,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError, _environment_identity
from phasesweep.mcp.recovery import RunRecoveryError, recover_run
from phasesweep.mcp.runs import RunStore
from phasesweep.runtime.process import write_attempt_lifecycle
from phasesweep.runtime.reaper import (
    PROCESS_IDENTITY_FILE,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    StaleProcessIdentity,
    _read_proc_stat,
    cleanup_stale_trial_process,
    is_same_live_process,
    kill_stale_group,
    read_proc_starttime,
    read_stale_process_identity,
)
from phasesweep.runtime.shutdown import PhaseSweepShutdown, ShutdownCleanupReport
from tests.conftest import (
    make_experiment,
    mark_current_format,
    patch_rejected_trial_user_attr,
    write_trainer,
)
from tests.ledger_fixtures import _write_config
from tests.mcp_helpers import stage_dead_run
from tests.recovery_helpers import (
    fabricate_registered_attempt,
    fabricate_stale_trial,
    stamp_artifact_root,
    write_trial_identity,
)


def test_read_proc_starttime_self():
    """We should be able to read our own process starttime on Linux."""
    import os

    st = read_proc_starttime(os.getpid())
    # On Linux this should be a positive integer. On non-Linux, None.
    if Path("/proc/self/stat").exists():
        assert st is not None and st > 0
    else:
        assert st is None


def test_read_proc_stat_tolerates_non_utf8_comm(tmp_path: Path) -> None:
    """``/proc/<pid>/stat`` comm bytes are not guaranteed to be UTF-8."""
    proc_entry = tmp_path / "123"
    proc_entry.mkdir()
    fields = [b"S", b"1", b"4321"] + [b"0"] * 16 + [b"987654"] + [b"0"] * 8
    (proc_entry / "stat").write_bytes(b"123 (trainer-\xff-worker) " + b" ".join(fields))

    stat = _read_proc_stat(proc_entry)

    assert stat is not None
    assert stat.state == "S"
    assert stat.pgrp == 4321
    assert stat.starttime == 987654


def test_is_same_live_process_fails_closed_when_proc_entry_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable ``/proc/<pid>/stat`` must never confirm process identity.

    The PID is alive and the saved starttime is genuinely ours, so the only
    thing that changes is that the ``/proc`` read is denied. Identity is then
    unverifiable, and callers (stale-trial cleanup, MCP run supervision) must
    see ``False`` - never a permissive ``True`` and never a raised
    ``PermissionError`` from a liveness probe.
    """
    pid = os.getpid()
    saved_starttime = read_proc_starttime(pid)
    if saved_starttime is None:
        pytest.skip("/proc process identity is unavailable on this platform")
    assert is_same_live_process(pid, saved_starttime) is True  # control: readable /proc matches

    real_read_bytes = Path.read_bytes

    def deny_proc_reads(self: Path, *args: object, **kwargs: object) -> bytes:
        if str(self).startswith("/proc/"):
            raise PermissionError(f"{self} is unreadable")
        return real_read_bytes(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_bytes", deny_proc_reads)

    assert is_same_live_process(pid, saved_starttime) is False


def _claimed_over(
    experiment: Experiment, studies: Mapping[str, object]
) -> engine_ledger.ClaimedLedger:
    """Hand preflight a claimed handle over exactly the given studies.

    Preflight only ever inspects the studies its handle carries, so tests of
    its error handling stand in for :func:`claim_ledger`'s discovery with
    hand-built (or deliberately broken) study objects instead of running it.

    :param Experiment experiment: Experiment the handle is claimed for.
    :param Mapping[str, object] studies: Study stand-ins keyed by phase name.
    :return engine_ledger.ClaimedLedger: Handle whose ``studies`` are ``studies``.
    """
    validated = engine_ledger.validate_ledger(experiment)
    return engine_ledger.ClaimedLedger(
        experiment=experiment,
        experiment_name=validated.experiment_name,
        storage_url=validated.storage_url,
        backend=validated.backend,
        ledger_path=validated.ledger_path,
        artifact_root=validated.artifact_root,
        binding_state="bound",
        format_scan_failure=None,
        studies=MappingProxyType(dict(studies)),  # type: ignore[arg-type]
    )


def test_reap_runs_before_fingerprint_check(tmp_path, monkeypatch):
    """If config changed AND a stale RUNNING trial exists, reap must happen first.

    Previously, _verify_fingerprint raised before _reap_stale_trials could run,
    leaving the prior orchestrator's training process holding the GPU.
    """
    trainer = tmp_path / "t.py"
    write_trainer(
        trainer,
        """
        import json, argparse
        ap = argparse.ArgumentParser()
        ap.add_argument('--out', required=True)
        args, _ = ap.parse_known_args()
        with open(args.out, 'w') as f: json.dump({'x': 1.0}, f)
        """,
    )
    db = tmp_path / "p.journal"
    storage = f"journal:///{db}"

    # Manually create a study and inject a RUNNING trial + fingerprint.
    study = optuna.create_study(
        study_name="t::a",
        storage=engine_ledger._resolve_storage(storage),
        direction="minimize",
    )
    study.set_user_attr("phasesweep_fingerprint", "OLD-FINGERPRINT")
    study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    t = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    # Don't call study.tell — leave it RUNNING.

    # The reaper is replaced below; the directory only makes the historical
    # workdir shape explicit for the fingerprint-order assertion.
    trial_dir = tmp_path / "runs" / "a" / f"trial_{t.number:05d}"
    trial_dir.mkdir(parents=True)

    reap_called = {"flag": False}
    fingerprint_called = {"flag": False}

    import phasesweep.engine.phase as orch

    real_verify = orch._verify_fingerprint

    def spy_reap(*args, **kwargs):
        reap_called["flag"] = True
        # Don't actually call the real reaper; just mark FAIL.
        for t_ in args[0].get_trials(deepcopy=False):
            if t_.state == optuna.trial.TrialState.RUNNING:
                _record_stale_trial_failure(args[0], t_)
                args[0].tell(t_.number, state=optuna.trial.TrialState.FAIL)
        return 1

    def spy_verify(*args, **kwargs):
        fingerprint_called["flag"] = True
        # Reap must have happened first.
        assert reap_called["flag"], (
            "_verify_fingerprint was called before _reap_stale_trials — "
            "config-mismatch errors will leave stale processes alive on the GPU."
        )
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(orch, "_reap_stale_trials", spy_reap)
    monkeypatch.setattr(orch, "_verify_fingerprint", spy_verify)

    exp = Experiment(
        experiment="t",
        storage=storage,
        provenance={"revision": "test-fixture-v1"},
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/result.json {{overrides}}",
        override_format="argparse",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
        phases=[
            Phase(
                name="a",
                n_trials=1,
                sampler=Sampler(type="random", seed=0),
                search_space={"x": IntParam(type="int", low=0, high=10)},
            )
        ],
    )

    stamp_artifact_root(study, exp)
    identity = _environment_identity(exp)
    t.set_user_attr(TRAINER_ENV_DIGEST_ATTR, identity.digest)
    t.set_user_attr(TRAINER_ENV_NAMES_ATTR, list(identity.names))
    claimed = engine_ledger.claim_ledger(engine_ledger.validate_ledger(exp))

    # Will raise on fingerprint mismatch, but reap must have run first.
    with pytest.raises(RuntimeError, match="different phase config"):
        _run_phase(
            exp,
            exp.phases[0],
            inherited_winners={},
            generation_id="generation-test",
            ledger=claimed,
        )
    assert reap_called["flag"]
    assert fingerprint_called["flag"]


@pytest.mark.integration
def test_run_reaps_later_phase_orphan_before_first_phase_launch(tmp_path: Path) -> None:
    """A new generation starts only after every existing phase is recovered."""
    trainer = write_trainer(
        tmp_path,
        """
        import os
        from pathlib import Path

        stat_path = Path("/proc") / os.environ["STALE_PID"] / "stat"
        if stat_path.exists():
            state = stat_path.read_text().rsplit(")", 1)[1].strip().split()[0]
            alive = state != "Z"
        else:
            alive = False
        Path(os.environ["PHASESWEEP_TRIAL_DIR"], "orphan_alive.txt").write_text(str(alive))
        print("metric=1.0")
        """,
    )
    storage = f"journal:///{tmp_path / 'studies.journal'}"
    experiment = Experiment(
        experiment="cross_phase_orphan",
        storage=storage,
        provenance={"revision": "test-fixture-v1"},
        workdir=str(tmp_path / "runs"),
        trial_command=f"{sys.executable} {trainer}",
        override_format="argparse",
        metric=Metric(
            name="metric",
            extractor=LogRegexExtractor(type="log_regex", pattern=r"metric=(?P<value>[0-9.eE+-]+)"),
        ),
        phases=[
            Phase(name="a", n_trials=1, sampler=Sampler(type="random", seed=0), search_space={}),
            Phase(
                name="b",
                n_trials=2,
                sampler=Sampler(type="random", seed=0),
                search_space={},
            ),
        ],
    )
    stale = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        starttime = read_proc_starttime(stale.pid)
        assert starttime is not None
        # The env names the orphan before the trial is fabricated, so its
        # recorded trainer-env identity matches the run that must reap it.
        experiment = experiment.model_copy(update={"env": {"STALE_PID": str(stale.pid)}})
        study, trial_dir, stale_number = fabricate_stale_trial(
            experiment, "b", attempt_id="old-attempt"
        )
        write_trial_identity(
            trial_dir,
            attempt_id="old-attempt",
            pid=stale.pid,
            pgid=os.getpgid(stale.pid),
            starttime=starttime,
        )
        study.set_user_attr(
            PHASE_FINGERPRINT_ATTR,
            _phase_fingerprint(experiment, experiment.phases[1], {}),
        )
        run_experiment(experiment)

        marker = next(
            (tmp_path / "runs" / experiment.experiment / "a").glob("trial_*/orphan_alive.txt")
        )
        assert marker.read_text() == "False"
        assert stale.poll() == -signal.SIGTERM
        assert study.get_trials(deepcopy=False)[stale_number].state == optuna.trial.TrialState.FAIL
    finally:
        if stale.poll() is None:
            os.killpg(stale.pid, signal.SIGKILL)
        stale.wait(timeout=5)


def test_current_schema_rejects_terminal_trial_without_policy_outcome(
    tmp_path: Path,
) -> None:
    """Current-schema terminal rows must carry reconstructable policy state."""
    storage = f"journal:///{tmp_path / 'studies.journal'}"
    experiment = make_experiment(
        experiment="missing_outcome",
        storage=storage,
        workdir=tmp_path / "runs",
        n_trials=1,
    )
    study = optuna.create_study(
        study_name="missing_outcome::p",
        storage=engine_ledger._resolve_storage(storage),
        direction="minimize",
    )
    study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    study.set_user_attr(TRIAL_TARGET_ATTR, experiment.phases[0].n_trials)
    stamp_artifact_root(study, experiment)
    study.add_trial(optuna.trial.create_trial(value=0.25, state=optuna.trial.TrialState.COMPLETE))

    with pytest.raises(
        RuntimeError,
        match=rf"terminal trial 0 has missing or malformed '{TRIAL_OUTCOME_ATTR}'",
    ):
        run_experiment(experiment)


@pytest.mark.parametrize("stage", ["load", "get_trials", "stale_reaper", "schema_validation"])
def test_recovery_preflight_preserves_shutdown_control_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    experiment = make_experiment(workdir=tmp_path / "runs")
    shutdown = PhaseSweepShutdown(
        signal.SIGINT,
        ShutdownCleanupReport(
            signum=signal.SIGINT,
            cleanup_confirmed=True,
            child_pgids=(),
        ),
    )

    if stage == "load":

        def claim(*_args: object, **_kwargs: object) -> engine_ledger.ClaimedLedger:
            raise shutdown

        monkeypatch.setattr("phasesweep.engine.guards.claim_ledger", claim)
    elif stage == "get_trials":
        study = SimpleNamespace(
            study_name="shutdown::p",
            user_attrs={},
            get_trials=lambda **_kwargs: (_ for _ in ()).throw(shutdown),
        )

        def claim(*_args: object, **_kwargs: object) -> engine_ledger.ClaimedLedger:
            return _claimed_over(experiment, {"p": study})

        monkeypatch.setattr("phasesweep.engine.guards.claim_ledger", claim)
    else:
        study = optuna.create_study(direction="minimize")

        def claim(*_args: object, **_kwargs: object) -> engine_ledger.ClaimedLedger:
            return _claimed_over(experiment, {"p": study})

        monkeypatch.setattr("phasesweep.engine.guards.claim_ledger", claim)
        if stage == "stale_reaper":
            monkeypatch.setattr(
                "phasesweep.engine.guards._reap_stale_trials",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(shutdown),
            )
        else:
            monkeypatch.setattr(
                "phasesweep.engine.guards._reap_stale_trials",
                lambda *_args, **_kwargs: 0,
            )
            monkeypatch.setattr(
                "phasesweep.engine.guards._validate_study_schema",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(shutdown),
            )

    cleanup = _PreflightCleanupReport()
    with pytest.raises(PhaseSweepShutdown) as exc_info:
        _reconcile_existing_studies(experiment, cleanup_report=cleanup)

    assert exc_info.value is shutdown
    assert cleanup.cleanup_confirmed is True
    assert cleanup.error is None


def test_mixed_preflight_errors_keep_cleanup_uncertainty_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        phases=[
            Phase(name="a", n_trials=1, search_space={}),
            Phase(name="b", n_trials=1, search_space={}),
        ],
    )
    studies = {
        phase.name: optuna.create_study(study_name=phase.name, direction="minimize")
        for phase in experiment.phases
    }

    def reap(_study: optuna.Study, _experiment: Experiment, phase_name: str, **_kwargs) -> int:
        if phase_name == "a":
            raise ProcessCleanupUncertainError("cleanup uncertain")
        return 0

    monkeypatch.setattr("phasesweep.engine.guards._reap_stale_trials", reap)
    monkeypatch.setattr(
        "phasesweep.engine.guards._validate_study_schema",
        lambda _study: (_ for _ in ()).throw(RuntimeError("schema read failed")),
    )

    with pytest.raises(ProcessCleanupUncertainError, match="multiple unsafe studies"):
        _preflight_existing_studies(_claimed_over(experiment, studies))


def test_registry_storage_failure_marks_cleanup_report_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An aborted registry scan cannot report cleanup as confirmed."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    failure = StudyStorageUnavailableError("registry storage unavailable")
    monkeypatch.setattr(
        "phasesweep.engine.guards._preflight_active_attempts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    report = _PreflightCleanupReport()

    with pytest.raises(StudyStorageUnavailableError, match="registry storage unavailable"):
        _preflight_existing_studies(_claimed_over(experiment, {}), cleanup_report=report)

    assert report.cleanup_confirmed is False
    assert report.error is failure


@pytest.mark.parametrize(
    ("secondary_error", "expected_type", "operational"),
    [
        pytest.param(
            StudyStorageUnavailableError("storage unavailable"),
            PhaseSweepError,
            True,
            id="all-operational",
        ),
        pytest.param(
            RuntimeError("injected implementation bug"),
            RuntimeError,
            False,
            id="contains-internal-error",
        ),
    ],
)
def test_mixed_preflight_error_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    secondary_error: Exception,
    expected_type: type[Exception],
    operational: bool,
) -> None:
    """A mixed aggregate is operational only when every cause is expected."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        phases=[
            Phase(name="a", n_trials=1, search_space={}),
            Phase(name="b", n_trials=1, search_space={}),
        ],
    )
    studies = {
        phase.name: optuna.create_study(study_name=phase.name, direction="minimize")
        for phase in experiment.phases
    }

    def reject_differently(study: optuna.Study) -> None:
        if study.study_name == "a":
            raise StudySchemaMismatchError("schema mismatch")
        raise secondary_error

    monkeypatch.setattr("phasesweep.engine.guards._validate_study_schema", reject_differently)
    with pytest.raises(expected_type, match="multiple unsafe studies") as exc_info:
        _preflight_existing_studies(_claimed_over(experiment, studies))
    assert isinstance(exc_info.value, PhaseSweepError) is operational
    if operational:
        assert type(exc_info.value) is PhaseSweepError


@pytest.mark.integration
def test_kill_stale_group_escalates_to_sigkill():
    """A child that ignores SIGTERM must still be killed within the grace window."""
    if not Path("/proc/self/stat").exists():
        pytest.skip("Linux-only test (uses /proc starttime)")

    # Spawn a child that ignores SIGTERM and only dies on SIGKILL.
    proc = subprocess.Popen(
        [
            "python3",
            "-c",
            ("import signal, time;signal.signal(signal.SIGTERM, signal.SIG_IGN);time.sleep(60)"),
        ],
        start_new_session=True,
    )
    try:
        # Give it a moment to install the handler.
        time.sleep(0.3)
        from phasesweep.runtime.reaper import read_proc_starttime

        st = read_proc_starttime(proc.pid)
        assert st is not None

        # Use a short grace window so the test runs fast.
        sent = kill_stale_group(proc.pid, st, grace_seconds=1.5)
        assert sent is True

        # After kill_stale_group returns, the process must actually be dead within
        # a brief follow-up window (SIGKILL is asynchronous from the kernel side).
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        assert proc.poll() is not None, "Child survived kill_stale_group despite SIGKILL escalation"
    finally:
        with contextlib.suppress(Exception):
            proc.kill()
            proc.wait(timeout=2)


@pytest.mark.integration
def test_kill_stale_group_uses_pgid_when_root_pid_gone() -> None:
    """If the root PID has exited but pgid is known, reaper still kills the group.

    Simulates the shell=True case where the shell exits but a long-lived
    descendant remains in the same process group.
    """
    # Spawn a process group, then kill only the root, leaving the descendant alive.
    parent = subprocess.Popen(
        [
            "python3",
            "-c",
            (
                "import os, subprocess, time, sys;"
                "child = subprocess.Popen(['sleep', '60']);"
                # parent prints child's PID then exits, leaving child in our PGID.
                "sys.stdout.write(str(child.pid) + '\\n'); sys.stdout.flush();"
                "time.sleep(0.3); sys.exit(0)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    pgid = os.getpgid(parent.pid)
    starttime = read_proc_starttime(parent.pid)
    assert starttime is not None
    child_pid = int(parent.stdout.readline().strip())
    parent.wait(timeout=5)  # parent is dead now
    assert parent.poll() is not None

    # Child should be alive in the same group.
    try:
        os.kill(child_pid, 0)
    except ProcessLookupError:
        pytest.fail("Test setup error: child died too early")

    # PID-based recovery would fail (parent's PID is dead), but pgid fallback works.
    sent = kill_stale_group(parent.pid, starttime, pgid=pgid, grace_seconds=1.0)
    assert sent is True

    # Confirm child is actually dead.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            return  # PASS
        time.sleep(0.05)
    pytest.fail(f"Descendant child {child_pid} survived pgid-based kill")


@pytest.mark.parametrize(
    "content",
    [
        None,
        "{",
        '{"schema_version": 1, "attempt_id": "attempt"}',
        (
            '{"schema_version": 1, "attempt_id": "attempt", "pid": 12345, '
            '"pid": 54321, "pgid": 12345, "proc_starttime": 111, '
            '"boot_id": "boot"}'
        ),
    ],
)
def test_read_stale_process_identity_rejects_malformed_or_partial_records(
    tmp_path: Path,
    content: str | None,
) -> None:
    if content is not None:
        (tmp_path / PROCESS_IDENTITY_FILE).write_text(content)

    with pytest.raises((OSError, ValueError)):
        read_stale_process_identity(tmp_path, expected_attempt_id="attempt")


def test_read_stale_process_identity_rejects_wrong_attempt(tmp_path: Path) -> None:
    write_trial_identity(
        tmp_path,
        attempt_id="first-attempt",
        pid=12345,
        pgid=12345,
        starttime=111,
    )

    with pytest.raises(ValueError, match="another attempt"):
        read_stale_process_identity(tmp_path, expected_attempt_id="second-attempt")


def test_cleanup_stale_trial_process_accepts_prior_boot_without_signalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = StaleProcessIdentity(
        schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
        attempt_id="old-boot-attempt",
        pid=12345,
        pgid=12345,
        proc_starttime=111,
        boot_id="old-boot",
    )
    monkeypatch.setattr("phasesweep.runtime.reaper.read_boot_id", lambda: "current-boot")
    monkeypatch.setattr(
        "phasesweep.runtime.reaper.kill_stale_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a prior-boot identity must never signal current processes")
        ),
    )

    assert cleanup_stale_trial_process(identity) is True


def test_cleanup_stale_trial_process_refuses_unverifiable_platform_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = StaleProcessIdentity(
        schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
        attempt_id="no-proc-attempt",
        pid=12345,
        pgid=12345,
        proc_starttime=None,
        boot_id=None,
    )
    monkeypatch.setattr("phasesweep.runtime.reaper.read_boot_id", lambda: None)

    assert cleanup_stale_trial_process(identity) is False


_MISSING = object()


def test_kill_stale_group_refuses_live_pid_without_starttime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr("phasesweep.runtime.reaper.is_pid_alive", lambda _pid: True)
    monkeypatch.setattr("phasesweep.runtime.reaper._process_group_exists", lambda _pgid: True)
    monkeypatch.setattr(
        "phasesweep.runtime.reaper._terminate_process_group",
        lambda pgid, *, grace_seconds: calls.append(pgid) or True,
    )

    assert kill_stale_group(pid=12345, saved_starttime=None, pgid=12345) is False
    assert calls == []


@dataclass(frozen=True)
class _KillStaleGroupCase:
    pid: int | None
    saved_starttime: int | None
    pgid: int | None
    pid_alive: bool | object = _MISSING
    proc_starttime: int | None | object = _MISSING
    group_exists: bool | object = _MISSING
    proc_stat: object = _MISSING
    group_alive: bool | object = _MISSING
    derived_pgid: int | object = _MISSING
    expected: bool = False
    expected_calls: tuple[int, ...] = ()


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _KillStaleGroupCase(
                pid=12345,
                saved_starttime=111,
                pgid=None,
                pid_alive=True,
                proc_starttime=999,
            ),
            id="refuse-pid-reuse-without-pgid",
        ),
        pytest.param(
            _KillStaleGroupCase(
                pid=12345,
                saved_starttime=111,
                pgid=12345,
                pid_alive=True,
                proc_starttime=None,
                derived_pgid=7777,
            ),
            id="refuse-unreadable-live-pid-identity",
        ),
        pytest.param(
            _KillStaleGroupCase(
                pid=None,
                saved_starttime=111,
                pgid=12345,
                pid_alive=True,
                group_exists=True,
                proc_stat=None,
                group_alive=True,
            ),
            id="refuse-unreadable-live-pgid-leader",
        ),
        pytest.param(
            _KillStaleGroupCase(
                pid=12345,
                saved_starttime=111,
                pgid=12345,
                pid_alive=False,
                group_exists=True,
                proc_stat=SimpleNamespace(state="S", pgrp=12345, starttime=999),
            ),
            id="refuse-reused-pgid-leader",
        ),
        pytest.param(
            _KillStaleGroupCase(
                pid=12345,
                saved_starttime=111,
                pgid=12345,
                pid_alive=True,
                proc_starttime=999,
                group_exists=False,
                expected=True,
            ),
            id="accept-reused-pgid-when-group-gone",
        ),
        pytest.param(
            _KillStaleGroupCase(
                pid=None,
                saved_starttime=111,
                pgid=12345,
                group_exists=False,
                proc_stat=SimpleNamespace(state="S", pgrp=7777, starttime=999),
                expected=True,
            ),
            id="accept-stored-pgid-when-group-gone",
        ),
        pytest.param(
            _KillStaleGroupCase(
                pid=12345,
                saved_starttime=111,
                pgid=12345,
                pid_alive=True,
                group_exists=True,
                proc_stat=SimpleNamespace(state="S", pgrp=7777, starttime=999),
                group_alive=True,
                expected=True,
                expected_calls=(12345,),
            ),
            id="use-pgid-when-reused-pid-is-outside-group",
        ),
        pytest.param(
            _KillStaleGroupCase(
                pid=99999,
                saved_starttime=111,
                pgid=42,
                pid_alive=False,
                group_exists=True,
                proc_stat=None,
                group_alive=True,
                expected=True,
                expected_calls=(42,),
            ),
            id="use-stored-pgid-when-pid-dead",
        ),
        pytest.param(
            _KillStaleGroupCase(
                pid=12345,
                saved_starttime=111,
                pgid=12345,
                pid_alive=True,
                proc_starttime=111,
                group_alive=True,
                derived_pgid=7777,
                expected=True,
                expected_calls=(7777,),
            ),
            id="use-live-pid-when-starttime-matches",
        ),
    ],
)
def test_kill_stale_group_pid_pgid_decision_matrix(
    monkeypatch: pytest.MonkeyPatch,
    case: _KillStaleGroupCase,
) -> None:
    """Exercise every PID/PGID identity branch and its signal decision."""
    calls: list[int] = []
    if case.pid_alive is not _MISSING:
        monkeypatch.setattr(
            "phasesweep.runtime.reaper.is_pid_alive",
            lambda _pid, value=case.pid_alive: value,
        )
    if case.proc_starttime is not _MISSING:
        monkeypatch.setattr(
            "phasesweep.runtime.reaper.read_proc_starttime",
            lambda _pid, value=case.proc_starttime: value,
        )
    if case.group_exists is not _MISSING:
        monkeypatch.setattr(
            "phasesweep.runtime.reaper._process_group_exists",
            lambda _pgid, value=case.group_exists: value,
        )
    if case.proc_stat is not _MISSING:
        monkeypatch.setattr(
            "phasesweep.runtime.reaper._read_proc_stat",
            lambda _entry, value=case.proc_stat: value,
        )
    if case.group_alive is not _MISSING:
        monkeypatch.setattr(
            "phasesweep.runtime.reaper._process_group_alive",
            lambda _pgid, value=case.group_alive: value,
        )
    if case.derived_pgid is not _MISSING:
        monkeypatch.setattr("os.getpgid", lambda _pid, value=case.derived_pgid: value)
    monkeypatch.setattr(
        "phasesweep.runtime.reaper._terminate_process_group",
        lambda pgid, *, grace_seconds: calls.append(pgid) or True,
    )

    confirmed = kill_stale_group(
        pid=case.pid,
        saved_starttime=case.saved_starttime,
        pgid=case.pgid,
    )

    assert confirmed is case.expected
    assert calls == list(case.expected_calls)


@pytest.mark.integration
def test_allocation_context_recovers_repeated_optuna_allocation_interruptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allocations interrupted before the objective retain environment provenance."""

    class PowerLoss(BaseException):
        pass

    trainer = write_trainer(tmp_path, "print('x=1.0')")
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'allocation.journal'}",
        trial_command=f"{sys.executable} {trainer} {{overrides}}",
        override_format="argparse",
        n_trials=3,
        sampler=Sampler(type="random", seed=1),
    )

    def allocate_then_die(self: optuna.Study, _objective: object, **_kwargs: object) -> None:
        self.ask()
        raise PowerLoss("after allocation, before objective metadata")

    for _ in range(2):
        with monkeypatch.context() as patch:
            patch.setattr(optuna.Study, "optimize", allocate_then_die)
            with pytest.raises(PowerLoss, match="after allocation"):
                run_experiment(exp)

    winners = run_experiment(exp)

    assert winners["p"].metric == pytest.approx(1.0)
    study = optuna.load_study(
        study_name="t::p", storage=engine_ledger._resolve_storage(exp.storage)
    )
    trials = study.get_trials(deepcopy=False)
    assert [trial.state for trial in trials] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]
    identity = _environment_identity(exp)
    assert all(trial.user_attrs[TRAINER_ENV_DIGEST_ATTR] == identity.digest for trial in trials)


@pytest.mark.integration
def test_allocation_context_covers_first_environment_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed first environment write does not poison abort recovery."""

    trainer = write_trainer(tmp_path, "print('x=1.0')")
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'first-write.journal'}",
        trial_command=f"{sys.executable} {trainer} {{overrides}}",
        override_format="argparse",
        n_trials=1,
        sampler=Sampler(type="random", seed=1),
    )

    real_set_user_attr = optuna.Trial.set_user_attr
    failed = False

    def fail_first_environment_write(self: optuna.Trial, key: str, value: object) -> None:
        nonlocal failed
        if key == TRAINER_ENV_DIGEST_ATTR and not failed:
            failed = True
            raise OSError("first environment write failed")
        real_set_user_attr(self, key, value)

    monkeypatch.setattr(optuna.Trial, "set_user_attr", fail_first_environment_write)

    with pytest.raises(OSError, match="first environment write failed"):
        run_experiment(exp)

    resumed = exp.model_copy(update={"phases": [exp.phases[0].model_copy(update={"n_trials": 2})]})
    winners = run_experiment(resumed)

    assert winners["p"].metric == pytest.approx(1.0)
    study = optuna.load_study(
        study_name="t::p", storage=engine_ledger._resolve_storage(exp.storage)
    )
    trials = study.get_trials(deepcopy=False)
    assert [trial.state for trial in trials] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert TRAINER_ENV_DIGEST_ATTR not in trials[0].user_attrs
    assert run_experiment(resumed)["p"].metric == pytest.approx(1.0)


@pytest.mark.parametrize("bad_value", ["", 123])
def test_reaper_raises_for_malformed_trial_dir_attr(
    tmp_path: Path,
    bad_value: object,
) -> None:
    """Malformed persisted trial dirs are storage corruption, not prelaunch recovery."""
    exp = make_experiment(workdir=tmp_path / "runs")
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial.set_user_attr(TRIAL_DIR_ATTR, bad_value)

    with pytest.raises(RuntimeError, match="invalid persisted"):
        _reap_stale_trials(study, exp, exp.phases[0].name)

    assert study.trials[trial.number].state == optuna.trial.TrialState.RUNNING


def test_kill_stale_group_returns_true_when_no_identity() -> None:
    """No PID, no PGID — nothing alive to clean up. Safe to mark FAIL."""
    assert kill_stale_group(pid=None, saved_starttime=None, pgid=None) is True


@pytest.mark.parametrize(
    ("failure_site", "match"),
    [
        pytest.param("outcome", "durable failure outcome could not be recorded", id="outcome"),
        pytest.param("ledger", "cleanup recovery ledger could not be updated", id="ledger"),
        pytest.param("tell", "Optuna state could not be updated", id="tell"),
    ],
)
def test_reaper_reports_storage_failures_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
    match: str,
) -> None:
    """Every failed durable recovery write is an operator-facing storage outage."""
    exp = make_experiment(workdir=tmp_path / "runs")
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial.set_user_attr(ATTEMPT_ID_ATTR, "tell-failure-attempt")
    trial.set_user_attr(TRIAL_DIR_ATTR, str(tmp_path / "runs" / "t" / "p" / "trial_00000"))

    def fake_identity(*_args: object, **_kwargs: object) -> StaleProcessIdentity:
        return StaleProcessIdentity(
            schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
            attempt_id="tell-failure-attempt",
            pid=99999,
            pgid=99999,
            proc_starttime=12345,
            boot_id="test-boot",
        )

    monkeypatch.setattr("phasesweep.engine.attempts._read_trial_process_identity", fake_identity)
    monkeypatch.setattr("phasesweep.engine.cleanup._read_trial_process_identity", fake_identity)
    monkeypatch.setattr("phasesweep.engine.attempts.cleanup_stale_trial_process", lambda _: True)
    monkeypatch.setattr("phasesweep.engine.cleanup.cleanup_stale_trial_process", lambda _: True)

    if failure_site == "outcome":
        patch_rejected_trial_user_attr(
            monkeypatch,
            TRIAL_OUTCOME_ATTR,
            "storage write failed",
        )
    elif failure_site == "ledger":
        monkeypatch.setattr(
            study,
            "set_user_attr",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("storage write failed")),
        )
    else:
        monkeypatch.setattr(
            study,
            "tell",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("storage write failed")),
        )

    with pytest.raises(StudyStorageUnavailableError, match=match):
        _reap_stale_trials(study, exp, exp.phases[0].name)

    # The trial must NOT have been marked FAIL (because tell raised).
    assert study.trials[trial.number].state == optuna.trial.TrialState.RUNNING


@pytest.mark.integration
def test_prelaunch_allocated_attempt_recovers_without_identity(tmp_path: Path) -> None:
    """A worker killed while queued for a GPU leaves 'allocated' and no identity.

    Recovery used to treat the missing process identity as unverifiable
    cleanup and fail closed forever (review v0.5.17 / blocker 2 gap A). The
    durable 'allocated' marker proves no process was ever created, so the
    stale trial is failed safely and the study unwedges.
    """
    trainer = write_trainer(tmp_path, "print('x=1.0')")
    exp = make_experiment(
        experiment="prelaunch",
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'p.journal'}",
        trial_command=f"{sys.executable} {trainer} {{overrides}}",
        override_format="argparse",
        n_trials=2,
        sampler=Sampler(type="random", seed=1),
    )
    study, trial_dir, stale_number = fabricate_stale_trial(exp, "p", attempt_id="queued-attempt")
    study.set_user_attr(PHASE_FINGERPRINT_ATTR, _phase_fingerprint(exp, exp.phases[0], {}))
    write_attempt_lifecycle(trial_dir, attempt_id="queued-attempt", state="allocated")

    winners = run_experiment(exp)

    assert "p" in winners
    states = {t.number: t.state for t in study.get_trials(deepcopy=False)}
    assert states[stale_number] == optuna.trial.TrialState.FAIL
    assert optuna.trial.TrialState.COMPLETE in states.values()


@pytest.mark.integration
def test_exited_attempt_recovers_without_signalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between process exit and the Optuna commit recovers via 'exited'.

    The retained identity plus the durable 'exited' transition prove the group
    is gone; recovery must fail the trial without sending any signal (review
    v0.5.17 / blocker 2 gap B).
    """
    exp = make_experiment(
        experiment="exited",
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'e.journal'}",
    )
    study, trial_dir, stale_number = fabricate_stale_trial(exp, "p", attempt_id="exited-attempt")
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        starttime = read_proc_starttime(child.pid)
        pgid = os.getpgid(child.pid)
    finally:
        child.kill()
        child.wait(timeout=5)
    write_trial_identity(
        trial_dir,
        attempt_id="exited-attempt",
        pid=child.pid,
        pgid=pgid,
        starttime=starttime,
    )
    write_attempt_lifecycle(
        trial_dir,
        attempt_id="exited-attempt",
        state="exited",
        return_code=0,
        cleanup_confirmed=True,
    )

    def _no_signal(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("an 'exited' attempt must never be signalled")

    monkeypatch.setattr("phasesweep.engine.attempts.cleanup_stale_trial_process", _no_signal)
    monkeypatch.setattr("phasesweep.engine.cleanup.cleanup_stale_trial_process", _no_signal)

    assert _reap_stale_trials(study, exp, "p") == 1
    assert study.get_trials(deepcopy=False)[stale_number].state == optuna.trial.TrialState.FAIL


def test_recovery_refuses_a_study_bound_to_another_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovering one workdir never reaps trials another workdir's run owns.

    Workdir B's run bound ``t::p`` to root B and left a reapable RUNNING trial. A run
    from workdir A on the same ledger was refused at claim, so tree A is still unbound
    and its format checks pass. Recovering that run opens ``t::p`` live; without the
    per-study ownership check ``claim_ledger`` applies, it would reap B's trial. It must
    refuse in the engine's own words instead, leaving the shared ledger untouched.
    """
    ledger_dir = tmp_path / "shared-ledger"
    ledger_dir.mkdir()
    owner = _write_config(
        tmp_path / "b.yaml", backend="journal", ledger_dir=ledger_dir, workdir=tmp_path / "b"
    )
    config_a = tmp_path / "a.yaml"
    recovering = _write_config(
        config_a, backend="journal", ledger_dir=ledger_dir, workdir=tmp_path / "a"
    )
    study, trial_dir, owned_number = fabricate_stale_trial(owner, "p", attempt_id="b-attempt")
    write_attempt_lifecycle(trial_dir, attempt_id="b-attempt", state="allocated")
    ledger_file = ledger_dir / "study.journal"
    before = ledger_file.read_bytes()

    state_dir = tmp_path / "mcp-state"
    run_id = "a-refused-run"
    stage_dead_run(
        RunStore(state_dir), run_id, config_a, recovering.experiment, cleanup_uncertain=True
    )
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", lambda *_a, **_k: True)

    with pytest.raises(RunRecoveryError) as excinfo:
        recover_run(state_dir, run_id, confirm=True, emit=lambda _message: None)

    states = {trial.number: trial.state for trial in study.get_trials(deepcopy=False)}
    assert states[owned_number] == optuna.trial.TrialState.RUNNING
    assert ledger_file.read_bytes() == before
    assert isinstance(excinfo.value.__cause__, ArtifactRootConflictError)
    assert str(excinfo.value) == str(excinfo.value.__cause__)
    assert f"publishes into artifact root {str(_experiment_dir(owner))!r}" in str(excinfo.value)


def _fabricate_registered_journal_attempt(
    tmp_path: Path,
    *,
    attempt_id: str,
) -> tuple[Experiment, Path, Path]:
    """Create an allocated registry entry backed by a journal study."""
    ledger = tmp_path / "attempts.journal"
    experiment = make_experiment(
        experiment="journal-attempt",
        workdir=tmp_path / "runs",
        storage=f"journal:///{ledger}",
    )
    study = optuna.create_study(
        study_name="journal-attempt::p",
        storage=engine_ledger._resolve_storage(experiment.resolved_storage),
        direction="minimize",
    )
    mark_current_format(experiment, study)
    trial = study.ask()
    trial_dir = _trial_dir_for(
        experiment,
        "p",
        trial.number,
        generation_id="old-generation",
        attempt_id=attempt_id,
    )
    trial_dir.mkdir(parents=True)
    _register_active_attempt(
        experiment,
        attempt_id=attempt_id,
        phase_name="p",
        study_name=study.study_name,
        trial_number=trial.number,
        trial_dir=trial_dir,
        generation_id="old-generation",
    )
    write_attempt_lifecycle(trial_dir, attempt_id=attempt_id, state="allocated")
    return experiment, ledger, _attempts_dir(experiment) / f"{attempt_id}.json"


@pytest.mark.parametrize("tail", [b"{}\n", b"not-json\n{}\n"])
def test_registry_retains_attempt_when_journal_snapshot_is_unreadable(
    tmp_path: Path, tail: bytes
) -> None:
    """A journal that stays unreadable cannot discard a stale-attempt recovery record."""
    experiment, ledger, entry_path = _fabricate_registered_journal_attempt(
        tmp_path, attempt_id="journal-attempt"
    )
    ledger.write_bytes(ledger.read_bytes() + tail)
    ledger_before = ledger.read_bytes()
    entry_before = entry_path.read_bytes()

    _preflight_active_attempts(experiment, _PreflightCleanupReport())

    assert ledger.read_bytes() == ledger_before
    assert entry_path.read_bytes() == entry_before


@pytest.mark.parametrize("tail", [b"not-json\n", b"not-json", b'{"unterminated"'])
def test_registry_reads_its_attempt_after_repairing_a_partial_journal_record(
    tmp_path: Path, tail: bytes
) -> None:
    """A partial final record is cut first, so the attempt is read as on a healthy journal."""
    (tmp_path / "healthy").mkdir()
    (tmp_path / "torn").mkdir()
    healthy_experiment, healthy_ledger, healthy_entry = _fabricate_registered_journal_attempt(
        tmp_path / "healthy", attempt_id="journal-attempt"
    )
    _preflight_active_attempts(healthy_experiment, _PreflightCleanupReport())
    experiment, ledger, entry_path = _fabricate_registered_journal_attempt(
        tmp_path / "torn", attempt_id="journal-attempt"
    )
    complete = ledger.read_bytes()
    ledger.write_bytes(complete + tail)

    _preflight_active_attempts(experiment, _PreflightCleanupReport())

    (backup,) = ledger.parent.glob(f"{ledger.name}.*.bak")
    assert backup.read_bytes() == complete + tail
    assert ledger.read_bytes().startswith(complete)
    assert not list(healthy_ledger.parent.glob(f"{healthy_ledger.name}.*.bak"))
    assert not entry_path.exists()
    assert not healthy_entry.exists()

    def trial_states(experiment: Experiment) -> list[optuna.trial.TrialState]:
        storage = engine_ledger._resolve_storage(experiment.resolved_storage)
        study = optuna.load_study(study_name="journal-attempt::p", storage=storage)
        return [trial.state for trial in study.get_trials(deepcopy=False)]

    assert trial_states(experiment) == trial_states(healthy_experiment)


def test_registry_retains_attempt_when_live_journal_loses_snapshotted_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later live replay cannot erase snapshot-proven recovery evidence."""
    experiment, ledger, entry_path = _fabricate_registered_journal_attempt(
        tmp_path, attempt_id="changed-journal-attempt"
    )
    create_study_only = ledger.read_bytes().splitlines(keepends=True)[0]
    entry_before = entry_path.read_bytes()
    real_snapshot = engine_ledger._load_journal_study_snapshot

    def snapshot_then_truncate(storage_url: str, study_name: str):
        snapshot = real_snapshot(storage_url, study_name)
        ledger.write_bytes(create_study_only)
        return snapshot

    monkeypatch.setattr(engine_ledger, "_load_journal_study_snapshot", snapshot_then_truncate)

    _preflight_active_attempts(experiment, _PreflightCleanupReport())

    assert ledger.read_bytes() == create_study_only
    assert entry_path.read_bytes() == entry_before


def test_registry_discards_attempt_for_a_confirmed_missing_journal_study(tmp_path: Path) -> None:
    """A valid journal can still prove that a named stale study is gone."""
    experiment, _ledger, entry_path = _fabricate_registered_journal_attempt(
        tmp_path, attempt_id="missing-journal-attempt"
    )
    storage = engine_ledger._resolve_storage(experiment.resolved_storage)
    optuna.delete_study(study_name="journal-attempt::p", storage=storage)

    _preflight_active_attempts(experiment, _PreflightCleanupReport())

    assert not entry_path.exists()


def test_registry_discards_attempt_whose_journal_ledger_is_gone_without_recreating_it(
    tmp_path: Path,
) -> None:
    """A deleted journal locator is confirmed absent, and inspecting it creates nothing.

    ``open_registry_study`` documents this directly: Optuna's own loader would
    create the missing file before reporting the study absent from it, leaving
    an empty ledger at a locator recovery was only inspecting; a missing file
    already proves the study is gone, and the journal snapshot reader never
    constructs anything that could bring it into existence.
    """

    def _exp(db_name: str) -> Experiment:
        return make_experiment(
            experiment="gone-ledger",
            workdir=tmp_path / "runs",
            storage=f"journal:///{tmp_path / db_name}",
            n_trials=1,
        )

    gone = tmp_path / "gone.journal"
    _study, trial_dir, _number = fabricate_registered_attempt(
        _exp("gone.journal"), "p", attempt_id="gone-attempt"
    )
    write_attempt_lifecycle(trial_dir, attempt_id="gone-attempt", state="allocated")
    entry_path = _attempts_dir(_exp("gone.journal")) / "gone-attempt.json"
    gone.unlink()

    report = _PreflightCleanupReport()
    _preflight_active_attempts(_exp("current.journal"), report)

    assert not entry_path.exists()
    assert not gone.exists()
    assert report.cleanup_confirmed is True
    assert report.recovered_attempt_ids == set()


def test_registry_refuses_a_foreign_ledger_holding_pre_cutover_state(tmp_path: Path) -> None:
    """A registry entry cannot reap through a ledger this release refuses.

    This release wrote the entry, so the ledger it names was current-format
    when the attempt registered. Pre-cutover state there now means the locator
    no longer names that ledger, so the run stops with the entry and the
    foreign ledger's bytes both intact instead of marking a trial FAIL in it.
    """

    def _exp(db_name: str) -> Experiment:
        return make_experiment(
            experiment="foreign-ledger",
            workdir=tmp_path / "runs",
            storage=f"journal:///{tmp_path / db_name}",
            n_trials=1,
        )

    current = _exp("current.journal")
    foreign_db = tmp_path / "foreign.journal"
    foreign_url = f"journal:///{foreign_db}"
    foreign_storage = engine_ledger._resolve_storage(foreign_url)
    # The tree belongs to the configured ledger, so the run passes its own
    # validation and claim and reaches the registry scan.
    mark_current_format(current)
    stale = optuna.create_study(
        study_name="foreign-ledger::p", storage=foreign_storage, direction="minimize"
    )
    stale.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    stale_trial = stale.ask()
    legacy = optuna.create_study(study_name="legacy::p", storage=foreign_storage)
    legacy.add_trial(optuna.trial.create_trial(value=0.5, state=optuna.trial.TrialState.COMPLETE))
    trial_dir = tmp_path / "foreign-attempt"
    trial_dir.mkdir()
    write_attempt_lifecycle(trial_dir, attempt_id="foreign-attempt", state="allocated")
    # Registered through the foreign config, so the entry records that ledger.
    _register_active_attempt(
        _exp("foreign.journal"),
        attempt_id="foreign-attempt",
        phase_name="p",
        study_name="foreign-ledger::p",
        trial_number=stale_trial.number,
        trial_dir=trial_dir,
        generation_id="old-generation",
    )
    entry_path = _attempts_dir(current) / "foreign-attempt.json"
    entry_before = entry_path.read_bytes()
    ledger_before = foreign_db.read_bytes()

    with pytest.raises(StudySchemaMismatchError, match=r"pre-cutover.*'legacy::p'"):
        run_experiment(current)

    assert entry_path.read_bytes() == entry_before
    assert foreign_db.read_bytes() == ledger_before


@pytest.mark.parametrize(
    "identity_change", [None, "attempt", "generation", "missing-attempt", "missing-generation"]
)
def test_registry_terminal_cleanup_requires_matching_attempt_identity(
    tmp_path: Path, identity_change: str | None
) -> None:
    """A stale registry entry cannot consume a replacement trial's cleanup evidence."""
    from phasesweep.engine.cleanup import _iter_cleanup_uncertain_trials

    experiment = make_experiment(
        experiment="terminal-identity",
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'ledger.journal'}",
    )
    old_dir = tmp_path / "old-attempt"
    old_dir.mkdir()
    write_attempt_lifecycle(old_dir, attempt_id="old-attempt", state="allocated")
    _register_active_attempt(
        experiment,
        attempt_id="old-attempt",
        phase_name="p",
        study_name="terminal-identity::p",
        trial_number=0,
        trial_dir=old_dir,
        generation_id="old-generation",
    )
    # A restored divergent ledger can reuse the number without preserving
    # the registry entry's attempt. Recovery must inspect that trial itself.
    storage = engine_ledger._resolve_storage(experiment.resolved_storage)
    study = optuna.create_study(study_name="terminal-identity::p", storage=storage)
    study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    trial = study.ask()
    new_dir = tmp_path / "ledger-attempt"
    new_dir.mkdir()
    if identity_change != "missing-attempt":
        trial.set_user_attr(
            ATTEMPT_ID_ATTR, "new-attempt" if identity_change == "attempt" else "old-attempt"
        )
    if identity_change != "missing-generation":
        trial.set_user_attr(
            GENERATION_ID_ATTR,
            "new-generation" if identity_change == "generation" else "old-generation",
        )
    trial.set_user_attr(TRIAL_DIR_ATTR, str(new_dir))
    trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(trial, state=optuna.trial.TrialState.FAIL)

    # Registry recovery must keep working when this phase is no longer
    # declared; there may be no later per-phase scan to catch missing identity.
    experiment = experiment.model_copy(
        update={"phases": [experiment.phases[0].model_copy(update={"name": "renamed"})]}
    )
    report = _PreflightCleanupReport()
    if identity_change == "generation":
        with pytest.raises(StudySchemaMismatchError, match="conflicting recovery identity"):
            _preflight_active_attempts(experiment, report)
    elif identity_change in {"missing-attempt", "missing-generation"}:
        with pytest.raises(ProcessCleanupUncertainError, match="identity is missing") as exc_info:
            _preflight_active_attempts(experiment, report)
        assert "Restore the original storage ledger" in str(exc_info.value)
        assert "before retrying recovery" in str(exc_info.value)
        assert report.uncertain_attempt_ids == {"old-attempt"}
    else:
        _preflight_active_attempts(experiment, report)

    if identity_change is None:
        assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [0]
        assert report.recovered_attempt_ids == {"old-attempt"}
        assert list(_iter_cleanup_uncertain_trials(study)) == []
    else:
        assert CLEANUP_RECOVERED_TRIALS_ATTR not in study.user_attrs
        assert report.recovered_attempt_ids == set()
        with pytest.raises(ProcessCleanupUncertainError) as exc_info:
            list(_iter_cleanup_uncertain_trials(study))
        assert "Restore the original storage ledger" in str(exc_info.value)
    assert bool(list(_attempts_dir(experiment).glob("*.json"))) == (
        identity_change in {"generation", "missing-attempt", "missing-generation"}
    )


@pytest.mark.parametrize(
    "mutation",
    ["directory_mode", "file_mode", "directory_symlink", "entry_symlink"],
)
def test_attempt_registry_refuses_unsafe_private_authority(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Recovery never follows or trusts a registry outside its private namespace."""
    experiment = make_experiment(
        experiment="unsafe-registry",
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'study.journal'}",
    )
    trial_dir = _experiment_dir(experiment) / "p" / "trial_00000__unsafe"
    trial_dir.mkdir(parents=True)
    _register_active_attempt(
        experiment,
        attempt_id="unsafe-attempt",
        phase_name="p",
        study_name="unsafe-registry::p",
        trial_number=0,
        trial_dir=trial_dir,
        generation_id="unsafe-generation",
    )
    attempts_dir = _attempts_dir(experiment)
    entry_path = attempts_dir / "unsafe-attempt.json"

    if mutation == "directory_mode":
        attempts_dir.chmod(0o755)
    elif mutation == "file_mode":
        entry_path.chmod(0o644)
    elif mutation == "directory_symlink":
        redirected = attempts_dir.with_name("redirected-attempts")
        attempts_dir.rename(redirected)
        attempts_dir.symlink_to(redirected, target_is_directory=True)
    else:
        target = attempts_dir.parent / "redirected-attempt.json"
        entry_path.rename(target)
        entry_path.symlink_to(target)

    with pytest.raises(ProcessCleanupUncertainError, match="registry|entry"):
        _inspect_active_attempts(experiment)


def test_relative_registry_storage_recovers_from_the_registration_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing cwd cannot redirect stale-trial repair into a new database."""
    registration_cwd = tmp_path / "registration"
    recovery_cwd = tmp_path / "recovery"
    registration_cwd.mkdir()
    recovery_cwd.mkdir()
    monkeypatch.chdir(registration_cwd)
    experiment = make_experiment(
        experiment="relative-registry",
        workdir=tmp_path / "runs",
        storage="journal:///studies.journal",
        n_trials=1,
    )
    _study, trial_dir, stale_number = fabricate_registered_attempt(
        experiment,
        "p",
        attempt_id="relative-attempt",
    )
    write_attempt_lifecycle(trial_dir, attempt_id="relative-attempt", state="allocated")

    monkeypatch.chdir(recovery_cwd)
    report = _PreflightCleanupReport()
    _preflight_active_attempts(experiment, report)

    # The journal backend reopens its file on every read, so a handle built on
    # the relative URL would now follow the new cwd; read the original ledger
    # through its absolute path instead.
    registered_ledger = f"journal:///{registration_cwd / 'studies.journal'}"
    study = optuna.load_study(
        study_name="relative-registry::p",
        storage=engine_ledger._resolve_storage(registered_ledger),
    )
    assert study.get_trials(deepcopy=False)[stale_number].state == optuna.trial.TrialState.FAIL
    assert report.recovered_attempt_ids == {"relative-attempt"}
    assert not (recovery_cwd / "studies.journal").exists()
    assert not list(_attempts_dir(experiment).glob("*.json"))


@pytest.mark.integration
def test_registry_repairs_partial_allocation_before_attempt_attr(tmp_path: Path) -> None:
    """A registry entry survives an Optuna failure before the first trial attr."""
    trainer = write_trainer(tmp_path, "print('x=1.0')")
    experiment = make_experiment(
        experiment="partial-registration",
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'partial.journal'}",
        trial_command=f"{sys.executable} {trainer} {{overrides}}",
        override_format="argparse",
        n_trials=2,
        sampler=Sampler(type="random", seed=1),
    )
    study, trial_dir, stale_number = fabricate_registered_attempt(
        experiment,
        "p",
        attempt_id="partial-attempt",
        persist_trial_attempt_id=False,
        persist_trial_generation_id=False,
    )
    study.set_user_attr(
        PHASE_FINGERPRINT_ATTR, _phase_fingerprint(experiment, experiment.phases[0], {})
    )
    write_attempt_lifecycle(trial_dir, attempt_id="partial-attempt", state="allocated")

    winners = run_experiment(experiment)

    assert "p" in winners
    trials = {trial.number: trial for trial in study.get_trials(deepcopy=False)}
    states = {number: trial.state for number, trial in trials.items()}
    assert states[stale_number] == optuna.trial.TrialState.FAIL
    assert optuna.trial.TrialState.COMPLETE in states.values()
    assert trials[stale_number].user_attrs[ATTEMPT_ID_ATTR] == "partial-attempt"
    assert trials[stale_number].user_attrs[GENERATION_ID_ATTR] == "old-generation"
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [stale_number]
    assert not list(_attempts_dir(experiment).glob("*.json"))


def test_in_memory_attempt_registry_accepts_locatorless_entries(tmp_path: Path) -> None:
    """In-memory attempts have an identity but intentionally no recovery URL."""
    experiment = make_experiment(
        experiment="memory-registry",
        workdir=tmp_path / "runs",
        storage=None,
    )
    trial_dir = _experiment_dir(experiment) / "p" / "trial_00000__memory-attempt"
    trial_dir.mkdir(parents=True)
    write_attempt_lifecycle(trial_dir, attempt_id="memory-attempt", state="allocated")
    _register_active_attempt(
        experiment,
        attempt_id="memory-attempt",
        phase_name="p",
        study_name="memory-registry::p",
        trial_number=0,
        trial_dir=trial_dir,
        generation_id="memory-generation",
    )

    _preflight_active_attempts(experiment, _PreflightCleanupReport())

    assert not list(_attempts_dir(experiment).glob("*.json"))


def test_registry_generation_conflict_is_study_schema_mismatch(tmp_path: Path) -> None:
    """Conflicting durable recovery identities are operator-visible corruption."""
    experiment = make_experiment(
        experiment="registry-conflict",
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'conflict.journal'}",
    )
    _study, trial_dir, _number = fabricate_registered_attempt(
        experiment,
        "p",
        attempt_id="conflicting-attempt",
    )
    write_attempt_lifecycle(trial_dir, attempt_id="conflicting-attempt", state="allocated")
    entry_path = _attempts_dir(experiment) / "conflicting-attempt.json"
    entry = json.loads(entry_path.read_text())
    entry["generation_id"] = "different-generation"
    entry_path.write_text(json.dumps(entry))

    with pytest.raises(StudySchemaMismatchError, match="conflicting recovery identity"):
        _preflight_active_attempts(experiment, _PreflightCleanupReport())


@pytest.mark.integration
def test_renamed_phase_cannot_hide_stale_trainer_from_recovery(tmp_path: Path) -> None:
    """The attempt registry finds stale work whose phase left the config.

    Reviewer repro (review v0.5.17 / blocker 3): kill the orchestrator while a
    trainer is alive, rename the phase in the YAML, run again. Recovery used to
    walk only the *current* phase list, so the old trainer stayed alive and
    overlapped the new sweep. The registry scan is phase-graph-independent.
    """
    trainer = write_trainer(tmp_path, "print('x=1.0')")
    storage = f"journal:///{tmp_path / 'r.journal'}"

    def _exp(phase_name: str) -> Experiment:
        return make_experiment(
            experiment="rename",
            workdir=tmp_path / "runs",
            storage=storage,
            trial_command=f"{sys.executable} {trainer} {{overrides}}",
            override_format="argparse",
            phases=[
                Phase(
                    name=phase_name,
                    n_trials=1,
                    sampler=Sampler(type="random", seed=0),
                    search_space={"x": IntParam(type="int", low=0, high=10)},
                )
            ],
        )

    old_exp = _exp("old_phase")
    old_study, trial_dir, stale_number = fabricate_registered_attempt(
        old_exp, "old_phase", attempt_id="renamed-attempt"
    )
    stale = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        starttime = read_proc_starttime(stale.pid)
        assert starttime is not None
        write_trial_identity(
            trial_dir,
            attempt_id="renamed-attempt",
            pid=stale.pid,
            pgid=os.getpgid(stale.pid),
            starttime=starttime,
        )

        winners = run_experiment(_exp("new_phase"))

        assert "new_phase" in winners
        assert stale.poll() is not None, "the renamed phase's trainer must be reaped"
        assert (
            old_study.get_trials(deepcopy=False)[stale_number].state == optuna.trial.TrialState.FAIL
        )
        assert not list((tmp_path / "runs" / "rename" / "attempts").glob("*.json"))
    finally:
        if stale.poll() is None:
            os.killpg(stale.pid, signal.SIGKILL)
        stale.wait(timeout=5)


@pytest.mark.integration
def test_wandb_worker_parent_death_is_recovered_after_phase_removal(
    tmp_path, monkeypatch, wandb_worker_sdk
):
    """The existing registry owns the polling worker after the trainer exits."""
    from phasesweep.runtime.reaper import read_stale_process_identity

    old = make_experiment(
        workdir=tmp_path / "runs", storage=f"journal:///{tmp_path / 'study.journal'}", n_trials=1
    )
    study, trial_dir, number = fabricate_registered_attempt(old, "p", attempt_id="remote-attempt")
    started = tmp_path / "worker-started"
    wandb_worker_sdk(f"""
        import time
        from pathlib import Path
        class Api:
            def __init__(self, **kwargs):
                Path({str(started)!r}).write_text("started")
                time.sleep(60)
    """)
    parent_code = f"""
from pathlib import Path
from phasesweep.evidence.wandb import poll_wandb_summary
poll_wandb_summary(base_url="https://example.test", entity="e", project="p",
    run_id="remote-attempt", trial_dir=Path({str(trial_dir)!r}),
    poll_seconds=0.01, timeout_seconds=120)
"""
    parent = subprocess.Popen([sys.executable, "-c", parent_code])
    identity = None
    try:
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            assert parent.poll() is None
            time.sleep(0.02)
        assert started.exists()
        identity = read_stale_process_identity(trial_dir, expected_attempt_id="remote-attempt")
        parent.kill()
        parent.wait(timeout=5)
        monkeypatch.setitem(sys.modules, "wandb", None)
        removed = old.model_copy(
            update={
                "phases": [
                    Phase(name="replacement", n_trials=1, sampler=Sampler(type="random", seed=0))
                ]
            }
        )
        report = _PreflightCleanupReport()
        _preflight_active_attempts(removed, report)
        assert report.cleanup_confirmed
        assert report.recovered_attempt_ids == {"remote-attempt"}
        assert study.get_trials()[number].state == optuna.trial.TrialState.FAIL
        assert not list(_attempts_dir(old).glob("*.json"))
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if identity is not None:
            cleanup_stale_trial_process(identity)


def test_storage_change_cannot_hide_stale_attempt_from_recovery(tmp_path: Path) -> None:
    """A registered attempt is repaired through its *recorded* storage URL.

    The current config points at a different storage; the registry entry keeps
    the producing study reachable so its RUNNING trial is still failed
    (review v0.5.17 / blocker 3).
    """
    trainer = write_trainer(tmp_path, "print('x=1.0')")

    def _exp(db_name: str) -> Experiment:
        return make_experiment(
            experiment="movedstorage",
            workdir=tmp_path / "runs",
            storage=f"journal:///{tmp_path / db_name}",
            trial_command=f"{sys.executable} {trainer} {{overrides}}",
            override_format="argparse",
            n_trials=1,
        )

    old_exp = _exp("old.journal")
    old_study, trial_dir, stale_number = fabricate_registered_attempt(
        old_exp, "p", attempt_id="moved-attempt"
    )
    write_attempt_lifecycle(trial_dir, attempt_id="moved-attempt", state="allocated")

    report = _PreflightCleanupReport()
    _preflight_active_attempts(_exp("new.journal"), report)

    assert old_study.get_trials(deepcopy=False)[stale_number].state == optuna.trial.TrialState.FAIL
    assert not list((tmp_path / "runs" / "movedstorage" / "attempts").glob("*.json"))
    with pytest.raises(ArtifactRootConflictError, match="different storage ledger"):
        run_experiment(_exp("new.journal"))


@pytest.mark.parametrize("failure_site", ["lifecycle", "registry"])
@pytest.mark.integration
def test_unpersistable_attempt_refuses_to_launch_its_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    """Lifecycle and registry writes are pre-launch durability requirements.

    Losing the registry makes a trainer undiscoverable after a phase rename;
    losing the allocated lifecycle marker makes a queued pre-launch crash
    indistinguishable from a torn process-identity write. Both failures must
    stop before the GPU lease and subprocess. The successful discovery path is
    covered by ``test_renamed_phase_cannot_hide_stale_trainer_from_recovery``.
    """
    launched = tmp_path / "trainer_ran"
    trainer = write_trainer(
        tmp_path,
        f"""
        import pathlib
        pathlib.Path({str(launched)!r}).touch()
        print("x=0.5")
        """,
    )
    storage = f"journal:///{tmp_path / 'registry.journal'}"

    def _exp(n_trials: int) -> Experiment:
        return make_experiment(
            experiment="unregistrable",
            workdir=tmp_path / "runs",
            storage=storage,
            trial_command=f"{sys.executable} {trainer} {{overrides}}",
            override_format="argparse",
            n_trials=n_trials,
        )

    attempts_dir = tmp_path / "runs" / "unregistrable" / "attempts"
    import phasesweep.engine.attempts as attempts_mod
    import phasesweep.engine.phase as phase_mod

    real_atomic_write_text = attempts_mod.private_atomic_write_text
    real_write_attempt_lifecycle = phase_mod.write_attempt_lifecycle
    if failure_site == "registry":

        def refuse_registry_writes(path: Path, text: str) -> None:
            if path.parent == attempts_dir:
                raise OSError(errno.EACCES, "injected registry write failure")
            real_atomic_write_text(path, text)

        monkeypatch.setattr(attempts_mod, "private_atomic_write_text", refuse_registry_writes)
    else:

        def refuse_lifecycle_write(*_args: object, **_kwargs: object) -> None:
            raise OSError(errno.EACCES, "injected lifecycle write failure")

        monkeypatch.setattr(phase_mod, "write_attempt_lifecycle", refuse_lifecycle_write)
    with pytest.raises(ActiveAttemptPersistenceError) as excinfo:
        run_experiment(_exp(2))

    message = str(excinfo.value)
    if failure_site == "registry":
        assert str(attempts_dir) in message
    else:
        assert "allocated lifecycle marker" in message
    assert "No trainer was started" in message
    # The refusal routes only the workdir repair; its message says when the
    # durable abort also needs a higher target, as the rerun below shows.
    assert "n_trials raised above its accepted target" in message
    # Nothing was launched: no marker, and no durable process identity.
    assert not launched.exists()
    phase_dir = tmp_path / "runs" / "unregistrable" / "p"
    assert not list(phase_dir.glob(f"trial_*/{PROCESS_IDENTITY_FILE}"))

    # The failure is a visible, valid terminal row - not an invisible RUNNING one.
    study = optuna.load_study(
        study_name="unregistrable::p", storage=engine_ledger._resolve_storage(storage)
    )
    (trial,) = study.get_trials(deepcopy=False)
    assert trial.state == optuna.trial.TrialState.FAIL
    assert trial.user_attrs[TRIAL_OUTCOME_ATTR]["outcome"] == "fatal"
    assert message in trial.user_attrs[TRIAL_OUTCOME_ATTR]["cause"]
    assert study.user_attrs[PHASE_ABORT_ATTR]["policy"] == "active_attempt_persistence"

    # Restoring writes does not erase the abort; a higher target explicitly recovers it.
    monkeypatch.setattr(attempts_mod, "private_atomic_write_text", real_atomic_write_text)
    monkeypatch.setattr(phase_mod, "write_attempt_lifecycle", real_write_attempt_lifecycle)
    with pytest.raises(NoFeasibleTrialError, match="Increase n_trials above 2"):
        run_experiment(_exp(2))
    assert not launched.exists()

    winners = run_experiment(_exp(3))

    assert winners["p"].metric == pytest.approx(0.5)
    assert launched.exists()
    assert not list(attempts_dir.glob("*.json"))
