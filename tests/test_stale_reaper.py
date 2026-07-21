"""Stale trial reaper: detect RUNNING Optuna trials whose orchestrator died, kill any leaked process group (PID-reuse-safe via starttime check), and mark the trial FAIL. Fail-closed throughout — never silently advance the study while a leaked process may be live."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from phasesweep.config import (
    Experiment,
    IntParam,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
)
from phasesweep.engine import run_experiment
from phasesweep.engine.guards import (
    _preflight_existing_studies,
    _PreflightCleanupReport,
    _reap_stale_trials,
)
from phasesweep.engine.phase import _run_phase
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GENERATION_ID_ATTR,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRIAL_DIR_ATTR,
    _generation_path,
    _trial_dir_for,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError
from phasesweep.runtime.process import (
    PROCESS_IDENTITY_FILE,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    PhaseSweepShutdown,
    ShutdownCleanupReport,
    StaleProcessIdentity,
    _read_proc_stat,
    _write_process_identity,
    cleanup_stale_trial_process,
    is_same_process,
    kill_stale_group,
    read_boot_id,
    read_proc_starttime,
    read_stale_process_identity,
)
from tests.conftest import make_experiment, write_trainer


def _write_test_process_identity(
    trial_dir: Path,
    *,
    attempt_id: str,
    pid: int,
    pgid: int,
    starttime: int | None,
    boot_id: str | None = None,
) -> None:
    _write_process_identity(
        trial_dir / PROCESS_IDENTITY_FILE,
        StaleProcessIdentity(
            schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
            attempt_id=attempt_id,
            pid=pid,
            pgid=pgid,
            proc_starttime=starttime,
            boot_id=read_boot_id() if boot_id is None else boot_id,
            launch_nonce="test-launch-nonce",
        ),
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


def test_is_same_process_rejects_dead_pid():
    """A definitely-dead PID should not be identified as the same process."""
    assert not is_same_process(999999999, saved_starttime=12345)


def test_is_same_process_with_matching_starttime():
    """Our own PID with our own starttime should match."""
    import os

    pid = os.getpid()
    st = read_proc_starttime(pid)
    if st is not None:
        assert is_same_process(pid, st)


def test_is_same_process_rejects_wrong_starttime():
    """Our own PID with a wrong starttime should NOT match."""
    import os

    pid = os.getpid()
    st = read_proc_starttime(pid)
    if st is not None:
        assert not is_same_process(pid, st + 999999)


def test_is_same_process_rejects_unreadable_current_starttime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("phasesweep.runtime.process.is_pid_alive", lambda _pid: True)
    monkeypatch.setattr("phasesweep.runtime.process.read_proc_starttime", lambda _pid: None)

    assert not is_same_process(12345, saved_starttime=111)


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
    db = tmp_path / "p.db"
    storage = f"sqlite:///{db}"

    # Manually create a study and inject a RUNNING trial + fingerprint.
    study = optuna.create_study(
        study_name="t::a",
        storage=storage,
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
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
        phases=[
            Phase(name="a", n_trials=1, search_space={"x": IntParam(type="int", low=0, high=10)})
        ],
    )

    # Will raise on fingerprint mismatch, but reap must have run first.
    with pytest.raises(RuntimeError, match="different phase config"):
        _run_phase(
            exp,
            exp.phases[0],
            inherited_winners={},
            generation_id="generation-test",
            dry_run=False,
        )
    assert reap_called["flag"]
    assert fingerprint_called["flag"]


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
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = Experiment(
        experiment="cross_phase_orphan",
        storage=storage,
        provenance={"revision": "test-fixture-v1"},
        workdir=str(tmp_path / "runs"),
        trial_command=f"{sys.executable} {trainer}",
        metric=Metric(
            name="metric",
            extractor=LogRegexExtractor(type="log_regex", pattern=r"metric=(?P<value>[0-9.eE+-]+)"),
        ),
        phases=[
            Phase(name="a", n_trials=1, search_space={}),
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
        study = optuna.create_study(
            study_name="cross_phase_orphan::b", storage=storage, direction="minimize"
        )
        study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
        trial = study.ask()
        trial_dir = _trial_dir_for(
            experiment,
            "b",
            trial.number,
            generation_id="old-generation",
            attempt_id="old-attempt",
        )
        trial_dir.mkdir(parents=True)
        trial.set_user_attr(GENERATION_ID_ATTR, "old-generation")
        trial.set_user_attr(ATTEMPT_ID_ATTR, "old-attempt")
        trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
        _write_test_process_identity(
            trial_dir,
            attempt_id="old-attempt",
            pid=stale.pid,
            pgid=os.getpgid(stale.pid),
            starttime=starttime,
        )

        experiment = experiment.model_copy(update={"env": {"STALE_PID": str(stale.pid)}})
        run_experiment(experiment)

        marker = next(
            (tmp_path / "runs" / experiment.experiment / "a").glob("trial_*/orphan_alive.txt")
        )
        assert marker.read_text() == "False"
        assert stale.poll() == -signal.SIGTERM
        assert study.get_trials(deepcopy=False)[trial.number].state == optuna.trial.TrialState.FAIL
    finally:
        if stale.poll() is None:
            os.killpg(stale.pid, signal.SIGKILL)
        stale.wait(timeout=5)


def test_populated_legacy_study_fails_before_counting_or_launch(tmp_path: Path) -> None:
    """Unscoped historical rows never satisfy a current phase budget."""
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = make_experiment(
        experiment="legacy_trial",
        storage=storage,
        workdir=tmp_path / "runs",
        trial_command=f"{sys.executable} -c \"print('metric=1')\"",
        metric=Metric(
            name="metric",
            extractor=LogRegexExtractor(type="log_regex", pattern=r"metric=(?P<value>[0-9.]+)"),
        ),
        phases=[Phase(name="p", n_trials=1, search_space={})],
    )
    study = optuna.create_study(study_name="legacy_trial::p", storage=storage, direction="minimize")
    study.add_trial(optuna.trial.create_trial(value=0.25, state=optuna.trial.TrialState.COMPLETE))

    with pytest.raises(
        RuntimeError,
        match=r"unsupported phasesweep storage schema missing.*Affected trial numbers: \[0\]",
    ):
        run_experiment(experiment)

    assert len(study.get_trials(deepcopy=False)) == 1
    assert not _generation_path(experiment).exists()


def test_storage_preflight_does_not_convert_shutdown_to_storage_failure(
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

    def interrupt_load(*args: object, **kwargs: object) -> None:
        raise shutdown

    monkeypatch.setattr(
        "phasesweep.engine.guards._load_existing_phase_study",
        interrupt_load,
    )
    cleanup = _PreflightCleanupReport()

    with pytest.raises(PhaseSweepShutdown) as exc_info:
        _preflight_existing_studies(experiment, cleanup_report=cleanup)

    assert exc_info.value is shutdown
    assert cleanup.cleanup_confirmed is True
    assert cleanup.error is None


@pytest.mark.parametrize("stage", ["get_trials", "stale_reaper", "schema_validation"])
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

    if stage == "get_trials":
        study = SimpleNamespace(
            study_name="shutdown::p",
            get_trials=lambda **_kwargs: (_ for _ in ()).throw(shutdown),
        )
        monkeypatch.setattr(
            "phasesweep.engine.guards._load_existing_phase_study",
            lambda *_args, **_kwargs: study,
        )
    else:
        study = optuna.create_study(direction="minimize")
        monkeypatch.setattr(
            "phasesweep.engine.guards._load_existing_phase_study",
            lambda *_args, **_kwargs: study,
        )
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

    with pytest.raises(PhaseSweepShutdown) as exc_info:
        _preflight_existing_studies(experiment)

    assert exc_info.value is shutdown


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
    monkeypatch.setattr(
        "phasesweep.engine.guards._load_existing_phase_study",
        lambda _experiment, phase: studies[phase.name],
    )

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
        _preflight_existing_studies(experiment)


def test_populated_legacy_study_reaps_orphan_before_schema_error(tmp_path: Path) -> None:
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = make_experiment(
        experiment="legacy_orphan",
        storage=storage,
        workdir=tmp_path / "runs",
        phases=[Phase(name="p", n_trials=1, search_space={})],
    )
    study = optuna.create_study(
        study_name="legacy_orphan::p",
        storage=storage,
        direction="minimize",
    )
    trial = study.ask()
    stale = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        starttime = read_proc_starttime(stale.pid)
        assert starttime is not None
        trial_dir = _trial_dir_for(
            experiment,
            "p",
            trial.number,
            generation_id="legacy-generation",
            attempt_id="legacy-attempt",
        )
        trial_dir.mkdir(parents=True)
        trial.set_user_attr(GENERATION_ID_ATTR, "legacy-generation")
        trial.set_user_attr(ATTEMPT_ID_ATTR, "legacy-attempt")
        trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
        _write_test_process_identity(
            trial_dir,
            attempt_id="legacy-attempt",
            pid=stale.pid,
            pgid=os.getpgid(stale.pid),
            starttime=starttime,
        )

        with pytest.raises(RuntimeError, match="unsupported phasesweep storage schema missing"):
            run_experiment(experiment)

        stale.wait(timeout=5)
        assert stale.returncode == -signal.SIGTERM
        recovered = optuna.load_study(study_name="legacy_orphan::p", storage=storage)
        assert recovered.trials[trial.number].state == optuna.trial.TrialState.FAIL
    finally:
        if stale.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(stale.pid, signal.SIGKILL)
            stale.wait(timeout=5)


def test_kill_stale_group_escalates_to_sigkill(tmp_path):
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
        from phasesweep.runtime.process import read_proc_starttime

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
    [None, "{", '{"schema_version": 1, "attempt_id": "attempt"}'],
)
def test_read_stale_process_identity_rejects_missing_or_partial_records(
    tmp_path: Path,
    content: str | None,
) -> None:
    if content is not None:
        (tmp_path / PROCESS_IDENTITY_FILE).write_text(content)

    with pytest.raises((OSError, ValueError)):
        read_stale_process_identity(tmp_path, expected_attempt_id="attempt")


def test_read_stale_process_identity_rejects_wrong_attempt(tmp_path: Path) -> None:
    _write_test_process_identity(
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
        launch_nonce="test-nonce",
    )
    monkeypatch.setattr("phasesweep.runtime.process.read_boot_id", lambda: "current-boot")
    monkeypatch.setattr(
        "phasesweep.runtime.process.kill_stale_group",
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
        launch_nonce="test-nonce",
    )
    monkeypatch.setattr("phasesweep.runtime.process.read_boot_id", lambda: None)

    assert cleanup_stale_trial_process(identity) is False


def test_kill_stale_group_refuses_live_pid_without_starttime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr("phasesweep.runtime.process.is_pid_alive", lambda _pid: True)
    monkeypatch.setattr("phasesweep.runtime.process._process_group_exists", lambda _pgid: True)
    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_group",
        lambda pgid, *, grace_seconds: calls.append(pgid) or True,
    )

    assert kill_stale_group(pid=12345, saved_starttime=None, pgid=12345) is False
    assert calls == []


def test_kill_stale_group_refuses_cleanup_on_pid_reuse_without_pgid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PID alive + starttime mismatch with no PGID leaves cleanup uncertain."""
    calls: list[int] = []

    monkeypatch.setattr("phasesweep.runtime.process.is_pid_alive", lambda pid: True)
    monkeypatch.setattr("phasesweep.runtime.process.read_proc_starttime", lambda pid: 999)
    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_group",
        lambda pgid, *, grace_seconds: calls.append(pgid) or True,
    )

    sent = kill_stale_group(pid=12345, saved_starttime=111, pgid=None)

    assert sent is False, "must refuse to advance when PID was reused and no PGID was saved"
    assert calls == [], "no kill signal should have been issued"


def test_kill_stale_group_refuses_unreadable_live_pid_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    monkeypatch.setattr("phasesweep.runtime.process.is_pid_alive", lambda _pid: True)
    monkeypatch.setattr("phasesweep.runtime.process.read_proc_starttime", lambda _pid: None)
    monkeypatch.setattr("os.getpgid", lambda _pid: 7777)
    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_group",
        lambda pgid, *, grace_seconds: calls.append(pgid) or True,
    )

    confirmed = kill_stale_group(pid=12345, saved_starttime=111, pgid=12345)

    assert confirmed is False
    assert calls == []


def test_kill_stale_group_refuses_unreadable_live_pgid_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    monkeypatch.setattr("phasesweep.runtime.process._process_group_exists", lambda _pgid: True)
    monkeypatch.setattr("phasesweep.runtime.process._read_proc_stat", lambda _entry: None)
    monkeypatch.setattr("phasesweep.runtime.process.is_pid_alive", lambda _pid: True)
    monkeypatch.setattr("phasesweep.runtime.process._process_group_alive", lambda _pgid: True)
    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_group",
        lambda pgid, *, grace_seconds: calls.append(pgid) or True,
    )

    confirmed = kill_stale_group(pid=None, saved_starttime=111, pgid=12345)

    assert confirmed is False
    assert calls == []


def test_kill_stale_group_refuses_pgid_fallback_when_group_leader_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dead root PID + reused PGID leader should fail closed."""
    calls: list[int] = []

    monkeypatch.setattr("phasesweep.runtime.process.is_pid_alive", lambda pid: False)
    monkeypatch.setattr("phasesweep.runtime.process._process_group_exists", lambda pgid: True)
    monkeypatch.setattr(
        "phasesweep.runtime.process._read_proc_stat",
        lambda proc_entry: SimpleNamespace(state="S", pgrp=12345, starttime=999),
    )
    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_group",
        lambda pgid, *, grace_seconds: calls.append(pgid) or True,
    )

    sent = kill_stale_group(pid=12345, saved_starttime=111, pgid=12345)

    assert sent is False, "must refuse reused PGID fallback"
    assert calls == [], "no kill signal should have been issued"


def test_kill_stale_group_accepts_pgid_reuse_when_group_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused PID with the saved PGID number is clean if no such group exists."""
    calls: list[int] = []

    monkeypatch.setattr("phasesweep.runtime.process.is_pid_alive", lambda pid: True)
    monkeypatch.setattr("phasesweep.runtime.process.read_proc_starttime", lambda pid: 999)
    monkeypatch.setattr("phasesweep.runtime.process._process_group_exists", lambda pgid: False)
    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_group",
        lambda pgid, *, grace_seconds: calls.append(pgid) or True,
    )

    sent = kill_stale_group(pid=12345, saved_starttime=111, pgid=12345)

    assert sent is True
    assert calls == [], "no signal is needed after confirming the saved group is gone"


def test_kill_stale_group_accepts_stored_pgid_reuse_when_group_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stored PGID fallback is complete when ``killpg(pgid, 0)`` says it is gone."""
    calls: list[int] = []

    monkeypatch.setattr("phasesweep.runtime.process._process_group_exists", lambda pgid: False)
    monkeypatch.setattr(
        "phasesweep.runtime.process._read_proc_stat",
        lambda proc_entry: SimpleNamespace(state="S", pgrp=7777, starttime=999),
    )
    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_group",
        lambda pgid, *, grace_seconds: calls.append(pgid) or True,
    )

    sent = kill_stale_group(pid=None, saved_starttime=111, pgid=12345)

    assert sent is True
    assert calls == [], "no signal is needed after confirming the saved group is gone"


def test_kill_stale_group_uses_pgid_when_reused_pid_is_not_group_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused PID outside the stored PGID must not block cleanup of that PGID."""
    calls: list[int] = []

    monkeypatch.setattr("phasesweep.runtime.process.is_pid_alive", lambda pid: True)
    monkeypatch.setattr("phasesweep.runtime.process._process_group_exists", lambda pgid: True)
    monkeypatch.setattr("phasesweep.runtime.process._process_group_alive", lambda pgid: True)
    monkeypatch.setattr(
        "phasesweep.runtime.process._read_proc_stat",
        lambda proc_entry: SimpleNamespace(state="S", pgrp=7777, starttime=999),
    )
    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_group",
        lambda pgid, *, grace_seconds: calls.append(pgid) or True,
    )

    sent = kill_stale_group(pid=12345, saved_starttime=111, pgid=12345)

    assert sent is True
    assert calls == [12345]


def test_kill_stale_group_uses_pgid_when_pid_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    """PID gone, no starttime check possible — PGID fallback is correct here."""
    calls: list[int] = []

    monkeypatch.setattr("phasesweep.runtime.process.is_pid_alive", lambda pid: False)
    monkeypatch.setattr("phasesweep.runtime.process._process_group_exists", lambda pgid: True)
    monkeypatch.setattr("phasesweep.runtime.process._read_proc_stat", lambda proc_entry: None)
    # Force the early-out gate to see the group as alive so the test exercises
    # the delegation to _terminate_process_group (post-v0.5.8 the gate would
    # otherwise short-circuit when the fake pgid 42 isn't a real process).
    monkeypatch.setattr("phasesweep.runtime.process._process_group_alive", lambda pgid: True)
    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_group",
        lambda pgid, *, grace_seconds: calls.append(pgid) or True,
    )

    sent = kill_stale_group(pid=99999, saved_starttime=111, pgid=42)

    assert sent is True
    assert calls == [42]


def test_kill_stale_group_uses_live_pid_when_starttime_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PID alive + starttime match => derive PGID and kill (the safe path)."""
    calls: list[int] = []

    monkeypatch.setattr("phasesweep.runtime.process.is_pid_alive", lambda pid: True)
    monkeypatch.setattr("phasesweep.runtime.process.read_proc_starttime", lambda pid: 111)
    monkeypatch.setattr("os.getpgid", lambda pid: 7777)
    # Force the early-out gate (post-v0.5.8) to see the group as alive so the
    # test exercises delegation to _terminate_process_group.
    monkeypatch.setattr("phasesweep.runtime.process._process_group_alive", lambda pgid: True)
    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_group",
        lambda pgid, *, grace_seconds: calls.append(pgid) or True,
    )

    sent = kill_stale_group(pid=12345, saved_starttime=111, pgid=12345)

    assert sent is True
    assert calls == [7777]


def test_reaper_uses_persisted_trial_dir_when_workdir_changes(tmp_path: Path) -> None:
    """A trial whose orchestrator died is reaped from its persisted dir, not a recomputed one.

    Simulates the failure mode: orchestrator A wrote identity files under
    workdir_A; the user re-runs from a different CWD/workdir as orchestrator B.
    Reaper must read trial_dir from the trial's user_attr, not reconstruct it.
    """
    workdir_A = tmp_path / "wd_a"
    workdir_B = tmp_path / "wd_b"
    persisted_trial_dir = workdir_A / "p" / "trial_00000"
    persisted_trial_dir.mkdir(parents=True)

    exp = make_experiment(workdir=workdir_B)

    # Set up an in-memory study with one RUNNING trial that has the user_attr.
    study = optuna.create_study(study_name="t::p")
    trial = study.ask()
    trial.set_user_attr(ATTEMPT_ID_ATTR, "persisted-attempt")
    trial.set_user_attr(TRIAL_DIR_ATTR, str(persisted_trial_dir))
    # Leave it RUNNING — that's what the reaper looks for.

    seen_dirs: list[str] = []

    def fake_read_identity(
        trial_dir: Path,
        *,
        expected_attempt_id: str,
    ) -> StaleProcessIdentity:
        seen_dirs.append(str(trial_dir))
        assert expected_attempt_id == "persisted-attempt"
        return StaleProcessIdentity(
            schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
            attempt_id=expected_attempt_id,
            pid=99999,
            pgid=99999,
            proc_starttime=12345,
            boot_id="test-boot",
            launch_nonce="test-nonce",
        )

    import phasesweep.engine.guards as _reaper

    real_read = _reaper.read_stale_process_identity
    _reaper.read_stale_process_identity = fake_read_identity  # type: ignore[assignment]
    real_cleanup = _reaper.cleanup_stale_trial_process
    _reaper.cleanup_stale_trial_process = lambda _identity: True
    try:
        _reap_stale_trials(study, exp, "p")
    finally:
        _reaper.read_stale_process_identity = real_read  # type: ignore[assignment]
        _reaper.cleanup_stale_trial_process = real_cleanup

    assert seen_dirs == [str(persisted_trial_dir)], (
        f"reaper should have used persisted trial_dir; got {seen_dirs}"
    )


def test_reaper_falls_back_for_prelaunch_trial_without_trial_dir_attr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RUNNING trial without ``phasesweep_trial_dir`` is safe to mark FAIL.

    Optuna creates the RUNNING trial before ``objective()`` persists the trial
    directory. A crash in that window cannot have launched a subprocess yet, so
    startup recovery must not strand the study.
    """
    exp = make_experiment(workdir=tmp_path / "runs")
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    expected_trial_dir = _trial_dir_for(exp, exp.phases[0].name, trial.number)

    monkeypatch.setattr(
        "phasesweep.engine.guards.read_stale_process_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prelaunch trials must not read process identity")
        ),
    )

    reaped = _reap_stale_trials(study, exp, exp.phases[0].name)

    assert reaped == 1
    assert expected_trial_dir.parent == tmp_path / "runs" / "t" / "p"
    assert study.trials[trial.number].state == optuna.trial.TrialState.FAIL


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


def test_reaper_raises_when_tell_fails_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``study.tell(... FAIL)`` fails after successful process cleanup,
    the reaper must raise — not silently skip. A phantom RUNNING trial would
    corrupt the ``remaining`` calculation and schedule extra trials against
    an inconsistent study state."""
    exp = make_experiment(workdir=tmp_path / "runs")
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial.set_user_attr(ATTEMPT_ID_ATTR, "tell-failure-attempt")
    trial.set_user_attr(TRIAL_DIR_ATTR, str(tmp_path / "runs" / "t" / "p" / "trial_00000"))

    monkeypatch.setattr(
        "phasesweep.engine.guards._read_trial_process_identity",
        lambda *_args, **_kwargs: StaleProcessIdentity(
            schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
            attempt_id="tell-failure-attempt",
            pid=99999,
            pgid=99999,
            proc_starttime=12345,
            boot_id="test-boot",
            launch_nonce="test-nonce",
        ),
    )
    monkeypatch.setattr("phasesweep.engine.guards.cleanup_stale_trial_process", lambda _: True)

    def fail_tell(*args: object, **kwargs: object) -> None:
        raise RuntimeError("storage write failed")

    monkeypatch.setattr(study, "tell", fail_tell)

    with pytest.raises(RuntimeError, match="Optuna state could not be updated"):
        _reap_stale_trials(study, exp, exp.phases[0].name)

    # The trial must NOT have been marked FAIL (because tell raised).
    assert study.trials[trial.number].state == optuna.trial.TrialState.RUNNING
