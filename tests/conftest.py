"""Shared test fixtures and helpers for phasesweep tests."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import signal
import stat
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NoReturn, ParamSpec, TypeVar

import optuna
import pytest
from pydantic import ValidationError

from phasesweep import load_experiment
from phasesweep.cli import main as cli_boundary
from phasesweep.config.models import (
    Constraint,
    ExecutionContext,
    Experiment,
    Metric,
    Observation,
    Phase,
)
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.artifact_roots import (
    _check_artifact_root_binding,
    _write_artifact_root_binding,
)
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
    FEASIBLE_ATTR,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRIAL_OUTCOME_SCHEMA_VERSION,
)
from phasesweep.evidence.evaluation import EVIDENCE_PROVENANCE_SCHEMA_VERSION, TrialContext
from phasesweep.evidence.models import JsonEnvelopeExtractor, LogRegexExtractor
from phasesweep.runtime.files import file_url_path
from phasesweep.runtime.json import JsonObject, JsonValue
from phasesweep.runtime.reaper import _read_proc_stat

# Repository root, derived from the conftest location. Tests that copy/edit
# the example experiment.yaml read this so they don't hard-code paths.
REPO = Path(__file__).resolve().parent.parent
_P = ParamSpec("_P")
_R = TypeVar("_R")

#: Skip marker for a test or parametrized case that needs the kernel to deny a
#: permission. Root bypasses file modes, so there the denial never happens.
requires_nonroot = pytest.mark.skipif(
    os.geteuid() == 0, reason="root bypasses file permissions, so no denial can be provoked"
)


def raise_after_first_successful_call(
    callback: Callable[_P, _R],
    error: BaseException,
) -> tuple[Callable[_P, _R], list[None]]:
    """Wrap ``callback`` so its first successful call raises ``error`` afterward."""
    calls: list[None] = []

    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        result = callback(*args, **kwargs)
        calls.append(None)
        if len(calls) == 1:
            raise error
        return result

    return wrapped, calls


def patch_directory_fsync_failure(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    """Make directory fsync fail while preserving regular-file fsync calls."""
    real_fsync = os.fsync

    def fail_directory_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(message)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)


def patch_path_method_failure(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    method_name: str,
    error: OSError,
) -> None:
    """Make one ``Path`` method fail only for ``target``."""
    original_method = getattr(Path, method_name)

    def fail_target(self: Path, *args: object, **kwargs: object):
        if self == target:
            raise error
        return original_method(self, *args, **kwargs)

    monkeypatch.setattr(Path, method_name, fail_target)


def patch_rejected_trial_user_attr(
    monkeypatch: pytest.MonkeyPatch,
    rejected_key: str,
    message: str,
) -> Callable[[optuna.Trial, str, Any], None]:
    """Make one Optuna trial-user-attribute key fail to persist."""
    real_set_user_attr = optuna.Trial.set_user_attr

    def reject_key(trial: optuna.Trial, key: str, value: Any) -> None:
        if key == rejected_key:
            raise RuntimeError(message)
        real_set_user_attr(trial, key, value)

    monkeypatch.setattr(optuna.Trial, "set_user_attr", reject_key)
    return real_set_user_attr


@contextlib.contextmanager
def temporary_umask(mask: int) -> Iterator[None]:
    """Set the process umask for one test and restore it afterward."""
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


def is_pid_zombie(pid: int) -> bool:
    """Return whether a test subprocess has exited but awaits parent reaping."""
    stat = _read_proc_stat(Path("/proc") / str(pid))
    return stat is not None and stat.state == "Z"


# Bound at import, so a test that patches ``subprocess.Popen`` or ``os.waitpid``
# for the code under test still gets a real, reaped child from ``reaped_pid``.
_spawn_child = os.posix_spawnp
_reap_child = os.waitpid


def reaped_pid() -> int:
    """Return the PID of a child this process spawned and has already reaped.

    It stands in for a runner or trainer that has exited. Linux assigns PIDs in
    increasing order, so this one stays unused until the PID space wraps, while
    a fixed "dead" number such as 999999 is below the default 64-bit
    ``pid_max`` and can belong to a live process on a busy host.
    """
    pid = _spawn_child("true", ["true"], os.environ)
    _reap_child(pid, 0)
    return pid


def file_mode(path: Path) -> int:
    """Return the permission bits for a test path."""
    return stat.S_IMODE(path.stat().st_mode)


#: Optuna journal op codes this helper needs. Mirrors
#: ``optuna.storages.journal._storage.JournalOperation`` and
#: ``tests/fixtures/make_ledger_fixtures.py``'s own local copies.
_JOURNAL_CREATE_STUDY = 0
_JOURNAL_SET_STUDY_USER_ATTR = 2


def drop_artifact_root_binding(storage: str, study_name: str) -> None:
    """Reconstruct a pre-binding study by deleting its artifact-root user attr.

    A journal ledger has no row to delete in place, so this replays the study
    id assignment the same way Optuna's journal replay does -- sequentially,
    in ``CREATE_STUDY`` record order -- then rewrites the ledger without the
    ``SET_STUDY_USER_ATTR`` record(s) that set ``ARTIFACT_ROOT_ATTR`` for that
    study id.

    :param str storage: ``journal:///`` storage URL for the study's ledger.
    :param str study_name: Study whose artifact-root user attribute is dropped.
    :raises RuntimeError: No matching attribute record was found to drop.
    """
    ledger = Path(file_url_path(storage))
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    study_id = None
    next_id = 0
    for record in records:
        if record.get("op_code") == _JOURNAL_CREATE_STUDY:
            if record.get("study_name") == study_name:
                study_id = next_id
            next_id += 1
    kept = [
        record
        for record in records
        if not (
            record.get("op_code") == _JOURNAL_SET_STUDY_USER_ATTR
            and record.get("study_id") == study_id
            and ARTIFACT_ROOT_ATTR in record.get("user_attr", {})
        )
    ]
    if len(kept) == len(records):
        raise RuntimeError(f"no {ARTIFACT_ROOT_ATTR!r} record found for study {study_name!r}")
    body = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in kept)
    ledger.write_text(body, encoding="utf-8")


def mark_current_format(experiment: Experiment, *studies: optuna.Study) -> None:
    """Stamp hand-built test studies and claim their artifact tree.

    A test that constructs studies through Optuna directly skips the engine's
    own create path, so nothing stamps the study schema and nothing records the
    tree's ownership. Every read path then correctly refuses the state as
    unmarked. This helper stamps the current study schema and records tree
    ownership before read paths inspect the state.

    The tree is only claimed when it has no owner yet, matching what the engine
    does: an already-bound tree is verified and left alone.

    :param Experiment experiment: Experiment whose artifact root is claimed.
    :param optuna.Study studies: Studies to stamp with the current study schema.
    """
    for study in studies:
        study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    if _check_artifact_root_binding(experiment) == "unbound":
        _write_artifact_root_binding(experiment)


def tell_feasible(study: optuna.Study, trial: optuna.Trial, value: float = 0.5) -> None:
    """Complete a hand-built trial as a run completes one inside every bound.

    A run records the feasibility attr before a trial completes, and a COMPLETE trial
    without it reads ``INFEASIBLE`` in counts and tables.

    :param optuna.Study study: Study that owns ``trial``.
    :param optuna.Trial trial: Asked trial to complete.
    :param float value: Objective value to tell.
    """
    trial.set_user_attr(FEASIBLE_ATTR, True)
    study.tell(trial, value)


def _bounded_metric(**bounds: float) -> Metric:
    """Return a log_regex ``x=<value>`` metric with the given viability bounds."""
    return Metric(
        extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        **bounds,
    )


def _phase_trials(experiment: Experiment, phase_name: str = "p") -> list[optuna.trial.FrozenTrial]:
    """Read one phase's trials from the experiment's ledger.

    :param Experiment experiment: Experiment whose ledger is read.
    :param str phase_name: Phase whose study is read.
    :return list[optuna.trial.FrozenTrial]: Its trials in number order.
    """
    storage = _resolve_storage(experiment.resolved_storage)
    return optuna.load_study(
        study_name=f"{experiment.experiment}::{phase_name}", storage=storage
    ).get_trials(deepcopy=False)


@pytest.fixture(autouse=True)
def isolate_phasesweep_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep scaffold state and default caches inside test-owned directories."""
    monkeypatch.delenv("PHASESWEEP_HOME", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state-home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache-home"))


@pytest.fixture(autouse=True)
def isolate_phasesweep_lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep host-wide test locks inside each test's temp directory."""
    lock_dir = tmp_path / "phasesweep-locks"
    lock_dir.mkdir(mode=0o700)
    lock_dir.chmod(0o700)
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(lock_dir))


@pytest.fixture(autouse=True)
def isolate_host_gpu_detection(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Keep ordinary tests independent of the host's NVIDIA driver state.

    Live-marked tests opt out and may explicitly exercise the real host.
    GPU behavior tests replace these defaults with the inventory and driver
    state required by each case.
    """
    if request.node.get_closest_marker("live") is not None:
        return
    # The empty CUDA sentinel also reaches runner/trainer subprocesses, where
    # this process's monkeypatches cannot. Individual GPU tests replace or
    # delete it when exercising ambient visibility and auto-detection.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))
    monkeypatch.setattr("phasesweep.runtime.gpu._nvidia_driver_reports_gpus", lambda: False)


@contextlib.contextmanager
def restored_signal_ownership() -> Iterator[None]:
    """Give the process back its shutdown-signal state when the block exits.

    Restores, in order: the shutdown signals blocked, so none is delivered to
    a half-restored handler set; the prior OS handlers; the prior kernel mask;
    and ``phasesweep.runtime.shutdown``'s ownership tokens and the signal its
    handler is servicing.

    :return Iterator[None]: Context in which PhaseSweep may take signal ownership.
    """
    import phasesweep.runtime.shutdown as shutdown

    # Bound at entry so a replacement a test installs cannot do the restoring.
    restore_signal = signal.signal
    restore_mask = signal.pthread_sigmask
    prior_owner = shutdown._process_lifetime_owner
    prior_depth = shutdown._scope_depth
    prior_servicing = shutdown._servicing_signum
    prior_handlers = {sig: signal.getsignal(sig) for sig in shutdown._SHUTDOWN_SIGNALS}
    prior_mask = restore_mask(signal.SIG_BLOCK, set())
    try:
        yield
    finally:
        restore_mask(signal.SIG_BLOCK, set(shutdown._SHUTDOWN_SIGNALS))
        for sig, handler in prior_handlers.items():
            restore_signal(sig, handler)
        restore_mask(signal.SIG_SETMASK, prior_mask)
        shutdown._process_lifetime_owner = prior_owner
        shutdown._scope_depth = prior_depth
        shutdown._servicing_signum = prior_servicing


@pytest.fixture(autouse=True)
def isolate_signal_ownership_tokens() -> Iterator[None]:
    """Snapshot and restore PhaseSweep's process-level signal state per test.

    ``phasesweep.runtime.shutdown._process_lifetime_owner`` and ``_scope_depth``
    are plain module globals, deliberately not re-derived from OS ground truth
    the way the actual signal handlers are.
    A test that calls ``install_signal_handlers()`` (or drives a CLI/MCP main
    in-process) would otherwise permanently replace pytest's SIGTERM/SIGINT/
    SIGHUP handlers and unblock those signals, as well as flipping the private
    ownership tokens. Later Ctrl-C and CI timeout signals would then enter
    PhaseSweep's shutdown handler instead of pytest's. Restore the kernel mask,
    OS handlers, and ownership bookkeeping as one fixture-level transaction.
    """
    with restored_signal_ownership():
        yield


#: Marker for a test that signals its own process on purpose, to drive a
#: shutdown handler. It lifts only that one target from the runner guard.
SIGNALS_OWN_PID = "signals_own_pid"


class RunnerSignalGuard:
    """``os.kill``/``os.killpg`` replacements that refuse to signal the test runner.

    Protected: pytest's PID and its parent's PID, and the process groups of
    both. Signal 0 only probes liveness and always passes through.
    """

    def __init__(self, *, allow_own_pid: bool) -> None:
        """Record the protected targets as they are when the test starts.

        :param bool allow_own_pid: Let signals reach pytest's own PID, but no
            other protected target.
        """
        own_pid, parent_pid = os.getpid(), os.getppid()
        self.pids = frozenset({parent_pid} if allow_own_pid else {own_pid, parent_pid})
        groups = {os.getpgrp()}
        with contextlib.suppress(OSError):
            groups.add(os.getpgid(parent_pid))
        self.groups = frozenset(groups)
        self.refused: list[str] = []
        # The guard's own tests swap these for recorders, so a regressed guard
        # still cannot deliver the signal it was meant to refuse.
        self.send_kill: Callable[[int, int], None] = os.kill
        self.send_killpg: Callable[[int, int], None] = os.killpg

    def _refuse(self, call: str) -> NoReturn:
        self.refused.append(call)
        # ``pytest.fail`` raises a BaseException, so the broad ``except
        # Exception`` around kills in the code under test cannot swallow it.
        pytest.fail(f"{call} would signal the test runner, its parent, or their process group")

    def kill(self, pid: int, sig: int) -> None:
        """Forward ``os.kill`` unless it would signal a protected target."""
        # 0 addresses the caller's own group, -1 every process, -N group N.
        if sig != 0 and (pid in self.pids or pid in (0, -1) or -pid in self.groups):
            self._refuse(f"os.kill({pid}, {sig!r})")
        self.send_kill(pid, sig)

    def killpg(self, pgid: int, sig: int) -> None:
        """Forward ``os.killpg`` unless it would signal a protected group."""
        if sig != 0 and (pgid == 0 or pgid in self.groups):
            self._refuse(f"os.killpg({pgid}, {sig!r})")
        self.send_killpg(pgid, sig)


@pytest.fixture(autouse=True)
def guard_runner_signals(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Iterator[RunnerSignalGuard]:
    """Fail any test whose code would send a real signal to the test runner.

    A default ``make_run_handle`` names pytest's live PID, and
    ``patch_popen_capture`` names pytest's parent, each with a matching start
    time. ``kill_stale_group`` trusts such an identity and signals
    ``os.getpgid(pid)``: pytest's real process group. Tests patch the reaper out
    of those paths; this guard catches the one that forgets. A refusal also
    fails the test at teardown, in case a ``BaseException`` boundary in the code
    under test swallowed the in-call failure. A test that signals itself on
    purpose opts out with ``@pytest.mark.signals_own_pid``.
    """
    guard = RunnerSignalGuard(
        allow_own_pid=request.node.get_closest_marker(SIGNALS_OWN_PID) is not None
    )
    monkeypatch.setattr(os, "kill", guard.kill)
    monkeypatch.setattr(os, "killpg", guard.killpg)
    yield guard
    if guard.refused:
        pytest.fail("refused signals to the test runner: " + ", ".join(guard.refused))


def copy_fake_train(tmp_path: Path) -> Path:
    trainer = tmp_path / "examples" / "fake_train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "src" / "phasesweep" / "examples" / "fake_train.py", trainer)
    return trainer


#: The objective the packaged ``fake_train`` example writes to ``result.json``.
FAKE_TRAIN_METRIC = Metric(
    name="eval_loss",
    extractor=JsonEnvelopeExtractor(
        type="json_envelope",
        objective_name="eval_loss",
        split="validation",
        policy="synthetic",
    ),
)

#: The command every runnable-trainer experiment uses, ``trainer`` still to fill in.
_RUNNABLE_TRIAL_COMMAND = "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"


def make_experiment(
    *,
    experiment: str = "t",
    workdir: str | Path | None = None,
    storage: str | None = None,
    persistent: Path | None = None,
    trainer: Path | None = None,
    trial_command: str | None = None,
    override_format: str = "argparse",
    trainer_config: dict[str, Any] | None = None,
    metric: Metric | None = None,
    constraints: list[Constraint] | None = None,
    observe: list[Observation] | None = None,
    phases: list[Phase] | None = None,
    env: dict[str, str] | None = None,
    execution: ExecutionContext | None = None,
    provenance: dict[str, str] | None = None,
    **phase_overrides: Any,
) -> Experiment:
    """Build a minimal valid Experiment for testing.

    If ``phases`` is not given, a single phase named ``"p"`` is created with
    ``search_space={"x": IntParam(0..10)}``. Extra ``**phase_overrides`` are
    forwarded to that default phase (e.g. ``n_trials=4``, ``gpu_ids=[0]``).
    With persistent ``storage`` and no caller-supplied sampler, that default
    phase uses ``Sampler(type="random", seed=0)`` so it satisfies the
    persistent-storage sampler policy.

    ``persistent`` names a directory that holds the journal ledger
    ``studies.journal`` and the workdir ``runs/``; an explicit ``workdir`` or
    ``storage`` overrides that half. ``trainer`` makes each trial run that
    script as ``python <trainer> --out {trial_dir}/r.json {overrides}``.

    :raises TypeError: ``trainer`` and ``trial_command`` are both given, or
        ``phases`` is given with phase keywords it would silently drop.
    """
    if trainer is not None and trial_command is not None:
        raise TypeError("pass trainer= or trial_command=, not both")
    if phases is not None and phase_overrides:
        raise TypeError(
            f"phase keywords {sorted(phase_overrides)} apply only to the default phase, "
            "not to explicit phases="
        )
    if persistent is not None:
        if workdir is None:
            workdir = persistent / "runs"
        if storage is None:
            storage = f"journal:///{persistent / 'studies.journal'}"
    if trial_command is None:
        trial_command = (
            "echo {overrides}"
            if trainer is None
            else _RUNNABLE_TRIAL_COMMAND.format(trainer=trainer)
        )
    if phases is None:
        base: dict[str, Any] = dict(
            name="p",
            n_trials=2,
            search_space={"x": IntParam(type="int", low=0, high=10)},
        )
        base.update(phase_overrides)
        if storage is not None and "sampler" not in base:
            # Persistent storage rejects an unseeded stochastic sampler, so the
            # default unseeded tpe sampler cannot be the fixture default here.
            base["sampler"] = Sampler(type="random", seed=0)
        phases = [Phase(**base)]  # type: ignore[arg-type]

    kwargs: dict[str, Any] = dict(
        experiment=experiment,
        trial_command=trial_command,
        override_format=override_format,
        metric=metric or _bounded_metric(),
        phases=phases,
    )
    if trainer_config is not None:
        kwargs["trainer_config"] = trainer_config
    if workdir is not None:
        kwargs["workdir"] = str(workdir)
    if storage is not None:
        kwargs["storage"] = storage
        kwargs["provenance"] = provenance or {"revision": "test-fixture-v1"}
    elif provenance is not None:
        kwargs["provenance"] = provenance
    if constraints is not None:
        kwargs["constraints"] = constraints
    if observe is not None:
        kwargs["observe"] = observe
    if env is not None:
        kwargs["env"] = env
    if execution is not None:
        kwargs["execution"] = execution

    return Experiment(**kwargs)


def write_yaml(tmp_path: Path, body: str) -> Path:
    """Write a YAML body, dedented, to ``tmp_path/exp.yaml``; return the path.

    Body is passed through ``textwrap.dedent`` so callers can use indented
    triple-quoted strings naturally. No ``.format()`` magic — if a test
    needs ``{tmp}`` substituted, it does so at the call site via
    ``body.format(tmp=tmp_path)`` before calling this helper.
    """
    p = tmp_path / "exp.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def assert_invalid_experiment_yaml(tmp_path: Path, body: str, match: str) -> None:
    """Assert that one YAML experiment fails model validation."""
    with pytest.raises(ValidationError, match=match):
        load_experiment(write_yaml(tmp_path, body))


def invoke_cli_boundary(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, *, debug: bool = False
) -> int:
    """Run the console-script entry point exactly as the installed command does; return its exit status."""
    # ``CliRunner`` invokes the Click group directly and so bypasses the
    # process-level error boundary under test; a patched ``sys.argv`` reaches it.
    monkeypatch.setattr(sys, "argv", ["phasesweep", *argv])
    logger = logging.getLogger()
    previous_level = logger.level
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    try:
        with pytest.raises(SystemExit) as excinfo:
            cli_boundary()
    finally:
        logger.setLevel(previous_level)
    code = excinfo.value.code
    return 0 if code is None else int(code)


def trial_outcome(sequence: int, outcome: str, **extra: JsonValue) -> JsonObject:
    """Build a current-schema outcome record.

    :param int sequence: Completion sequence.
    :param str outcome: Outcome label.
    :param JsonValue extra: Further fields, such as ``cause`` or an abort record.
    :return JsonObject: The record as the objective writes it.
    """
    return {
        "schema_version": TRIAL_OUTCOME_SCHEMA_VERSION,
        "sequence": sequence,
        "outcome": outcome,
        **extra,
    }


def log_regex_provenance() -> JsonObject:
    """Return a current-format ``log_regex`` objective provenance record."""
    return {
        "schema_version": EVIDENCE_PROVENANCE_SCHEMA_VERSION,
        "extractor": {"kind": "log_regex", "config_sha256": "0" * 64},
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "source": {"kind": "file", "path": "stdout.log", "size_bytes": 0, "sha256": "0" * 64},
    }


def single_trial_confound(trial_number: int) -> dict[str, Any]:
    """Return the confound block selection records for a one-trial population."""
    return {
        "schema_version": 2,
        "ranked_trials": [trial_number],
        "ranked_groups": None,
        "checks": {
            "evaluation_point": {"verdict": "n/a", "reason": "only one trial was ranked"},
            "survivorship": {
                "verdict": "ok",
                "detail": {
                    "ranked": 1,
                    "infeasible": 0,
                    "incomplete_group": 0,
                    "failed": 0,
                },
            },
            "tie": {"verdict": "n/a", "reason": "only one trial was ranked"},
        },
    }


def write_trainer(path: Path, body: str) -> Path:
    """Write an executable Python trainer script to ``path``.

    ``path`` may be a directory (the trainer is placed at ``path/trainer.py``)
    or a file path. Returns the resolved file path. Body is dedented and
    given a ``#!/usr/bin/env python3`` shebang.
    """
    if path.is_dir():
        path = path / "trainer.py"
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
    path.chmod(0o755)
    return path


def write_flaky_trainer(tmp_path: Path, *, failures: int, value: float = 1.0) -> tuple[Path, Path]:
    """Write a trainer that fails its first ``failures`` calls, then logs ``x=<value>``.

    :param Path tmp_path: Directory holding the trainer and its call log.
    :param int failures: How many leading invocations exit nonzero.
    :param float value: The objective value every later invocation logs.
    :return tuple[Path, Path]: The trainer script and the log that gains one line per call.
    """
    calls = tmp_path / "calls.log"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import sys
        from pathlib import Path

        calls = Path({str(calls)!r})
        seen = len(calls.read_text().splitlines()) if calls.exists() else 0
        with calls.open("a") as log:
            log.write("run\\n")
        if seen < {failures}:
            sys.exit(1)
        print("x={value}")
        """,
    )
    return trainer, calls


def write_constant_trainer(tmp_path: Path, *, key: str = "x", value: float = 0.5) -> Path:
    """Drop a trainer that writes ``{key: value}`` to ``--out`` and logs ``key=value``."""
    logged = f"{key}={value!r}"
    return write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({{{key!r}: {value!r}}}))
        print({logged!r})
        """,
    )


def write_param_echo_trainer(tmp_path: Path) -> Path:
    """Drop a trainer that logs ``x=<--x>``, so each trial's objective is its own sampled value."""
    return write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--out")
        parser.add_argument("--x", type=int, default=0)
        args, _ = parser.parse_known_args()
        print(f"x={args.x}")
        """,
    )


def write_trial_zero_trainer(tmp_path: Path, *, otherwise: str) -> Path:
    """Drop a trainer whose trial 0 writes and logs ``x=1.0`` and whose other trials run ``otherwise``."""
    return write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json, os, sys, time
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        if os.environ["PHASESWEEP_TRIAL_ID"] == "0":
            with open(args.out, "w") as f:
                json.dump({{"x": 1.0}}, f)
            print("x=1.0")
        else:
            {otherwise}
        """,
    )


def write_flag_gated_trainer(tmp_path: Path, flag: Path) -> Path:
    """Drop a trainer that exits 1 until ``flag`` exists, then logs ``x=0.5``."""
    return write_trainer(
        tmp_path / "trainer.py",
        f"""
        import pathlib, sys
        if not pathlib.Path({str(flag)!r}).exists():
            sys.exit(1)
        print("x=0.5")
        """,
    )


def make_trial_context(
    tmp_path: Path,
    *,
    experiment: str = "t",
    phase: str = "p",
    trial_id: int = 0,
    run_name: str | None = None,
) -> TrialContext:
    """Build a minimal extractor trial context for unit tests."""
    return TrialContext(
        experiment=experiment,
        phase=phase,
        trial_id=trial_id,
        generation_id="generation-test",
        attempt_id="attempt-test",
        overrides_sha256="a" * 64,
        trial_dir=tmp_path,
        run_name=run_name or f"{experiment}-{phase}-{trial_id}-attempt-test",
        return_code=0,
        duration_seconds=0.0,
    )


@pytest.fixture
def wandb_worker_sdk(tmp_path, monkeypatch):
    """Supply controlled API responses to the real supervised evidence worker."""
    import sys
    from types import ModuleType

    # Availability preflight is independent of the worker fixtures and must
    # also work in the core-only test environment.
    monkeypatch.setitem(sys.modules, "wandb.apis.public", ModuleType("wandb.apis.public"))
    root = tmp_path / "sdk"
    apis = root / "wandb" / "apis"
    apis.mkdir(parents=True)
    (root / "wandb" / "__init__.py").write_text("")
    (root / "wandb" / "errors.py").write_text(
        "class AuthenticationError(Exception): pass\nclass UsageError(Exception): pass\n"
    )
    (apis / "__init__.py").write_text("")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]))

    def install(source):
        (apis / "public.py").write_text(textwrap.dedent(source))

    return install
