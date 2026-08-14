"""Shared test fixtures and helpers for phasesweep tests.

One experiment factory to replace the 5+ near-identical _minimal_experiment /
_exp / _make_exp helpers scattered across test files. Tests that need specialized construction can pass explicit phases or phase keyword overrides.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import sqlite3
import stat
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import optuna
import pytest
import yaml
from pydantic import ValidationError

from phasesweep import load_experiment
from phasesweep.config import (
    Constraint,
    ExecutionContext,
    Experiment,
    IntParam,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
)
from phasesweep.engine.state import ARTIFACT_ROOT_ATTR
from phasesweep.evidence import TrialContext
from phasesweep.runtime.process import _read_proc_stat

# Repository root, derived from the conftest location. Tests that copy/edit
# the example experiment.yaml read this so they don't hard-code paths.
REPO = Path(__file__).resolve().parent.parent
_P = ParamSpec("_P")
_R = TypeVar("_R")


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


def file_mode(path: Path) -> int:
    """Return the permission bits for a test path."""
    return stat.S_IMODE(path.stat().st_mode)


def drop_artifact_root_binding(storage: str, study_name: str) -> None:
    """Reconstruct a pre-binding study by deleting its artifact-root user attr."""
    with sqlite3.connect(storage.removeprefix("sqlite:///")) as connection:
        connection.execute(
            "DELETE FROM study_user_attributes WHERE key = ? AND study_id = "
            "(SELECT study_id FROM studies WHERE study_name = ?)",
            (ARTIFACT_ROOT_ATTR, study_name),
        )


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

    Hardware-marked tests opt out and may explicitly exercise the real host.
    GPU behavior tests replace these defaults with the inventory and driver
    state required by each case.
    """
    if request.node.get_closest_marker("hardware") is not None:
        return
    # The empty CUDA sentinel also reaches runner/trainer subprocesses, where
    # this process's monkeypatches cannot. Individual GPU tests replace or
    # delete it when exercising ambient visibility and auto-detection.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))
    monkeypatch.setattr("phasesweep.runtime.gpu._nvidia_driver_reports_gpus", lambda: False)


@pytest.fixture(autouse=True)
def isolate_signal_ownership_tokens() -> Iterator[None]:
    """Snapshot and restore PhaseSweep's process-level signal state per test.

    ``phasesweep.runtime.process._process_lifetime_owner`` and ``_scope_depth``
    are plain module globals (review v0.5.15 / blocker 2B), deliberately not
    re-derived from OS ground truth the way the actual signal handlers are.
    A test that calls ``install_signal_handlers()`` (or drives a CLI/MCP main
    in-process) would otherwise permanently replace pytest's SIGTERM/SIGINT/
    SIGHUP handlers and unblock those signals, as well as flipping the private
    ownership tokens. Later Ctrl-C and CI timeout signals would then enter
    PhaseSweep's shutdown handler instead of pytest's. Restore the kernel mask,
    OS handlers, and ownership bookkeeping as one fixture-level transaction.
    """
    import phasesweep.runtime.process as process

    restore_signal = signal.signal
    restore_mask = getattr(signal, "pthread_sigmask", None)
    prior_owner = process._process_lifetime_owner
    prior_depth = process._scope_depth
    prior_handlers = {sig: signal.getsignal(sig) for sig in process._SHUTDOWN_SIGNALS}
    prior_mask = restore_mask(signal.SIG_BLOCK, set()) if restore_mask is not None else None
    try:
        yield
    finally:
        if restore_mask is not None and prior_mask is not None:
            restore_mask(signal.SIG_BLOCK, set(process._SHUTDOWN_SIGNALS))
        for sig, handler in prior_handlers.items():
            restore_signal(sig, handler)
        if restore_mask is not None and prior_mask is not None:
            restore_mask(signal.SIG_SETMASK, prior_mask)
        process._process_lifetime_owner = prior_owner
        process._scope_depth = prior_depth


def copy_fake_train(tmp_path: Path) -> Path:
    trainer = tmp_path / "examples" / "fake_train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "src" / "phasesweep" / "examples" / "fake_train.py", trainer)
    return trainer


def make_experiment(
    *,
    experiment: str = "t",
    workdir: str | Path | None = None,
    storage: str | None = None,
    trial_command: str = "echo {overrides}",
    override_format: str = "argparse",
    trainer_config: dict[str, Any] | None = None,
    metric: Metric | None = None,
    constraints: list[Constraint] | None = None,
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
    """
    if phases is None:
        base: dict[str, Any] = dict(
            name="p",
            n_trials=2,
            search_space={"x": IntParam(type="int", low=0, high=10)},
        )
        base.update(phase_overrides)
        if storage is not None and "sampler" not in base:
            # Persistent storage rejects an unseeded stochastic sampler and
            # requires acknowledge_nonresumable for tpe/cmaes, so the default
            # tpe sampler cannot be the fixture default here. Seeded random is
            # stochastic, reproducible, and resumable: it satisfies the policy
            # without asking every caller to acknowledge anything.
            base["sampler"] = Sampler(type="random", seed=0)
        phases = [Phase(**base)]  # type: ignore[arg-type]

    kwargs: dict[str, Any] = dict(
        experiment=experiment,
        trial_command=trial_command,
        override_format=override_format,
        metric=metric
        or Metric(
            extractor=LogRegexExtractor(
                type="log_regex",
                pattern=r"x=(?P<value>[0-9.eE+-]+)",
            )
        ),
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
    if env is not None:
        kwargs["env"] = env
    if execution is not None:
        kwargs["execution"] = execution

    return Experiment(**kwargs)


def write_yaml(tmp_path: Path, body: str) -> Path:
    """Write a YAML body verbatim to ``tmp_path/exp.yaml``; return the path.

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


def write_constant_trainer(tmp_path: Path) -> Path:
    """Drop a minimal trainer that writes and logs a constant objective.

    Cheap enough for tests that need a real subprocess run before mutating
    the parent config and re-running with ``--from-phase``.
    """
    return write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse, json
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"x": 0.5}))
        print("x=0.5")
        """,
    )


def assert_published_winner_evidence_local(experiment_dir: Path) -> None:
    """Assert every published winner's source generation exists in this same tree.

    The invariant a second, divergent artifact tree breaks: a published
    winner's ``winner_source.generation_id`` names the immutable generation
    namespace holding the evidence behind that number, so it must resolve
    *inside the tree the reader is reading*. When one study is allowed to back
    two roots, both report ``publication_integrity: ok`` while the winner in
    one of them cites a generation that only exists in the other (re-review
    v0.5.19 / blocker B1).

    :param Path experiment_dir: Experiment artifact namespace to check, i.e.
        ``<workdir>/<experiment>``.
    """
    pointer = experiment_dir / "last_successful_generation.yaml"
    assert pointer.is_file(), f"{experiment_dir} has never published"
    published = yaml.safe_load(pointer.read_text())["generation_id"]
    generations = experiment_dir / "generations"
    assert (generations / published).is_dir(), (
        f"published generation {published!r} is missing from {generations}"
    )
    winners = sorted((generations / published / "phases").glob("*/winner.yaml"))
    assert winners, f"published generation {published!r} exposes no winner"
    for winner_path in winners:
        source = yaml.safe_load(winner_path.read_text())["winner_source"]
        source_generation = source["generation_id"]
        assert (generations / source_generation).is_dir(), (
            f"{winner_path} cites generation {source_generation!r}, which does not exist "
            f"under {generations}: its evidence lives in another artifact tree"
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
