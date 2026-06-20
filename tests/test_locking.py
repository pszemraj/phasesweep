"""Same-host advisory run locks: output namespace and storage identity."""

from __future__ import annotations

import stat
import threading
from pathlib import Path

import pytest

from phasesweep.config import FloatParam, IntParam, Phase
from phasesweep.engine import run_experiment
from phasesweep.engine.guards import (
    _experiment_lock,
    _run_lock_paths,
)
from phasesweep.runtime import files as runtime_files
from tests.conftest import make_experiment


def test_lock_dir_defaults_to_host_stable_root_not_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_tmp = tmp_path / "job-tmp"
    job_tmp.mkdir()
    default_lock_dir = tmp_path / "host-var-tmp" / "phasesweep-locks"
    monkeypatch.delenv("PHASESWEEP_LOCK_DIR", raising=False)
    monkeypatch.setenv("TMPDIR", str(job_tmp))
    monkeypatch.setattr(runtime_files, "_DEFAULT_LOCK_DIR", default_lock_dir)

    path = runtime_files.lock_dir()

    assert path == default_lock_dir
    assert job_tmp not in path.parents
    assert path.is_dir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o1777


def test_lock_dir_honors_explicit_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "scheduler-shared-locks"
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(override))

    path = runtime_files.lock_dir()

    assert path == override
    assert path.is_dir()


def test_run_lock_collides_for_same_storage_different_workdirs(
    tmp_path: Path,
) -> None:
    """Two configs sharing the same SQLite storage but different workdirs
    target the same phase-chained experiment and must collide on the run lock.

    This is the v0.5.5 reviewer's primary scenario: process A and process B
    each write to their own workdir but share an Optuna backend. Without the
    run lock, they could top-up phase ``arch`` from B while A was already
    fingerprinted and running phase ``lr`` against an older arch winner.
    """
    storage = f"sqlite:///{tmp_path / 'shared.db'}"
    exp_a = make_experiment(workdir=str(tmp_path / "runs_a"), storage=storage)
    exp_b = make_experiment(workdir=str(tmp_path / "runs_b"), storage=storage)

    held = threading.Event()
    released = threading.Event()

    def hold_first() -> None:
        with _experiment_lock(exp_a):
            held.set()
            released.wait(timeout=5.0)

    t = threading.Thread(target=hold_first, daemon=True)
    t.start()
    assert held.wait(timeout=2.0)

    try:
        with (  # noqa: SIM117 — testing that the inner enter raises
            pytest.raises(RuntimeError, match="Another phasesweep process"),
            _experiment_lock(exp_b),
        ):
            pass
    finally:
        released.set()
        t.join(timeout=2.0)


def test_run_lock_blocks_even_when_processes_target_different_phases(
    tmp_path: Path,
) -> None:
    """Two configs with the same storage but different phase orderings still
    collide on the run lock — because the lock is experiment-scoped, not
    phase-scoped.

    This is the v0.5.5 reviewer's interleaving scenario: process A is on
    phase ``lr`` while process B starts a top-up of phase ``arch``. The phase
    lock would not catch this; the run lock does.
    """
    storage = f"journal:///{tmp_path / 'shared.journal'}"
    phases_a = [
        Phase(
            name="arch",
            n_trials=1,
            search_space={"depth": IntParam(type="int", low=1, high=2)},
        ),
        Phase(
            name="lr",
            inherits=["arch"],
            n_trials=1,
            search_space={
                "lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True),
            },
        ),
    ]
    phases_b = [
        Phase(
            name="arch",
            n_trials=2,  # top-up
            search_space={"depth": IntParam(type="int", low=1, high=2)},
        ),
    ]

    exp_a = make_experiment(workdir=str(tmp_path / "runs_a"), storage=storage, phases=phases_a)
    exp_b = make_experiment(workdir=str(tmp_path / "runs_b"), storage=storage, phases=phases_b)

    with (
        _experiment_lock(exp_a),
        pytest.raises(  # noqa: SIM117
            RuntimeError, match="Another phasesweep process"
        ),
        _experiment_lock(exp_b),
    ):
        pass


def test_run_lock_collides_for_different_storage_same_output_dir(
    tmp_path: Path,
) -> None:
    """v0.5.6 missed this: two configs sharing workdir + experiment but pointing
    at different storage backends *would* collide on filesystem outputs, but
    not on the lock. v0.5.7 introduces an output-namespace lock that catches
    this (review v0.5.6 / blocker 1).
    """
    exp_a = make_experiment(
        workdir=str(tmp_path / "runs"), storage=f"sqlite:///{tmp_path / 'a.db'}"
    )
    exp_b = make_experiment(
        workdir=str(tmp_path / "runs"), storage=f"sqlite:///{tmp_path / 'b.db'}"
    )

    with (  # noqa: SIM117 — testing that the inner enter raises
        _experiment_lock(exp_a),
        pytest.raises(RuntimeError, match="output namespace|backend"),
        _experiment_lock(exp_b),
    ):
        pass


def test_run_lock_does_not_collide_for_different_experiment_dirs(
    tmp_path: Path,
) -> None:
    """Same workdir but different experiment names → different output dirs →
    no collision (output lock identities differ).
    """
    exp_a = make_experiment(
        workdir=str(tmp_path / "runs"), storage=f"sqlite:///{tmp_path / 'a.db'}"
    )
    exp_b = make_experiment(
        workdir=str(tmp_path / "runs"), storage=f"sqlite:///{tmp_path / 'b.db'}"
    )
    exp_b = exp_b.model_copy(update={"experiment": "other"})

    with _experiment_lock(exp_a), _experiment_lock(exp_b):
        pass  # must not raise


def test_run_lock_does_not_collide_for_different_experiment_names(
    tmp_path: Path,
) -> None:
    """Same storage, distinct experiment namespaces → no collision.

    A shared SQLite store can hold multiple independent experiments; locking
    them out of running concurrently would over-restrict the user.
    """
    storage = f"sqlite:///{tmp_path / 'shared.db'}"
    exp_a = make_experiment(workdir=str(tmp_path / "runs"), storage=storage)
    exp_b = make_experiment(workdir=str(tmp_path / "runs"), storage=storage)
    # make_experiment hardcodes experiment="t"; clone exp_b with another name.
    exp_b = exp_b.model_copy(update={"experiment": "other"})

    # Both output and storage lock identities differ — sets share no element.
    assert set(_run_lock_paths(exp_a)).isdisjoint(_run_lock_paths(exp_b))


def test_in_memory_run_lock_keyed_by_workdir(tmp_path: Path) -> None:
    """In-memory storage: only the output lock applies (no shared backend).
    Two same-workdir+experiment configs share the same output lock path.
    """
    exp_a = make_experiment(workdir=str(tmp_path / "runs"))
    exp_b = make_experiment(workdir=str(tmp_path / "runs"))

    paths_a = _run_lock_paths(exp_a)
    paths_b = _run_lock_paths(exp_b)
    # In-memory storage means only the output lock is taken — single path.
    assert len(paths_a) == 1
    assert paths_a == paths_b


def test_in_memory_run_lock_does_not_collide_for_different_workdirs(
    tmp_path: Path,
) -> None:
    """In-memory storage with different workdirs = independent output dirs."""
    exp_a = make_experiment(workdir=str(tmp_path / "runs_a"))
    exp_b = make_experiment(workdir=str(tmp_path / "runs_b"))

    assert set(_run_lock_paths(exp_a)).isdisjoint(_run_lock_paths(exp_b))


def test_run_experiment_holds_experiment_lock_for_duration(tmp_path: Path) -> None:
    """A second concurrent ``run_experiment`` against the same experiment
    fails fast with the expected error while the run lock is held.
    """
    storage = f"sqlite:///{tmp_path / 'shared.db'}"
    exp_a = make_experiment(
        workdir=str(tmp_path / "runs_a"),
        storage=storage,
        trial_command="true {overrides}",
        n_trials=1,
    )
    exp_b = make_experiment(
        workdir=str(tmp_path / "runs_b"),
        storage=storage,
        trial_command="true {overrides}",
        n_trials=1,
    )

    held = threading.Event()
    released = threading.Event()

    def hold() -> None:
        with _experiment_lock(exp_a):
            held.set()
            released.wait(timeout=5.0)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    assert held.wait(timeout=2.0)

    try:
        with pytest.raises(RuntimeError, match="Another phasesweep process"):
            run_experiment(exp_b, dry_run=False)
    finally:
        released.set()
        t.join(timeout=2.0)


def test_run_experiment_dry_run_does_not_take_experiment_lock(tmp_path: Path) -> None:
    """Dry-run is read-only: it must not require or take the experiment lock.
    A user inspecting an experiment's plan while a real run is in progress is
    a legitimate workflow.
    """
    storage = f"sqlite:///{tmp_path / 'shared.db'}"
    exp_a = make_experiment(workdir=str(tmp_path / "runs_a"), storage=storage)
    exp_b = make_experiment(workdir=str(tmp_path / "runs_b"), storage=storage)

    with _experiment_lock(exp_a):
        # Dry-run must succeed with the run lock held by another caller.
        winners = run_experiment(exp_b, dry_run=True)
        assert "p" in winners
