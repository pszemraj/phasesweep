"""Same-host advisory locks: phase lock, experiment-level run lock, output namespace lock, and storage-identity lock. Two orchestrators against the same on-disk state must collide; against unrelated state they must not."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from phasesweep.config import (
    Experiment,
    FloatParam,
    IntParam,
    JsonExtractor,
    Metric,
    Phase,
)
from phasesweep.orchestrator import (
    _experiment_lock,
    _phase_lock,
    _phase_lock_path,
    _run_lock_paths,
    run_experiment,
)
from tests.conftest import make_experiment


def test_phase_lock_blocks_second_acquirer(tmp_path: Path) -> None:
    """Two processes targeting the same experiment::phase fail fast on the second."""
    exp = make_experiment(workdir=tmp_path / "wd")
    phase = exp.phases[0]

    held = threading.Event()
    released = threading.Event()
    second_error: list[BaseException] = []

    def hold_first() -> None:
        with _phase_lock(exp, phase):
            held.set()
            released.wait(timeout=5.0)

    t = threading.Thread(target=hold_first, daemon=True)
    t.start()
    assert held.wait(timeout=2.0)

    try:
        with _phase_lock(exp, phase):
            pytest.fail("second lock acquisition should have raised")
    except RuntimeError as exc:  # noqa: BLE001
        second_error.append(exc)

    released.set()
    t.join(timeout=2.0)

    assert len(second_error) == 1
    assert "Another phasesweep process" in str(second_error[0])


def test_phase_lock_distinct_phases_dont_collide(tmp_path: Path) -> None:
    """Lock is per-phase, not per-experiment — different phases can coexist."""
    base: dict[str, object] = dict(
        n_trials=1,
        search_space={"x": IntParam(type="int", low=0, high=10)},
    )
    exp = Experiment(
        experiment="t",
        workdir=str(tmp_path / "wd"),
        trial_command="echo {overrides}",
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
        phases=[
            Phase(name="a", **base),  # type: ignore[arg-type]
            Phase(name="b", **base),  # type: ignore[arg-type]
        ],
    )

    # Different phase, same experiment, same host — must succeed.
    with _phase_lock(exp, exp.phases[0]), _phase_lock(exp, exp.phases[1]):
        pass


@pytest.mark.parametrize("storage_name", ["shared.db", "shared.journal"])
def test_phase_lock_collides_for_same_storage_different_workdirs(
    tmp_path: Path,
    storage_name: str,
) -> None:
    """Two configs sharing storage but different workdirs must collide."""
    scheme = "journal" if storage_name.endswith(".journal") else "sqlite"
    storage = f"{scheme}:///{tmp_path / storage_name}"
    exp_a = make_experiment(workdir=str(tmp_path / "runs_a"), storage=storage)
    exp_b = make_experiment(workdir=str(tmp_path / "runs_b"), storage=storage)

    held = threading.Event()
    released = threading.Event()

    def hold_first() -> None:
        with _phase_lock(exp_a, exp_a.phases[0]):
            held.set()
            released.wait(timeout=5.0)

    t = threading.Thread(target=hold_first, daemon=True)
    t.start()
    assert held.wait(timeout=2.0)

    try:
        with _phase_lock(exp_b, exp_b.phases[0]):
            pytest.fail("second lock should have been blocked")
    except RuntimeError as exc:
        assert "Another phasesweep process" in str(exc)

    released.set()
    t.join(timeout=2.0)


def test_phase_lock_does_not_collide_for_different_storage(
    tmp_path: Path,
) -> None:
    """Different storage backends = different studies = no lock collision."""
    exp_a = make_experiment(
        workdir=str(tmp_path / "runs"), storage=f"sqlite:///{tmp_path / 'a.db'}"
    )
    exp_b = make_experiment(
        workdir=str(tmp_path / "runs"), storage=f"sqlite:///{tmp_path / 'b.db'}"
    )

    with _phase_lock(exp_a, exp_a.phases[0]), _phase_lock(exp_b, exp_b.phases[0]):
        pass  # must not raise


def test_in_memory_lock_collides_on_same_workdir(tmp_path: Path) -> None:
    """In-memory storage: no shared study, so lock falls back to workdir+study."""
    exp_a = make_experiment(workdir=str(tmp_path / "runs"))
    exp_b = make_experiment(workdir=str(tmp_path / "runs"))

    path_a = _phase_lock_path(exp_a, exp_a.phases[0])
    path_b = _phase_lock_path(exp_b, exp_b.phases[0])
    assert path_a == path_b


def test_in_memory_lock_does_not_collide_on_different_workdir(
    tmp_path: Path,
) -> None:
    """In-memory storage with different workdirs = separate studies."""
    exp_a = make_experiment(workdir=str(tmp_path / "runs_a"))
    exp_b = make_experiment(workdir=str(tmp_path / "runs_b"))

    path_a = _phase_lock_path(exp_a, exp_a.phases[0])
    path_b = _phase_lock_path(exp_b, exp_b.phases[0])
    assert path_a != path_b


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


def test_run_lock_paths_differ_from_phase_lock_path(tmp_path: Path) -> None:
    """Run lock(s) and phase lock must be distinct files for the same
    experiment, so a phase lock held internally never blocks an outer run lock
    of the same identity (and vice versa).
    """
    storage = f"sqlite:///{tmp_path / 'shared.db'}"
    exp = make_experiment(workdir=str(tmp_path / "runs"), storage=storage)
    assert _phase_lock_path(exp, exp.phases[0]) not in _run_lock_paths(exp)


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


def test_phase_lock_still_works_under_held_run_lock(tmp_path: Path) -> None:
    """The two locks have distinct files, so holding the run lock for an
    experiment must not block a phase lock for the same experiment.
    """
    storage = f"sqlite:///{tmp_path / 'shared.db'}"
    exp = make_experiment(workdir=str(tmp_path / "runs"), storage=storage)
    with _experiment_lock(exp), _phase_lock(exp, exp.phases[0]):
        pass  # must not raise
