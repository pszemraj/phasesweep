"""A dry run previews each phase from the trials its ledger already records.

A real run counts the terminal trials a phase's study holds: it launches what the target still
lacks and replays a phase that lacks none. The preview has to say the same, so it reads the ledger
through the write-nothing door. The integration tests compare its claims with what a real run then
launches; the fast ones build the recorded studies by hand.
"""

from __future__ import annotations

import logging
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config import Experiment, IntParam, Phase, Sampler
from phasesweep.engine import NoFeasibleTrialError
from phasesweep.engine.ledger import _resolve_storage
from tests.conftest import make_experiment, mark_current_format, write_trainer
from tests.ledger_fixtures import tree_changes, tree_snapshot

_STATE = optuna.trial.TrialState

_UNREADABLE = (
    "DRY RUN warning: phase={name} recorded trial counts could not be read, so it is previewed "
    "as a fresh phase and a real run may launch fewer trials"
)


def _phase(
    name: str,
    n_trials: int,
    *,
    inherits: tuple[str, ...] = (),
    max_consecutive_failures: int = 5,
) -> Phase:
    """Build a seeded random phase that searches one integer of its own."""
    return Phase(
        name=name,
        n_trials=n_trials,
        gpu_policy="none",
        inherits=list(inherits),
        max_consecutive_failures=max_consecutive_failures,
        sampler=Sampler(type="random", seed=0),
        search_space={f"{name}_x": IntParam(type="int", low=0, high=10)},
    )


def _persistent(tmp_path: Path, *phases: Phase, trainer: Path | None = None) -> Experiment:
    """Build an experiment whose ledger lives beside its workdir under ``tmp_path``."""
    return make_experiment(persistent=tmp_path, trainer=trainer, phases=list(phases))


def _record(experiment: Experiment, phase: str, *states: optuna.trial.TrialState) -> None:
    """Write a phase study holding one trial per state, as a prior run left it, with no trainer."""
    study = optuna.create_study(
        study_name=f"{experiment.experiment}::{phase}",
        storage=_resolve_storage(experiment.resolved_storage),
        direction="minimize",
    )
    mark_current_format(experiment, study)
    for state in states:
        value = 1.0 if state == _STATE.COMPLETE else None
        study.add_trial(optuna.trial.create_trial(state=state, value=value))


def _terminal_trials(experiment: Experiment, phase: str) -> int:
    """Count the terminal trials the ledger holds for one phase."""
    study = optuna.load_study(
        study_name=f"{experiment.experiment}::{phase}",
        storage=_resolve_storage(experiment.resolved_storage),
    )
    return sum(trial.state.is_finished() for trial in study.get_trials(deepcopy=False))


def _lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the dry run's per-phase count lines, launch sentences and read warnings."""
    return [
        message
        for message in (record.getMessage() for record in caplog.records)
        if message.startswith(("phase=", "DRY RUN phase=", "DRY RUN warning: phase="))
    ]


def _preview(experiment: Experiment, root: Path, caplog: pytest.LogCaptureFixture) -> list[str]:
    """Dry-run an experiment, assert it wrote nothing under ``root``, and return its lines."""
    before = tree_snapshot(root)
    caplog.clear()
    caplog.set_level(logging.INFO)
    run_experiment(experiment, dry_run=True)
    assert tree_changes(before, tree_snapshot(root)) == {}
    return _lines(caplog)


def _flaky_trainer(tmp_path: Path, *, failures: int) -> tuple[Path, Path]:
    """Write a trainer that fails its first ``failures`` calls, then logs ``x=1``.

    :param Path tmp_path: Directory holding the trainer and its call log.
    :param int failures: How many leading invocations exit nonzero.
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
        print("x=1")
        """,
    )
    return trainer, calls


@pytest.mark.integration
def test_finished_phase_previews_a_replay_and_the_appended_phase_its_trials(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A phase that finished and published launches nothing, as a real run then confirms."""
    first = _phase("a", 3)
    run_experiment(_persistent(tmp_path, first))
    appended = _persistent(tmp_path, first, _phase("b", 2, inherits=("a",)))

    assert _preview(appended, tmp_path, caplog) == [
        "phase=a study=t::a completed=3 remaining=0 n_jobs=1",
        "DRY RUN phase=a target met (3 of 3 trials recorded): would replay its existing "
        "result and launch no trials",
        "phase=b study=t::b completed=0 remaining=2 n_jobs=1",
        "DRY RUN phase=b would launch 2 trials",
    ]

    run_experiment(appended)
    assert _terminal_trials(appended, "a") == 3
    assert _terminal_trials(appended, "b") == 2


@pytest.mark.integration
def test_raised_target_previews_only_the_trials_it_adds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A phase stopped at 2 of 4 previews the 2 a real run launches, not all 4."""
    run_experiment(_persistent(tmp_path, _phase("a", 2)))
    raised = _persistent(tmp_path, _phase("a", 4))

    assert _preview(raised, tmp_path, caplog) == [
        "phase=a study=t::a completed=2 remaining=2 n_jobs=1",
        "DRY RUN phase=a would launch 2 trials",
    ]

    run_experiment(raised)
    assert _terminal_trials(raised, "a") == 4


@pytest.mark.integration
def test_aborted_phase_with_a_raised_target_previews_the_remaining_trials(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failure abort at 2 of 4 and a target raised to 6 previews the 4 a real run launches."""
    trainer, calls = _flaky_trainer(tmp_path, failures=2)

    def attempt(n_trials: int) -> Experiment:
        phase = _phase("a", n_trials, max_consecutive_failures=2)
        return _persistent(tmp_path, phase, trainer=trainer)

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(attempt(4))
    raised = attempt(6)

    assert _preview(raised, tmp_path, caplog) == [
        "phase=a study=t::a completed=2 remaining=4 n_jobs=1",
        "DRY RUN phase=a would launch 4 trials",
    ]
    assert len(calls.read_text().splitlines()) == 2

    run_experiment(raised)
    assert len(calls.read_text().splitlines()) == 6
    assert _terminal_trials(raised, "a") == 6


@pytest.mark.parametrize(
    ("states", "n_trials", "counts", "sentence"),
    [
        pytest.param(
            (_STATE.COMPLETE, _STATE.COMPLETE),
            2,
            "completed=2 remaining=0",
            "target met (2 of 2 trials recorded): would replay its existing result and "
            "launch no trials",
            id="met",
        ),
        pytest.param(
            (_STATE.COMPLETE, _STATE.FAIL),
            4,
            "completed=2 remaining=2",
            "would launch 2 trials",
            id="failed-attempt-is-spent",
        ),
        pytest.param(
            (_STATE.PRUNED, _STATE.FAIL, _STATE.RUNNING),
            4,
            "completed=3 remaining=1",
            "would launch 1 trials",
            id="stale-running-attempt-is-recovered-as-failed",
        ),
    ],
)
def test_dry_run_counts_the_terminal_trials_a_launch_would(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    states: tuple[optuna.trial.TrialState, ...],
    n_trials: int,
    counts: str,
    sentence: str,
) -> None:
    """Every terminal state, and a stale RUNNING attempt, uses up one trial of the target."""
    experiment = _persistent(tmp_path, _phase("a", n_trials))
    _record(experiment, "a", *states)

    assert _preview(experiment, tmp_path, caplog) == [
        f"phase=a study=t::a {counts} n_jobs=1",
        f"DRY RUN phase=a {sentence}",
    ]


@pytest.mark.parametrize("persistent", [True, False], ids=["persistent", "in-memory"])
def test_fresh_experiment_previews_every_phase_as_before(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, persistent: bool
) -> None:
    """With nothing recorded, the dry run reports every trial as one to launch and writes nothing."""
    phases = (_phase("a", 3), _phase("b", 2, inherits=("a",)))
    experiment = (
        _persistent(tmp_path, *phases)
        if persistent
        else make_experiment(workdir=tmp_path / "runs", phases=list(phases))
    )

    assert _preview(experiment, tmp_path, caplog) == [
        "phase=a study=t::a completed=0 remaining=3 n_jobs=1",
        "DRY RUN phase=a would launch 3 trials",
        "phase=b study=t::b completed=0 remaining=2 n_jobs=1",
        "DRY RUN phase=b would launch 2 trials",
    ]
    assert not (tmp_path / "studies.journal").exists()
    assert not (tmp_path / "runs").exists()


def test_unreadable_trial_data_previews_a_fresh_phase_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A bound tree whose ledger cannot be read keeps the fresh preview and notes the gap."""
    experiment = _persistent(tmp_path, _phase("a", 2))
    _record(experiment, "a", _STATE.COMPLETE, _STATE.COMPLETE)
    ledger = tmp_path / "studies.journal"
    mode = ledger.stat().st_mode
    caplog.set_level(logging.INFO)
    ledger.chmod(0)
    try:
        run_experiment(experiment, dry_run=True)
    finally:
        ledger.chmod(mode)

    assert _lines(caplog) == [
        "phase=a study=t::a completed=0 remaining=2 n_jobs=1",
        _UNREADABLE.format(name="a"),
        "DRY RUN phase=a would launch 2 trials",
    ]


def test_uninspectable_ledger_on_an_unbound_tree_previews_fresh_phases_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An inherited ledger with no tree to name its owner, and unreadable, is noted, not refused."""
    _record(_persistent(tmp_path, _phase("a", 2)), "a", _STATE.COMPLETE, _STATE.COMPLETE)
    ledger = tmp_path / "studies.journal"
    inheriting = make_experiment(
        storage=f"journal:///{ledger}", workdir=tmp_path / "elsewhere", phases=[_phase("a", 2)]
    )
    mode = ledger.stat().st_mode
    caplog.set_level(logging.INFO)
    ledger.chmod(0)
    try:
        run_experiment(inheriting, dry_run=True)
    finally:
        ledger.chmod(mode)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert warnings[0].startswith("DRY RUN warning: the ledger could not be inspected: ")
    assert warnings[1] == _UNREADABLE.format(name="a")
    assert _lines(caplog)[-1] == "DRY RUN phase=a would launch 2 trials"
    assert not (tmp_path / "elsewhere").exists()
