"""A dry run previews each phase from the trials its ledger already records.

A real run judges a phase by the trials its study holds: it refuses what its rules refuse, replays
a phase whose target is met, and launches what the target still lacks. The preview has to say the
same, so it reads the ledger through the write-nothing door and applies the run's own rules to
what it read. The integration tests compare its claims with what a real run then launches; the
fast ones build the recorded studies by hand and compare its refusals with a real run's.
"""

from __future__ import annotations

import logging
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config import Experiment, IntParam, LogRegexExtractor, Metric, Phase, Sampler
from phasesweep.engine import (
    NoFeasibleTrialError,
    SamplerContinuationUnsupportedError,
    TrialTargetRefusedError,
    WinnerIntegrityError,
)
from phasesweep.engine.fingerprints import _phase_fingerprint
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.state import (
    PHASE_DECISION_ATTR,
    PHASE_DECISION_SCHEMA_VERSION,
    PHASE_FINGERPRINT_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_ENV_NAMES_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_OUTCOME_SCHEMA_VERSION,
    TRIAL_TARGET_ATTR,
)
from phasesweep.engine.study_policy import _load_phase_policy_state
from phasesweep.engine.trial import _environment_identity
from tests.conftest import make_experiment, mark_current_format, write_trainer
from tests.ledger_fixtures import tree_changes, tree_snapshot
from tests.recovery_helpers import stamp_artifact_root

_STATE = optuna.trial.TrialState

_OUTCOME = {_STATE.COMPLETE: "success", _STATE.FAIL: "failure", _STATE.PRUNED: "pruned"}

_UNREADABLE = (
    "DRY RUN warning: phase={name} recorded trial counts could not be read, so it is previewed "
    "as a fresh phase and a real run may launch fewer trials"
)

_PLACEHOLDERS = "inherited values are placeholders until 'a' publishes a winner"


def _phase(
    name: str,
    n_trials: int,
    *,
    inherits: tuple[str, ...] = (),
    max_consecutive_failures: int = 5,
    sampler: Sampler | None = None,
    allow_incomplete_on_timeout: bool = False,
) -> Phase:
    """Build a phase that searches one integer of its own, by seeded random sampling by default."""
    return Phase(
        name=name,
        n_trials=n_trials,
        gpu_policy="none",
        inherits=list(inherits),
        max_consecutive_failures=max_consecutive_failures,
        allow_incomplete_on_timeout=allow_incomplete_on_timeout,
        sampler=Sampler(type="random", seed=0) if sampler is None else sampler,
        search_space={f"{name}_x": IntParam(type="int", low=0, high=10)},
    )


def _persistent(
    tmp_path: Path, *phases: Phase, trainer: Path | None = None, maximize: bool = False
) -> Experiment:
    """Build an experiment whose ledger lives beside its workdir under ``tmp_path``."""
    metric = Metric(
        goal="maximize" if maximize else "minimize",
        extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
    )
    return make_experiment(persistent=tmp_path, trainer=trainer, phases=list(phases), metric=metric)


def _study(experiment: Experiment, phase: str) -> optuna.Study:
    """Load one phase's live study from the experiment's ledger."""
    return optuna.load_study(
        study_name=f"{experiment.experiment}::{phase}",
        storage=_resolve_storage(experiment.resolved_storage),
    )


def _record(
    experiment: Experiment, phase: str, *states: optuna.trial.TrialState, target: int
) -> None:
    """Write a phase study as a prior run left it, one trial per state, with no trainer.

    It carries what a launch reads back: the format stamp, the artifact-root claim, the
    accepted target, the environment cohort, and a terminal outcome for every finished trial.
    Its COMPLETE trials hold no objective evidence, so selecting a winner from them fails.
    """
    study = optuna.create_study(
        study_name=f"{experiment.experiment}::{phase}",
        storage=_resolve_storage(experiment.resolved_storage),
        direction="minimize",
    )
    mark_current_format(experiment, study)
    stamp_artifact_root(study, experiment)
    study.set_user_attr(TRIAL_TARGET_ATTR, target)
    recorded_phase = next(item for item in experiment.phases if item.name == phase)
    study.set_user_attr(PHASE_FINGERPRINT_ATTR, _phase_fingerprint(experiment, recorded_phase, {}))
    identity = _environment_identity(experiment, phase)
    for sequence, state in enumerate(states, 1):
        attrs: dict[str, object] = {
            TRAINER_ENV_DIGEST_ATTR: identity.digest,
            TRAINER_ENV_NAMES_ATTR: list(identity.names),
        }
        if state != _STATE.RUNNING:
            attrs[TRIAL_OUTCOME_ATTR] = {
                "schema_version": TRIAL_OUTCOME_SCHEMA_VERSION,
                "sequence": sequence,
                "outcome": _OUTCOME[state],
                "cause": None,
            }
        value = 1.0 if state == _STATE.COMPLETE else None
        study.add_trial(optuna.trial.create_trial(state=state, value=value, user_attrs=attrs))


def _terminal_trials(experiment: Experiment, phase: str) -> int:
    """Count the terminal trials the ledger holds for one phase."""
    trials = _study(experiment, phase).get_trials(deepcopy=False)
    return sum(trial.state.is_finished() for trial in trials)


def _lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the dry run's per-phase count lines, launch sentences and read warnings."""
    return [
        message
        for message in (record.getMessage() for record in caplog.records)
        if message.startswith(("phase=", "DRY RUN phase=", "DRY RUN warning: phase="))
    ]


def _commands(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the dry run's example-command messages."""
    return [
        message
        for message in (record.getMessage() for record in caplog.records)
        if message.startswith("DRY RUN example command")
    ]


def _preview(experiment: Experiment, root: Path, caplog: pytest.LogCaptureFixture) -> list[str]:
    """Dry-run an experiment, assert it wrote nothing under ``root``, and return its lines."""
    before = tree_snapshot(root)
    caplog.clear()
    caplog.set_level(logging.INFO)
    run_experiment(experiment, dry_run=True)
    assert tree_changes(before, tree_snapshot(root)) == {}
    return _lines(caplog)


def _refusals_match(experiment: Experiment, root: Path, error: type[Exception], match: str) -> None:
    """Assert the dry run refuses an experiment with the error a real run then gives.

    The dry run must write nothing under ``root`` and must give the real run's message verbatim.
    """
    before = tree_snapshot(root)
    with pytest.raises(error, match=match) as dry:
        run_experiment(experiment, dry_run=True)
    assert tree_changes(before, tree_snapshot(root)) == {}
    with pytest.raises(error) as real:
        run_experiment(experiment)
    assert str(dry.value) == str(real.value)


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
def test_finished_phase_previews_a_replay_and_hands_its_child_the_real_winner(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A phase that finished launches nothing, and its child previews with the winner it replays.

    The metric is maximized so the winner is the largest draw, never the search space's low
    bound that a stand-in winner would carry.
    """
    first = _phase("a", 3)
    finished = run_experiment(_persistent(tmp_path, first, maximize=True))
    appended = _persistent(tmp_path, first, _phase("b", 2, inherits=("a",)), maximize=True)
    assert finished["a"].params["a_x"] != 0

    assert _preview(appended, tmp_path, caplog) == [
        "phase=a study=t::a completed=3 remaining=0 n_jobs=1",
        "DRY RUN phase=a target met (3 of 3 trials recorded): would replay its existing "
        "result and launch no trials",
        "phase=b study=t::b completed=0 remaining=2 n_jobs=1",
        "DRY RUN phase=b would launch 2 trials",
    ]
    (command,) = _commands(caplog)
    assert f"--a_x={finished['a'].params['a_x']}" in command
    assert "placeholders" not in command

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
def test_aborted_phase_is_refused_at_its_target_and_previews_a_raised_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failure abort at 2 of 4 refuses a rerun; a target raised to 6 previews the 4 it launches."""
    trainer, calls = _flaky_trainer(tmp_path, failures=2)

    def attempt(n_trials: int) -> Experiment:
        phase = _phase("a", n_trials, max_consecutive_failures=2)
        return _persistent(tmp_path, phase, trainer=trainer)

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(attempt(4))
    _refusals_match(attempt(4), tmp_path, NoFeasibleTrialError, "aborted")
    assert len(calls.read_text().splitlines()) == 2

    raised = attempt(6)
    assert _preview(raised, tmp_path, caplog) == [
        "phase=a study=t::a completed=2 remaining=4 n_jobs=1",
        "DRY RUN phase=a would launch 4 trials",
    ]
    assert len(calls.read_text().splitlines()) == 2

    run_experiment(raised)
    assert len(calls.read_text().splitlines()) == 6
    assert _terminal_trials(raised, "a") == 6


@pytest.mark.integration
def test_accepted_partial_timeout_previews_a_replay_and_launches_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A partial result a timeout accepted at the same target replays, as a real run replays it."""
    run_experiment(_persistent(tmp_path, _phase("a", 2)))
    study = _study(_persistent(tmp_path, _phase("a", 2)), "a")
    study.set_user_attr(TRIAL_TARGET_ATTR, 4)
    study.set_user_attr(
        PHASE_DECISION_ATTR,
        {
            "schema_version": PHASE_DECISION_SCHEMA_VERSION,
            "decision": "accepted_partial_timeout",
            "trial_target": 4,
            "outcome_sequence": _load_phase_policy_state(study).max_sequence,
            "finished_trials": 2,
            "completed_trials": 2,
            "timeout_scope": "phase",
            "recovered_abort_sequence": None,
        },
    )
    accepting = _persistent(tmp_path, _phase("a", 4, allow_incomplete_on_timeout=True))

    assert _preview(accepting, tmp_path, caplog) == [
        "phase=a study=t::a completed=2 remaining=2 n_jobs=1",
        "DRY RUN phase=a accepted a partial result at 2 of 4 trials: would replay it and "
        "launch no trials",
    ]
    assert _commands(caplog) == []

    run_experiment(accepting)
    assert _terminal_trials(accepting, "a") == 2

    refusing = _persistent(tmp_path, _phase("a", 4))
    _refusals_match(refusing, tmp_path, WinnerIntegrityError, "allow_incomplete_on_timeout: true")


def test_dry_run_counts_a_failed_attempt_as_spent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed attempt uses up one trial of the target, so a phase at 2 of 4 launches 2."""
    experiment = _persistent(tmp_path, _phase("a", 4))
    _record(experiment, "a", _STATE.COMPLETE, _STATE.FAIL, target=4)

    assert _preview(experiment, tmp_path, caplog) == [
        "phase=a study=t::a completed=2 remaining=2 n_jobs=1",
        "DRY RUN phase=a would launch 2 trials",
    ]


def test_one_remaining_trial_reads_in_the_singular(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The launch sentence says "1 trial", not "1 trials"."""
    experiment = _persistent(tmp_path, _phase("a", 3))
    _record(experiment, "a", _STATE.COMPLETE, _STATE.FAIL, target=3)

    assert _preview(experiment, tmp_path, caplog)[-1] == "DRY RUN phase=a would launch 1 trial"


def test_running_attempt_is_noted_and_its_phase_previews_from_counts_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A real run recovers a RUNNING attempt first, so the dry run counts it but checks nothing."""
    experiment = _persistent(tmp_path, _phase("a", 2), _phase("b", 2, inherits=("a",)))
    _record(experiment, "a", _STATE.COMPLETE, _STATE.RUNNING, target=2)

    assert _preview(experiment, tmp_path, caplog) == [
        "phase=a study=t::a completed=2 remaining=0 n_jobs=1",
        "DRY RUN warning: phase=a has 1 RUNNING attempt(s) that a real run recovers as failed "
        "before it checks the phase, so its recorded target, sampler and abort state are not "
        "previewed",
        "DRY RUN phase=a target met (2 of 2 trials recorded): would replay its existing "
        "result and launch no trials",
        "phase=b study=t::b completed=0 remaining=2 n_jobs=1",
        "DRY RUN phase=b would launch 2 trials",
    ]
    (command,) = _commands(caplog)
    assert _PLACEHOLDERS in command


def test_met_target_without_a_feasible_trial_is_refused_as_a_run_refuses(
    tmp_path: Path,
) -> None:
    """A phase that replays a result with no feasible trial gives the real run's refusal."""
    experiment = _persistent(tmp_path, _phase("a", 2))
    _record(experiment, "a", _STATE.FAIL, _STATE.FAIL, target=2)

    _refusals_match(experiment, tmp_path, NoFeasibleTrialError, "No feasible completed trials")


def test_lowered_target_is_refused_as_a_run_refuses(tmp_path: Path) -> None:
    """A target below the one the study accepted gives the real run's refusal."""
    experiment = _persistent(tmp_path, _phase("a", 2))
    _record(experiment, "a", _STATE.COMPLETE, _STATE.FAIL, target=4)

    _refusals_match(experiment, tmp_path, TrialTargetRefusedError, "already accepted a target of 4")


@pytest.mark.parametrize(
    ("target", "match"),
    [
        pytest.param(4, "interrupted at 2/4", id="interrupted-mid-target"),
        pytest.param(2, "raises its accepted target from 2 to 4", id="target-raised"),
    ],
)
def test_stateful_sampler_continuation_is_refused_as_a_run_refuses(
    tmp_path: Path, target: int, match: str
) -> None:
    """A tpe phase cut short, or raised past its target, gives the real run's refusal."""
    tpe = Sampler(type="tpe", seed=0, acknowledge_nonresumable=True)
    experiment = _persistent(tmp_path, _phase("a", 4, sampler=tpe))
    _record(experiment, "a", _STATE.COMPLETE, _STATE.FAIL, target=target)

    _refusals_match(experiment, tmp_path, SamplerContinuationUnsupportedError, match)


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
    """A bound tree whose ledger cannot be read keeps the fresh preview and the placeholder note."""
    experiment = _persistent(tmp_path, _phase("a", 2), _phase("b", 2, inherits=("a",)))
    _record(experiment, "a", _STATE.COMPLETE, _STATE.COMPLETE, target=2)
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
        "phase=b study=t::b completed=0 remaining=2 n_jobs=1",
        _UNREADABLE.format(name="b"),
        "DRY RUN phase=b would launch 2 trials",
    ]
    assert _PLACEHOLDERS in _commands(caplog)[-1]


def test_uninspectable_ledger_on_an_unbound_tree_previews_fresh_phases_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An inherited ledger with no tree to name its owner, and unreadable, is noted, not refused."""
    _record(_persistent(tmp_path, _phase("a", 2)), "a", _STATE.COMPLETE, _STATE.COMPLETE, target=2)
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
