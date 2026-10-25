"""A dry run previews each phase from the trials its ledger already records.

A real run judges a phase by the trials its study holds: it refuses what its rules refuse, replays
a phase whose target is met, and launches what the target still lacks. The preview has to say the
same, so it reads the ledger through the write-nothing door and applies the run's own rules to
what it read, in the run's order.
"""

from __future__ import annotations

import logging
from pathlib import Path

import optuna
import pytest

import phasesweep.engine.ledger as ledger_module
from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.errors import (
    PublicationIntegrityError,
    PublishedStudyMissingError,
    ResumeWinnerMissingError,
    StudyContextConflictError,
    StudySchemaMismatchError,
    TrialEvidenceMissingError,
    TrialTargetRefusedError,
    WinnerIntegrityError,
)
from phasesweep.engine.fingerprints import _phase_fingerprint
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.paths import _last_successful_generation_path, _trial_dir_for
from phasesweep.engine.preview import _placeholder_winner
from phasesweep.engine.selection import NoFeasibleTrialError, PhaseAbortRefusedError
from phasesweep.engine.state import (
    PHASE_FINGERPRINT_ATTR,
    TRAINER_ENV_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_TARGET_ATTR,
    Winner,
)
from phasesweep.engine.trial import _environment_identity
from tests.conftest import (
    bounded_metric,
    make_experiment,
    phase_study,
    phase_trials,
    record_accepted_partial_timeout,
    requires_nonroot,
    trial_outcome,
    write_flaky_trainer,
)
from tests.ledger_fixtures import forbid_file_backed_storage, tree_changes, tree_snapshot
from tests.recovery_helpers import new_phase_study, stamp_artifact_root

_STATE = optuna.trial.TrialState

_OUTCOME = {_STATE.COMPLETE: "success", _STATE.FAIL: "failure"}

_UNREADABLE = (
    "DRY RUN warning: phase={name} recorded trials could not be read, so it is previewed as a "
    "fresh phase; a real run refuses a ledger it cannot read"
)

_LEDGER_REFUSED = (
    "DRY RUN warning: the ledger could not be inspected, and a real run refuses a ledger it "
    "cannot read: "
)

_PLACEHOLDERS = "placeholders until 'a' publishes a winner: a_x"

_RUNNING_WARNING = (
    "DRY RUN warning: phase={name} has 1 RUNNING attempt(s) that a real run recovers before it "
    "checks the phase, as failed where the trial recorded a failure and otherwise as "
    "interrupted, which reruns; they are counted that way, and its accepted target, abort "
    "state and partial decision are not checked"
)


def _phase(
    name: str,
    n_trials: int,
    *,
    inherits: tuple[str, ...] = (),
    max_consecutive_failures: int = 5,
    sampler: Sampler | None = None,
    allow_incomplete_on_timeout: bool = False,
    fixed: bool = False,
) -> Phase:
    """Build a phase that searches one integer of its own, by seeded random sampling by default.

    With ``fixed`` the search space is empty, which makes it a fixed configuration.
    """
    return Phase(
        name=name,
        n_trials=n_trials,
        gpu_policy="none",
        inherits=list(inherits),
        max_consecutive_failures=max_consecutive_failures,
        allow_incomplete_on_timeout=allow_incomplete_on_timeout,
        sampler=Sampler(type="random", seed=0) if sampler is None else sampler,
        search_space={} if fixed else {f"{name}_x": IntParam(type="int", low=0, high=10)},
    )


def _persistent(
    tmp_path: Path, *phases: Phase, trainer: Path | None = None, maximize: bool = False
) -> Experiment:
    """Build an experiment whose ledger lives beside its workdir under ``tmp_path``."""
    metric = bounded_metric().model_copy(update={"goal": "maximize" if maximize else "minimize"})
    return make_experiment(persistent=tmp_path, trainer=trainer, phases=list(phases), metric=metric)


def _stand_ins(experiment: Experiment, phase: Phase) -> dict[str, Winner]:
    """Build stand-in winners for every phase ``phase`` inherits, so it can be fingerprinted."""
    winners: dict[str, Winner] = {}
    for parent_name in phase.inherits:
        parent = next(item for item in experiment.phases if item.name == parent_name)
        winners[parent_name] = _placeholder_winner(
            experiment, parent, _stand_ins(experiment, parent)
        )
    return winners


def _record(
    experiment: Experiment,
    phase: str,
    *states: optuna.trial.TrialState,
    target: int,
    outcomes: bool = True,
    running_outcome: str | None = None,
    fail_outcome: str = "failure",
) -> None:
    """Write a phase study as a prior run left it, one trial per state, with no trainer.

    It carries what a launch reads back: the format stamp, the artifact-root claim, the
    accepted target, the environment cohort, and a terminal outcome for every finished trial
    (none with ``outcomes`` off, which is malformed failure-policy state). A FAIL trial
    records ``fail_outcome``, and a RUNNING trial carries ``running_outcome`` when given,
    as an objective that recorded its outcome just before its orchestrator died. Its
    COMPLETE trials hold no objective evidence, so selecting a winner from them fails.
    """
    study = new_phase_study(experiment, phase)
    stamp_artifact_root(study, experiment)
    study.set_user_attr(TRIAL_TARGET_ATTR, target)
    recorded_phase = next(item for item in experiment.phases if item.name == phase)
    study.set_user_attr(
        PHASE_FINGERPRINT_ATTR,
        _phase_fingerprint(experiment, recorded_phase, _stand_ins(experiment, recorded_phase)),
    )
    identity = _environment_identity(experiment, phase)
    for sequence, state in enumerate(states, 1):
        attrs: dict[str, object] = {
            TRAINER_ENV_DIGEST_ATTR: identity.digest,
            TRAINER_ENV_ATTR: dict(identity.tracked),
        }
        outcome = {_STATE.RUNNING: running_outcome, _STATE.FAIL: fail_outcome}.get(
            state, _OUTCOME.get(state)
        )
        if outcomes and outcome is not None:
            attrs[TRIAL_OUTCOME_ATTR] = trial_outcome(sequence, outcome, cause=None)
        value = 1.0 if state == _STATE.COMPLETE else None
        study.add_trial(optuna.trial.create_trial(state=state, value=value, user_attrs=attrs))


def _terminal_trials(experiment: Experiment, phase: str) -> int:
    """Count the terminal trials the ledger holds for one phase."""
    return sum(trial.state.is_finished() for trial in phase_trials(experiment, phase))


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


def _refusals_match(
    experiment: Experiment,
    root: Path,
    error: type[Exception],
    match: str,
    *,
    from_phase: str | None = None,
) -> None:
    """Assert the dry run refuses an experiment with the error a real run then gives.

    The dry run must write nothing under ``root`` and must give the real run's message verbatim.
    """
    before = tree_snapshot(root)
    with pytest.raises(error, match=match) as dry:
        run_experiment(experiment, from_phase=from_phase, dry_run=True)
    assert tree_changes(before, tree_snapshot(root)) == {}
    with pytest.raises(error) as real:
        run_experiment(experiment, from_phase=from_phase)
    assert str(dry.value) == str(real.value)


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
        "phase=a study=t::a finished=3 remaining=0 reruns=0 n_jobs=1",
        "DRY RUN phase=a target met (3 of 3 trials recorded): would replay its existing "
        "result and launch no trials",
        "phase=b study=t::b finished=0 remaining=2 reruns=0 n_jobs=1",
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
        "phase=a study=t::a finished=2 remaining=2 reruns=0 n_jobs=1",
        "DRY RUN phase=a would launch 2 trials",
    ]

    run_experiment(raised)
    assert _terminal_trials(raised, "a") == 4


@pytest.mark.integration
def test_aborted_phase_is_refused_at_its_target_and_previews_a_raised_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failure abort at 2 of 4 refuses a rerun; a target raised to 6 previews the 4 it launches."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=2, value=1)

    def attempt(n_trials: int) -> Experiment:
        phase = _phase("a", n_trials, max_consecutive_failures=2)
        return _persistent(tmp_path, phase, trainer=trainer)

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(attempt(4))
    _refusals_match(attempt(4), tmp_path, PhaseAbortRefusedError, "aborted")
    assert len(calls.read_text().splitlines()) == 2

    raised = attempt(6)
    assert _preview(raised, tmp_path, caplog) == [
        "phase=a study=t::a finished=2 remaining=4 reruns=0 n_jobs=1",
        "DRY RUN phase=a would launch 4 trials",
    ]
    assert len(calls.read_text().splitlines()) == 2

    run_experiment(raised)
    assert len(calls.read_text().splitlines()) == 6
    assert _terminal_trials(raised, "a") == 6


def _accept_partial_timeout(tmp_path: Path) -> tuple[Experiment, Winner]:
    """Run a phase to 2 trials, then record that a timeout accepted that partial result at 4.

    :return tuple[Experiment, Winner]: The experiment that accepts the partial result, and the
        winner the first run selected from its two trials.
    """
    winner = run_experiment(_persistent(tmp_path, _phase("a", 2)))["a"]
    study = phase_study(_persistent(tmp_path, _phase("a", 2)), "a")
    record_accepted_partial_timeout(study, 4)
    return _persistent(tmp_path, _phase("a", 4, allow_incomplete_on_timeout=True)), winner


@pytest.mark.integration
def test_accepted_partial_timeout_previews_a_replay_and_launches_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A partial result a timeout accepted at the same target replays, as a real run replays it.

    Like a run, the preview prints no header with trials ``remaining`` for a phase that replays.
    """
    accepting, _winner = _accept_partial_timeout(tmp_path)

    assert _preview(accepting, tmp_path, caplog) == [
        "DRY RUN phase=a accepted a partial result at 2 of 4 trials: would replay it and "
        "launch no trials",
    ]
    assert _commands(caplog) == []

    run_experiment(accepting)
    assert _terminal_trials(accepting, "a") == 2

    refusing = _persistent(tmp_path, _phase("a", 4))
    _refusals_match(refusing, tmp_path, WinnerIntegrityError, "allow_incomplete_on_timeout: true")


@pytest.mark.integration
def test_accepted_partial_replay_proves_the_winners_evidence_but_the_preview_leaves_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A live replay of an accepted partial result re-proves its winner's bytes; the preview does not.

    A same-size edit of the winning trial's objective source passes every check but the
    selection digest, so only a replay that verifies evidence refuses it. The preview reads no
    trial directory, so it still previews the replay.
    """
    accepting, winner = _accept_partial_timeout(tmp_path)
    assert winner.source is not None
    source = (
        _trial_dir_for(
            accepting,
            "a",
            winner.source.trial_number,
            generation_id=winner.source.generation_id,
            attempt_id=winner.source.attempt_id,
        )
        / "stdout.log"
    )
    original = source.read_bytes()
    source.write_bytes(b"y" + original[1:])
    assert source.stat().st_size == len(original)

    assert _preview(accepting, tmp_path, caplog)[-1] == (
        "DRY RUN phase=a accepted a partial result at 2 of 4 trials: would replay it and "
        "launch no trials"
    )
    with pytest.raises(TrialEvidenceMissingError, match="bytes behind the published metric"):
        run_experiment(accepting)


@pytest.mark.parametrize(
    ("target", "remaining", "expected_launch"),
    [
        pytest.param(4, 2, "DRY RUN phase=a would launch 2 trials", id="plural-two-remaining"),
        pytest.param(3, 1, "DRY RUN phase=a would launch 1 trial", id="singular-one-remaining"),
    ],
)
def test_dry_run_counts_a_failed_attempt_as_spent(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    target: int,
    remaining: int,
    expected_launch: str,
) -> None:
    """Failed attempts consume the target; the remaining-work sentence uses matching grammar."""
    experiment = _persistent(tmp_path, _phase("a", target))
    _record(experiment, "a", _STATE.COMPLETE, _STATE.FAIL, target=target)

    assert _preview(experiment, tmp_path, caplog) == [
        f"phase=a study=t::a finished=2 remaining={remaining} reruns=0 n_jobs=1",
        expected_launch,
    ]


def test_running_attempt_is_noted_and_its_phase_previews_from_counts_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A real run recovers a RUNNING attempt first, so the dry run counts it as recovery will and says so."""
    experiment = _persistent(tmp_path, _phase("a", 2), _phase("b", 2, inherits=("a",)))
    _record(experiment, "a", _STATE.COMPLETE, _STATE.RUNNING, target=2, running_outcome="failure")

    assert _preview(experiment, tmp_path, caplog) == [
        "phase=a study=t::a finished=2 remaining=0 reruns=0 n_jobs=1",
        _RUNNING_WARNING.format(name="a"),
        "DRY RUN phase=a target met (2 of 2 trials recorded): would replay its existing "
        "result and launch no trials",
        "phase=b study=t::b finished=0 remaining=2 reruns=0 n_jobs=1",
        "DRY RUN phase=b would launch 2 trials",
    ]
    (command,) = _commands(caplog)
    assert _PLACEHOLDERS in command


@pytest.mark.parametrize(
    ("recorded_state", "target", "expected_lines"),
    [
        pytest.param(
            _STATE.RUNNING,
            2,
            [
                "phase=a study=t::a finished=1 remaining=1 reruns=1 n_jobs=1",
                _RUNNING_WARNING.format(name="a"),
                "DRY RUN phase=a would launch 1 trial, 1 of them a rerun",
            ],
            id="running-without-outcome",
        ),
        pytest.param(
            _STATE.FAIL,
            3,
            [
                "phase=a study=t::a finished=1 remaining=2 reruns=1 n_jobs=1",
                "DRY RUN phase=a would launch 2 trials, 1 of them a rerun",
            ],
            id="interrupted",
        ),
    ],
)
def test_interrupted_attempts_preview_as_reruns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    recorded_state: optuna.trial.TrialState,
    target: int,
    expected_lines: list[str],
) -> None:
    """Unfinished or interrupted attempts are reruns rather than completed budget."""
    experiment = _persistent(tmp_path, _phase("a", target))
    _record(
        experiment, "a", _STATE.COMPLETE, recorded_state, target=target, fail_outcome="interrupted"
    )

    assert _preview(experiment, tmp_path, caplog) == expected_lines


def test_total_line_sums_the_launches_the_phase_lines_print(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The total counts launching and replaying phases, leaves skipped ones out, and a refusal prints none."""
    experiment = _persistent(
        tmp_path, _phase("a", 2), _phase("b", 2, inherits=("a",)), _phase("c", 3, inherits=("b",))
    )
    _record(experiment, "a", _STATE.COMPLETE, _STATE.RUNNING, target=2)

    def totals() -> list[str]:
        return [
            m for m in (r.getMessage() for r in caplog.records) if m.startswith("DRY RUN total")
        ]

    _preview(experiment, tmp_path, caplog)
    assert totals() == ["DRY RUN total: would launch 6 trials (1 rerun) across 3 phases"]

    caplog.clear()
    with pytest.raises(ResumeWinnerMissingError, match="Winner file missing for phase 'a'"):
        run_experiment(experiment, from_phase="b", dry_run=True)
    assert totals() == []

    caplog.clear()
    lowered = _persistent(tmp_path, _phase("d", 2))
    _record(lowered, "d", _STATE.COMPLETE, _STATE.FAIL, target=4)
    with pytest.raises(TrialTargetRefusedError):
        run_experiment(lowered, dry_run=True)
    assert totals() == []


def test_running_attempt_counts_as_failed_toward_a_fixed_configurations_retry_limit(
    tmp_path: Path,
) -> None:
    """Over a RUNNING attempt that recorded a failure, a fixed configuration may reach two attempts, never three."""
    allowed = _persistent(tmp_path, _phase("a", 2, fixed=True))
    _record(allowed, "a", _STATE.RUNNING, target=1, running_outcome="failure")
    run_experiment(allowed, dry_run=True)

    with pytest.raises(TrialTargetRefusedError, match="1 failed attempt") as refused:
        run_experiment(_persistent(tmp_path, _phase("a", 3, fixed=True)), dry_run=True)
    assert refused.value.remediation == "Use n_trials at most 2, not 3."


def test_running_attempt_that_will_rerun_leaves_a_fixed_configuration_at_one_attempt(
    tmp_path: Path,
) -> None:
    """A RUNNING attempt that recorded no failure reruns, so a fixed configuration stays at n_trials 1."""
    experiment = _persistent(tmp_path, _phase("a", 1, fixed=True))
    _record(experiment, "a", _STATE.RUNNING, target=1)
    run_experiment(experiment, dry_run=True)

    with pytest.raises(TrialTargetRefusedError, match="would run 2 identical trials"):
        run_experiment(_persistent(tmp_path, _phase("a", 2, fixed=True)), dry_run=True)


def test_met_target_without_a_feasible_trial_is_refused_as_a_run_refuses(
    tmp_path: Path,
) -> None:
    """A phase that replays a result with no feasible trial gives the real run's refusal."""
    experiment = _persistent(tmp_path, _phase("a", 2))
    _record(experiment, "a", _STATE.FAIL, _STATE.FAIL, target=2)

    _refusals_match(experiment, tmp_path, NoFeasibleTrialError, "No completed trials")


@pytest.mark.parametrize(
    "case", ["populated-target", "first-refusal-before-fixed-phase", "target-only-empty-study"]
)
def test_lowered_target_is_refused_as_a_run_refuses(tmp_path: Path, case: str) -> None:
    """An accepted target binds empty and populated studies, before later phase refusals."""
    phases = [_phase("a", 2)]
    if case == "first-refusal-before-fixed-phase":
        phases.append(_phase("b", 3, fixed=True))
    experiment = _persistent(tmp_path, *phases)
    states = () if case == "target-only-empty-study" else (_STATE.COMPLETE, _STATE.FAIL)
    _record(experiment, "a", *states, target=4)

    _refusals_match(
        experiment,
        tmp_path,
        TrialTargetRefusedError,
        r"already accepted a target of 4 .* requests 2\. Set phases\[a\]\.n_trials to at least 4 "
        "in the config",
    )


@pytest.mark.parametrize("target", [4, 2], ids=["failed-mid-target", "target-raised"])
def test_tpe_phase_cut_short_or_raised_previews_its_remaining_trials(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, target: int
) -> None:
    """A tpe phase cut short, or raised past its target, previews the trials it still needs."""
    experiment = _persistent(tmp_path, _phase("a", 4, sampler=Sampler(type="tpe", seed=0)))
    _record(experiment, "a", _STATE.COMPLETE, _STATE.FAIL, target=target)

    assert _preview(experiment, tmp_path, caplog)[-1] == "DRY RUN phase=a would launch 2 trials"


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
        "phase=a study=t::a finished=0 remaining=3 reruns=0 n_jobs=1",
        "DRY RUN phase=a would launch 3 trials",
        "phase=b study=t::b finished=0 remaining=2 reruns=0 n_jobs=1",
        "DRY RUN phase=b would launch 2 trials",
    ]
    assert not (tmp_path / "studies.journal").exists()
    assert not (tmp_path / "runs").exists()


@requires_nonroot
def test_unreadable_trial_data_previews_a_fresh_phase_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound tree whose ledger cannot be read keeps the fresh preview, writes nothing, and says so."""
    experiment = _persistent(tmp_path, _phase("a", 2), _phase("b", 2, inherits=("a",)))
    _record(experiment, "a", _STATE.COMPLETE, _STATE.COMPLETE, target=2)
    ledger = tmp_path / "studies.journal"
    mode = ledger.stat().st_mode
    before = tree_snapshot(tmp_path)
    attempts = forbid_file_backed_storage(monkeypatch, allow_memory_studies=True)
    caplog.set_level(logging.INFO)
    ledger.chmod(0)
    try:
        run_experiment(experiment, dry_run=True)
    finally:
        ledger.chmod(mode)

    assert attempts == []
    assert tree_changes(before, tree_snapshot(tmp_path)) == {}
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings[0].startswith(_LEDGER_REFUSED)
    assert _lines(caplog) == [
        "phase=a study=t::a finished=0 remaining=2 reruns=0 n_jobs=1",
        _UNREADABLE.format(name="a"),
        "DRY RUN phase=a would launch 2 trials",
        "phase=b study=t::b finished=0 remaining=2 reruns=0 n_jobs=1",
        _UNREADABLE.format(name="b"),
        "DRY RUN phase=b would launch 2 trials",
    ]
    assert _PLACEHOLDERS in _commands(caplog)[-1]


@requires_nonroot
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
    assert warnings[0].startswith(_LEDGER_REFUSED)
    assert warnings[1] == _UNREADABLE.format(name="a")
    assert _lines(caplog)[-1] == "DRY RUN phase=a would launch 2 trials"
    assert not (tmp_path / "elsewhere").exists()


def test_raised_upstream_target_over_a_bound_descendant_is_refused_as_a_run_refuses(
    tmp_path: Path,
) -> None:
    """Raising a finished phase's target under studies bound to its winner is refused, not previewed."""
    recorded = _persistent(
        tmp_path,
        _phase("a", 2),
        _phase("b", 2, inherits=("a",)),
        _phase("c", 2, inherits=("b",)),
    )
    for name in ("a", "b", "c"):
        _record(recorded, name, _STATE.FAIL, _STATE.FAIL, target=2)
    raised = _persistent(
        tmp_path,
        _phase("a", 3),
        _phase("b", 2, inherits=("a",)),
        _phase("c", 2, inherits=("b",)),
    )

    _refusals_match(
        raised,
        tmp_path,
        StudyContextConflictError,
        r"dependent phase study/studies \['b', 'c'\] are already bound",
    )


@pytest.mark.integration
@pytest.mark.parametrize("damage", ["ledger-deleted", "pointer-corrupted"])
def test_publication_the_ledger_cannot_back_is_refused_as_a_run_refuses(
    tmp_path: Path, damage: str
) -> None:
    """A deleted ledger, or an unreadable publication pointer, is refused before any phase previews."""
    experiment = _persistent(tmp_path, _phase("a", 2))
    run_experiment(experiment)
    error: type[Exception]
    if damage == "ledger-deleted":
        (tmp_path / "studies.journal").unlink()
        error, match = PublishedStudyMissingError, "its persistent study is missing"
    else:
        _last_successful_generation_path(experiment).write_text("not: [a valid pointer")
        error, match = PublicationIntegrityError, "last-success pointer"

    _refusals_match(experiment, tmp_path, error, match)


def test_several_refused_studies_are_reported_together_as_a_run_reports_them(
    tmp_path: Path,
) -> None:
    """Two lowered targets come back as the one combined refusal a run gives, naming both phases."""
    experiment = _persistent(tmp_path, _phase("a", 2), _phase("b", 2))
    for name in ("a", "b"):
        _record(experiment, name, _STATE.COMPLETE, _STATE.FAIL, target=4)

    _refusals_match(
        experiment,
        tmp_path,
        TrialTargetRefusedError,
        r"multiple unsafe studies: Phase 'a' .*; Phase 'b' ",
    )


def test_flipped_goal_is_refused_as_a_run_refuses(tmp_path: Path) -> None:
    """A study that optimizes the other direction is refused, not replayed under the wrong goal."""
    _record(_persistent(tmp_path, _phase("a", 2)), "a", _STATE.FAIL, _STATE.FAIL, target=2)

    _refusals_match(
        _persistent(tmp_path, _phase("a", 2), maximize=True),
        tmp_path,
        StudySchemaMismatchError,
        "optimizes minimize, but the current config requires maximize",
    )


def test_malformed_failure_policy_state_in_a_skipped_phase_is_refused_as_a_run_refuses(
    tmp_path: Path,
) -> None:
    """A phase before ``--from-phase`` still has its recorded outcomes validated, as a run validates them."""
    experiment = _persistent(tmp_path, _phase("a", 2), _phase("b", 2, inherits=("a",)))
    _record(experiment, "a", _STATE.FAIL, _STATE.FAIL, target=2, outcomes=False)

    _refusals_match(
        experiment, tmp_path, StudySchemaMismatchError, "missing or malformed", from_phase="b"
    )


def test_failure_limit_met_with_work_remaining_is_refused_as_a_run_refuses(tmp_path: Path) -> None:
    """A streak that meets a lowered limit launches nothing until the target is raised past it."""
    experiment = _persistent(tmp_path, _phase("a", 5, max_consecutive_failures=3))
    _record(experiment, "a", _STATE.FAIL, _STATE.FAIL, _STATE.FAIL, target=5)

    _refusals_match(
        experiment, tmp_path, PhaseAbortRefusedError, "meets max_consecutive_failures=3"
    )


def test_refusal_is_given_before_any_phase_is_previewed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A lowered target on the second phase refuses before the first phase logs a line."""
    experiment = _persistent(tmp_path, _phase("a", 2), _phase("b", 2))
    _record(experiment, "b", _STATE.COMPLETE, _STATE.FAIL, target=4)
    caplog.set_level(logging.INFO)

    with pytest.raises(TrialTargetRefusedError, match="already accepted a target of 4"):
        run_experiment(experiment, dry_run=True)

    assert _lines(caplog) == []


def test_dry_run_reads_the_journal_once_however_many_phases_there_are(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every phase's study comes from one capture of the ledger, so the counts and the checks agree."""
    reads: list[Path] = []
    real_capture = ledger_module._capture_journal

    def counting_capture(path: Path) -> bytes | None:
        reads.append(path)
        return real_capture(path)

    def captures(root: Path, *phases: Phase) -> int:
        root.mkdir()
        experiment = _persistent(root, *phases)
        for phase in phases:
            _record(experiment, phase.name, _STATE.FAIL, target=2)
        reads.clear()
        monkeypatch.setattr(ledger_module, "_capture_journal", counting_capture)
        run_experiment(experiment, dry_run=True)
        monkeypatch.setattr(ledger_module, "_capture_journal", real_capture)
        return len(reads)

    one = captures(tmp_path / "one", _phase("a", 2))
    three = captures(tmp_path / "three", _phase("a", 2), _phase("b", 2), _phase("c", 2))

    assert one == three >= 1


def test_empty_unstamped_study_previews_as_a_fresh_phase(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A study a crash left empty and unstamped is one the launch would stamp; the preview just reads it."""
    experiment = _persistent(tmp_path, _phase("a", 2))
    optuna.create_study(
        study_name="t::a",
        storage=_resolve_storage(experiment.resolved_storage),
        direction="minimize",
    )

    assert _preview(experiment, tmp_path, caplog) == [
        "phase=a study=t::a finished=0 remaining=2 reruns=0 n_jobs=1",
        "DRY RUN phase=a would launch 2 trials",
    ]
