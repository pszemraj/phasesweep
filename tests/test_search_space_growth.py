"""A search space that only gains values continues its study, keeping every finished trial.

Removing or changing a recorded value is refused with text that names it, a grown grid runs only
the cells it has not drawn, and a draw position keeps its cell across restarts and growth.
"""

from __future__ import annotations

import logging
import signal
from collections.abc import Iterator
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import (
    CategoricalParam,
    FloatParam,
    IntParam,
    Repeats,
    Sampler,
    SearchParam,
    sampler_search_space,
)
from phasesweep.engine.errors import StudyFingerprintMismatchError
from phasesweep.engine.fingerprints import _verify_fingerprint
from phasesweep.engine.growth import CHOICES_ROUTE, CMAES_ROUTE, _search_space_growth
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.optuna import _build_sampler, _draw_positions, _suggest_all
from phasesweep.engine.preview import validate_trial_targets
from phasesweep.engine.state import RERUN_OF_ATTR, TRIAL_DIR_ATTR, _recorded_identity
from phasesweep.runtime.json import dump_json
from tests.conftest import make_experiment, phase_study, phase_trials, write_trainer
from tests.test_rerun_real_runs import _Harness

_ORIGINAL = [1e-4, 3e-4, 1e-3, 3e-3]
_GROWN = [1e-5, *_ORIGINAL, 1e-2, 3e-2]
_RENAME = "give `p` a new `name`, which runs it again from scratch"


def _trainer(tmp_path: Path, *, fail_once: float | None = None) -> tuple[Path, Path]:
    """Write a trainer that logs its ``--lr`` as the objective and records each call.

    :param Path tmp_path: Directory holding the trainer and its call log.
    :param float | None fail_once: An lr whose first call exits nonzero.
    :return tuple[Path, Path]: The trainer and the log that gains one ``lr`` line per call.
    """
    calls = tmp_path / "calls.log"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, sys
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out")
        ap.add_argument("--lr", type=float, default=0.0)
        args, _ = ap.parse_known_args()
        calls = Path({str(calls)!r})
        seen = calls.read_text().split() if calls.exists() else []
        with calls.open("a") as log:
            log.write(repr(args.lr) + "\\n")
        if args.lr == {fail_once!r} and repr(args.lr) not in seen:
            sys.exit(1)
        print(f"x={{args.lr}}")
        """,
    )
    return trainer, calls


def _calls(calls: Path) -> list[float]:
    """Read the lr of every trainer call so far.

    :param Path calls: The trainer's call log.
    :return list[float]: One lr per call, in call order.
    """
    return [float(line) for line in calls.read_text().split()] if calls.exists() else []


def _phase(
    values: list[float],
    *,
    n_trials: int | None = None,
    sampler: Sampler | None = None,
    repeats: Repeats | None = None,
    param: SearchParam | None = None,
) -> Phase:
    """Build phase ``p`` searching ``lr``, by default a full seeded grid over ``values``.

    :param list[float] values: The lr choices.
    :param int | None n_trials: The trial target; the grid's size by default.
    :param Sampler | None sampler: The sampler; a seeded grid by default.
    :param Repeats | None repeats: An optional repeat axis.
    :param SearchParam | None param: An lr parameter to use instead of the choices.
    :return Phase: The phase.
    """
    count = 1 if repeats is None else len(repeats.values)
    return Phase(
        name="p",
        n_trials=n_trials or len(values) * count,
        gpu_policy="none",
        max_consecutive_failures=3,
        sampler=sampler or Sampler(type="grid", seed=0),
        search_space={"lr": param or CategoricalParam(type="categorical", choices=values)},
        repeats=repeats,
    )


def _experiment(tmp_path: Path, trainer: Path, *phases: Phase) -> Experiment:
    """Build a persistent experiment under ``tmp_path`` running ``trainer``.

    :param Path tmp_path: Directory holding the ledger and workdir.
    :param Path trainer: The trainer script.
    :param Phase phases: The experiment's phases.
    :return Experiment: The experiment.
    """
    return make_experiment(persistent=tmp_path, trainer=trainer, phases=list(phases))


def _complete_lrs(experiment: Experiment) -> list[float]:
    """List the lr of every COMPLETE trial of phase ``p``, sorted.

    :param Experiment experiment: The experiment.
    :return list[float]: The lrs.
    """
    return sorted(t.params["lr"] for t in phase_trials(experiment) if t.state.name == "COMPLETE")


def _stamped(phase: Phase, *, trials: int = 1) -> optuna.Study:
    """Return an in-memory study bound to ``phase`` holding ``trials`` finished trials.

    :param Phase phase: The phase the study is bound to.
    :param int trials: How many trials to run under it.
    :return optuna.Study: The study.
    """
    study = optuna.create_study(
        sampler=_build_sampler(phase.sampler, sampler_search_space(phase), repeats=phase.repeats)
    )
    _verify_fingerprint(study, make_experiment(phases=[phase]), phase, {})
    study.optimize(lambda trial: float(len(_suggest_all(trial, phase))), n_trials=trials)
    return study


def _refusal(study: optuna.Study, phase: Phase) -> StudyFingerprintMismatchError:
    """Verify ``phase`` against ``study`` and return the refusal it raises.

    :param optuna.Study study: The bound study.
    :param Phase phase: The changed phase.
    :return StudyFingerprintMismatchError: The refusal.
    """
    with pytest.raises(StudyFingerprintMismatchError) as caught:
        _verify_fingerprint(study, make_experiment(phases=[phase]), phase, {})
    return caught.value


@pytest.mark.integration
def test_grown_grid_runs_exactly_the_added_cells(tmp_path: Path) -> None:
    """A grid that gains three values keeps its four trials, runs the three new cells once each, and selects over all seven."""
    trainer, calls = _trainer(tmp_path)
    original = _experiment(tmp_path, trainer, _phase(_ORIGINAL))
    run_experiment(original)
    before = {t.number: t.params for t in phase_trials(original)}

    grown = _experiment(tmp_path, trainer, _phase(_GROWN))
    winners = run_experiment(grown)

    assert sorted(_calls(calls)[len(_ORIGINAL) :]) == [1e-5, 1e-2, 3e-2]
    trials = phase_trials(grown)
    assert {t.number: t.params for t in trials[: len(_ORIGINAL)]} == before
    assert _complete_lrs(grown) == sorted(_GROWN)
    assert sorted(t.system_attrs["grid_id"] for t in trials) == list(range(len(_GROWN)))
    assert winners["p"].params["lr"] == 1e-5


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[_Harness]:
    """Yield a real-run harness and kill whatever its runs leave alive.

    :param Path tmp_path: The test's scratch directory.
    :return Iterator[_Harness]: The harness.
    """
    built = _Harness(tmp_path)
    try:
        yield built
    finally:
        built.kill_survivors()
        built.release()


def _gated_phase(values: list[float]) -> Phase:
    """Build the grid phase over ``optimizer.lr``, the override the gated trainer accepts.

    :param list[float] values: The grid's lr choices.
    :return Phase: The phase.
    """
    return _phase(values).model_copy(
        update={
            "search_space": {"optimizer.lr": CategoricalParam(type="categorical", choices=values)}
        }
    )


@pytest.mark.integration
def test_grown_grid_resumed_after_sigkill_covers_each_new_cell_once(harness: _Harness) -> None:
    """A grown grid whose run is SIGKILLed on a new cell resumes and measures every cell exactly once."""
    run_experiment(harness.experiment("interrupted", _gated_phase(_ORIGINAL)))
    grown = harness.experiment("interrupted", _gated_phase(_GROWN))
    config = harness.config(grown)

    harness.interrupt(config, len(_ORIGINAL) + 1, signal.SIGKILL)
    harness.complete(config)

    trials = phase_trials(grown)
    complete = [t for t in trials if t.state.name == "COMPLETE"]
    assert sorted(t.params["optimizer.lr"] for t in complete) == sorted(_GROWN)
    assert sorted(t.system_attrs["grid_id"] for t in complete) == list(range(len(_GROWN)))
    assert [t.user_attrs[RERUN_OF_ATTR] for t in trials if RERUN_OF_ATTR in t.user_attrs] == [
        len(_ORIGINAL) + 1
    ]


@pytest.mark.integration
def test_interrupted_old_cell_reruns_in_place_before_the_grown_cells(harness: _Harness) -> None:
    """A cell interrupted before its grid grew reruns first, in its source's directory with its params and grid_id, and the grown run then draws every other cell once."""
    original = harness.experiment("interrupted", _gated_phase(_ORIGINAL))
    harness.interrupt(harness.config(original), 1, signal.SIGKILL)
    grown = harness.experiment("interrupted", _gated_phase(_GROWN))
    harness.complete(harness.config(grown))

    trials = phase_trials(grown)
    source = trials[1]
    rerun = next(t for t in trials if t.user_attrs.get(RERUN_OF_ATTR) == source.number)
    assert rerun.number == 2
    assert rerun.params == source.params
    assert rerun.system_attrs["grid_id"] == source.system_attrs["grid_id"]
    assert rerun.user_attrs[TRIAL_DIR_ATTR] == source.user_attrs[TRIAL_DIR_ATTR]
    complete = [t for t in trials if t.state.name == "COMPLETE"]
    assert sorted(t.params["optimizer.lr"] for t in complete) == sorted(_GROWN)
    assert sorted(t.system_attrs["grid_id"] for t in complete) == list(range(len(_GROWN)))


@pytest.mark.integration
def test_failed_old_cell_retries_above_the_grown_size(tmp_path: Path) -> None:
    """A recorded failed cell is not retried by the grown grid's own size, only by a target above it, at its own grid_id."""
    trainer, calls = _trainer(tmp_path, fail_once=3e-4)
    original = _experiment(tmp_path, trainer, _phase(_ORIGINAL))
    run_experiment(original)
    failed = next(t for t in phase_trials(original) if t.state.name == "FAIL")

    run_experiment(_experiment(tmp_path, trainer, _phase(_GROWN)))
    assert sorted(_calls(calls)[len(_ORIGINAL) :]) == [1e-5, 1e-2, 3e-2]

    raised = _experiment(tmp_path, trainer, _phase(_GROWN, n_trials=len(_GROWN) + 1))
    run_experiment(raised)
    assert _calls(calls)[-1] == 3e-4
    retry = phase_trials(raised)[-1]
    assert retry.user_attrs[RERUN_OF_ATTR] == failed.number
    assert retry.system_attrs["grid_id"] == failed.system_attrs["grid_id"]
    assert _complete_lrs(raised) == sorted(_GROWN)


@pytest.mark.integration
def test_grown_numeric_choices_survive_replay_by_another_worker(tmp_path: Path) -> None:
    """Another process replaying the journal of a grown grid still sees the grown param on every trial."""
    trainer, _calls_log = _trainer(tmp_path)
    run_experiment(_experiment(tmp_path, trainer, _phase(_ORIGINAL)))
    grown = _experiment(tmp_path, trainer, _phase(_GROWN))
    run_experiment(grown)

    # A fresh storage replays the journal as a worker that wrote none of it.
    replayed = optuna.load_study(
        study_name="t::p", storage=_resolve_storage(grown.resolved_storage)
    ).get_trials(deepcopy=False)
    assert sorted(t.params["lr"] for t in replayed) == sorted(_GROWN)
    assert {type(t.params["lr"]) for t in replayed} == {float}


def test_int_to_float_choice_is_a_removal() -> None:
    """Rewriting int choices as floats drops the ints, whose JSON text differs, and refuses naming them."""
    study = _stamped(_phase([4, 8]))
    error = _refusal(study, _phase([4.0, 8.0, 12.0]))

    assert "search_space 'lr' dropped 4, 8;" in str(error)
    assert error.remediation == f"Put back lr 4, 8, or {_RENAME}."


@pytest.mark.integration
def test_dropping_a_value_refuses_and_names_it(tmp_path: Path) -> None:
    """A removal beside an addition refuses before any trial launches, names the dropped value, and leaves the record as it was."""
    trainer, calls = _trainer(tmp_path)
    original = _experiment(tmp_path, trainer, _phase(_ORIGINAL))
    run_experiment(original)
    recorded = _recorded_identity(phase_study(original))

    swapped = [1e-4, 1e-3, 3e-3, 1e-2]
    with pytest.raises(StudyFingerprintMismatchError) as caught:
        run_experiment(_experiment(tmp_path, trainer, _phase(swapped)))

    assert str(caught.value).endswith(
        "search_space 'lr' dropped 0.0003; added values alone keep the study and run only the "
        "new ones."
    )
    assert caught.value.remediation == f"Put back lr 0.0003, or {_RENAME}."
    assert len(_calls(calls)) == len(_ORIGINAL)
    assert _recorded_identity(phase_study(original)) == recorded


@pytest.mark.integration
def test_unused_growth_reverts_without_refusal(tmp_path: Path) -> None:
    """Growth no trial was drawn under, as a run that stopped right after binding it leaves, reverts or regrows freely."""
    trainer, calls = _trainer(tmp_path)
    original = _experiment(tmp_path, trainer, _phase(_ORIGINAL))
    run_experiment(original)
    grown = _experiment(tmp_path, trainer, _phase(_GROWN))
    _verify_fingerprint(phase_study(grown), grown, grown.phases[0], {})
    record = _recorded_identity(phase_study(grown))
    assert record is not None and len(record["search_spaces"]) == 2

    run_experiment(original)
    record = _recorded_identity(phase_study(original))
    assert record is not None and len(record["search_spaces"]) == 1
    assert len(_calls(calls)) == len(_ORIGINAL)

    other = [*_ORIGINAL, 1e-2]
    _verify_fingerprint(phase_study(grown), grown, grown.phases[0], {})
    run_experiment(_experiment(tmp_path, trainer, _phase(other)))
    record = _recorded_identity(phase_study(original))
    assert record is not None
    assert [entry["first_position"] for entry in record["search_spaces"]] == [0, len(_ORIGINAL)]
    assert _calls(calls)[len(_ORIGINAL) :] == [1e-2]


@pytest.mark.integration
def test_dry_run_of_grown_phase_leaves_the_journal_unchanged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A dry run and a target validation of a grown phase write nothing and say what the phase keeps and would launch."""
    trainer, calls = _trainer(tmp_path)
    run_experiment(_experiment(tmp_path, trainer, _phase(_ORIGINAL)))
    journal = tmp_path / "studies.journal"
    before = journal.read_bytes()
    grown = _experiment(tmp_path, trainer, _phase(_GROWN))
    caplog.set_level(logging.INFO, logger="phasesweep")

    run_experiment(grown, dry_run=True)
    validate_trial_targets(grown)

    assert journal.read_bytes() == before
    assert len(_calls(calls)) == len(_ORIGINAL)
    assert (
        "DRY RUN phase=p search_space gains lr 1e-05, 0.01, 0.03: keeps its 4 finished trials, "
        "would launch 3 new cells."
    ) in caplog.messages


@pytest.mark.integration
def test_from_phase_past_grown_phase_names_from_phase_step(tmp_path: Path) -> None:
    """Skipping a grown phase is refused with the step that runs it; running from it, with the dependent left out and put back, keeps every trial."""
    trainer, calls = _trainer(tmp_path)
    child = Phase(
        name="q",
        inherits=["p"],
        n_trials=2,
        gpu_policy="none",
        sampler=Sampler(type="random", seed=0),
        search_space={"wd": FloatParam(type="float", low=0.0, high=0.1)},
    )
    run_experiment(_experiment(tmp_path, trainer, _phase(_ORIGINAL), child))
    # The added values all lose to the recorded winner, so the dependent's inherited lr holds.
    worse = [*_ORIGINAL, 1e-2, 3e-2]
    grown = _experiment(tmp_path, trainer, _phase(worse), child)

    with pytest.raises(StudyFingerprintMismatchError) as caught:
        run_experiment(grown, from_phase="q")
    assert caught.value.remediation is not None
    assert caught.value.remediation.endswith(
        "If its search_space only gained values, run from `p` (`--from-phase p`)."
    )

    run_experiment(_experiment(tmp_path, trainer, _phase(worse)))
    winners = run_experiment(grown)
    assert len(_calls(calls)) == len(worse) + child.n_trials
    assert winners["p"].params["lr"] == 1e-4
    assert set(winners) == {"p", "q"}


def test_cmaes_range_change_refused() -> None:
    """Under CMA-ES a widened range refuses with the step that keeps every trial."""
    narrow = FloatParam(type="float", low=0.0, high=1.0)
    sampler = Sampler(type="cmaes", seed=0)
    study = _stamped(_phase([], sampler=sampler, param=narrow, n_trials=2))
    wide = narrow.model_copy(update={"high": 2.0})

    error = _refusal(study, _phase([], sampler=sampler, param=wide, n_trials=2))

    assert "search_space 'lr' changed under CMA-ES" in str(error)
    assert error.remediation == f"{CMAES_ROUTE}, or {_RENAME}."


def test_repeat_values_change_refused() -> None:
    """A grown repeats.values regroups every recorded trial, so it refuses as any other change."""
    study = _stamped(_phase([1e-4, 1e-3], repeats=Repeats(key="seed", values=[0, 1])))

    error = _refusal(study, _phase([1e-4, 1e-3], repeats=Repeats(key="seed", values=[0, 1, 2])))

    assert "Changed: phase.repeats.values: [0,1] -> [0,1,2]." in str(error)
    assert error.remediation == f"Restore the recorded value(s), or {_RENAME}."


def test_partly_drawn_repeat_slot_keeps_its_range_after_growth() -> None:
    """A seeded random repeats configuration drawn partly before its range widened draws its other members from the old range."""
    repeats = Repeats(key="seed", values=[0, 1, 2])
    sampler = Sampler(type="random", seed=0)
    narrow = FloatParam(type="float", low=0.0, high=1.0)
    phase = _phase([], sampler=sampler, param=narrow, repeats=repeats, n_trials=6)
    study = _stamped(phase, trials=1)
    wide = phase.model_copy(
        update={"search_space": {"lr": narrow.model_copy(update={"high": 1000.0})}}
    )
    _verify_fingerprint(study, make_experiment(phases=[wide]), wide, {})
    record = _recorded_identity(study)
    assert record is not None and record["search_spaces"][-1]["first_position"] == 3

    study.sampler = _build_sampler(sampler, sampler_search_space(wide), repeats=repeats)
    study.optimize(lambda trial: float(len(_suggest_all(trial, wide))), n_trials=5)

    positions = _draw_positions(study.get_trials())
    by_slot: dict[int, set[float]] = {}
    for trial in study.get_trials():
        by_slot.setdefault(positions[trial.number] // 3, set()).add(trial.params["lr"])
    assert len(by_slot[0]) == 1 and next(iter(by_slot[0])) <= 1.0
    assert len(by_slot[1]) == 1


_GRID = Sampler(type="grid", seed=0)
_RANDOM = Sampler(type="random", seed=0)


def _int(low: int, high: int, step: int = 1) -> SearchParam:
    """Build an int range.

    :param int low: Lower bound.
    :param int high: Upper bound.
    :param int step: Step.
    :return SearchParam: The range.
    """
    return IntParam(type="int", low=low, high=high, step=step)


def _float(low: float, high: float, *, step: float | None = None, log: bool = False) -> SearchParam:
    """Build a float range.

    :param float low: Lower bound.
    :param float high: Upper bound.
    :param float | None step: Optional step.
    :param bool log: Log scale.
    :return SearchParam: The range.
    """
    return FloatParam(type="float", low=low, high=high, step=step, log=log)


def _choices(*values: object) -> SearchParam:
    """Build a choice list.

    :param object values: The choices.
    :return SearchParam: The choices.
    """
    return CategoricalParam(type="categorical", choices=list(values))


@pytest.mark.contract
@pytest.mark.parametrize(
    ("old", "new", "sampler", "expected"),
    [
        (_int(0, 4), _int(0, 8), _GRID, "grows"),
        (_int(0, 8, 2), _int(0, 8), _GRID, "grows"),
        (_int(0, 8), _int(0, 8, 2), _GRID, "dropped"),
        (_float(0.1, 0.4, step=0.1), _float(0.0, 0.4, step=0.05), _GRID, "grows"),
        (_choices(1e-4, 1e-3), _choices(1e-4, 1e-3, 1e-2), _GRID, "grows"),
        (_choices(4, 8), _int(0, 12, 4), _GRID, "grows"),
        (_choices(1, 2), _choices(1.0, 2.0, 3.0), _GRID, "dropped"),
        (_choices("a", "b"), _choices("a", "b", "c"), _GRID, CHOICES_ROUTE),
        (_choices(True, False), _choices(True, False, None), _GRID, CHOICES_ROUTE),
        (_choices(1, 2), _choices(1, 2, 2.5), _GRID, CHOICES_ROUTE),
        (_float(1e-4, 1e-3, log=True), _float(1e-5, 1e-2, log=True), _RANDOM, "grows"),
        (_float(0.0, 1.0, step=0.2), _float(0.0, 2.0, step=0.1), _RANDOM, "grows"),
        (_float(0.0, 1.0, step=0.2), _float(0.0, 1.2, step=0.3), _RANDOM, "dropped"),
        (_float(0.0, 1.0), _float(0.0, 1.0, step=0.5), _RANDOM, "dropped"),
        (_float(0.0, 1.0, step=0.5), _float(-1.0, 1.0), _RANDOM, "grows"),
        (_float(0.1, 1.0), _float(0.2, 1.0), _RANDOM, "dropped"),
        (_int(0, 10), _int(1, 10), _RANDOM, "dropped"),
        (_int(0, 10, 2), _int(1, 11, 2), _RANDOM, "dropped"),
        (_float(0.1, 1.0, log=True), _float(0.1, 2.0), _RANDOM, "refused"),
        (_int(0, 10), _float(0.0, 10.0), _RANDOM, "refused"),
        (_choices(1, 2), _choices(1, 2, 3), Sampler(type="tpe", seed=0), CHOICES_ROUTE),
        (_float(0.0, 1.0), _float(0.0, 2.0), Sampler(type="cmaes", seed=0), CMAES_ROUTE),
    ],
)
def test_range_superset_rules(
    old: SearchParam, new: SearchParam, sampler: Sampler, expected: str
) -> None:
    """Each kind of change is growth, a removal, or a refusal with its own route, as the growth table says."""
    verdict = _search_space_growth(dump_json(old), dump_json(new), sampler=sampler)
    if expected == "grows":
        assert verdict.grows and verdict.gained
    elif expected == "dropped":
        assert verdict.dropped and not verdict.grows
    elif expected == "refused":
        assert verdict.refusal is not None and verdict.route is None and not verdict.dropped
    else:
        assert verdict.refusal is not None and verdict.route == expected and not verdict.dropped
