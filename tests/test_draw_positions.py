"""Draw positions: a rerun takes its source's position, so every seeded sampler draws the same configurations however often a phase is interrupted or enlarged."""

from __future__ import annotations

import contextlib
import inspect
import pickle
from collections.abc import Callable

import optuna
import pytest
from optuna.samplers._lazy_random_state import LazyRandomState

from phasesweep.config.search import (
    CategoricalParam,
    FloatParam,
    IntParam,
    Repeats,
    Sampler,
    SearchParam,
)
from phasesweep.engine.attempts import _record_stale_trial_outcome
from phasesweep.engine.optuna import (
    _build_sampler,
    _draw_positions,
    _position_seed,
    _PositionCmaEsSampler,
    _suggest,
)
from phasesweep.engine.state import RERUN_OF_ATTR
from phasesweep.engine.study_policy import _record_reruns

_GRID_SPACE: dict[str, SearchParam] = {
    "x": CategoricalParam(type="categorical", choices=[1, 2, 3]),
    "y": CategoricalParam(type="categorical", choices=["a", "b"]),
}
_RANDOM_SPACE: dict[str, SearchParam] = {
    "lr": FloatParam(type="float", low=1e-4, high=1e-1, log=True),
    "width": IntParam(type="int", low=8, high=64),
}


class _Interrupt(BaseException):
    """Stops ``optimize`` and leaves the trial RUNNING, as a killed orchestrator does."""


def _trial(number: int, rerun_of: int | None = None) -> optuna.trial.FrozenTrial:
    """Build a finished trial, optionally marked as a rerun.

    :param int number: Trial number.
    :param int | None rerun_of: Source trial number, or ``None`` for a new draw.
    :return optuna.trial.FrozenTrial: The trial.
    """
    trial = optuna.trial.create_trial(
        state=optuna.trial.TrialState.FAIL,
        user_attrs={} if rerun_of is None else {RERUN_OF_ATTR: rerun_of},
    )
    trial.number = number
    return trial


def test_positions_count_new_draws_and_give_each_rerun_its_source_position() -> None:
    """Without reruns position equals number; a rerun chain shares its root's position."""
    assert _draw_positions([_trial(n) for n in range(4)]) == {0: 0, 1: 1, 2: 2, 3: 3}
    trials = [_trial(0), _trial(1), _trial(2, rerun_of=1), _trial(3), _trial(4, rerun_of=2)]
    assert _draw_positions(reversed(trials)) == {0: 0, 1: 1, 2: 1, 3: 2, 4: 1}


@pytest.mark.parametrize("source", [5, "1", None, 2], ids=["later", "text", "null", "self"])
def test_a_rerun_must_name_an_earlier_trial(source: object) -> None:
    """A marker that names no earlier trial is refused instead of shifting every later position."""
    trial = _trial(2)
    trial.user_attrs[RERUN_OF_ATTR] = source
    if source is None:
        assert _draw_positions([_trial(0), _trial(1), trial])[2] == 2
        return
    with pytest.raises(ValueError, match="not an earlier trial"):
        _draw_positions([_trial(0), _trial(1), trial])


def _study(sampler: optuna.samplers.BaseSampler) -> optuna.Study:
    """Create an in-memory study under ``sampler`` with PhaseSweep's study name shape.

    :param optuna.samplers.BaseSampler sampler: Sampler to attach.
    :return optuna.Study: The study.
    """
    return optuna.create_study(study_name="t::p", sampler=sampler)


def _run(
    study: optuna.Study,
    space: dict[str, SearchParam],
    n_trials: int,
    *,
    interrupt_at: frozenset[int] = frozenset(),
    n_jobs: int = 1,
) -> None:
    """Run ``n_trials`` trials, stopping when a new draw reaches a trial number in ``interrupt_at``.

    :param optuna.Study study: Study to optimize.
    :param dict[str, SearchParam] space: Search space the objective suggests.
    :param int n_trials: Trials to ask for.
    :param frozenset[int] interrupt_at: Trial numbers whose first attempt is interrupted.
    :param int n_jobs: Optuna worker threads.
    """

    def objective(trial: optuna.Trial) -> float:
        for name, param in space.items():
            _suggest(trial, name, param)
        if trial.number in interrupt_at and RERUN_OF_ATTR not in trial.user_attrs:
            raise _Interrupt
        return sum(
            (value - 0.3) ** 2
            for value in trial.params.values()
            if isinstance(value, int | float) and not isinstance(value, bool)
        )

    with contextlib.suppress(_Interrupt):
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)


def _interrupt_and_rerun(study: optuna.Study) -> int:
    """Recover every RUNNING trial as stale recovery does, then schedule reruns as a run does.

    :param optuna.Study study: Study left with RUNNING trials by an interruption.
    :return int: Number of reruns production scheduled.
    """
    for trial in [t for t in study.get_trials(deepcopy=False) if t.state.name == "RUNNING"]:
        _record_stale_trial_outcome(study, trial)
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    reruns = _record_reruns(study)
    trials = study.get_trials(deepcopy=False)
    # A rerun reuses its source's configuration; it never redraws one.
    assert all(trials[rerun].params == trials[source].params for source, rerun in reruns)
    return len(reruns)


def _finished_draws(study: optuna.Study) -> list[tuple[int, tuple[object, ...]]]:
    """Return the completed configurations by draw position.

    :param optuna.Study study: Study to read.
    :return list[tuple[int, tuple[object, ...]]]: ``(position, sorted params)`` of COMPLETE trials.
    """
    trials = study.get_trials(deepcopy=False)
    positions = _draw_positions(trials)
    return sorted(
        (positions[t.number], tuple(sorted(t.params.items())))
        for t in trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )


def _resume(
    study: optuna.Study,
    space: dict[str, SearchParam],
    target: int,
    *,
    interrupt_at: frozenset[int] = frozenset(),
    n_jobs: int = 1,
) -> None:
    """Schedule reruns for the interrupted trials, then run what the target still needs.

    :param optuna.Study study: Study to resume.
    :param dict[str, SearchParam] space: Search space the objective suggests.
    :param int target: Trial target.
    :param frozenset[int] interrupt_at: Trial numbers whose first attempt is interrupted.
    :param int n_jobs: Optuna worker threads.
    """
    _interrupt_and_rerun(study)
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.get_trials())
    _run(study, space, target - done, interrupt_at=interrupt_at, n_jobs=n_jobs)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_grid_covers_every_point_once_across_interruptions(n_jobs: int) -> None:
    """Each interrupted grid point reruns, the grid never stops early, and no point repeats."""
    sampler = _build_sampler(Sampler(type="grid", seed=3), _GRID_SPACE)
    control = _study(_build_sampler(Sampler(type="grid", seed=3), _GRID_SPACE))
    _run(control, _GRID_SPACE, 6)

    study = _study(sampler)
    _run(study, _GRID_SPACE, 6, interrupt_at=frozenset({1}))
    _resume(study, _GRID_SPACE, 6, n_jobs=n_jobs)

    assert _finished_draws(study) == _finished_draws(control)
    grid_ids = [t.system_attrs["grid_id"] for t in study.get_trials() if t.state.name == "COMPLETE"]
    assert sorted(grid_ids) == list(range(6))


def test_a_fixed_configuration_gets_no_grid_and_its_retries_no_grid_attributes() -> None:
    """An empty search space consults no sampler, so even a grid phase stamps no grid point on its attempts."""
    sampler = _build_sampler(Sampler(type="grid"), {})
    assert not isinstance(sampler, optuna.samplers.GridSampler)
    study = _study(sampler)
    _run(study, {}, 2)
    assert [t.system_attrs for t in study.get_trials()] == [{}, {}]


def test_grid_runs_every_pending_rerun_after_the_last_new_point() -> None:
    """Two pending reruns both run, although every grid point was assigned before them."""
    study = _study(_build_sampler(Sampler(type="grid", seed=0), _GRID_SPACE))
    _run(study, _GRID_SPACE, 6, interrupt_at=frozenset({4}))
    # Trial 4 was interrupted; leave trial 5 RUNNING as a second in-flight trial.
    fifth = study.ask()
    for name, param in _GRID_SPACE.items():
        _suggest(fifth, name, param)
    assert _interrupt_and_rerun(study) == 2

    _run(study, _GRID_SPACE, 2)

    states = [t.state.name for t in study.get_trials()]
    assert states.count("COMPLETE") == 6
    assert "WAITING" not in states


def test_a_rerun_without_params_draws_its_source_grid_point() -> None:
    """A source stopped before its first suggestion reruns at its own position's grid point."""
    control = _study(_build_sampler(Sampler(type="grid", seed=0), _GRID_SPACE))
    _run(control, _GRID_SPACE, 3)
    study = _study(_build_sampler(Sampler(type="grid", seed=0), _GRID_SPACE))
    _run(study, _GRID_SPACE, 2)
    assert study.ask().params == {}
    assert _interrupt_and_rerun(study) == 1
    _run(study, _GRID_SPACE, 1)

    rerun = study.get_trials()[3]
    assert rerun.user_attrs[RERUN_OF_ATTR] == 2
    assert rerun.system_attrs["grid_id"] == 2
    assert rerun.params == control.get_trials()[2].params


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize(
    "repeats",
    [None, Repeats(key="seed", values=[0, 1])],
    ids=["no-repeats", "repeats"],
)
def test_seeded_random_draws_the_same_set_however_it_is_interrupted(
    n_jobs: int, repeats: Repeats | None
) -> None:
    """Seeded random keys every draw by position, so interruptions never change what is drawn."""
    space = dict(_RANDOM_SPACE)
    if repeats is not None:
        space["seed"] = CategoricalParam(type="categorical", choices=[0, 1])
    cfg = Sampler(type="random", seed=11)
    control = _study(_build_sampler(cfg, space, repeats=repeats))
    _run(control, space, 8)

    study = _study(_build_sampler(cfg, space, repeats=repeats))
    _run(study, space, 8, interrupt_at=frozenset({2}))
    _resume(study, space, 8, interrupt_at=frozenset({6}), n_jobs=1)
    _resume(study, space, 8, n_jobs=n_jobs)

    assert _finished_draws(study) == _finished_draws(control)


def test_a_seeded_draw_is_pinned_by_seed_study_position_and_label() -> None:
    """A seeded draw is a fixed function of the seed, study name, position and label, so any change to that derivation fails here."""
    assert _position_seed(11, "t::p", 3, "lr") == 2070613593
    space = {"lr": _RANDOM_SPACE["lr"]}
    study = _study(_build_sampler(Sampler(type="random", seed=11), space))
    _run(study, space, 4)
    assert study.get_trials()[3].params["lr"] == 0.0030591554960112855


_MODEL_SPACE: dict[str, SearchParam] = {
    "a": FloatParam(type="float", low=-1.0, high=1.0),
    "b": FloatParam(type="float", low=-1.0, high=1.0),
    "k": IntParam(type="int", low=0, high=4),
}
_CMA_SPACE: dict[str, SearchParam] = {name: _MODEL_SPACE[name] for name in ("a", "b")}
_TPE = Sampler(type="tpe", seed=5, n_startup_trials=4)
_CMAES = Sampler(type="cmaes", seed=5)


def _reopen(
    study: optuna.Study, cfg: Sampler, space: dict[str, SearchParam], n_jobs: int = 1
) -> optuna.Study:
    """Open ``study`` again under a new sampler object, as a new process would.

    :param optuna.Study study: Study whose storage is reopened.
    :param Sampler cfg: Sampler config to build the new sampler from.
    :param dict[str, SearchParam] space: The phase's search space.
    :param int n_jobs: Optuna worker threads the sampler is built for.
    :return optuna.Study: The same study under a fresh sampler.
    """
    return optuna.load_study(
        study_name=study.study_name,
        storage=study._storage,
        sampler=_build_sampler(cfg, space, n_jobs),
    )


@pytest.mark.parametrize(
    ("cfg", "space", "target", "interrupts"),
    [
        # Trial 2 is position 2, inside the four startup draws; trial 7 is position 6, after them.
        pytest.param(_TPE, _MODEL_SPACE, 12, (2, 7), id="tpe"),
        # Two floats give popsize 6: position 0 draws independently, positions 1-6 are generation 0,
        # trial 3 is mid-generation, and trial 8 is position 7, the trial that tells generation 0.
        pytest.param(_CMAES, _CMA_SPACE, 14, (3, 8), id="cmaes"),
    ],
)
def test_model_samplers_resume_exactly_as_an_uninterrupted_run(
    cfg: Sampler, space: dict[str, SearchParam], target: int, interrupts: tuple[int, int]
) -> None:
    """TPE and CMA-ES interrupted twice, each resume in a fresh sampler, draw what an uninterrupted run draws."""
    control = _study(_build_sampler(cfg, space))
    _run(control, space, target)

    study = _study(_build_sampler(cfg, space))
    _run(study, space, target, interrupt_at=frozenset({interrupts[0]}))
    _resume(_reopen(study, cfg, space), space, target, interrupt_at=frozenset({interrupts[1]}))
    _resume(_reopen(study, cfg, space), space, target)

    assert len(study.get_trials()) == target + 2
    assert _finished_draws(study) == _finished_draws(control)


@pytest.mark.parametrize(
    ("cfg", "space", "first", "target"),
    [
        # Six trials put the first process past TPE's four startup draws, into its model.
        pytest.param(_TPE, _MODEL_SPACE, 6, 12, id="tpe"),
        pytest.param(_CMAES, _CMA_SPACE, 4, 14, id="cmaes"),
    ],
)
def test_model_samplers_top_up_exactly_as_one_larger_run(
    cfg: Sampler, space: dict[str, SearchParam], first: int, target: int
) -> None:
    """Raising the target of a finished TPE or CMA-ES phase draws what one run of the larger target draws."""
    control = _study(_build_sampler(cfg, space))
    _run(control, space, target)

    study = _study(_build_sampler(cfg, space))
    _run(study, space, first)
    _resume(_reopen(study, cfg, space), space, target)

    assert _finished_draws(study) == _finished_draws(control)


# A categorical CMA-ES cannot model, so every trial draws it from the independent sampler.
_CMA_WITH_CHOICE: dict[str, SearchParam] = {
    **_CMA_SPACE,
    "c": CategoricalParam(type="categorical", choices=list(range(1000))),
}


@pytest.mark.parametrize(
    ("cfg", "space", "independent"),
    [
        # Positions 0-3 are TPE's startup draws whichever trials have finished.
        pytest.param(_TPE, _MODEL_SPACE, lambda position, _name: position < 4, id="tpe"),
        # Position 0 draws every parameter independently, and every position draws ``c``.
        pytest.param(
            _CMAES,
            _CMA_WITH_CHOICE,
            lambda position, name: position == 0 or name == "c",
            id="cmaes",
        ),
    ],
)
def test_model_samplers_resume_under_two_workers_drawing_by_position(
    cfg: Sampler,
    space: dict[str, SearchParam],
    independent: Callable[[int, str], bool],
) -> None:
    """Under ``n_jobs: 2`` every independent draw is the one its position gives, and each rerun keeps its source's configuration.

    Stock seeded samplers reseed per worker thread, so their independent draws would
    differ from an uninterrupted ``n_jobs: 1`` run's.
    """
    control = _study(_build_sampler(cfg, space))
    _run(control, space, 12)
    study = _study(_build_sampler(cfg, space, 2))
    _run(study, space, 12, interrupt_at=frozenset({5}), n_jobs=2)
    _resume(_reopen(study, cfg, space, 2), space, 12, n_jobs=2)

    trials = study.get_trials()
    positions = _draw_positions(trials)
    # The control has no reruns, so each of its trials sits at the position of its number.
    expected = {t.number: t.params for t in control.get_trials()}
    complete = [t for t in trials if t.state.name == "COMPLETE"]
    assert len(complete) == len({tuple(sorted(t.params.items())) for t in complete}) == 12
    drawn = [
        (positions[t.number], name, value)
        for t in complete
        for name, value in t.params.items()
        if independent(positions[t.number], name)
    ]
    assert drawn
    assert all(expected[position][name] == value for position, name, value in drawn)
    reruns = [t for t in trials if RERUN_OF_ATTR in t.user_attrs]
    assert [t.user_attrs[RERUN_OF_ATTR] for t in reruns] == [5]
    assert all(t.params == trials[t.user_attrs[RERUN_OF_ATTR]].params for t in reruns)


def _carrier(
    sampler: _PositionCmaEsSampler, number: int, generation: int, snapshot: str
) -> optuna.trial.FrozenTrial:
    """Build a completed trial holding a CMA-ES optimizer snapshot.

    :param _PositionCmaEsSampler sampler: Sampler whose attribute keys are used.
    :param int number: Trial number.
    :param int generation: The trial's ``cma:generation``.
    :param str snapshot: Stand-in object pickled as the optimizer.
    :return optuna.trial.FrozenTrial: The trial.
    """
    attrs = sampler._split_optimizer_str(pickle.dumps(snapshot).hex())
    trial = optuna.trial.create_trial(
        value=0.0, system_attrs={**attrs, sampler._attr_key_generation: generation}
    )
    trial.number = number
    return trial


def test_cmaes_restores_the_newest_generation_not_the_highest_numbered_snapshot() -> None:
    """A rerun of an older generation's tell trial, numbered after a newer tell, never rolls the optimizer back."""
    sampler = _PositionCmaEsSampler(seed=0)
    told_three = _carrier(sampler, 20, 3, "generation 3")
    rerun_of_two = _carrier(sampler, 21, 2, "generation 2")
    plain = optuna.trial.create_trial(value=0.0, system_attrs={sampler._attr_key_generation: 3})

    assert sampler._restore_optimizer([told_three, rerun_of_two, plain]) == "generation 3"
    assert sampler._restore_optimizer([plain]) is None


def test_the_optuna_internals_the_position_samplers_build_on_exist() -> None:
    """Fail loudly when an Optuna release renames a private name the position samplers override."""
    tpe = optuna.samplers.TPESampler(seed=0)
    assert isinstance(tpe._random_sampler, optuna.samplers.RandomSampler)
    assert isinstance(tpe._rng, LazyRandomState)

    independent = optuna.samplers.RandomSampler(seed=0)
    cma = optuna.samplers.CmaEsSampler(seed=0, independent_sampler=independent)
    assert cma._independent_sampler is independent
    assert isinstance(cma._cma_rng, LazyRandomState)
    assert (cma._attr_key_generation, cma._attr_key_optimizer) == (
        "cma:generation",
        "cma:optimizer",
    )
    assert list(inspect.signature(cma._restore_optimizer).parameters) == ["completed_trials"]
    grid = optuna.samplers.GridSampler({"x": [0, 1]}, seed=0)
    assert grid._search_space == {"x": [0, 1]}
    assert len(grid._all_grids) == 2
