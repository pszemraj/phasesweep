"""Draw positions: a rerun takes its source's position, so seeded random and grid draw the same set however often a phase is interrupted."""

from __future__ import annotations

import contextlib

import optuna
import pytest

from phasesweep.config import CategoricalParam, FloatParam, IntParam, Repeats, Sampler, SearchParam
from phasesweep.engine.optuna import _build_sampler, _draw_positions, _suggest
from phasesweep.engine.state import RERUN_OF_ATTR

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
        return 0.0

    with contextlib.suppress(_Interrupt):
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)


def _interrupt_and_rerun(study: optuna.Study) -> int:
    """Fail every RUNNING trial and add one WAITING rerun of it carrying its params and system attrs.

    :param optuna.Study study: Study left with RUNNING trials by an interruption.
    :return int: Number of reruns added.
    """
    stale = [t for t in study.get_trials(deepcopy=False) if t.state.name == "RUNNING"]
    for trial in stale:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.WAITING,
                params=trial.params,
                distributions=trial.distributions,
                system_attrs=dict(trial.system_attrs),
                user_attrs={RERUN_OF_ATTR: trial.number},
            )
        )
    return len(stale)


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
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.WAITING, user_attrs={RERUN_OF_ATTR: 2}
        )
    )
    _run(study, _GRID_SPACE, 1)

    rerun = study.get_trials()[3]
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
