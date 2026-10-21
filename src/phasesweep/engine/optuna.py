"""Optuna sampler, trial-identity, and study-name helpers.

Storage construction lives in :mod:`phasesweep.engine.ledger`, the single
chokepoint; nothing here opens a ledger.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, assert_never

import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.samplers._lazy_random_state import LazyRandomState

from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import (
    CategoricalParam,
    FloatParam,
    IntParam,
    Repeats,
    Sampler,
    SearchParam,
    grid_search_space,
)
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    RERUN_OF_ATTR,
    TERMINAL_TRIAL_STATES,
    TRIAL_STATE_NAMES,
    RecoveredTrialState,
    TrialStateName,
    _trial_interrupted,
)

if TYPE_CHECKING:
    from optuna.samplers._cmaes import CmaClass


@dataclass(frozen=True)
class _TrialRef:
    """Identity of one trial recorded in a publication or storage snapshot.

    Reconciles a row against published evidence or external cleanup evidence:
    which trial it is, and which generation/attempt claimed it.
    Values are ``None`` for an allocation interrupted before its identity
    attributes were written. Published references additionally
    carry their terminal and complete count boundary, and a RUNNING row the
    state stale recovery will record for it. Those fields are not part of a
    trial-row identity comparison.
    """

    trial_number: int
    generation_id: str | None
    attempt_id: str | None
    finished_trials: int | None = field(default=None, compare=False)
    completed_trials: int | None = field(default=None, compare=False)
    recovers_as: RecoveredTrialState | None = field(default=None, compare=False)


@dataclass(frozen=True)
class _PhaseTrialStats:
    """One read-only storage snapshot for phase counts.

    ``running_attempts`` is part of the *same* snapshot as ``counts``, not a
    second read: it lists the RUNNING trials those counts describe, and is
    ``None`` exactly when ``available`` is ``False`` (the counts are unknown).
    A confirmed absent study has known zero counts and an empty attempt list.
    ``published_trial_available`` checks the requested published trial identity
    in that same snapshot; false also covers unreadable or absent storage.
    """

    counts: dict[str, int]
    available: bool
    generation_counts: dict[str, dict[str, int]]
    running_attempts: list[_TrialRef] | None
    published_trial_available: bool = False


def _trial_count_state(trial: optuna.trial.FrozenTrial) -> TrialStateName:
    """Return the state a trial is counted under: ``INTERRUPTED``, or its Optuna state.

    The one classifier behind every state-keyed count, so an interrupted trial
    (:func:`~phasesweep.engine.state._trial_interrupted`) is never counted as
    finished anywhere.

    :param optuna.trial.FrozenTrial trial: Trial to classify.
    :return TrialStateName: ``INTERRUPTED`` for an interrupted trial, otherwise
        the Optuna state name.
    """
    if _trial_interrupted(trial):
        return "INTERRUPTED"
    name = trial.state.name
    for state in TRIAL_STATE_NAMES:
        if state == name:
            return state
    raise ValueError(f"Trial {trial.number} has an unknown Optuna state {name!r}.")


def _finished_trial_count(trials: Iterable[optuna.trial.FrozenTrial]) -> int:
    """Return the number of trials that count toward ``n_trials``.

    A trial counts once it is terminal and was not interrupted: an interrupted
    trial reruns instead, so it never consumes the trial target.

    :param Iterable[optuna.trial.FrozenTrial] trials: Trials whose states should be counted.
    :return int: Number of trials in a state of :data:`~phasesweep.engine.state.TERMINAL_TRIAL_STATES`.
    """
    return sum(1 for trial in trials if _trial_count_state(trial) in TERMINAL_TRIAL_STATES)


def _completed_trial_count(trials: Iterable[optuna.trial.FrozenTrial]) -> int:
    """Return the number of COMPLETE trials in ``trials``.

    :param Iterable[optuna.trial.FrozenTrial] trials: Trials whose states should be counted.
    :return int: Number of trials in :attr:`optuna.trial.TrialState.COMPLETE`.
    """
    return sum(1 for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE)


def _trial_state_label(trial: optuna.trial.FrozenTrial) -> str:
    """Return the state a trial is reported under: its counted state, or ``INFEASIBLE``.

    A COMPLETE trial that violated a constraint or the metric's bounds is never
    eligible for selection, so run logs and trial tables name it apart from the
    trials that ranked, and an interrupted trial reads ``INTERRUPTED``, as it
    is counted.

    :param optuna.trial.FrozenTrial trial: Trial whose state is labeled.
    :return str: :func:`_trial_count_state`, except ``INFEASIBLE`` for a COMPLETE
        trial without a truthy feasibility attr.
    """
    if trial.state == optuna.trial.TrialState.COMPLETE and not trial.user_attrs.get(
        FEASIBLE_ATTR, False
    ):
        return "INFEASIBLE"
    return _trial_count_state(trial)


def _published_phase_trial_refs(
    summary: Mapping[str, Any] | None,
) -> dict[str, _TrialRef | None]:
    """Return each published phase's local winning-trial identity.

    :param Mapping[str, Any] | None summary: Already-resolved publication summary.
    :return dict[str, _TrialRef | None]: Local trial identities by phase, with
        their known terminal-count boundary; None when a published phase does
        not record a complete identity.
    """
    refs: dict[str, _TrialRef | None] = {}
    for item in (summary or {}).get("phases", ()):
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            continue
        name = item["name"]
        number = item.get("trial_number")
        generation = item.get("generation_id")
        attempt = item.get("attempt_id")
        completion = item.get("completion")
        finished_trials = (
            completion.get("finished_trials")
            if isinstance(completion, Mapping)
            and type(completion.get("finished_trials")) is int
            and completion["finished_trials"] >= 0
            else None
        )
        completed_trials = (
            completion.get("completed_trials")
            if isinstance(completion, Mapping)
            and type(completion.get("completed_trials")) is int
            and completion["completed_trials"] >= 0
            else None
        )
        refs[name] = (
            _TrialRef(number, generation, attempt, finished_trials, completed_trials)
            if isinstance(number, int)
            and not isinstance(number, bool)
            and isinstance(generation, str)
            and generation
            and isinstance(attempt, str)
            and attempt
            else None
        )
    return refs


def _meets_published_trial_history_boundary(
    finished: int, completed: int, expected: _TrialRef
) -> bool:
    """Return whether a study still holds the history a publication recorded.

    :param int finished: Terminal trials the study holds now.
    :param int completed: COMPLETE trials the study holds now.
    :param _TrialRef expected: Published trial whose recorded counts, when
        present, are the minimum.
    :return bool: ``True`` when both counts meet the recorded boundary.
    """
    return (expected.finished_trials is None or finished >= expected.finished_trials) and (
        expected.completed_trials is None or completed >= expected.completed_trials
    )


def _published_trial_history_available(
    stats: _PhaseTrialStats, published_trial: _TrialRef | None
) -> bool:
    """Return whether one storage snapshot retains a published trial history.

    :param _PhaseTrialStats stats: Counts and identity result from one storage read.
    :param _TrialRef | None published_trial: Published trial and count boundary.
    :return bool: ``True`` when identity and every recorded count boundary survive.
    """
    if not stats.available or not stats.published_trial_available or published_trial is None:
        return False
    finished = sum(stats.counts.get(state, 0) for state in TERMINAL_TRIAL_STATES)
    completed = stats.counts.get("COMPLETE", 0)
    return _meets_published_trial_history_boundary(finished, completed, published_trial)


def _published_trial_matches(trial: optuna.trial.FrozenTrial, expected: _TrialRef) -> bool:
    """Return whether a completed trial has the published identity.

    :param optuna.trial.FrozenTrial trial: Ledger trial to compare.
    :param _TrialRef expected: Published trial number, generation, and attempt identity.
    :return bool: ``True`` when the trial is complete and all identity fields match.
    """
    return (
        trial.state == optuna.trial.TrialState.COMPLETE
        and trial.number == expected.trial_number
        and trial.user_attrs.get(GENERATION_ID_ATTR) == expected.generation_id
        and trial.user_attrs.get(ATTEMPT_ID_ATTR) == expected.attempt_id
    )


def _draw_positions(trials: Iterable[optuna.trial.FrozenTrial]) -> dict[int, int]:
    """Map each trial number to its draw position.

    Walking trials in number order, a trial that is not a rerun takes the next
    position and a rerun (one carrying :data:`~phasesweep.engine.state.RERUN_OF_ATTR`)
    takes its source's. Optuna numbers trials in creation order and the marker
    is part of a trial's creation record, so a position never changes once its
    trial exists, and the trials that are not reruns hold exactly positions
    ``0..M-1`` however many interruptions came between them.

    :param Iterable[optuna.trial.FrozenTrial] trials: Every trial of one study.
    :raises ValueError: A rerun names a source that is not an earlier trial.
    :return dict[int, int]: Draw position by trial number.
    """
    positions: dict[int, int] = {}
    next_position = 0
    for trial in sorted(trials, key=lambda candidate: candidate.number):
        source = trial.user_attrs.get(RERUN_OF_ATTR)
        if source is None:
            positions[trial.number] = next_position
            next_position += 1
        elif type(source) is int and source in positions:
            positions[trial.number] = positions[source]
        else:
            raise ValueError(
                f"Trial {trial.number} names {source!r} under {RERUN_OF_ATTR!r}, "
                "which is not an earlier trial of its study."
            )
    return positions


def _draw_position(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> int:
    """Return one trial's draw position, read from the study's storage.

    :param optuna.Study study: Study holding ``trial``.
    :param optuna.trial.FrozenTrial trial: Trial whose position is wanted.
    :return int: The trial's draw position (:func:`_draw_positions`).
    """
    trials = study._storage.get_all_trials(study._study_id, deepcopy=False)
    return _draw_positions(trials)[trial.number]


def _position_seed(base_seed: int, study_name: str, position: int, label: str) -> int:
    """Derive the seed of one random choice from the draw position that owns it.

    :param int base_seed: The phase's ``sampler.seed``.
    :param str study_name: The phase's study name.
    :param int position: The trial's draw position.
    :param str label: What the seed is for, such as a parameter name.
    :return int: A 32-bit seed, the same in every process for the same inputs.
    """
    material = json.dumps(
        [base_seed, study_name, position, label],
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


class _PositionRandomSampler(optuna.samplers.RandomSampler):
    """Seeded random sampler whose every draw is a function of the trial's draw position."""

    def __init__(self, *, seed: int, repeats: Repeats | None = None) -> None:
        """Store the caller's seed without handing it to the base sampler.

        The base ``RandomSampler`` gets no seed because this sampler never draws
        through its stream: each draw seeds a fresh one from the draw position,
        so a restarted process draws exactly what the first one would have.

        :param int seed: Base seed supplied by the phase's sampler config.
        :param Repeats | None repeats: The phase's repeat axis, whose values cycle
            through each configuration slot of ``len(values)`` consecutive positions.
        """
        super().__init__(seed=None)
        self._base_seed = seed
        self._repeats = repeats

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> object:
        """Draw one parameter from the stream its draw position owns.

        The position is :func:`_draw_position`, or under repeats the
        configuration slot ``position // len(values)``, so every trial of a slot
        draws one configuration and its repeat value is
        ``values[position % len(values)]``. A rerun has its source's position and
        therefore draws its source's values.

        :param optuna.Study study: The active study; its name is part of the seed.
        :param optuna.trial.FrozenTrial trial: The trial being sampled for.
        :param str param_name: Name of the parameter being sampled.
        :param optuna.distributions.BaseDistribution param_distribution: The
            parameter's distribution.
        :return object: The repeat value for the repeat key, otherwise the value a
            fresh ``RandomSampler`` seeded by :func:`_position_seed` draws.
        """
        position = _draw_position(study, trial)
        if self._repeats is not None:
            count = len(self._repeats.values)
            if param_name == self._repeats.key:
                return self._repeats.values[position % count]
            position //= count
        seed = _position_seed(self._base_seed, study.study_name, position, param_name)
        sampler = optuna.samplers.RandomSampler(seed=seed)
        return sampler.sample_independent(study, trial, param_name, param_distribution)


class _PositionGridSampler(optuna.samplers.GridSampler):
    """Grid sampler that gives each trial the grid point at its draw position and never stops a study."""

    def before_trial(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Assign the grid point at the trial's draw position, unless the trial already has one.

        A rerun copies its source's ``grid_id``. New draws take positions
        ``0..M-1`` in the grid's seeded shuffle, so no two trials that are not
        reruns share a point under any ``n_jobs``.

        :param optuna.Study study: The active study.
        :param optuna.trial.FrozenTrial trial: The trial about to run.
        :raises RuntimeError: The position lies past the grid, which target
            validation and queued failed-cell retries rule out.
        """
        if "grid_id" in trial.system_attrs:
            return
        grid_id = _draw_position(study, trial)
        if grid_id >= len(self._all_grids):
            raise RuntimeError(
                f"Trial {trial.number} of study {study.study_name!r} has draw position "
                f"{grid_id}, past the {len(self._all_grids)}-point grid."
            )
        study._storage.set_trial_system_attr(trial._trial_id, "search_space", self._search_space)
        study._storage.set_trial_system_attr(trial._trial_id, "grid_id", grid_id)

    def after_trial(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        values: Sequence[float] | None,
    ) -> None:
        """Leave the study running: the trial target, not the grid, bounds the loop.

        Optuna's own early stop counts an interrupted point as visited and
        would strand its pending rerun.

        :param optuna.Study study: The active study.
        :param optuna.trial.FrozenTrial trial: The trial that finished.
        :param optuna.trial.TrialState state: Its terminal state.
        :param Sequence[float] | None values: Its objective values.
        """
        return


class _ThreadRandomState(threading.local):
    """The random state the calling thread's draw in progress seeded."""

    state: LazyRandomState | None = None

    def current(self) -> LazyRandomState:
        """Return the calling thread's random state.

        :raises RuntimeError: No draw on this thread seeded one.
        :return LazyRandomState: The state the draw in progress seeded.
        """
        if self.state is None:
            raise RuntimeError("A position-seeded sampler drew before it seeded its random state.")
        return self.state


class _PositionTPESampler(optuna.samplers.TPESampler):
    """Seeded TPE whose every random choice is a function of the trial's draw position.

    Stock TPE draws its startup trials and its candidates from two streams
    that restart at the seed in every process, so a resumed study would draw
    its first configurations again. Here the startup draws come from
    :class:`_PositionRandomSampler` and each candidate draw from a stream
    seeded by its position, per thread, so a resumed or enlarged study draws
    no configuration twice, and at ``n_jobs: 1`` draws what an uninterrupted
    one would have, given the same trial values.
    """

    def __init__(self, *, seed: int, n_startup_trials: int, constant_liar: bool) -> None:
        """Build the stock sampler, then key its two random sources by draw position.

        :param int seed: Base seed supplied by the phase's sampler config.
        :param int n_startup_trials: Random trials before TPE fits its model.
        :param bool constant_liar: Whether running trials count as pessimistic observations.
        """
        self._thread_rng = _ThreadRandomState()
        super().__init__(seed=seed, n_startup_trials=n_startup_trials, constant_liar=constant_liar)
        self._base_seed = seed
        self._random_sampler = _PositionRandomSampler(seed=seed)

    @property
    def _rng(self) -> LazyRandomState:
        """Return the stream the calling thread's candidate draw seeded.

        :return LazyRandomState: That thread's position-seeded state.
        """
        return self._thread_rng.current()

    @_rng.setter
    def _rng(self, value: LazyRandomState) -> None:
        """Set the calling thread's candidate stream.

        :param LazyRandomState value: The state its next candidate draw uses.
        """
        self._thread_rng.state = value

    def reseed_rng(self) -> None:
        """Do nothing: every draw seeds its own stream from its draw position."""
        return

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> object:
        """Seed this thread's candidate stream from the draw position, then sample as TPE does.

        :param optuna.Study study: The active study; its name is part of the seed.
        :param optuna.trial.FrozenTrial trial: The trial being sampled for.
        :param str param_name: Name of the parameter being sampled.
        :param optuna.distributions.BaseDistribution param_distribution: The
            parameter's distribution.
        :return object: The sampled value.
        """
        position = _draw_position(study, trial)
        label = f"tpe:{param_name}"
        self._rng = LazyRandomState(
            _position_seed(self._base_seed, study.study_name, position, label)
        )
        return super().sample_independent(study, trial, param_name, param_distribution)


class _PositionCmaEsSampler(optuna.samplers.CmaEsSampler):
    """Seeded CMA-ES whose every random choice is a function of the trial's draw position.

    The optimizer itself already lives in storage. Stock CMA-ES seeds each ask
    from a process-local stream plus the trial number, so a resumed study
    searches differently from an uninterrupted one. Here each ask is seeded
    from the trial's draw position, the independent draws come from
    :class:`_PositionRandomSampler`, and the optimizer is restored from the
    newest generation's snapshot.
    """

    def __init__(self, *, seed: int) -> None:
        """Build the stock sampler with a position-keyed independent sampler and ask stream.

        :param int seed: Base seed supplied by the phase's sampler config.
        """
        self._thread_rng = _ThreadRandomState()
        super().__init__(seed=seed, independent_sampler=_PositionRandomSampler(seed=seed))
        self._base_seed = seed

    @property
    def _cma_rng(self) -> LazyRandomState:
        """Return the stream the calling thread's relative draw seeded.

        :return LazyRandomState: That thread's position-seeded state.
        """
        return self._thread_rng.current()

    @_cma_rng.setter
    def _cma_rng(self, value: LazyRandomState) -> None:
        """Set the calling thread's relative-draw stream.

        :param LazyRandomState value: The state its next relative draw uses.
        """
        self._thread_rng.state = value

    def sample_relative(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, object]:
        """Sample as CMA-ES does, with the ask seeded by the draw position instead of the trial number.

        The trial handed on is a copy numbered by position; its trial id is the
        real one, so every storage write lands on the real trial.

        :param optuna.Study study: The active study; its name is part of the seed.
        :param optuna.trial.FrozenTrial trial: The trial being sampled for.
        :param dict[str, optuna.distributions.BaseDistribution] search_space: The
            relative search space.
        :return dict[str, object]: The sampled parameters, empty when CMA-ES defers
            to the independent sampler.
        """
        position = _draw_position(study, trial)
        self._cma_rng = LazyRandomState(
            _position_seed(self._base_seed, study.study_name, position, "cmaes")
        )
        positioned = copy.copy(trial)
        positioned.number = position
        return super().sample_relative(study, positioned, search_space)

    def _restore_optimizer(
        self, completed_trials: list[optuna.trial.FrozenTrial]
    ) -> CmaClass | None:
        """Restore the optimizer from the completed trial holding the newest generation's snapshot.

        Stock CMA-ES restores from the highest-numbered snapshot. A rerun of the
        trial that told a generation copies that trial's snapshot, so under
        ``n_jobs > 1`` it can hold an older snapshot under a higher number than
        a generation told meanwhile; the newest generation wins, ties to the
        highest number.

        :param list[optuna.trial.FrozenTrial] completed_trials: The study's
            completed trials.
        :return CmaClass | None: The restored optimizer, or None before any tell.
        """
        carriers = [
            completed
            for completed in completed_trials
            if any(key.startswith(self._attr_key_optimizer) for key in completed.system_attrs)
        ]
        if not carriers:
            return None
        newest = max(
            carriers,
            key=lambda carrier: (
                carrier.system_attrs[self._attr_key_generation],
                carrier.number,
            ),
        )
        return super()._restore_optimizer([newest])


def _build_sampler(
    cfg: Sampler,
    search_space: dict[str, SearchParam],
    n_jobs: int = 1,
    *,
    repeats: Repeats | None = None,
) -> optuna.samplers.BaseSampler:
    """Construct the Optuna sampler for a phase from its YAML ``sampler`` block.

    Args:
        cfg: Parsed sampler config (type, seed, startup-trials, etc.).
        search_space: The phase's validated sampler search space
            (:func:`phasesweep.config.search.sampler_search_space`), used to build the
            ``GridSampler`` grid; it includes the repeat axis.
        n_jobs: Phase parallelism; enables TPE's ``constant_liar`` heuristic
            when ``n_jobs > 1``.
        repeats: The phase's repeats block, which seeded random draws by
            configuration slot.

    Returns:
        A configured :class:`optuna.samplers.BaseSampler` subclass instance.
        A seeded stochastic sampler draws by draw position, so a study resumes
        and grows without repeating a draw; an unseeded one is Optuna's own,
        which only in-memory storage allows. A fixed configuration, whose
        search space is empty, gets a plain random sampler that is never
        consulted.

    Raises:
        ValueError: The validated grid search space cannot be enumerated.

    """
    if not search_space:
        return optuna.samplers.RandomSampler(seed=cfg.seed)
    if cfg.type == "tpe":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=ExperimentalWarning,
                message=r"Argument ``constant_liar`` is an experimental feature.*",
            )
            if cfg.seed is None:
                return optuna.samplers.TPESampler(
                    n_startup_trials=cfg.n_startup_trials, constant_liar=(n_jobs > 1)
                )
            return _PositionTPESampler(
                seed=cfg.seed,
                n_startup_trials=cfg.n_startup_trials,
                constant_liar=(n_jobs > 1),
            )
    if cfg.type == "random":
        if cfg.seed is None:
            return optuna.samplers.RandomSampler()
        return _PositionRandomSampler(seed=cfg.seed, repeats=repeats)
    if cfg.type == "grid":
        return _PositionGridSampler(grid_search_space(search_space), seed=cfg.seed)
    if cfg.type == "cmaes":
        if cfg.seed is None:
            return optuna.samplers.CmaEsSampler()
        return _PositionCmaEsSampler(seed=cfg.seed)
    assert_never(cfg.type)


def _suggest(trial: optuna.Trial, name: str, p: SearchParam) -> Any:
    """Dispatch to the right Optuna ``trial.suggest_*`` based on param type.

    Args:
        trial: The active Optuna trial.
        name: Parameter name (used as the Optuna key).
        p: The concrete search parameter from the phase's ``search_space``.

    Returns:
        The sampled value. Type matches ``p`` (``float``/``int``/categorical scalar).

    """
    if isinstance(p, FloatParam):
        return trial.suggest_float(name, p.low, p.high, step=p.step, log=p.log)
    if isinstance(p, IntParam):
        return trial.suggest_int(name, p.low, p.high, step=p.step, log=p.log)
    if isinstance(p, CategoricalParam):
        return trial.suggest_categorical(name, p.choices)
    assert_never(p)


def _phase_study_name(experiment: Experiment, phase: Phase | str) -> str:
    """Return the stable Optuna study name for a phase.

    :param Experiment experiment: Parsed experiment config supplying the experiment name.
    :param Phase | str phase: Phase or historical phase name appended to the namespace.
    :return str: Stable Optuna study name for the experiment/phase pair.
    """
    name = phase if isinstance(phase, str) else phase.name
    return f"{experiment.experiment}::{name}"
