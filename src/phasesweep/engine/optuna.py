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
from collections.abc import Collection, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, assert_never

import optuna
from optuna._transform import _SearchSpaceTransform
from optuna.exceptions import ExperimentalWarning
from optuna.samplers._lazy_random_state import LazyRandomState

from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import (
    _OPTUNA_EXACT_INT_LIMIT,
    CategoricalParam,
    FloatParam,
    IntParam,
    Repeats,
    Sampler,
    SearchParam,
    grid_search_space,
    sampler_search_space,
)
from phasesweep.engine.bounds import trial_bound_violations
from phasesweep.engine.grid_order import (
    GridCell,
    grid_axis_names,
    grid_cell_order,
    recorded_search_space,
)
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FAILURE_REASON_ATTR,
    FINISHED_TRIAL_STATES,
    GENERATION_ID_ATTR,
    METRIC_EXTRACTOR_LABEL,
    REREAD_FILL_ATTR,
    RERUN_OF_ATTR,
    TRIAL_STATE_NAMES,
    RecoveredTrialState,
    SearchSpaceEntry,
    TrialStateName,
    _recorded_identity,
    _trial_interrupted,
)
from phasesweep.runtime.json import dump_json, json_object
from phasesweep.runtime.process import TIMEOUT_REASON_PREFIX

if TYPE_CHECKING:
    from optuna.samplers._cmaes import CmaClass

    from phasesweep.engine.publication_validation import ValidatedWinner


@dataclass(frozen=True)
class _TrialRef:
    """Identity of one trial recorded in a publication or storage snapshot.

    Reconciles a row against published evidence or external cleanup evidence:
    which trial it is, and which generation/attempt claimed it.
    Values are ``None`` for an allocation interrupted before its identity
    attributes were written. Published references additionally
    carry their selecting study and finished and measured count boundary, and a RUNNING row the
    state stale recovery will record for it. Those fields are not part of a
    trial-row identity comparison.
    """

    trial_number: int
    generation_id: str | None
    attempt_id: str | None
    finished_trials: int | None = field(default=None, compare=False)
    measured_trials: int | None = field(default=None, compare=False)
    recovers_as: RecoveredTrialState | None = field(default=None, compare=False)
    study_name: str | None = field(default=None, compare=False)


@dataclass(frozen=True)
class _PhaseTrialStats:
    """One read-only storage snapshot for phase counts.

    ``running_attempts`` is part of the *same* snapshot as ``counts``, not a
    second read: it lists the RUNNING trials those counts describe, and is
    ``None`` exactly when ``available`` is ``False`` (the counts are unknown).
    A confirmed absent addressed study has zero progress and an empty attempt list.
    ``generation_counts`` counts producing rows from every study, without counting their copies.
    ``published_trial_available`` says whether the same snapshot holds the requested published
    trial identity, in whichever of the phase's studies it was selected from, with that study's
    recorded count boundary met; false also covers unreadable or absent storage.
    ``studies`` holds every study of the phase in that snapshot, oldest first, so a status
    read can tell from the same capture whether the addressed one waits to be filled from
    another (:func:`phasesweep.engine.reread.pending_reread`).
    """

    counts: dict[str, int]
    available: bool
    generation_counts: dict[str, dict[str, int]]
    running_attempts: list[_TrialRef] | None
    published_trial_available: bool = False
    studies: tuple[optuna.Study, ...] = ()


def _trial_run_state(trial: optuna.trial.FrozenTrial) -> TrialStateName:
    """Return the state a run left a trial in, before any bound is applied.

    An interrupted trial (:func:`~phasesweep.engine.state._trial_interrupted`) reads
    ``INTERRUPTED``, so it is never counted as finished anywhere. A trial that measured its
    values reads ``COMPLETE`` whatever the bounds say; :func:`_trial_count_state` judges it.

    :param optuna.trial.FrozenTrial trial: Trial to classify.
    :return TrialStateName: ``INTERRUPTED`` for an interrupted trial, otherwise the Optuna
        state name; never ``INFEASIBLE``.
    """
    if _trial_interrupted(trial):
        return "INTERRUPTED"
    name = trial.state.name
    for state in TRIAL_STATE_NAMES:
        if state == name:
            return state
    raise ValueError(f"Trial {trial.number} has an unknown Optuna state {name!r}.")


def _trial_count_state(trial: optuna.trial.FrozenTrial, experiment: Experiment) -> TrialStateName:
    """Return the state a trial is counted and reported under.

    The one classifier behind every state-keyed count, run log line and trial table. A
    measured trial whose recorded values violate a current bound
    (:func:`~phasesweep.engine.bounds.trial_bound_violations`) reads ``INFEASIBLE``.

    :param optuna.trial.FrozenTrial trial: Trial to classify.
    :param Experiment experiment: Config supplying the current metric and constraint bounds.
    :return TrialStateName: ``INFEASIBLE`` for a measured trial outside the bounds,
        otherwise :func:`_trial_run_state`.
    """
    state = _trial_run_state(trial)
    if state == "COMPLETE" and trial_bound_violations(trial, experiment):
        return "INFEASIBLE"
    return state


def _finished_trial_count(trials: Iterable[optuna.trial.FrozenTrial]) -> int:
    """Return the number of trials that count toward ``n_trials``.

    A trial counts once it is finished: an interrupted trial reruns instead, so it
    never consumes the trial target. No bound changes the count.

    :param Iterable[optuna.trial.FrozenTrial] trials: Trials whose states should be counted.
    :return int: Number of trials in a state of :data:`~phasesweep.engine.state.FINISHED_TRIAL_STATES`.
    """
    return sum(1 for trial in trials if _trial_run_state(trial) in FINISHED_TRIAL_STATES)


def _measured_trial_count(trials: Iterable[optuna.trial.FrozenTrial]) -> int:
    """Return the Optuna COMPLETE trial count of ``trials``, infeasible ones included.

    No bound changes it, so a publication and an accepted partial decision record it as
    ``measured_trials`` and later reads compare a recount with that record.

    :param Iterable[optuna.trial.FrozenTrial] trials: Trials whose states should be counted.
    :return int: Number of trials in :attr:`optuna.trial.TrialState.COMPLETE`.
    """
    return sum(1 for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE)


def _measured_draw_count(
    trials: Sequence[optuna.trial.FrozenTrial],
    *,
    excluded: Collection[int] = (),
    assumed_complete: Collection[int] = (),
) -> int:
    """Count draws already measured, even if a reread recovered several attempts per draw.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial of one study.
    :param Collection[int] excluded: Trial numbers still awaiting a measurement verdict.
    :param Collection[int] assumed_complete: Trial numbers counted as COMPLETE to bound the
        capacity left after re-reading their evidence.
    :return int: Distinct draw positions that have a COMPLETE trial, feasible or not.
    """
    positions = _draw_positions(trials)
    return len(
        {
            positions[trial.number]
            for trial in trials
            if trial.number not in excluded
            and (
                trial.state == optuna.trial.TrialState.COMPLETE or trial.number in assumed_complete
            )
        }
    )


def _any_trial_timed_out(trials: Iterable[optuna.trial.FrozenTrial]) -> bool:
    """Say whether any trial was killed at the phase's per-trial timeout.

    The phase and run deadlines word their reasons differently, and raising either is not
    a change of study identity, so only the supervisor's per-trial prefix counts.

    :param Iterable[optuna.trial.FrozenTrial] trials: Trials of one phase.
    :return bool: ``True`` when a recorded failure reason says the trial timed out.
    """
    for trial in trials:
        reason = trial.user_attrs.get(FAILURE_REASON_ATTR)
        if isinstance(reason, str) and reason.startswith(TIMEOUT_REASON_PREFIX):
            return True
    return False


def _any_objective_unreadable(trials: Iterable[optuna.trial.FrozenTrial]) -> bool:
    """Say whether any trial ran its trainer but failed to yield its objective.

    Only the metric extractor's own failure counts. A non-finite value was read, and a
    constraint extractor's failure names its constraint rather than the objective.

    :param Iterable[optuna.trial.FrozenTrial] trials: Trials of one phase.
    :return bool: ``True`` when a recorded failure reason says the metric extractor failed.
    """
    prefix = f"{METRIC_EXTRACTOR_LABEL}: "
    for trial in trials:
        reason = trial.user_attrs.get(FAILURE_REASON_ATTR)
        if isinstance(reason, str) and reason.startswith(prefix):
            return True
    return False


def _published_phase_trial_refs(
    summary: Mapping[str, Any] | None,
    *,
    winners: Mapping[str, ValidatedWinner] | None = None,
) -> dict[str, _TrialRef | None]:
    """Return each published phase's local winning-trial identity.

    :param Mapping[str, Any] | None summary: Already-resolved publication summary.
    :param Mapping[str, ValidatedWinner] | None winners: Validated winner artifacts that
        name their selecting studies, for callers checking the ledger.
    :return dict[str, _TrialRef | None]: Local trial identities by phase, with
        their known finished and measured count boundary; None when a published
        phase does not record a complete identity.
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
        measured_trials = (
            completion.get("measured_trials")
            if isinstance(completion, Mapping)
            and type(completion.get("measured_trials")) is int
            and completion["measured_trials"] >= 0
            else None
        )
        winner = None if winners is None else winners.get(name)
        refs[name] = (
            _TrialRef(
                number,
                generation,
                attempt,
                finished_trials,
                measured_trials,
                study_name=None if winner is None else winner.study_name,
            )
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
    finished: int, measured: int, expected: _TrialRef
) -> bool:
    """Return whether a study still holds the history a publication recorded.

    :param int finished: Finished trials the study holds now.
    :param int measured: Trials the study holds now that measured their values
        (:func:`_measured_trial_count`).
    :param _TrialRef expected: Published trial whose recorded counts, when
        present, are the minimum.
    :return bool: ``True`` when both counts meet the recorded boundary.
    """
    return (expected.finished_trials is None or finished >= expected.finished_trials) and (
        expected.measured_trials is None or measured >= expected.measured_trials
    )


def _published_trial_history_available(
    stats: _PhaseTrialStats, published_trial: _TrialRef | None
) -> bool:
    """Return whether one storage snapshot retains a published trial history.

    :param _PhaseTrialStats stats: Counts and identity result from one storage read.
    :param _TrialRef | None published_trial: Published trial and count boundary.
    :return bool: ``True`` when identity and every recorded count boundary survive.
    """
    return stats.available and stats.published_trial_available and published_trial is not None


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
    :param str study_name: The phase's namespace (:func:`_study_namespace`), so every study of
        a phase draws the same configurations whatever values it inherits.
    :param int position: The trial's draw position.
    :param str label: What the seed is for, such as a parameter name.
    :return int: A 32-bit seed, the same in every process for the same inputs.
    """
    material = json.dumps(
        [base_seed, study_name, position, label],
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _distribution(param: SearchParam) -> optuna.distributions.BaseDistribution:
    """Return the Optuna distribution ``_suggest`` registers for a search parameter.

    :param SearchParam param: A search parameter.
    :return optuna.distributions.BaseDistribution: Its distribution.
    """
    if isinstance(param, FloatParam):
        return optuna.distributions.FloatDistribution(
            param.low, param.high, log=param.log, step=param.step
        )
    if isinstance(param, IntParam):
        return optuna.distributions.IntDistribution(
            param.low, param.high, log=param.log, step=param.step
        )
    return optuna.distributions.CategoricalDistribution(param.choices)


def _distribution_at(
    study: optuna.Study,
    position: int,
    param_name: str,
    current: optuna.distributions.BaseDistribution,
) -> optuna.distributions.BaseDistribution:
    """Return the distribution a configuration starting at ``position`` draws ``param_name`` from.

    A configuration belongs to the search space that was in effect at its first position, so a
    repeats slot drawn partly before its range widened draws its remaining members from the
    range its first members drew from. Every other new draw is at or past the newest space's
    first position and draws from ``current``.

    :param optuna.Study study: The active study, whose identity record holds its search spaces.
    :param int position: The configuration's first draw position.
    :param str param_name: The parameter being drawn.
    :param optuna.distributions.BaseDistribution current: The configured distribution.
    :return optuna.distributions.BaseDistribution: The distribution to draw from.
    """
    record = _recorded_identity(study)
    if record is None:
        return current
    entries = [entry for entry in record["search_spaces"] if entry["first_position"] <= position]
    if not entries or entries[-1] is record["search_spaces"][-1]:
        return current
    recorded = entries[-1]["search_space"].get(param_name)
    if recorded is None:
        return current
    return _distribution(recorded_search_space({param_name: recorded})[param_name])


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
            fresh ``RandomSampler`` seeded by :func:`_position_seed` draws from the search space
            in effect at the configuration's first position (:func:`_distribution_at`).
        """
        position = _draw_position(study, trial)
        first = position
        if self._repeats is not None:
            count = len(self._repeats.values)
            if param_name == self._repeats.key:
                return self._repeats.values[position % count]
            position //= count
            first = position * count
        seed = _position_seed(
            self._base_seed, _study_namespace(study.study_name), position, param_name
        )
        sampler = optuna.samplers.RandomSampler(seed=seed)
        distribution = _distribution_at(study, first, param_name, param_distribution)
        return sampler.sample_independent(study, trial, param_name, distribution)


class _PositionGridSampler(optuna.samplers.GridSampler):
    """Grid sampler that gives each trial the cell at its draw position and never stops a study.

    The cell order comes from the study's search-space history
    (:func:`~phasesweep.engine.grid_order.grid_cell_order`), so a grown grid keeps every drawn
    position's cell and runs only cells it has not drawn. A study with no recorded identity,
    such as a dry run's in-memory sample, orders the sampler's own search space alone.
    """

    def __init__(
        self,
        search_space: Mapping[str, SearchParam],
        *,
        seed: int | None,
        repeats: Repeats | None = None,
    ) -> None:
        """Keep configurations shuffled and their repeat values consecutive.

        :param Mapping[str, SearchParam] search_space: The phase's sampler search space,
            repeat axis included.
        :param int | None seed: Seed for the configuration shuffle; None uses zero.
        :param Repeats | None repeats: Repeat axis to group in its configured value order.
        """
        super().__init__(grid_search_space(dict(search_space)), seed=seed)
        self._seed = seed
        self._repeats = repeats
        own = {
            name: dump_json(param)
            for name, param in search_space.items()
            if repeats is None or name != repeats.key
        }
        self._own_history: list[SearchSpaceEntry] = [
            {"search_space": json_object(own), "first_position": 0}
        ]
        self._names = grid_axis_names(list(own), repeats)
        self._orders: dict[str, list[GridCell]] = {}

    def _order(self, study: optuna.Study) -> list[GridCell]:
        """Return the study's cell order, computed once per recorded history.

        :param optuna.Study study: The active study.
        :return list[GridCell]: Cells by draw position.
        """
        record = _recorded_identity(study)
        history = self._own_history if record is None else record["search_spaces"]
        key = json.dumps(history, sort_keys=True, separators=(",", ":"))
        if key not in self._orders:
            self._orders[key] = grid_cell_order(
                history, seed=self._seed, repeats=self._repeats, phase_name=study.study_name
            )
        return self._orders[key]

    def before_trial(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Assign the cell at the trial's draw position, unless the trial already has one.

        The ``grid_id`` is the draw position itself, an index into the study's cell order. A
        rerun copies its source's ``grid_id``. New draws take positions ``0..M-1``, so no two
        trials that are not reruns share a cell under any ``n_jobs``.

        :param optuna.Study study: The active study.
        :param optuna.trial.FrozenTrial trial: The trial about to run.
        :raises RuntimeError: The position lies past the grid, which target
            validation and queued failed-cell retries rule out.
        """
        if "grid_id" in trial.system_attrs:
            return
        position = _draw_position(study, trial)
        cells = len(self._order(study))
        if position >= cells:
            raise RuntimeError(
                f"Trial {trial.number} of study {study.study_name!r} has draw position "
                f"{position}, past the {cells}-point grid."
            )
        study._storage.set_trial_system_attr(trial._trial_id, "grid_id", position)

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> object:
        """Return the value of ``param_name`` in the cell at the trial's ``grid_id``.

        :param optuna.Study study: The active study.
        :param optuna.trial.FrozenTrial trial: The trial being sampled for.
        :param str param_name: Name of the axis being sampled.
        :param optuna.distributions.BaseDistribution param_distribution: The axis's
            distribution, which the cell's value lies in.
        :raises RuntimeError: The trial has no ``grid_id``, which :meth:`before_trial` or its
            rerun source always assigns.
        :return object: The cell's value for the axis.
        """
        grid_id = trial.system_attrs.get("grid_id")
        if type(grid_id) is not int:
            raise RuntimeError(f"Trial {trial.number} has no grid_id to sample from.")
        return self._order(study)[grid_id][self._names.index(param_name)]

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

    Startup draws use :class:`_PositionRandomSampler`; candidate draws use
    thread-local streams seeded by position. At ``n_jobs: 1``, resume and top-up
    reproduce an uninterrupted run given the same trial values. Distinct draws
    can sample the same configuration.
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
            _position_seed(self._base_seed, _study_namespace(study.study_name), position, label)
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
        self._reread_draw: ContextVar[
            tuple[dict[str, optuna.distributions.BaseDistribution], optuna.study.StudyDirection]
            | None
        ] = ContextVar("cmaes_reread_draw", default=None)
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
        real one, so every storage write lands on the real trial. A re-read study
        supplies the current direction and search space when Optuna restores its optimizer.

        :param optuna.Study study: The active study; its name is part of the seed.
        :param optuna.trial.FrozenTrial trial: The trial being sampled for.
        :param dict[str, optuna.distributions.BaseDistribution] search_space: The
            relative search space.
        :return dict[str, object]: The sampled parameters, empty when CMA-ES defers
            to the independent sampler.
        """
        position = _draw_position(study, trial)
        self._cma_rng = LazyRandomState(
            _position_seed(self._base_seed, _study_namespace(study.study_name), position, "cmaes")
        )
        positioned = copy.copy(trial)
        positioned.number = position
        replay = (search_space, study.direction) if REREAD_FILL_ATTR in study.user_attrs else None
        token = self._reread_draw.set(replay)
        try:
            return super().sample_relative(study, positioned, search_space)
        finally:
            self._reread_draw.reset(token)

    def _restore_optimizer(
        self, completed_trials: list[optuna.trial.FrozenTrial]
    ) -> CmaClass | None:
        """Restore the optimizer from the completed trial holding the newest generation's snapshot.

        Stock CMA-ES restores from the highest-numbered snapshot. A rerun of the
        trial that told a generation copies that trial's snapshot, so under
        ``n_jobs > 1`` it can hold an older snapshot under a higher number than
        a generation told meanwhile; the newest generation wins, ties to the
        highest number.

        A re-read can have several complete batches after that snapshot. Replay all
        but the last consecutive complete batch, which Optuna tells and saves as
        part of the ask. When an observation is missing, the first incomplete batch
        receives new draws before any later batch can affect the optimizer.

        :param list[optuna.trial.FrozenTrial] completed_trials: The study's
            completed trials.
        :return CmaClass | None: Optimizer immediately before Optuna's next ask, or
            None when there is no snapshot to restore in an ordinary study.
        """
        carriers = [
            completed
            for completed in completed_trials
            if any(key.startswith(self._attr_key_optimizer) for key in completed.system_attrs)
        ]
        optimizer = None
        if carriers:
            newest = max(
                carriers,
                key=lambda carrier: (
                    carrier.system_attrs[self._attr_key_generation],
                    carrier.number,
                ),
            )
            optimizer = super()._restore_optimizer([newest])
        replay = self._reread_draw.get()
        if replay is None:
            return optimizer

        search_space, direction = replay
        transform = _SearchSpaceTransform(search_space, transform_step=True, transform_0_1=True)
        if optimizer is None:
            optimizer = self._init_optimizer(transform, direction)
        population = optimizer.population_size
        sign = 1 if direction == optuna.study.StudyDirection.MINIMIZE else -1
        batch = self._get_solution_trials(completed_trials, optimizer.generation)
        while len(batch) >= population:
            following = self._get_solution_trials(completed_trials, optimizer.generation + 1)
            if len(following) < population:
                break
            solutions = []
            for trial in batch[:population]:
                assert trial.value is not None
                solutions.append((transform.transform(trial.params), sign * trial.value))
            optimizer.tell(solutions)
            batch = following
        return optimizer


def _sampling_trial_for_reread(
    trial: optuna.trial.FrozenTrial, sampler: optuna.samplers.BaseSampler
) -> optuna.trial.FrozenTrial:
    """Keep measured draw positions without importing a fitted optimizer.

    CMA-ES's generation identifies the draw batch. Its optimizer snapshot stores the
    updates learned from scores and direction; later asks reconstruct updates from the
    new study's batches and values. TPE fits directly from the new study's history.

    :param optuna.trial.FrozenTrial trial: The trial the new reading study will add.
    :param optuna.samplers.BaseSampler sampler: The new reading study's sampler.
    :return optuna.trial.FrozenTrial: A trial carrying only reusable sampling state.
    """
    if not isinstance(sampler, _PositionCmaEsSampler):
        return trial
    retained = copy.copy(trial)
    retained.system_attrs = {
        key: value
        for key, value in trial.system_attrs.items()
        if not key.startswith(sampler._attr_key_optimizer)
    }
    return retained


def _build_sampler(
    cfg: Sampler,
    search_space: dict[str, SearchParam],
    n_jobs: int = 1,
    *,
    repeats: Repeats | None = None,
) -> optuna.samplers.BaseSampler:
    """Construct the Optuna sampler for a phase from its YAML ``sampler`` block.

    :param Sampler cfg: Parsed sampler config (type, seed, startup-trials, etc.).
    :param dict[str, SearchParam] search_space: The phase's validated sampler search space
        (:func:`phasesweep.config.search.sampler_search_space`), used to build the
        ``GridSampler`` grid; it includes the repeat axis.
    :param int n_jobs: Phase parallelism; enables TPE's ``constant_liar`` heuristic when
        ``n_jobs > 1``.
    :param Repeats | None repeats: The phase's repeats block; grid and seeded random run
        one configuration per slot of ``len(values)`` consecutive positions.
    :raises ValueError: The validated grid search space cannot be enumerated.
    :return optuna.samplers.BaseSampler: A configured sampler instance. A seeded stochastic
        sampler draws by draw position, so a study resumes and grows without repeating a
        draw; an unseeded one is Optuna's own, which only in-memory storage allows. A fixed
        configuration, whose search space is empty, gets a plain random sampler that is never
        consulted.
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
        return _PositionGridSampler(search_space, seed=cfg.seed, repeats=repeats)
    if cfg.type == "cmaes":
        if cfg.seed is None:
            return optuna.samplers.CmaEsSampler()
        return _PositionCmaEsSampler(seed=cfg.seed)
    assert_never(cfg.type)


def _suggest(trial: optuna.Trial, name: str, p: SearchParam) -> Any:
    """Dispatch to the right Optuna ``trial.suggest_*`` based on param type.

    :param optuna.Trial trial: The active Optuna trial.
    :param str name: Parameter name (used as the Optuna key).
    :param SearchParam p: The concrete search parameter from the phase's ``search_space``.
    :return float | int | str | bool | None: The sampled value. Type matches ``p``
        (``float``/``int``/categorical scalar).
    """
    if isinstance(p, FloatParam):
        return trial.suggest_float(name, p.low, p.high, step=p.step, log=p.log)
    if isinstance(p, IntParam):
        return trial.suggest_int(name, p.low, p.high, step=p.step, log=p.log)
    if isinstance(p, CategoricalParam):
        return trial.suggest_categorical(name, p.choices)
    assert_never(p)


GridAxisKind = Literal["int", "float", "categorical"]
"""How a grid axis of choices is registered with Optuna."""


def _grid_axis_kind(values: Sequence[object]) -> GridAxisKind:
    """Say how a grid axis's choices are registered: as an int or float range, or as choices.

    Optuna requires a categorical parameter's choices to be identical across a study's trials
    but lets an int or float range change its bounds, so a numeric axis registered as a range
    can gain values in place. Booleans are not numbers here, and an int past Optuna's exact
    float range stays a choice.

    :param Sequence[object] values: The axis's choices.
    :return GridAxisKind: ``int`` when every choice is an int within +-2^53, ``float`` when
        every choice is a float, otherwise ``categorical``.
    """
    if all(type(value) is int and abs(value) <= _OPTUNA_EXACT_INT_LIMIT for value in values):
        return "int"
    if all(type(value) is float for value in values):
        return "float"
    return "categorical"


def _suggest_all(trial: optuna.Trial, phase: Phase) -> dict[str, object]:
    """Suggest every sampler axis of a phase, the one path a trial's params are drawn through.

    A grid's numeric choices register as an int or float range spanning them
    (:func:`_grid_axis_kind`); the grid sampler returns the cell's value, so the trial records
    the configured value with its type. A param the trial already holds, as a rerun holds its
    source's, is returned as recorded.

    :param optuna.Trial trial: The active Optuna trial.
    :param Phase phase: Phase whose ``search_space`` and repeat axis are drawn.
    :return dict[str, object]: The drawn value of every axis, by name.
    """
    grid = phase.sampler.type == "grid"
    recorded = trial.params
    sampled: dict[str, object] = {}
    for name, param in sampler_search_space(phase).items():
        if name in recorded:
            # A rerun carries its source's value under its source's distribution, which a grown
            # space widened; suggesting it again would return the same value and only warn.
            sampled[name] = recorded[name]
            continue
        if grid and isinstance(param, CategoricalParam):
            kind = _grid_axis_kind(param.choices)
            if kind == "int":
                sampled[name] = trial.suggest_int(name, min(param.choices), max(param.choices))
                continue
            if kind == "float":
                sampled[name] = trial.suggest_float(name, min(param.choices), max(param.choices))
                continue
        sampled[name] = _suggest(trial, name, param)
    return sampled


def _phase_namespace(experiment_name: str, phase_name: str) -> str:
    """Return the name every study of one phase starts with.

    Each study of the phase appends ``::`` and a digest of the values it inherits and of its
    reading settings (:func:`phasesweep.engine.fingerprints.phase_study_name`); the namespace
    alone seeds the phase's draws, so every study of a phase draws alike. Experiment and phase
    names are safe names, so they hold no ``:``.

    :param str experiment_name: The experiment's name.
    :param str phase_name: The phase's name, configured or historical.
    :return str: ``<experiment>::<phase>``.
    """
    return f"{experiment_name}::{phase_name}"


def _study_namespace(study_name: str) -> str:
    """Return the phase namespace a study name starts with, without its address digest.

    :param str study_name: A phase study's name.
    :return str: Its ``<experiment>::<phase>`` part.
    """
    return "::".join(study_name.split("::")[:2])


def _belongs_to_phase(study_name: str, experiment_name: str, phase_name: str) -> bool:
    """Say whether a study holds trials of one phase, under any inherited values and reading.

    :param str study_name: Study name read from storage.
    :param str experiment_name: The experiment's name.
    :param str phase_name: The phase's name.
    :return bool: Whether the name is the phase's namespace with an address digest.
    """
    return study_name.startswith(f"{_phase_namespace(experiment_name, phase_name)}::")
