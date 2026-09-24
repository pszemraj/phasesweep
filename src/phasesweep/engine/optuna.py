"""Optuna sampler, trial-identity, and study-name helpers.

Storage construction lives in :mod:`phasesweep.engine.ledger`, the single
chokepoint; nothing here opens a ledger.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, assert_never

import optuna
from optuna.exceptions import ExperimentalWarning

from phasesweep.config import (
    CategoricalParam,
    Experiment,
    FloatParam,
    IntParam,
    Phase,
    Sampler,
    SearchParam,
    grid_search_space,
)
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GENERATION_ID_ATTR,
)


@dataclass(frozen=True)
class _TrialRef:
    """Identity of one trial recorded in a publication or storage snapshot.

    Reconciles a row against published evidence or external cleanup evidence:
    which trial it is, and which generation/attempt claimed it.
    Values are ``None`` when the trial never recorded that attribute (e.g. a
    row written by an older PhaseSweep). Published references additionally
    carry their terminal and complete count boundary. Those count fields are
    not part of a trial-row identity comparison.
    """

    trial_number: int
    generation_id: str | None
    attempt_id: str | None
    finished_trials: int | None = field(default=None, compare=False)
    completed_trials: int | None = field(default=None, compare=False)


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
    finished = sum(stats.counts.get(state, 0) for state in ("COMPLETE", "PRUNED", "FAIL"))
    completed = stats.counts.get("COMPLETE", 0)
    return (
        published_trial.finished_trials is None or finished >= published_trial.finished_trials
    ) and (
        published_trial.completed_trials is None or completed >= published_trial.completed_trials
    )


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


class _TrialNumberRandomSampler(optuna.samplers.RandomSampler):
    """Seeded random sampler whose draws survive process-local RNG restarts."""

    def __init__(self, *, seed: int) -> None:
        """Store the caller's seed without handing it to the base sampler.

        The base ``RandomSampler`` is constructed with ``seed=None`` because
        this sampler never draws through it directly; ``seed`` is instead
        mixed into a fresh per-draw seed by :meth:`sample_independent`, so
        continuation across process restarts stays deterministic.

        Args:
            seed: Base seed supplied by the phase's sampler config.

        """
        super().__init__(seed=None)
        self._base_seed = seed

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        """Sample from a deterministic stream position owned by the durable trial.

        Derives a fresh per-parameter seed from the base seed, study name,
        trial number, and parameter name, so the same (trial, param) pair
        always draws the same value regardless of process restarts or call
        ordering.

        Args:
            study: The active Optuna study (contributes ``study_name`` to the
                derived seed).
            trial: The trial being sampled for (contributes ``trial.number``).
            param_name: Name of the parameter being sampled.
            param_distribution: The parameter's Optuna distribution.

        Returns:
            The sampled value, drawn from a fresh :class:`optuna.samplers.RandomSampler`
            seeded deterministically from ``(base_seed, study_name, trial.number, param_name)``.

        """
        material = json.dumps(
            [self._base_seed, study.study_name, trial.number, param_name],
            separators=(",", ":"),
        ).encode()
        derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
        sampler = optuna.samplers.RandomSampler(seed=derived_seed)
        return sampler.sample_independent(study, trial, param_name, param_distribution)


def _build_sampler(
    cfg: Sampler, search_space: dict[str, SearchParam], n_jobs: int = 1
) -> optuna.samplers.BaseSampler:
    """Construct the Optuna sampler for a phase from its YAML ``sampler`` block.

    Args:
        cfg: Parsed sampler config (type, seed, startup-trials, etc.).
        search_space: The phase's validated search space, used to build the
            ``GridSampler`` grid.
        n_jobs: Phase parallelism; enables TPE's ``constant_liar`` heuristic
            when ``n_jobs > 1``.

    Returns:
        A configured :class:`optuna.samplers.BaseSampler` subclass instance.

    Raises:
        ValueError: The validated grid search space cannot be enumerated.

    """
    if cfg.type == "tpe":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=ExperimentalWarning,
                message=r"Argument ``constant_liar`` is an experimental feature.*",
            )
            return optuna.samplers.TPESampler(
                seed=cfg.seed,
                n_startup_trials=cfg.n_startup_trials,
                constant_liar=(n_jobs > 1),
            )
    if cfg.type == "random":
        if cfg.seed is None:
            return optuna.samplers.RandomSampler()
        return _TrialNumberRandomSampler(seed=cfg.seed)
    if cfg.type == "grid":
        return optuna.samplers.GridSampler(grid_search_space(search_space), seed=cfg.seed)
    if cfg.type == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=cfg.seed)
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
