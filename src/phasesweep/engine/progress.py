"""Engine progress lines and process-scoped warnings.

A phase writes its header, one RERUN line per rerun and one TRIAL line per trial.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Sequence

import optuna

from phasesweep.config.models import Phase
from phasesweep.engine.optuna import _trial_state_label
from phasesweep.engine.repeats import repeat_axis
from phasesweep.engine.state import RERUN_OF_ATTR, TRIAL_OUTCOME_ATTR, _parsed_trial_outcome

# The lines keep the phase channel that docs and run.log readers match on.
log = logging.getLogger("phasesweep.engine.phase")
_WARNED: set[Hashable] = set()


def _warn_once(logger: logging.Logger, key: Hashable, message: str, *args: object) -> None:
    """Log a warning once per process for its namespaced identity.

    Callers include the warning kind in ``key`` so separate diagnostics never
    suppress one another. The supplied logger retains each diagnostic's channel.

    :param logging.Logger logger: Logger that emits the warning.
    :param Hashable key: Warning kind and the identities defining its scope.
    :param str message: Logging format string.
    :param object args: Values substituted into the format string.
    """
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message, *args)


def _log_phase_header(
    phase: Phase,
    study_name: str,
    *,
    finished: int,
    remaining: int,
    reruns: int,
    logger: logging.Logger = log,
) -> None:
    """Log the line that opens a phase's work in run.log.

    :param Phase phase: The phase about to run or replay.
    :param str study_name: Its Optuna study name.
    :param int finished: Trials that count toward its target.
    :param int remaining: Trials still needed to reach ``phase.n_trials``.
    :param int reruns: Reruns it owes for interrupted trials.
    :param logging.Logger logger: Progress channel for a live phase or its preview.
    """
    logger.info(
        "phase=%s study=%s finished=%d remaining=%d reruns=%d n_jobs=%d",
        phase.name,
        study_name,
        finished,
        remaining,
        reruns,
        phase.n_jobs,
    )


def _log_finished_trial(phase: Phase, trial: optuna.trial.FrozenTrial, finished: int) -> None:
    """Log one finished trial's state, outcome reason, and duration.

    Use ``INFEASIBLE`` or ``INTERRUPTED`` where applicable. Repeat values and rerun
    sources are included; params remain on the phase's WINNER line.

    :param Phase phase: The phase the trial belongs to.
    :param optuna.trial.FrozenTrial trial: The just-finished trial.
    :param int finished: Trials the phase's study holds that count toward its
        target, this one included unless it was interrupted.
    """
    parsed = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    value = "none" if trial.value is None else f"{trial.value:g}"
    duration = "none" if trial.duration is None else f"{trial.duration.total_seconds():.1f}s"
    axis = repeat_axis(phase)
    repeat = ""
    if axis is not None and axis.key in trial.params:
        repeat = f" repeat={axis.key}={trial.params[axis.key]}"
    source = trial.user_attrs.get(RERUN_OF_ATTR)
    rerun = "" if source is None else f" rerun_of={source}"
    reason = _first_cause_line(trial)
    log.info(
        "phase=%s TRIAL trial=%d state=%s outcome=%s value=%s duration=%s%s%s (%d/%d finished)%s",
        phase.name,
        trial.number,
        _trial_state_label(trial),
        "unknown" if parsed is None else parsed.outcome,
        value,
        duration,
        repeat,
        rerun,
        finished,
        phase.n_trials,
        "" if reason is None else f" reason={reason}",
    )


def _first_cause_line(trial: optuna.trial.FrozenTrial) -> str | None:
    """Return the first line of a trial's recorded outcome cause.

    :param optuna.trial.FrozenTrial trial: Trial whose outcome record is read.
    :return str | None: The cause's first nonblank line, or None when it recorded none.
    """
    parsed = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    lines = parsed.cause.strip().splitlines() if parsed and parsed.cause else []
    return lines[0] if lines else None


def _log_reruns(phase: Phase, study: optuna.Study, reruns: Sequence[tuple[int, int]]) -> None:
    """Log each interrupted rerun or failed grid-cell retry and its source's cause.

    :param Phase phase: The phase whose study scheduled the reruns.
    :param optuna.Study study: That study.
    :param Sequence[tuple[int, int]] reruns: ``(source, rerun)`` trial numbers, as
        :func:`~phasesweep.engine.study_policy._record_reruns` returns them.
    """
    if not reruns:
        return
    trials = {trial.number: trial for trial in study.get_trials(deepcopy=False)}
    for source, rerun in reruns:
        reason = _first_cause_line(trials[source])
        log.info(
            "phase=%s RERUN trial=%d of=%d%s",
            phase.name,
            rerun,
            source,
            "" if reason is None else f" reason={reason}",
        )
