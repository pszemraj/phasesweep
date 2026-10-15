"""The run.log progress lines a phase writes: its header, one RERUN line per rerun, one TRIAL line per trial."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import optuna

from phasesweep.config.models import Phase
from phasesweep.engine.optuna import _trial_state_label
from phasesweep.engine.repeats import repeat_axis
from phasesweep.engine.state import RERUN_OF_ATTR, TRIAL_OUTCOME_ATTR, _parsed_trial_outcome

# The lines keep the phase channel that docs and run.log readers match on.
log = logging.getLogger("phasesweep.engine.phase")


def _log_phase_header(
    phase: Phase, study_name: str, *, finished: int, remaining: int, reruns: int
) -> None:
    """Log the line that opens a phase's work in run.log.

    :param Phase phase: The phase about to run or replay.
    :param str study_name: Its Optuna study name.
    :param int finished: Trials that count toward its target.
    :param int remaining: Trials still needed to reach ``phase.n_trials``.
    :param int reruns: Reruns it owes for interrupted trials.
    """
    log.info(
        "phase=%s study=%s finished=%d remaining=%d reruns=%d n_jobs=%d",
        phase.name,
        study_name,
        finished,
        remaining,
        reruns,
        phase.n_jobs,
    )


def _log_finished_trial(phase: Phase, trial: optuna.trial.FrozenTrial, finished: int) -> None:
    """Log one INFO line for a finished trial, the progress stream a ``tail -f`` follows.

    A COMPLETE trial that is infeasible, whether by a constraint or by the
    metric's bounds, reports ``INFEASIBLE`` in place of its Optuna state, as
    selection treats it, and an interrupted trial reports ``INTERRUPTED``. The
    reason is the first line of the recorded outcome cause. A repeats phase adds
    the trial's repeat value so a reader can see which seed ran, and a rerun
    names the interrupted trial it reran. Params stay out of the line; the
    phase's WINNER line carries the winner's.

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
    """Log one INFO line per rerun scheduled, naming the interrupted trial and why it stopped.

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
