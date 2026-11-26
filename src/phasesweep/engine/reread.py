"""Re-read a phase's kept attempts into the study its current reading settings address.

Optuna keeps a finished trial's value fixed, so results read under other reading settings
(:func:`phasesweep.engine.fingerprints.reading_settings`) live in another study of the same
phase, which the same address finds (:func:`phasesweep.engine.fingerprints.phase_study_name`).
The trainer's work does not change with the reading, so a study never retrains an attempt
another study of its phase already holds under the same training identity and inherited values.

One procedure serves an empty study, a returning one, the run between launch batches and every
read that writes nothing (dry run, preflight, ``validate`` and ``status``): :func:`walk` follows
the study's next slots in the order a run fills them (:func:`next_slot`) and gives each slot the
oldest compatible attempt that trained it. A terminal attempt is read again under the current
settings through the one path recovery reads a finished attempt by
(:func:`phasesweep.engine.attempts.read_finished_attempt`); an interrupted one is copied as
interrupted, so the ordinary rerun continues it from its checkpoint. The walk stops at the first
slot no kept attempt fills, where the run launches. A run writes each adopted row in one storage
write, so an interruption keeps every row written and the next run continues from them.
"""

from __future__ import annotations

import copy
import heapq
import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple

import optuna

from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import phase_draws, sampler_search_space
from phasesweep.engine.attempts import (
    _pending_evidence_refusal,
    _record_stale_trial_outcome,
    _trial_attempt_record,
    read_finished_attempt,
)
from phasesweep.engine.errors import (
    PhaseTimeoutError,
    StudyFingerprintMismatchError,
    WinnerIntegrityError,
    phase_timeout,
)
from phasesweep.engine.evidence import _earlier_attempts
from phasesweep.engine.fingerprints import (
    READING_KEY,
    _changed_payload_keys,
    _identity_payload,
    _identity_record,
    _inherited_values,
    _judge_identity,
    _judge_record,
    _payload_search_space,
    _recorded_inheritance,
    _same_value,
    _semantic_payload_digest,
    _stamp_identity,
    _with_search_space,
    format_changed_keys,
    phase_study_name,
    reading_settings,
)
from phasesweep.engine.grid_order import grid_cell_order
from phasesweep.engine.optuna import (
    _distribution,
    _draw_positions,
    _finished_trial_count,
    _measured_trial_count,
    _sampling_trial_for_reread,
    _suggest_all,
    _trial_run_state,
)
from phasesweep.engine.paths import _experiment_dir
from phasesweep.engine.provenance import _load_generation_experiment
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FAILURE_REASON_ATTR,
    FINISHED_TRIAL_STATES,
    OVERRIDES_ATTR,
    PHASE_DECISION_ATTR,
    RERUN_OF_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRIAL_OUTCOME_ATTR,
    PhaseIdentityRecord,
    _completed_trainer_duration,
    _parsed_trial_outcome,
    _recorded_identity,
    _required_trial_dir,
    _trial_outcome_payload,
    constraint_attr,
)
from phasesweep.engine.study_policy import (
    _abort_decision,
    _at_accepted_partial_target,
    _config_retires_abort,
    _counted_outcome,
    _initialize_study_schema,
    _load_accepted_partial_decision,
    _load_phase_policy_state,
    _parse_accepted_partial_decision,
    _prelaunch_trial_environment,
    _record_accepted_partial_decision,
    _record_recovery_boundary,
    _recorded_outcome_sequence,
    _replayed_streak_abort,
    _scheduled_reruns,
)
from phasesweep.engine.trial import TrialResult, _failed_trial, record_trial_result
from phasesweep.engine.trial_dirs import (
    DirectoryLaunch,
    ProducingAttempt,
    directory_history,
    directory_launches,
    producing_attempts,
)
from phasesweep.errors import OperatorAction, PhaseSweepError
from phasesweep.evidence.baseline import read_evidence_baseline
from phasesweep.runtime.json import JsonObject, json_object
from phasesweep.runtime.terminal import REPORT_LOGGER, plural

log = logging.getLogger(__name__)
report = logging.getLogger(REPORT_LOGGER)

#: The recorded outcomes a re-read replaces; an interruption or a fatal outcome stands as it was.
_REREAD_OUTCOMES = frozenset({"success", "failure"})
#: Outcomes of attempts another study can take: a fatal outcome is no training result.
_ADOPTED_OUTCOMES = _REREAD_OUTCOMES | {"interrupted"}
_ADAPTIVE_SAMPLERS = frozenset({"tpe", "cmaes"})
_COMPLETE = optuna.trial.TrialState.COMPLETE


@dataclass
class _AttrSheet:
    """A trial's user attributes being rebuilt before the trial is written in one storage write."""

    user_attrs: dict[str, object]

    def set_user_attr(self, key: str, value: object) -> None:
        """Set one attribute, as :meth:`optuna.Trial.set_user_attr` would.

        :param str key: Attribute name.
        :param object value: JSON-serializable value.
        """
        self.user_attrs[key] = value


def _reread_record(record: PhaseIdentityRecord, reading: JsonObject) -> PhaseIdentityRecord:
    """Return a study's identity record as it reads under other reading settings.

    :param PhaseIdentityRecord record: The study's record.
    :param JsonObject reading: The reading settings to record.
    :return PhaseIdentityRecord: The record with its search-space history and new settings.
    """
    payload = {**record["payload"], READING_KEY: reading}
    return _identity_record(
        _semantic_payload_digest(payload),
        payload,
        list(record["search_spaces"]),
    )


def _trainer_seconds(
    trial: optuna.trial.FrozenTrial, *, attempts: Mapping[str, ProducingAttempt] | None = None
) -> float | None:
    """Return a finished trial's trainer time when its trainer exited successfully.

    The committed trainer exit comes from the ledger. An evidence worker may have replaced
    the attempt's process slot after the trainer exited. Only a trial whose recorded outcome
    is a success or failure is read again.

    :param optuna.trial.FrozenTrial trial: A trial holding a physical attempt.
    :param Mapping[str, ProducingAttempt] | None attempts: Resolved launch rows, when available.
    :return float | None: The trainer's seconds, or ``None`` when the trial is copied unread.
    """
    attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    producer = (
        attempts.get(attempt_id) if attempts is not None and isinstance(attempt_id, str) else None
    )
    producing = trial if producer is None else producer.trial
    outcome = _parsed_trial_outcome(producing.user_attrs.get(TRIAL_OUTCOME_ATTR))
    if (
        producing.state not in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL}
        or outcome is None
        or outcome.outcome not in _REREAD_OUTCOMES
    ):
        return None
    return _completed_trainer_duration(producing, None, rechecking_evidence=True)


def _reread_trial(
    experiment: Experiment,
    phase: Phase,
    trial: optuna.trial.FrozenTrial,
    trials: Mapping[int, optuna.trial.FrozenTrial],
    producing: dict[str, Experiment | WinnerIntegrityError],
    attempts: Mapping[str, ProducingAttempt],
    launches: Mapping[str, Sequence[DirectoryLaunch]],
    *,
    deadline: float | None,
    reader_generation_id: str,
) -> optuna.trial.FrozenTrial | None:
    """Read one finished attempt again under the current settings.

    Evidence that is gone or unreadable now fails the trial with the reason its reader gives,
    as a live extraction would, and the phase's retry rules decide whether it runs again.

    :param Experiment experiment: Current config, whose reading settings apply.
    :param Phase phase: The trial's phase.
    :param optuna.trial.FrozenTrial trial: The row holding the attempt.
    :param Mapping[int, optuna.trial.FrozenTrial] trials: That row's study's trials by number.
    :param dict[str, Experiment | WinnerIntegrityError] producing: Producing configurations by
        generation, filled as they are loaded.
    :param Mapping[str, ProducingAttempt] attempts: Producing rows by physical attempt ID.
    :param Mapping[str, Sequence[DirectoryLaunch]] launches: Physical launches by directory.
    :param float | None deadline: Absolute invocation deadline.
    :param str reader_generation_id: Generation running the reader.
    :raises PhaseTimeoutError: The invocation deadline stopped a read.
    :raises TrackingTargetUnavailableError: The tracker gave no answer, so the verdict waits for
        a later run; the rows already adopted stay and the next run continues.
    :return optuna.trial.FrozenTrial | None: The re-read row, or ``None`` when the attempt has
        no finished trainer to read.
    """
    seconds = _trainer_seconds(trial, attempts=attempts)
    outcome = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    owner = attempts.get(attempt_id) if isinstance(attempt_id, str) else None
    physical = trial if owner is None else owner.trial
    overrides_json = physical.user_attrs.get(OVERRIDES_ATTR)
    if seconds is None or outcome is None or not isinstance(overrides_json, str):
        return None
    try:
        directory = _required_trial_dir(physical)
        record = _trial_attempt_record(physical, phase.name, directory)
    except ValueError:
        return None
    if record is None or record.generation_id is None:
        return None
    result: TrialResult
    try:
        baseline = read_evidence_baseline(record.attempt_dir)
        if record.generation_id not in producing:
            try:
                producing[record.generation_id] = _load_generation_experiment(
                    _experiment_dir(experiment), record.generation_id
                )
            except WinnerIntegrityError as exc:
                producing[record.generation_id] = exc
        launched = producing[record.generation_id]
        if isinstance(launched, WinnerIntegrityError):
            raise ValueError(str(launched))
        result, _ = read_finished_attempt(
            directory_history(
                launches.get(directory.name, ()),
                attempt_id=record.attempt_id,
                launched_at=baseline.launched_at,
                ancestors=_earlier_attempts(trials, trial, attempts=attempts),
            ),
            physical,
            record,
            producing=launched,
            reading=experiment,
            current=experiment,
            trainer_duration=seconds,
            baseline=baseline,
            deadline=deadline,
            reader_generation_id=reader_generation_id,
        )
    except ValueError as exc:
        result = _failed_trial(rc=0, duration=seconds, failure_reason=str(exc))
    if result.deadline_exhausted:
        raise _reread_timeout(experiment, phase)
    if result.evidence_unreadable and result.failure_reason is not None:
        raise _pending_evidence_refusal({record.attempt_id: result.failure_reason})
    sheet = _AttrSheet(dict(trial.user_attrs))
    record_trial_result(sheet, result, overrides_json=overrides_json)
    failure = result.failure_reason
    if failure is None:
        sheet.user_attrs.pop(FAILURE_REASON_ATTR, None)
    else:
        sheet.user_attrs[FAILURE_REASON_ATTR] = failure
    sheet.user_attrs[TRIAL_OUTCOME_ATTR] = _trial_outcome_payload(
        outcome.sequence,
        "success" if failure is None else "failure",
        metric=result.metric if failure is None else None,
        cause=failure,
    )
    return optuna.trial.FrozenTrial(
        number=trial.number,
        state=optuna.trial.TrialState.COMPLETE if failure is None else optuna.trial.TrialState.FAIL,
        value=result.metric if failure is None else None,
        datetime_start=trial.datetime_start,
        datetime_complete=trial.datetime_complete,
        params=trial.params,
        distributions=trial.distributions,
        user_attrs=sheet.user_attrs,
        system_attrs=trial.system_attrs,
        intermediate_values=trial.intermediate_values,
        trial_id=trial._trial_id,
    )


def _reread_timeout(experiment: Experiment, phase: Phase) -> PhaseTimeoutError:
    """Describe a run budget that stopped re-reading before it committed every row.

    :param Experiment experiment: The invocation whose run deadline expired.
    :param Phase phase: The phase reading its kept evidence.
    :return PhaseTimeoutError: The run's timeout refusal and resume instructions.
    """
    return phase_timeout(
        f"Phase {phase.name!r} hit its run deadline while re-reading kept evidence. "
        "The study keeps the rows already re-read.",
        scope="run",
        experiment=experiment,
    )


def _require_reread_time(experiment: Experiment, phase: Phase, deadline: float | None) -> None:
    """Refuse to start or commit a re-read after the invocation's deadline.

    :param Experiment experiment: Current configuration for recovery instructions.
    :param Phase phase: Phase re-reading evidence.
    :param float | None deadline: Absolute invocation deadline.
    :raises PhaseTimeoutError: The invocation deadline has passed.
    """
    if deadline is not None and time.monotonic() >= deadline:
        raise _reread_timeout(experiment, phase)


def adoption_identity(
    experiment: Experiment,
    phase: Phase,
    inherited: Mapping[str, object],
    others: Sequence[optuna.Study],
) -> tuple[PhaseIdentityRecord, optuna.Study] | None:
    """Choose the identity an empty study takes so its draws line up with kept attempts.

    A seeded draw depends on the search-space history in effect at its position, so an empty
    study takes the newest study of the phase holding trials under the same inherited values and
    a training identity the current config continues, judged with the current reading settings.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase whose addressed study holds no trials.
    :param Mapping[str, object] inherited: The parent layer the phase inherits now.
    :param Sequence[optuna.Study] others: The phase's other studies, oldest first.
    :raises StudyFingerprintMismatchError: Studies with these inherited values exist but each one's
        training identity differs from the current config; the refusal names what changed.
    :return tuple[PhaseIdentityRecord, optuna.Study] | None: The record and the study it comes
        from, or ``None`` when no study holds these values.
    """
    reading = reading_settings(experiment, phase)
    values = _inherited_values(phase, inherited)
    refusal: StudyFingerprintMismatchError | None = None
    for study in reversed(others):
        record = _recorded_identity(study)
        trials = study.get_trials(deepcopy=False)
        if (
            record is None
            or not trials
            or not _same_value(record["payload"].get("inherited_effective_overrides"), values)
        ):
            continue
        candidate = _reread_record(record, reading)
        try:
            verdict = _judge_record(
                candidate, trials, study.study_name, experiment, phase, inherited
            )
        except StudyFingerprintMismatchError as exc:
            refusal = refusal or exc
            continue
        return verdict.record or candidate, study
    if refusal is not None:
        raise refusal
    return None


def stamp_adoption_identity(
    experiment: Experiment,
    phase: Phase,
    inherited: Mapping[str, object],
    study: optuna.Study,
    others: Sequence[optuna.Study],
) -> None:
    """Bind an empty study to the identity :func:`adoption_identity` chooses, before any row.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param Mapping[str, object] inherited: The parent layer the phase inherits now.
    :param optuna.Study study: The addressed study, writable.
    :param Sequence[optuna.Study] others: The phase's other studies, oldest first.
    :raises StudyFingerprintMismatchError: See :func:`adoption_identity`.
    """
    if _recorded_identity(study) is not None or study.get_trials(deepcopy=False):
        return
    chosen = adoption_identity(experiment, phase, inherited, others)
    if chosen is not None:
        _initialize_study_schema(study)
        _stamp_identity(study, chosen[0])


class _Candidate(NamedTuple):
    """A kept attempt another study of the phase holds, as a slot of this study can take it."""

    owner: ProducingAttempt
    #: Draw position in the study that launched it.
    position: int
    #: Whether it finished with a success or failure; otherwise it was interrupted.
    terminal: bool

    @property
    def rereads(self) -> bool:
        """Say whether taking the attempt reads its evidence again under the current settings."""
        return self.terminal and _trainer_seconds(self.owner.trial) is not None


def _compatible_studies(
    experiment: Experiment, phase: Phase, study: optuna.Study, others: Sequence[optuna.Study]
) -> list[optuna.Study]:
    """Return the other studies whose attempts trained what this study's would.

    Their recorded payload, given this study's reading settings and search space, must equal the
    current one: the same trainer identity and inherited parent values. A draw trained under
    another parent value is never adopted, since its score would describe other training.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param optuna.Study study: Study with a recorded identity.
    :param Sequence[optuna.Study] others: The phase's studies, oldest first.
    :return list[optuna.Study]: Compatible studies, oldest first.
    """
    record = _recorded_identity(study)
    if record is None:
        return []
    payload = _identity_payload(experiment, phase, _recorded_inheritance(record))
    space = _payload_search_space(payload)
    compatible = []
    for other in others:
        if other.study_name == study.study_name or (identity := _recorded_identity(other)) is None:
            continue
        source = _with_search_space(
            {**identity["payload"], READING_KEY: payload[READING_KEY]}, space
        )
        if _same_value(source, payload):
            compatible.append(other)
    return compatible


def _allocation_order(pool: Sequence[_Candidate]) -> list[_Candidate]:
    """Order kept attempts oldest first: each study's by row number, the studies by start time.

    Optuna records a naive local start time, which a changed time zone can move past a later
    row's, so it only interleaves studies and never reorders rows within one.

    :param Sequence[_Candidate] pool: Kept attempts in any order.
    :return list[_Candidate]: The attempts in allocation order.
    """
    by_study: dict[str, list[_Candidate]] = {}
    for candidate in pool:
        by_study.setdefault(candidate.owner.study.study_name, []).append(candidate)
    rows = [
        sorted(candidates, key=lambda candidate: candidate.owner.trial.number)
        for _, candidates in sorted(by_study.items())
    ]

    def started(candidate: _Candidate) -> float:
        start = candidate.owner.trial.datetime_start
        return 0.0 if start is None else start.timestamp()

    return list(heapq.merge(*rows, key=started))


def _draw_params_in_domain(
    params: Mapping[str, object], domain: Mapping[str, optuna.distributions.BaseDistribution]
) -> bool:
    """Check whether the current search space could draw these exact parameters.

    :param Mapping[str, object] params: Parameters of a kept attempt.
    :param Mapping[str, optuna.distributions.BaseDistribution] domain: Current search parameters.
    :return bool: Whether the parameters match the keys, values and types the study can draw.
    """
    if params.keys() != domain.keys():
        return False
    for name, value in params.items():
        distribution = domain[name]
        try:
            internal = distribution.to_internal_repr(value)
        except (ValueError, OverflowError):
            return False
        # Optuna can coerce equal values such as 1.0 and 1, but slot matching cannot.
        if not distribution._contains(internal) or not _same_value(
            value, distribution.to_external_repr(internal)
        ):
            return False
    return True


def _domain(phase: Phase) -> dict[str, optuna.distributions.BaseDistribution]:
    """Return the distributions the phase's sampler draws from.

    :param Phase phase: The phase.
    :return dict[str, optuna.distributions.BaseDistribution]: Distribution by parameter name.
    """
    return {name: _distribution(param) for name, param in sampler_search_space(phase).items()}


def kept_attempts(
    experiment: Experiment, phase: Phase, study: optuna.Study, others: Sequence[optuna.Study]
) -> list[_Candidate]:
    """Collect the attempts compatible studies hold and this study does not, oldest first.

    An attempt comes from the study that launched it. When that study is gone, the oldest
    compatible study holding a copy supplies it, so deleting a study never retrains its work.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param optuna.Study study: Study with a recorded identity.
    :param Sequence[optuna.Study] others: The phase's studies.
    :return list[_Candidate]: Terminal and interrupted attempts with parameters the current
        search space can draw.
    """
    compatible = _compatible_studies(experiment, phase, study, others)
    if not compatible:
        return []
    names = {other.study_name for other in compatible}
    present = {trial.user_attrs.get(ATTEMPT_ID_ATTR) for trial in study.get_trials(deepcopy=False)}
    domain = _domain(phase)
    positions: dict[str, dict[int, int]] = {}
    pool = []
    owners = producing_attempts((study, *compatible))
    for other in compatible:
        rows = {trial.number: trial for trial in other.get_trials(deepcopy=False)}
        for trial in rows.values():
            attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
            if isinstance(attempt_id, str) and attempt_id not in owners:
                owners[attempt_id] = ProducingAttempt(other, trial, rows)
    for attempt_id, owner in owners.items():
        if attempt_id in present or owner.study.study_name not in names:
            continue
        outcome = _parsed_trial_outcome(owner.trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
        if (
            not owner.trial.state.is_finished()
            or outcome is None
            or outcome.outcome not in _ADOPTED_OUTCOMES
            or not _draw_params_in_domain(owner.trial.params, domain)
        ):
            continue
        name = owner.study.study_name
        if name not in positions:
            positions[name] = _draw_positions(owner.rows.values())
        pool.append(
            _Candidate(owner, positions[name][owner.trial.number], outcome.outcome != "interrupted")
        )
    return _allocation_order(pool)


class Slot(NamedTuple):
    """The next trial a study's run would allocate."""

    #: The row this slot reruns, or ``None`` for a new draw.
    rerun: optuna.trial.FrozenTrial | None
    #: The queued WAITING rerun that holds the slot, if one does.
    waiting: optuna.trial.FrozenTrial | None = None


def next_slot(study: optuna.Study, phase: Phase) -> Slot | None:
    """Return the trial a run of the study allocates next: the one budget rule.

    A queued rerun runs first, then the reruns :func:`_scheduled_reruns` owes (interrupted trials,
    then failed cells or a failed fixed configuration a raised target retries), then a new draw.
    Every slot spends one trial of the remaining target. A study that accepted a partial result
    at the requested target replays that result and allocates nothing until ``n_trials`` rises.

    :param optuna.Study study: A live study or an in-memory copy of one.
    :param Phase phase: The phase whose target applies.
    :return Slot | None: The slot, or ``None`` when the finished trials meet the target or an
        accepted partial result holds it.
    """
    trials = study.get_trials(deepcopy=False)
    if _finished_trial_count(trials) >= phase.n_trials or _at_accepted_partial_target(
        _load_accepted_partial_decision(study), phase
    ):
        return None
    by_number = {trial.number: trial for trial in trials}
    for trial in trials:
        if trial.state == optuna.trial.TrialState.WAITING and RERUN_OF_ATTR in trial.user_attrs:
            source = trial.user_attrs[RERUN_OF_ATTR]
            assert type(source) is int  # _draw_positions validated the link.
            return Slot(by_number[source], trial)
    scheduled = _scheduled_reruns(study, trials, phase)
    return Slot(scheduled[0]) if scheduled else Slot(None)


class Adopt(NamedTuple):
    """A slot a kept attempt fills."""

    candidate: _Candidate
    #: The row to append, after the WAITING rerun it settles, if any.
    row: optuna.trial.FrozenTrial
    waiting: optuna.trial.FrozenTrial | None
    #: The abort the target's failure policy records with the row if a launch follows it.
    abort: JsonObject | None


class Launch(NamedTuple):
    """A slot no kept attempt fills, so the run launches a trainer for it."""

    slot: Slot
    #: The in-memory trial the slot's proposal was asked on, which no run holds.
    asked: int | None


#: Gives the row content an attempt reads as under the current settings.
Reader = Callable[[_Candidate], optuna.trial.FrozenTrial]


def mirror_study(
    experiment: Experiment,
    phase: Phase,
    study: optuna.Study | None,
    *,
    study_name: str,
    record: PhaseIdentityRecord | None = None,
    upto: int | None = None,
) -> optuna.Study:
    """Copy a study into memory, so a walk can project its next slots without writing it.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param optuna.Study | None study: The study to copy, or ``None`` for one that does not exist.
    :param str study_name: The study's address.
    :param PhaseIdentityRecord | None record: Identity to bind the copy to, defaults to the study's.
    :param int | None upto: Copy only the trials numbered below this, defaults to all.
    :return optuna.Study: The in-memory copy.
    """
    from phasesweep.engine.ledger import open_preview_study

    mirror = open_preview_study(experiment, phase, study_name=study_name)
    if study is not None:
        mirror.add_trials(study.get_trials(deepcopy=False)[:upto])
        for key, value in json_object(study.user_attrs).items():
            mirror.set_user_attr(key, value)
    if record is not None:
        _stamp_identity(mirror, record)
    return mirror


def _settle_waiting(study: optuna.Study, number: int) -> None:
    """Settle a queued rerun that a kept attempt fills, so it cannot launch again.

    :param optuna.Study study: The study holding the WAITING rerun.
    :param int number: Its trial number.
    """
    trial = study.get_trials(deepcopy=False)[number]
    study._storage.set_trial_state_values(trial._trial_id, optuna.trial.TrialState.RUNNING)
    _record_stale_trial_outcome(
        study, trial, cause="a kept attempt of the phase finished this rerun"
    )
    study.tell(number, state=optuna.trial.TrialState.FAIL)


def _finish_asked(study: optuna.Study, trial_id: int, row: optuna.trial.FrozenTrial) -> None:
    """Turn the in-memory trial a slot's proposal was asked on into the row that fills it.

    :param optuna.Study study: An in-memory copy.
    :param int trial_id: The asked trial's storage ID.
    :param optuna.trial.FrozenTrial row: The row, whose parameters the trial already holds.
    """
    for key, value in row.user_attrs.items():
        study._storage.set_trial_user_attr(trial_id, key, value)
    study._storage.set_trial_state_values(
        trial_id, row.state, None if row.value is None else [row.value]
    )


def _adopted_row(
    phase: Phase,
    study: optuna.Study,
    slot: Slot,
    candidate: _Candidate,
    content: optuna.trial.FrozenTrial,
    proposal: optuna.trial.FrozenTrial | None,
    sequence: int,
) -> optuna.trial.FrozenTrial:
    """Build the row a slot takes from a kept attempt's content as read under this study.

    The slot's place comes from this study: a rerun names the row it reruns and keeps its
    sampling state; a new draw takes the proposal's state, or, for an adaptive sampler, the
    attempt's own without any CMA-ES optimizer fitted to another reading. The outcome takes this
    study's next completion sequence, and the trainer environment stays the attempt's own.

    :param Phase phase: The phase, whose search space an adaptive new draw takes.
    :param optuna.Study study: The study, or its in-memory copy.
    :param Slot slot: The slot being filled.
    :param _Candidate candidate: The kept attempt.
    :param optuna.trial.FrozenTrial content: The attempt's row as read under this study.
    :param optuna.trial.FrozenTrial | None proposal: The study's own draw for a seeded new slot.
    :param int sequence: The completion sequence the outcome takes.
    :return optuna.trial.FrozenTrial: The row.
    """
    attrs = dict(content.user_attrs)
    outcome = _parsed_trial_outcome(attrs.get(TRIAL_OUTCOME_ATTR))
    assert outcome is not None  # kept_attempts takes only attempts with an outcome.
    attrs[TRIAL_OUTCOME_ATTR] = _trial_outcome_payload(
        sequence, outcome.outcome, metric=outcome.metric, cause=outcome.cause, policy=outcome.policy
    )
    if not attrs.get(TRAINER_ENV_DIGEST_ATTR):
        environment = _prelaunch_trial_environment(candidate.owner.study, candidate.owner.trial)
        if environment is not None:
            attrs[TRAINER_ENV_DIGEST_ATTR] = environment
    if slot.rerun is not None:
        source = slot.waiting or slot.rerun
        attrs[RERUN_OF_ATTR] = source.number
        distributions, system_attrs = source.distributions, dict(source.system_attrs)
    else:
        attrs.pop(RERUN_OF_ATTR, None)
        if proposal is not None:
            distributions, system_attrs = proposal.distributions, dict(proposal.system_attrs)
        else:
            sampled = _sampling_trial_for_reread(content, study.sampler)
            distributions = _domain(phase)
            system_attrs = dict(sampled.system_attrs)
    return optuna.trial.FrozenTrial(
        number=len(study.get_trials(deepcopy=False)) if proposal is None else proposal.number,
        state=content.state,
        value=content.value,
        datetime_start=content.datetime_start,
        datetime_complete=content.datetime_complete,
        params=dict(content.params),
        distributions=distributions,
        user_attrs=attrs,
        system_attrs=system_attrs,
        intermediate_values=dict(content.intermediate_values),
        trial_id=-1,
    )


def _placeholder(
    experiment: Experiment, study: optuna.Study, slot: Slot, assumed: optuna.trial.TrialState
) -> optuna.trial.FrozenTrial:
    """Build the row a projected launch leaves in an in-memory copy.

    :param Experiment experiment: Config whose bounds a stand-in measurement satisfies.
    :param optuna.Study study: The in-memory copy.
    :param Slot slot: A rerun slot.
    :param optuna.trial.TrialState assumed: The state the projection assumes the launch ends in.
    :return optuna.trial.FrozenTrial: The row.
    """
    source = slot.waiting or slot.rerun
    assert source is not None
    value, attrs = _assumed_verdict(experiment, study, assumed)
    return optuna.trial.create_trial(
        state=assumed,
        value=value,
        params=dict(source.params),
        distributions=source.distributions,
        system_attrs=dict(source.system_attrs),
        user_attrs={RERUN_OF_ATTR: source.number, **attrs},
    )


def _inside(low: float | None, high: float | None) -> float:
    """Return a value no bound excludes.

    :param float | None low: Inclusive lower bound, or ``None``.
    :param float | None high: Inclusive upper bound, or ``None``.
    :return float: The midpoint of two bounds, the one bound given, or zero.
    """
    if low is not None and high is not None:
        return (low + high) / 2
    return next((bound for bound in (low, high) if bound is not None), 0.0)


def _assumed_verdict(
    experiment: Experiment, study: optuna.Study, assumed: optuna.trial.TrialState
) -> tuple[float | None, dict[str, object]]:
    """Return the value and attributes a projection records for a verdict it cannot know.

    An assumed measurement satisfies every current bound, so a projection never counts it
    toward a failure streak.

    :param Experiment experiment: Config whose bounds the measurement satisfies.
    :param optuna.Study study: The study whose next completion sequence the outcome takes.
    :param optuna.trial.TrialState assumed: COMPLETE or FAIL.
    :return tuple[float | None, dict[str, object]]: The value, and the outcome and constraint
        attributes.
    """
    sequence = _recorded_outcome_sequence(study) + 1
    if assumed != optuna.trial.TrialState.COMPLETE:
        outcome = _trial_outcome_payload(sequence, "failure", cause="assumed by the projection")
        return None, {TRIAL_OUTCOME_ATTR: outcome}
    value = _inside(experiment.metric.min, experiment.metric.max)
    attrs: dict[str, object] = {
        constraint_attr(constraint.name): _inside(constraint.min, constraint.max)
        for constraint in experiment.constraints
    }
    attrs[TRIAL_OUTCOME_ATTR] = _trial_outcome_payload(sequence, "success", metric=value)
    return value, attrs


def walk(
    experiment: Experiment,
    phase: Phase,
    mirror: optuna.Study,
    pool: list[_Candidate],
    read: Reader,
    *,
    through_launches: bool = False,
    assumed: optuna.trial.TrialState = optuna.trial.TrialState.COMPLETE,
) -> Iterator[Adopt | Launch]:
    """Walk a study's next slots, filling each with the oldest kept attempt that trained it.

    A grid slot is its cell. A seeded random slot is its draw position and the configuration the
    study draws there. An adaptive new draw takes any kept attempt whose configuration the study
    holds no row of. A rerun takes a terminal attempt of its own configuration only; an
    interrupted attempt fills only a new draw, so the ordinary rerun continues its checkpoint and
    no slot holds two. Each filled slot is applied to ``mirror`` before it is yielded.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase whose target and failure policy apply.
    :param optuna.Study mirror: In-memory copy of the study, advanced as slots fill.
    :param list[_Candidate] pool: :func:`kept_attempts`, consumed as slots take them.
    :param Reader read: Gives each taken attempt's content under the current settings.
    :param bool through_launches: Keep projecting past unfilled slots, assuming each launch ends
        in ``assumed``, until no kept attempt can fill a later slot; otherwise stop at the first.
    :param optuna.trial.TrialState assumed: State a projected launch ends in.
    :return Iterator[Adopt | Launch]: The slots in allocation order.
    """
    adaptive = phase.sampler.type in _ADAPTIVE_SAMPLERS
    positioned = phase.sampler.type == "random" and phase_draws(phase)
    cells = None
    if phase.sampler.type == "grid" and phase_draws(phase):
        record = _recorded_identity(mirror)
        assert record is not None
        cells = len(
            grid_cell_order(
                record["search_spaces"],
                seed=phase.sampler.seed,
                repeats=phase.repeats,
                phase_name=mirror.study_name,
            )
        )
    while pool and (slot := next_slot(mirror, phase)) is not None:
        trials = mirror.get_trials(deepcopy=False)
        positions = _draw_positions(trials)
        asked: optuna.trial.FrozenTrial | None = None
        if slot.rerun is not None:
            at = positions[slot.rerun.number]
            params = slot.rerun.params
            match = next(
                (
                    candidate
                    for candidate in pool
                    if candidate.terminal
                    and _same_value(candidate.owner.trial.params, params)
                    and (not positioned or candidate.position == at)
                ),
                None,
            )
        else:
            at = max(positions.values(), default=-1) + 1
            if cells is not None and at >= cells:
                return
            if adaptive:
                held = {_semantic_payload_digest(trial.params) for trial in trials}
                match = next(
                    (
                        candidate
                        for candidate in pool
                        if _semantic_payload_digest(candidate.owner.trial.params) not in held
                    ),
                    None,
                )
            else:
                trial = mirror.ask()
                _suggest_all(trial, phase)
                asked = mirror._storage.get_trial(trial._trial_id)
                match = next(
                    (
                        candidate
                        for candidate in pool
                        if _same_value(candidate.owner.trial.params, asked.params)
                        and (not positioned or candidate.position == at)
                    ),
                    None,
                )
        if match is None:
            yield Launch(slot, None if asked is None else asked.number)
            if (
                not through_launches
                or (slot.rerun is None and adaptive)
                or (positioned and all(candidate.position <= at for candidate in pool))
            ):
                return
            if asked is not None:
                value, attrs = _assumed_verdict(experiment, mirror, assumed)
                for key, attr in attrs.items():
                    mirror._storage.set_trial_user_attr(asked._trial_id, key, attr)
                mirror._storage.set_trial_state_values(
                    asked._trial_id, assumed, None if value is None else [value]
                )
                continue
            if slot.waiting is not None:
                _settle_waiting(mirror, slot.waiting.number)
            mirror.add_trial(_placeholder(experiment, mirror, slot, assumed))
            continue
        content = read(match)
        if slot.waiting is not None:
            _settle_waiting(mirror, slot.waiting.number)
        sequence = _recorded_outcome_sequence(mirror) + 1
        row = _adopted_row(phase, mirror, slot, match, content, asked, sequence)
        state = _load_phase_policy_state(mirror, experiment)
        measured = _parsed_trial_outcome(row.user_attrs[TRIAL_OUTCOME_ATTR])
        assert measured is not None
        abort = _abort_decision(
            phase,
            row.number,
            _counted_outcome(measured.outcome, measured.metric, row, experiment),
            completion_sequence=sequence - 1,
            consecutive_failures=state.consecutive_failures,
            abort_recorded=state.abort is not None,
        )
        if asked is not None:
            _finish_asked(mirror, asked._trial_id, row)
        else:
            mirror.add_trial(row)
        pool.remove(match)
        yield Adopt(match, row, slot.waiting, abort)


def assumed_reader(experiment: Experiment, assumed: optuna.trial.TrialState) -> Reader:
    """Read every attempt as a projection must: re-reads end in ``assumed``, the rest as recorded.

    A re-read assumed to measure keeps the producer's measurement when it has one.

    :param Experiment experiment: Config whose bounds a stand-in measurement satisfies.
    :param optuna.trial.TrialState assumed: COMPLETE or FAIL.
    :return Reader: The reader.
    """

    def read(candidate: _Candidate) -> optuna.trial.FrozenTrial:
        trial = copy.deepcopy(candidate.owner.trial)
        if candidate.rereads and trial.state != assumed:
            value, attrs = _assumed_verdict(experiment, candidate.owner.study, assumed)
            trial.state, trial.value = assumed, value
            trial.user_attrs.update(attrs)
        return trial

    return read


def _evidence_reader(
    experiment: Experiment,
    phase: Phase,
    study: optuna.Study,
    others: Sequence[optuna.Study],
    deadline: float | None,
    reader_generation_id: str,
) -> Reader:
    """Read each taken attempt from its kept evidence, or copy what has no trainer result to read.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param optuna.Study study: The adopting study.
    :param Sequence[optuna.Study] others: The phase's other studies.
    :param float | None deadline: Absolute invocation deadline.
    :param str reader_generation_id: Generation that owns any supervised tracker work.
    :return Reader: The reader.
    """
    attempts = producing_attempts(
        (study, *(other for other in others if other.study_name != study.study_name))
    )
    launches = directory_launches(attempts)
    producing: dict[str, Experiment | WinnerIntegrityError] = {}

    def read(candidate: _Candidate) -> optuna.trial.FrozenTrial:
        trial = candidate.owner.trial
        if not candidate.terminal:
            return copy.deepcopy(trial)
        if not candidate.rereads:
            return copy.deepcopy(trial)
        _require_reread_time(experiment, phase, deadline)
        log.info(
            "Phase %s re-reading trial %d of study %s.",
            phase.name,
            trial.number,
            candidate.owner.study.study_name,
        )
        row = _reread_trial(
            experiment,
            phase,
            trial,
            candidate.owner.rows,
            producing,
            attempts,
            launches,
            deadline=deadline,
            reader_generation_id=reader_generation_id,
        )
        _require_reread_time(experiment, phase, deadline)
        # A trainer that failed or timed out holds no result for any reading.
        return copy.deepcopy(trial) if row is None else row

    return read


@dataclass
class Adoption:
    """What adoption appended to a study: rows by kind, and outcomes a re-read changed."""

    rereads: int = 0
    changed: int = 0
    failures: int = 0
    interrupted: int = 0
    #: The distinct reading settings the adopted attempts' studies recorded.
    readings: list[object] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Return the number of rows appended."""
        return self.rereads + self.failures + self.interrupted

    def count(self, step: Adopt) -> None:
        """Count one adopted row.

        :param Adopt step: The row and the attempt it came from.
        """
        identity = _recorded_identity(step.candidate.owner.study)
        reading = None if identity is None else identity["payload"].get(READING_KEY)
        if reading not in self.readings:
            self.readings.append(reading)
        if step.candidate.rereads:
            self.rereads += 1
            producer = step.candidate.owner.trial
            self.changed += (step.row.state, step.row.value) != (producer.state, producer.value)
        elif step.candidate.terminal:
            self.failures += 1
        else:
            self.interrupted += 1

    def note(self, reading: JsonObject) -> str:
        """Say what adoption did, for the run's line about the phase.

        :param JsonObject reading: The phase's current reading settings.
        :return str: The sentence.
        """
        return (
            f"{self.changed_reading(reading)}{self.describe(planned=False)}; nothing was retrained."
        )

    def changed_reading(self, reading: JsonObject) -> str:
        """Name the reading settings that changed, when every attempt comes from one reading.

        :param JsonObject reading: The phase's current reading settings.
        :return str: ``reading settings changed (<keys>): ``, or empty for several readings.
        """
        if len(self.readings) != 1 or not isinstance(self.readings[0], dict):
            return ""
        changes = format_changed_keys(_changed_payload_keys(self.readings[0], reading))
        return f"reading settings changed ({changes}): " if changes else ""

    def describe(self, *, planned: bool) -> str:
        """Name the adopted rows by kind, which a dry run and a run both report.

        :param bool planned: Whether the rows are a dry run's plan, which cannot say how many
            re-reads change their outcome, rather than rows a run wrote.
        :return str: The rows, joined into one clause.
        """
        parts = []
        if self.rereads:
            parts.append(
                f"re-read {plural(self.rereads, 'kept trial')} from other reading studies"
                + ("" if planned else f" ({self.changed} changed outcome)")
            )
        if self.failures:
            keep = "keep" if planned else "kept"
            parts.append(f"{keep} {plural(self.failures, 'trainer failure')} as recorded")
        if self.interrupted:
            take = "take" if planned else "took"
            parts.append(
                f"{take} {plural(self.interrupted, 'interrupted trial')} to continue from "
                "its checkpoint"
            )
        return ", ".join(parts)


def _write_adopted(
    experiment: Experiment,
    phase: Phase,
    study: optuna.Study,
    step: Adopt,
    *,
    launch_follows: bool,
    deadline: float | None,
) -> None:
    """Append one adopted row to the live study, after settling the queued rerun it fills.

    The failure policy exists to stop launches, and taking a kept attempt launches nothing, so a
    row records the abort its streak trips only when the next slot needs a launch.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param optuna.Study study: The live study.
    :param Adopt step: The row.
    :param bool launch_follows: Whether the slot after this row needs a launch.
    :param float | None deadline: Absolute invocation deadline.
    :raises PhaseTimeoutError: The deadline passed before the row was written.
    """
    _require_reread_time(experiment, phase, deadline)
    if step.waiting is not None:
        _settle_waiting(study, step.waiting.number)
    study.add_trial(_with_abort(step) if launch_follows else step.row)


def _with_abort(step: Adopt) -> optuna.trial.FrozenTrial:
    """Return an adopted row carrying the abort its streak trips, for a launch to stop at.

    :param Adopt step: The row and the abort its outcome trips, if any.
    :return optuna.trial.FrozenTrial: The row, with the abort in its outcome when there is one.
    """
    if step.abort is None:
        return step.row
    row = copy.copy(step.row)
    outcome = _parsed_trial_outcome(row.user_attrs[TRIAL_OUTCOME_ATTR])
    assert outcome is not None
    row.user_attrs = {
        **row.user_attrs,
        TRIAL_OUTCOME_ATTR: _trial_outcome_payload(
            outcome.sequence,
            outcome.outcome,
            metric=outcome.metric,
            cause=outcome.cause,
            policy=outcome.policy,
            abort=step.abort,
        ),
    }
    return row


def carry_partial_decision(
    experiment: Experiment, phase: Phase, study: optuna.Study, others: Sequence[optuna.Study]
) -> bool:
    """Accept a compatible study's partial-timeout boundary when this study holds its attempts.

    An operator who accepted a phase cut short at its deadline accepted those attempts, whatever
    reading scored them, so a reading change launches nothing a rerun would not. The newest
    compatible study with an accepted partial decision for the target the config still requests,
    whose counted attempts this study all holds, gives this study the same boundary over its own
    rows and measurements.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param optuna.Study study: The study after adoption, live or an in-memory copy.
    :param Sequence[optuna.Study] others: The phase's other studies.
    :raises StudySchemaMismatchError: A compatible study's decision is malformed.
    :return bool: Whether the study now records the boundary.
    """
    trials = study.get_trials(deepcopy=False)
    finished = _finished_trial_count(trials)
    if study.user_attrs.get(PHASE_DECISION_ATTR) is not None or finished >= phase.n_trials:
        return False
    state = _load_phase_policy_state(study, experiment)
    if state.abort is not None:
        return False
    present = {trial.user_attrs.get(ATTEMPT_ID_ATTR) for trial in trials}
    for other in reversed(_compatible_studies(experiment, phase, study, others)):
        decision = _parse_accepted_partial_decision(
            other.user_attrs.get(PHASE_DECISION_ATTR), other.study_name
        )
        if decision is None or decision.trial_target != phase.n_trials:
            continue
        counted = {
            trial.user_attrs.get(ATTEMPT_ID_ATTR)
            for trial in other.get_trials(deepcopy=False)
            if isinstance(trial.user_attrs.get(ATTEMPT_ID_ATTR), str)
            and _trial_run_state(trial) in FINISHED_TRIAL_STATES
            and (outcome := _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR)))
            is not None
            and outcome.sequence <= decision.outcome_sequence
        }
        if not counted <= present:
            continue
        _record_accepted_partial_decision(
            study,
            phase.n_trials,
            outcome_sequence=state.max_sequence,
            finished_trials=finished,
            measured_trials=_measured_trial_count(trials),
            timeout_scope=decision.timeout_scope,
            recovered_abort_sequence=None,
        )
        return True
    return False


def adopt_kept_attempts(
    experiment: Experiment,
    phase: Phase,
    study: optuna.Study,
    others: Sequence[optuna.Study],
    deadline: float | None,
    reader_generation_id: str,
    *,
    before_launch: bool,
) -> Adoption:
    """Append the kept attempts that fill the study's next slots, re-reading finished ones.

    Rows are written one at a time, each once the slot after it is known, so a stop at any
    point keeps every row written. Before the first launch, an accepted partial-timeout boundary
    of a compatible study carries over (:func:`carry_partial_decision`).

    :param Experiment experiment: Current reading and budget settings.
    :param Phase phase: Phase whose target and failure policy apply.
    :param optuna.Study study: The live study, with a recorded identity.
    :param Sequence[optuna.Study] others: The phase's other studies.
    :param float | None deadline: Absolute invocation deadline.
    :param str reader_generation_id: Generation that owns any tracker reader.
    :param bool before_launch: Whether the run has accepted no target yet.
    :raises PhaseTimeoutError: The deadline stopped a read before its row was committed.
    :raises StudySchemaMismatchError: A kept success lacks its training record.
    :raises TrackingTargetUnavailableError: A tracker gave no answer for a finished attempt.
    :return Adoption: What was appended.
    """
    adoption = Adoption()
    if _recorded_identity(study) is None:
        return adoption
    pool = kept_attempts(experiment, phase, study, others)
    state = _load_phase_policy_state(study, experiment)
    abort = state.abort or _replayed_streak_abort(phase, study, state)
    if pool and abort is not None:
        # Rows past an abort are new work: only a config that retires the abort adds them, and
        # the recovery boundary comes first, as before a launch.
        probe = mirror_study(experiment, phase, study, study_name=study.study_name)
        first = next(
            walk(experiment, phase, probe, list(pool), assumed_reader(experiment, _COMPLETE)), None
        )
        if not _config_retires_abort(phase, study, abort) or not isinstance(first, Adopt):
            pool = []
        else:
            sequence = abort["completion_sequence"]
            assert type(sequence) is int
            _record_recovery_boundary(study, phase, sequence)
    if pool:
        mirror = mirror_study(experiment, phase, study, study_name=study.study_name)
        read = _evidence_reader(experiment, phase, study, others, deadline, reader_generation_id)
        held: list[Adopt] = []

        def flush(launch_follows: bool) -> None:
            if held:
                row = held.pop()
                _write_adopted(
                    experiment, phase, study, row, launch_follows=launch_follows, deadline=deadline
                )
                adoption.count(row)

        def read_after_flush(candidate: _Candidate) -> optuna.trial.FrozenTrial:
            # The walk reads only for a slot a kept attempt fills, so the held row is followed by
            # no launch, and writing it first keeps it across a stop during this read.
            flush(False)
            return read(candidate)

        for step in walk(experiment, phase, mirror, pool, read_after_flush):
            if isinstance(step, Launch):
                flush(True)
            else:
                held.append(step)
        # The kept attempts ran out, so any slot left launches.
        flush(next_slot(mirror, phase) is not None)
    if adoption.total:
        report.info("%s  %s", phase.name, adoption.note(reading_settings(experiment, phase)))
    if before_launch:
        carry_partial_decision(experiment, phase, study, others)
    return adoption


def launch_batch(
    experiment: Experiment, phase: Phase, study: optuna.Study, others: Sequence[optuna.Study]
) -> int:
    """Bound launches before the next slot a kept attempt could fill.

    Slots fill in allocation order, so launches stop at that slot and adoption runs again; with
    no such slot the whole remaining target launches.

    :param Experiment experiment: Current config.
    :param Phase phase: Phase whose target applies.
    :param optuna.Study study: The study about to launch trials.
    :param Sequence[optuna.Study] others: The phase's other studies.
    :return int: Maximum launches before checking kept attempts again.
    """
    remaining = max(0, phase.n_trials - _finished_trial_count(study.get_trials(deepcopy=False)))
    pool = kept_attempts(experiment, phase, study, others)
    if not pool:
        return remaining
    mirror = mirror_study(experiment, phase, study, study_name=study.study_name)
    launches = 0
    for step in walk(
        experiment,
        phase,
        mirror,
        pool,
        assumed_reader(experiment, optuna.trial.TrialState.COMPLETE),
        through_launches=True,
    ):
        if isinstance(step, Launch):
            launches += 1
        else:
            return max(1, launches)
    return remaining


@dataclass(frozen=True)
class AdoptionPlan:
    """What adoption would do to a study, projected without reading evidence or writing."""

    #: The study an empty study takes its identity from, or ``None`` for one that has one.
    source: optuna.Study | None
    #: Rows adopted before the first launch, by kind.
    adoption: Adoption
    #: The study at its first launch when every re-read measures, and when every one fails.
    complete: optuna.Study
    failed: optuna.Study
    #: Fewest and most launches to reach the target, over those two assumptions.
    launches: tuple[int, int]
    #: Whether a compatible study's accepted partial boundary carries over.
    partial: bool


def _left_for_launch(
    experiment: Experiment,
    phase: Phase,
    mirror: optuna.Study,
    last: Adopt | None,
    study_name: str,
    upto: int | None,
) -> optuna.Study:
    """Copy a walked mirror as a run leaves the study before its first launch.

    The run's last adopted row carries the abort its streak trips (:func:`_write_adopted`), so
    the projection judges the same refusal.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param optuna.Study mirror: The walked in-memory study.
    :param Adopt | None last: The last row adopted before the launch, if any.
    :param str study_name: The study's address.
    :param int | None upto: Number of the trial the launch was asked on, or ``None`` for all.
    :return optuna.Study: The projected study.
    """
    if last is None or last.abort is None:
        return mirror_study(experiment, phase, mirror, study_name=study_name, upto=upto)
    left = mirror_study(experiment, phase, mirror, study_name=study_name, upto=last.row.number)
    left.add_trial(_with_abort(last))
    return left


def _identity_after_growth(
    experiment: Experiment,
    phase: Phase,
    inherited: Mapping[str, object],
    study: optuna.Study,
    record: PhaseIdentityRecord,
) -> PhaseIdentityRecord:
    """Return the identity a run binds the study to before it adopts, growth included.

    The run's fingerprint check records a grown search space before adoption
    (:func:`~phasesweep.engine.fingerprints._judge_identity`), so a projection walks the cells and
    draws the grown space adds. A config the study cannot continue under keeps the recorded
    identity; the ordinary fingerprint check reports that refusal.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param Mapping[str, object] inherited: The parent layer the phase inherits now.
    :param optuna.Study study: The addressed study.
    :param PhaseIdentityRecord record: Its recorded identity.
    :return PhaseIdentityRecord: The identity the run would hold when it adopts.
    """
    try:
        verdict = _judge_identity(study, experiment, phase, inherited)
    except StudyFingerprintMismatchError:
        return record
    return record if verdict.record is None else verdict.record


def plan_adoption(
    experiment: Experiment,
    phase: Phase,
    inherited: Mapping[str, object],
    study: optuna.Study | None,
    others: Sequence[optuna.Study],
    *,
    study_name: str,
) -> AdoptionPlan | None:
    """Project adoption for a read that writes nothing: dry run, preflight, ``validate``, status.

    The walk is the one a run takes; only the verdicts of re-reads are unknown, so it runs once
    assuming each measures and once assuming each fails. A launch is assumed to end the same
    way, and the failure policy is applied to adopted rows only, so the most launches is an
    upper bound.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param Mapping[str, object] inherited: The parent layer the phase inherits now.
    :param optuna.Study | None study: The addressed study as captured, or ``None`` if absent.
    :param Sequence[optuna.Study] others: The phase's studies.
    :param str study_name: The addressed study's name.
    :raises StudyFingerprintMismatchError: See :func:`adoption_identity`.
    :return AdoptionPlan | None: The projection, or ``None`` when no compatible study exists.
    """
    others = [other for other in others if other.study_name != study_name]
    if not others:
        return None
    source = None
    record = None if study is None else _recorded_identity(study)
    if record is None:
        if study is not None and study.get_trials(deepcopy=False):
            return None
        chosen = adoption_identity(experiment, phase, inherited, others)
        if chosen is None:
            return None
        record, source = chosen
    elif study is not None:
        record = _identity_after_growth(experiment, phase, inherited, study, record)
    projections = []
    for assumed in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL):
        mirror = mirror_study(experiment, phase, study, study_name=study_name, record=record)
        if not _compatible_studies(experiment, phase, mirror, others):
            return None
        pool = kept_attempts(experiment, phase, mirror, others)
        adoption = Adoption()
        first: optuna.Study | None = None
        last: Adopt | None = None
        abort: JsonObject | None = None
        # Every slot is adopted or launched, so the launches are the finished trials the target
        # still needs after the adopted ones, unless a stop comes first.
        launches = 0
        unfilled = phase.n_trials - _finished_trial_count(mirror.get_trials(deepcopy=False))
        stopped = partial = False
        for step in walk(
            experiment,
            phase,
            mirror,
            pool,
            assumed_reader(experiment, assumed),
            through_launches=True,
            assumed=assumed,
        ):
            if isinstance(step, Adopt):
                if first is None:
                    adoption.count(step)
                    last = step
                abort = step.abort
                unfilled -= _trial_run_state(step.row) in FINISHED_TRIAL_STATES
                continue
            if first is None:
                first = _left_for_launch(experiment, phase, mirror, last, study_name, step.asked)
                partial = carry_partial_decision(experiment, phase, first, others)
            if partial or abort is not None:
                stopped = True
                break
            launches += 1
        if first is None:
            launch_follows = next_slot(mirror, phase) is not None
            first = (
                _left_for_launch(experiment, phase, mirror, last, study_name, None)
                if launch_follows
                else mirror_study(experiment, phase, mirror, study_name=study_name)
            )
            partial = carry_partial_decision(experiment, phase, first, others)
            stopped = partial or (abort is not None and launch_follows)
        if not stopped:
            launches = max(0, unfilled)
        projections.append((first, adoption, launches, partial))
    (complete, adoption, low, partial), (failed, _, high, _) = projections
    return AdoptionPlan(
        source=source,
        adoption=adoption,
        complete=complete,
        failed=failed,
        launches=(min(low, high), max(low, high)),
        partial=partial,
    )


class PendingRereadNotice(PhaseSweepError):
    """A phase not started under the current reading settings, which a run re-reads, not retrains.

    ``status`` and ``status --trials`` print it where they report a phase's refusal, with the
    changed values and the command that runs the phase. The status payload, which MCP serves,
    carries :attr:`statement` instead: the same sentence naming only the changed settings,
    since their values are config-authored text such as file names and patterns.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RETRY
    #: The value-free sentence the status payload carries (:func:`_pending_notice`).
    statement: str = ""
    #: Finished trials the next run re-reads (:attr:`AdoptionPlan.rereads`).
    finished_trials: int = 0


def _pending_notice(
    rereads: int,
    changes: Sequence[tuple[str, object, object]],
    phase: str,
    unsettled: Sequence[str] = (),
) -> PendingRereadNotice:
    """Build both texts of a :class:`PendingRereadNotice` from one description of the re-read.

    A parent that is itself re-read first may select another winner, and then this phase
    inherits other values and runs in another study, so the re-read here is conditional.

    :param int rereads: Finished trials the next run re-reads.
    :param Sequence[tuple[str, object, object]] changes: Each changed reading setting, with its
        recorded and current value.
    :param str phase: The phase's name.
    :param Sequence[str] unsettled: Parents whose winner the next run selects again first,
        defaults to none.
    :return PendingRereadNotice: The notice, with its message and step for the CLI and its
        :attr:`PendingRereadNotice.statement` for the status payload.
    """
    opening = f"Phase {phase!r} has not started under the current reading settings"
    trials = plural(rereads, "finished trial")
    settings = ", ".join(path for path, _old, _new in changes) or "reading settings"
    if unsettled:
        names = " and ".join(repr(name) for name in unsettled)
        keeps = "keeps its winner" if len(unsettled) == 1 else "keep their winners"
        remediation = (
            f"Run the same configuration. If {names} {keeps}, this phase re-reads {trials} "
            "from kept evidence without retraining them. Different inherited values select a different study."
        )
        statement = (
            f"{opening} ({settings} changed); the next run re-reads {trials} from kept evidence "
            f"if {names} {keeps}. Different inherited values select a different study."
        )
    else:
        remediation = (
            f"Run the same configuration to re-read {trials} from kept evidence without "
            "retraining those trials."
        )
        statement = (
            f"{opening} ({settings} changed); the next run re-reads {trials} from kept evidence "
            "and retrains nothing."
        )
    notice = PendingRereadNotice(
        f"{opening} ({format_changed_keys(changes) or 'reading settings'}).",
        remediation=remediation,
    )
    notice.statement = statement
    notice.finished_trials = rereads
    return notice


def pending_reread(
    experiment: Experiment,
    phase: Phase,
    study_name: str,
    studies: Sequence[optuna.Study],
    *,
    unsettled: Sequence[str] = (),
) -> PendingRereadNotice | None:
    """Say that a phase's addressed study holds no rows yet and would re-read kept attempts.

    The inherited values the address stands for are the ones a study of the phase recorded
    whose values, read under the current settings, give that address. A study that already holds
    rows reports them instead.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param str study_name: The study it addresses (:func:`phase_study_name`).
    :param Sequence[optuna.Study] studies: Every study of the phase, oldest first.
    :param Sequence[str] unsettled: Parents whose winner the next run selects again before this
        phase, so the address holds only if they keep it; defaults to none.
    :raises StudyFingerprintMismatchError: A study with these inherited values has another
        training identity (:func:`adoption_identity`).
    :return PendingRereadNotice | None: The notice, or ``None`` when the addressed study holds
        rows or no study holds trials to re-read.
    """
    addressed = next((study for study in studies if study.study_name == study_name), None)
    if addressed is not None and addressed.get_trials(deepcopy=False):
        return None
    for study in reversed(studies):
        record = _recorded_identity(study)
        if study is addressed or record is None:
            continue
        inherited = _recorded_inheritance(record)
        if phase_study_name(experiment, phase, inherited) != study_name:
            continue
        plan = plan_adoption(
            experiment, phase, inherited, addressed, studies, study_name=study_name
        )
        if plan is None or not plan.adoption.total:
            return None
        recorded = record["payload"].get(READING_KEY)
        changes = _changed_payload_keys(
            recorded if isinstance(recorded, dict) else {}, reading_settings(experiment, phase)
        )
        return _pending_notice(plan.adoption.rereads, tuple(changes), phase.name, unsettled)
    return None
