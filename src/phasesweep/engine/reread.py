"""Fill a study addressed under changed reading settings from the kept evidence of its trials.

Optuna keeps a finished trial's value fixed, so results read under other reading settings
(:func:`phasesweep.engine.fingerprints.reading_settings`) live in another study, which the same
address finds (:func:`phasesweep.engine.fingerprints.phase_study_name`). When a run reaches a
phase whose addressed study records no identity yet, and another study of the phase holds
trials measured under the same inherited values and the same training identity, the new study
is filled from it before anything launches: each trial is copied in number order, so trial
numbers, params, distributions, grid positions, rerun links and trial directories all hold,
and each trial whose trainer finished is read again from its kept evidence under the current
settings, through the one path recovery reads a finished attempt by
(:func:`phasesweep.engine.attempts.read_finished_attempt`). No trainer launches.

The fill is resumable. The source name is written before copying, each trial is appended in
one storage write, and the identity is written last. A killed fill resumes from its source's
current ledger, including trials the source finished while the fill was stopped.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import optuna

from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import retry_target_ceiling
from phasesweep.engine.attempts import (
    _pending_evidence_refusal,
    _record_stale_trial_outcome,
    _trial_attempt_record,
    read_finished_attempt,
)
from phasesweep.engine.errors import (
    PhaseTimeoutError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    WinnerIntegrityError,
    phase_timeout,
)
from phasesweep.engine.evidence import _earlier_attempts
from phasesweep.engine.fingerprints import (
    _MAX_REPORTED_CHANGES,
    READING_KEY,
    _changed_payload_keys,
    _identity_payload,
    _identity_record,
    _inherited_values,
    _judge_record,
    _payload_search_space,
    _recorded_inheritance,
    _render_changed_values,
    _same_value,
    _semantic_payload_digest,
    _stamp_identity,
    _with_search_space,
    phase_study_name,
    reading_settings,
)
from phasesweep.engine.grid_order import grid_axis_names, grid_cell_order
from phasesweep.engine.optuna import (
    _draw_positions,
    _finished_trial_count,
    _measured_draw_count,
    _measured_trial_count,
    _sampling_trial_for_reread,
    _suggest_all,
    _trial_run_state,
)
from phasesweep.engine.paths import _experiment_dir
from phasesweep.engine.provenance import _load_generation_experiment
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
    ATTEMPT_ID_ATTR,
    FAILURE_REASON_ATTR,
    GATES_ATTR,
    OVERRIDES_ATTR,
    PHASE_DECISION_ATTR,
    PHASE_IDENTITY_ATTR,
    PHASE_RECOVERY_ATTR,
    REREAD_FILL_ATTR,
    RERUN_OF_ATTR,
    STUDY_SCHEMA_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_TARGET_ATTR,
    PhaseIdentityRecord,
    _completed_trainer_duration,
    _parsed_trial_outcome,
    _recorded_identity,
    _required_trial_dir,
    _trial_outcome_payload,
)
from phasesweep.engine.study_policy import (
    _abort_decision,
    _accepted_trial_target,
    _counted_outcome,
    _initialize_study_schema,
    _load_phase_policy_state,
    _parse_accepted_partial_decision,
    _pending_rerun_count,
    _record_recovery_boundary,
    _record_trial_target,
    _recorded_outcome_sequence,
    _scheduled_reruns,
)
from phasesweep.engine.trial import TrialResult, _failed_trial, record_trial_result
from phasesweep.engine.trial_dirs import (
    REREAD_FILL_SCHEMA_VERSION,
    DirectoryLaunch,
    ProducingAttempt,
    _fill_source_name,
    _rerun_chain,
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

#: The source's identity, root, schema and any earlier fill belong to the source study.
_NOT_COPIED = frozenset(
    {REREAD_FILL_ATTR, PHASE_IDENTITY_ATTR, ARTIFACT_ROOT_ATTR, STUDY_SCHEMA_ATTR}
)
#: The recorded outcomes a re-read replaces; an interruption or a fatal outcome stands as it was.
_REREAD_OUTCOMES = frozenset({"success", "failure"})


@dataclass(frozen=True)
class FillSource:
    """The study a new one is filled from, and what its reading settings were."""

    study: optuna.Study
    record: PhaseIdentityRecord
    #: Which trials still have a finished trainer to read.
    reread_trials: frozenset[int]
    #: Each reading setting that changed, with its recorded and current value.
    changes: tuple[tuple[str, object, object], ...]
    #: The source's current trials and study attributes.
    trials: list[optuna.trial.FrozenTrial]
    attrs: JsonObject

    @property
    def rereads(self) -> int:
        """Count the finished trials the fill still re-reads.

        :return int: Number of trials with a finished trainer and no row in the target.
        """
        return len(self.reread_trials)

    @property
    def copyable_trials(self) -> Sequence[optuna.trial.FrozenTrial]:
        """Return the rows a fill copies.

        The ordinary retry policy schedules the trailing WAITING reruns again after
        judging the new measurements.

        :return Sequence[optuna.trial.FrozenTrial]: Stored history, before queued reruns.
        """
        end = len(self.trials)
        while end > 0 and (
            (trial := self.trials[end - 1]).state == optuna.trial.TrialState.WAITING
            and RERUN_OF_ATTR in trial.user_attrs
        ):
            end -= 1
        return self.trials[:end]


def _fill_candidates(name: str | None, others: Sequence[optuna.Study]) -> Sequence[optuna.Study]:
    """Use only the already chosen source when resuming a fill.

    :param str | None name: The pinned source name, if there is one.
    :param Sequence[optuna.Study] others: Other studies of the phase, oldest first.
    :raises StudySchemaMismatchError: The saved source is unavailable.
    :return Sequence[optuna.Study]: Every candidate or just the saved source.
    """
    if name is None:
        return others
    for study in others:
        if study.study_name == name:
            return [study]
    raise StudySchemaMismatchError(f"Saved fill source {name!r} is missing.")


@dataclass(frozen=True)
class FillResult:
    """What a fill wrote: the trials it copied, re-read, and whose outcome the re-read changed."""

    copied: int
    rereads: int
    changed: int
    #: The source's accepted target and the ceiling after re-reading, when they conflict.
    adjusted_target: tuple[int, int] | None


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
    """Return a source's identity record as it reads under other reading settings.

    :param PhaseIdentityRecord record: The source study's record.
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

    :param optuna.trial.FrozenTrial trial: A source study's trial.
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


def fill_source(
    experiment: Experiment,
    phase: Phase,
    inherited: Mapping[str, object],
    others: Sequence[optuna.Study],
    *,
    target: optuna.Study | None = None,
) -> FillSource | None:
    """Pick the study a phase's new study is filled from, judging without writing anything.

    A candidate holds trials measured under the same inherited values and would continue
    under the current config once its reading settings are replaced.
    Prefer histories within the requested trial budget. Among them, keep the most finished
    trainers to read, then the most spent attempts and retained rows, so a shorter copy
    cannot erase a failure or an interrupted checkpoint.
    Equally complete histories use the most recently created study.
    An interrupted fill keeps its source and uses its latest trials and study metadata.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase whose addressed study records no identity.
    :param Mapping[str, object] inherited: The parent layer the phase inherits now.
    :param Sequence[optuna.Study] others: The phase's other studies, oldest first.
    :param optuna.Study | None target: The addressed study, when already created.
    :raises StudyFingerprintMismatchError: A study with these inherited values exists but its
        training identity differs from the current config; the refusal names what changed.
    :raises StudySchemaMismatchError: The saved source is unavailable or a record is malformed.
    :return FillSource | None: The source, or ``None`` when no study holds these values.
    """
    reading = reading_settings(experiment, phase)
    values = _inherited_values(phase, inherited)
    pinned = _fill_source_name(target)
    copied = 0 if target is None else len(target.get_trials(deepcopy=False))
    found: list[FillSource] = []
    refusal: StudyFingerprintMismatchError | None = None
    attempts = producing_attempts(others)
    for study in _fill_candidates(pinned, others):
        record = _recorded_identity(study)
        trials = study.get_trials(deepcopy=False)
        if record is None or not trials:
            if pinned is not None:
                raise StudySchemaMismatchError(
                    f"Saved fill source {study.study_name!r} has no identity or trials."
                )
            continue
        payload = record["payload"]
        if not _same_value(payload.get("inherited_effective_overrides"), values):
            if pinned is not None:
                raise StudySchemaMismatchError(
                    f"Saved fill source {study.study_name!r} records different inherited values."
                )
            continue
        try:
            _judge_record(
                _reread_record(record, reading),
                trials,
                study.study_name,
                experiment,
                phase,
                inherited,
            )
        except StudyFingerprintMismatchError as exc:
            refusal = exc
            continue
        recorded = payload.get(READING_KEY)
        changes = _changed_payload_keys(recorded if isinstance(recorded, dict) else {}, reading)
        rereads = frozenset(
            trial.number
            for trial in trials[copied:]
            if _trainer_seconds(trial, attempts=attempts) is not None
        )
        attrs = json_object(
            {key: value for key, value in study.user_attrs.items() if key not in _NOT_COPIED}
        )
        found.append(FillSource(study, record, rereads, tuple(changes), trials, attrs))
    if not found:
        if refusal is not None:
            raise refusal
        return None

    def history_rank(source: FillSource) -> tuple[bool, bool, int, int, int]:
        spent = _finished_trial_count(source.trials)
        return (
            _accepted_trial_target(source.study) <= phase.n_trials,
            spent <= phase.n_trials,
            source.rereads,
            spent,
            len(source.copyable_trials),
        )

    return max(reversed(found), key=history_rank)


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
    """Read one finished trial again under the current settings, as the trial a fill writes.

    Evidence that is gone or unreadable now fails the trial with the reason its reader gives,
    as a live extraction would, and the phase's retry rules decide whether it runs again.

    :param Experiment experiment: Current config, whose reading settings apply.
    :param Phase phase: The trial's phase.
    :param optuna.trial.FrozenTrial trial: The source trial.
    :param Mapping[int, optuna.trial.FrozenTrial] trials: The source study's trials by number.
    :param dict[str, Experiment | WinnerIntegrityError] producing: Producing configurations by
        generation, filled as they are loaded.
    :param Mapping[str, ProducingAttempt] attempts: Producing rows by physical attempt ID.
    :param Mapping[str, Sequence[DirectoryLaunch]] launches: Physical launches by directory.
    :param float | None deadline: Absolute invocation deadline.
    :param str reader_generation_id: Generation running the reader.
    :raises PhaseTimeoutError: The invocation deadline stopped a read.
    :raises TrackingTargetUnavailableError: The tracker gave no answer, so the verdict waits for
        a later run; the trials already copied stay and the fill resumes.
    :return optuna.trial.FrozenTrial | None: The trial to write, or ``None`` when the trial is
        copied as it was recorded.
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
        raise _fill_timeout(experiment, phase)
    if result.evidence_unreadable and result.failure_reason is not None:
        raise _pending_evidence_refusal({record.attempt_id: result.failure_reason})
    sheet = _AttrSheet(dict(trial.user_attrs))
    record_trial_result(sheet, result, overrides_json=overrides_json)
    failure = result.failure_reason
    if failure is None:
        sheet.user_attrs.pop(FAILURE_REASON_ATTR, None)
    else:
        sheet.user_attrs[FAILURE_REASON_ATTR] = failure
    # The outcome keeps its completion sequence and any abort it recorded: the failure-policy
    # replay judges a recorded streak abort against the outcomes as they read now.
    sheet.user_attrs[TRIAL_OUTCOME_ATTR] = _trial_outcome_payload(
        outcome.sequence,
        "success" if failure is None else "failure",
        metric=result.metric if failure is None else None,
        cause=failure,
        abort=outcome.abort,
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


def _copy_study_attrs(
    phase: Phase, target: optuna.Study, source: FillSource
) -> tuple[int, int] | None:
    """Copy the source policy with the retry capacity and measurements of the filled study.

    :param Phase phase: The phase whose retry ceiling applies to the re-read results.
    :param optuna.Study target: The study with all of the copied attempts.
    :param FillSource source: The source study and its current attributes.
    :raises StudySchemaMismatchError: The source's accepted target or timeout is malformed.
    :return tuple[int, int] | None: The source target and reduced target, if it changed.
    """
    source_target = _accepted_trial_target(source.study)
    trials = target.get_trials(deepcopy=False)
    ceiling = retry_target_ceiling(
        phase, finished=_finished_trial_count(trials), measured=_measured_draw_count(trials)
    )
    accepted = source_target if ceiling is None else min(source_target, ceiling)
    attrs = {**source.attrs, TRIAL_TARGET_ATTR: accepted}
    recovery = attrs.get(PHASE_RECOVERY_ATTR)
    if accepted < source_target and isinstance(recovery, dict):
        recovered_target = recovery["trial_target"]
        assert type(recovered_target) is int  # Checked with the accepted target.
        attrs[PHASE_RECOVERY_ATTR] = {**recovery, "trial_target": min(recovered_target, accepted)}
    partial = _parse_accepted_partial_decision(
        attrs.get(PHASE_DECISION_ATTR), source.study.study_name
    )
    if partial is not None:
        retargeted = min(partial.trial_target, accepted)
        if retargeted <= partial.finished_trials or (
            partial.trial_target < source_target and retargeted == accepted
        ):
            attrs[PHASE_DECISION_ATTR] = None
        else:
            measured = _measured_trial_count(
                trial
                for trial in trials
                if (outcome := _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR)))
                is not None
                and outcome.sequence <= partial.outcome_sequence
            )
            decision = json_object(attrs[PHASE_DECISION_ATTR])
            attrs[PHASE_DECISION_ATTR] = {
                **decision,
                "trial_target": retargeted,
                "measured_trials": measured,
            }
    stored = target.user_attrs
    for key, value in attrs.items():
        if stored.get(key) != value:
            target.set_user_attr(key, value)
    return (source_target, accepted) if accepted < source_target else None


def _fill_timeout(experiment: Experiment, phase: Phase) -> PhaseTimeoutError:
    """Describe a run budget that stopped the fill before it committed all the rows.

    :param Experiment experiment: The invocation whose run deadline expired.
    :param Phase phase: The phase reading its kept evidence.
    :return PhaseTimeoutError: The run's timeout refusal and resume instructions.
    """
    return phase_timeout(
        f"Phase {phase.name!r} hit its run deadline while re-reading kept evidence. "
        "The study keeps the rows already copied.",
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
        raise _fill_timeout(experiment, phase)


def _finish_fill(
    experiment: Experiment,
    phase: Phase,
    target: optuna.Study,
    source: FillSource,
    *,
    deadline: float | None = None,
) -> tuple[int, int] | None:
    """Import the phase policy and seal a fully decided fill.

    :param Experiment experiment: Current reading and budget settings.
    :param Phase phase: Phase whose new study is filled.
    :param optuna.Study target: Filled live study or in-memory preview.
    :param FillSource source: Recorded phase policy and training identity.
    :param float | None deadline: Absolute invocation deadline, defaults to None.
    :raises PhaseTimeoutError: The invocation deadline passed before sealing.
    :return tuple[int, int] | None: The source and adjusted trial targets, if changed.
    """
    adjusted = _copy_study_attrs(phase, target, source)
    _require_reread_time(experiment, phase, deadline)
    _stamp_identity(target, _reread_record(source.record, reading_settings(experiment, phase)))
    return adjusted


def preview_filled_study(
    experiment: Experiment,
    phase: Phase,
    study_name: str,
    source: FillSource,
    trials: Sequence[optuna.trial.FrozenTrial],
) -> optuna.Study:
    """Project already-known fill verdicts using the same policy as the live fill.

    :param Experiment experiment: Current reading settings.
    :param Phase phase: Phase whose run is being previewed.
    :param str study_name: Addressed study's name.
    :param FillSource source: Source study and its recorded phase policy.
    :param Sequence[optuna.trial.FrozenTrial] trials: Already-known rows of the addressed study.
    :return optuna.Study: An in-memory study ready for the ordinary recorded-phase checks.
    """
    from phasesweep.engine.ledger import open_preview_study

    study = open_preview_study(experiment, phase, study_name=study_name)
    study.add_trials(trials)
    _finish_fill(experiment, phase, study, source)
    return study


def fill_study(
    experiment: Experiment,
    phase: Phase,
    target: optuna.Study,
    source: FillSource,
    other_studies: Sequence[optuna.Study],
    run_deadline: float | None,
    reader_generation_id: str,
) -> FillResult:
    """Fill a phase's addressed study from its source, re-reading every finished trainer.

    The target is the study a run opened for the phase and records no identity: empty, or left
    part-way by a killed fill, whose trials are then the source's first ones, so the fill
    appends the rest. Queued reruns at the end have yet to launch. Scheduling them after the
    fill uses the re-read outcomes, and a stopped fill does not copy a row that could gain
    an outcome in the source before it resumes. The identity, the source's with the current
    reading settings, is written last and marks the fill complete.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param optuna.Study target: The addressed study, writable.
    :param FillSource source: The study to fill from (:func:`fill_source`).
    :param Sequence[optuna.Study] other_studies: Every study of the phase except the target.
    :param float | None run_deadline: Absolute invocation deadline.
    :param str reader_generation_id: The generation that owns any supervised tracker work.
    :raises PhaseTimeoutError: The deadline stopped the fill. Committed rows remain reusable.
    :raises StudySchemaMismatchError: The target holds trials the source does not, or the
        source's accepted partial decision is malformed.
    :raises TrackingTargetUnavailableError: A tracker gave no answer for a finished trial.
    :return FillResult: What the fill wrote in this call.
    """
    _require_reread_time(experiment, phase, run_deadline)
    trials = source.copyable_trials
    present = target.get_trials(deepcopy=False)
    if [(t.number, t.params, t.user_attrs.get(ATTEMPT_ID_ATTR)) for t in present] != [
        (t.number, t.params, t.user_attrs.get(ATTEMPT_ID_ATTR)) for t in trials[: len(present)]
    ]:
        raise StudySchemaMismatchError(
            f"Study {target.study_name!r} holds trials that study {source.study.study_name!r} "
            "does not, so it cannot be filled from it."
        )
    _initialize_study_schema(target)
    stored = target.user_attrs
    if REREAD_FILL_ATTR not in stored:
        target.set_user_attr(
            REREAD_FILL_ATTR,
            {
                "schema_version": REREAD_FILL_SCHEMA_VERSION,
                "study_name": source.study.study_name,
            },
        )
    by_number = {trial.number: trial for trial in trials}
    attempts = producing_attempts(other_studies)
    launches = directory_launches(attempts)
    producing: dict[str, Experiment | WinnerIntegrityError] = {}
    rereads = changed = 0
    for trial in trials[len(present) :]:
        _require_reread_time(experiment, phase, run_deadline)
        if trial.number in source.reread_trials:
            log.info(
                "Phase %s re-reading trial %d (%d/%d).",
                phase.name,
                trial.number,
                rereads + 1,
                source.rereads,
            )
        reread = _reread_trial(
            experiment,
            phase,
            trial,
            by_number,
            producing,
            attempts,
            launches,
            deadline=run_deadline,
            reader_generation_id=reader_generation_id,
        )
        _require_reread_time(experiment, phase, run_deadline)
        if reread is not None:
            rereads += 1
            changed += (reread.state, reread.value) != (trial.state, trial.value)
        target.add_trial(
            _sampling_trial_for_reread(trial if reread is None else reread, target.sampler)
        )
    _require_reread_time(experiment, phase, run_deadline)
    adjusted_target = _finish_fill(experiment, phase, target, source, deadline=run_deadline)
    return FillResult(len(trials) - len(present), rereads, changed, adjusted_target)


class SiblingTrial(NamedTuple):
    """One target-local slot and its finished physical candidates, newest first."""

    source: optuna.trial.FrozenTrial
    producers: list[ProducingAttempt]
    requires_measurement: bool
    new_draw: bool = False

    @classmethod
    def draw(
        cls, source: optuna.trial.FrozenTrial, producers: list[ProducingAttempt]
    ) -> SiblingTrial:
        """Require a usable reading for a budgeted draw unless training itself failed.

        :param optuna.trial.FrozenTrial source: The target sampler's proposed draw.
        :param list[ProducingAttempt] producers: Matching attempts, newest first.
        :return SiblingTrial: The target slot and the evidence available for it.
        """
        needs_read = all(_trainer_seconds(owner.trial) is not None for owner in producers)
        return cls(source, producers, needs_read, True)


def pending_sibling_retries(
    phase: Phase, study: optuna.Study, others: Sequence[optuna.Study]
) -> tuple[SiblingTrial, ...]:
    """Resolve outstanding reruns that already finished training in sibling studies.

    A shared physical ancestor identifies the draw even when a continuation opened a
    fresh directory. Studies can number the same draw differently and can also sample
    the same parameters independently.

    :param Phase phase: Config whose remaining target and rerun policy choose the slots.
    :param optuna.Study study: Target study whose reruns are examined.
    :param Sequence[optuna.Study] others: The phase's other reading studies.
    :return tuple[SiblingTrial, ...]: Completed training matching each outstanding slot.
    """
    others = tuple(other for other in others if other.study_name != study.study_name)
    if not others or _recorded_identity(study) is None:
        return ()
    trials = study.get_trials(deepcopy=False)
    room = phase.n_trials - _finished_trial_count(trials)
    if room <= 0:
        return ()
    queued = [
        trial
        for trial in trials
        if trial.state == optuna.trial.TrialState.WAITING and RERUN_OF_ATTR in trial.user_attrs
    ]
    pending = [*queued, *_scheduled_reruns(study, trials, phase)][:room]
    if not pending:
        return ()
    by_number = {trial.number: trial for trial in trials}
    attempts = producing_attempts((study, *others))
    slots: dict[str, tuple[optuna.trial.FrozenTrial, bool]] = {}
    for trial in pending:
        source_number = trial.user_attrs.get(RERUN_OF_ATTR)
        source = (
            by_number.get(source_number)
            if trial.state == optuna.trial.TrialState.WAITING and type(source_number) is int
            else trial
        )
        requires_measurement = source is not None and _trial_run_state(source) == "FAIL"
        for ancestor in _rerun_chain(by_number, trial, attempts=attempts):
            identity = ancestor.user_attrs.get(ATTEMPT_ID_ATTR)
            if isinstance(identity, str):
                slots[identity] = trial, requires_measurement
    if not slots:
        return ()
    present = {trial.user_attrs.get(ATTEMPT_ID_ATTR) for trial in trials}
    sibling_names = {other.study_name for other in others}
    candidates = [
        owner
        for attempt_id, owner in attempts.items()
        if attempt_id not in present
        and owner.study.study_name in sibling_names
        and RERUN_OF_ATTR in owner.trial.user_attrs
        and _trainer_seconds(owner.trial) is not None
    ]

    # Physical study rows carry the time of allocation even when the directory was reopened.
    def launch_order(owner: ProducingAttempt) -> tuple[float, str, int]:
        start = owner.trial.datetime_start
        return (
            0.0 if start is None else start.timestamp(),
            owner.study.study_name,
            owner.trial.number,
        )

    matches: dict[int, SiblingTrial] = {}
    for owner in sorted(candidates, key=launch_order, reverse=True):
        # The first row is this attempt. An earlier row must establish a shared draw.
        ancestors = _rerun_chain(owner.rows, owner.trial, attempts=attempts)
        next(ancestors)
        for ancestor in ancestors:
            identity = ancestor.user_attrs.get(ATTEMPT_ID_ATTR)
            slot = slots.get(identity) if isinstance(identity, str) else None
            if slot is None:
                continue
            source, requires_measurement = slot
            if _same_value(source.params, owner.trial.params):
                retry = matches.setdefault(
                    source.number, SiblingTrial(source, [], requires_measurement)
                )
                retry.producers.append(owner)
                break
    return tuple(matches[trial.number] for trial in pending if trial.number in matches)


def _sibling_draw_candidates(
    experiment: Experiment, phase: Phase, study: optuna.Study, others: Sequence[optuna.Study]
) -> list[ProducingAttempt]:
    """Resolve terminal physical draws from studies with compatible training identities.

    :param Experiment experiment: Current trainer and reading settings.
    :param Phase phase: Phase whose identity the siblings must share.
    :param optuna.Study study: Initialized target study.
    :param Sequence[optuna.Study] others: Other reading studies of the phase.
    :return list[ProducingAttempt]: Terminal candidates, newest first, excluding attempts present.
    """
    record = _recorded_identity(study)
    if record is None:
        return []
    payload = _identity_payload(experiment, phase, _recorded_inheritance(record))
    compatible = []
    for other in others:
        if other.study_name == study.study_name or (identity := _recorded_identity(other)) is None:
            continue
        # An individual draw need not adopt its source's wider search space. Its exact
        # parameters must match the target proposal; every other training input must agree.
        source = _with_search_space(
            {**identity["payload"], READING_KEY: payload[READING_KEY]},
            _payload_search_space(payload),
        )
        if not _same_value(source, payload):
            continue
        compatible.append(other)
    present = {trial.user_attrs.get(ATTEMPT_ID_ATTR) for trial in study.get_trials(deepcopy=False)}
    candidates = [
        owner
        for identity, owner in producing_attempts(compatible).items()
        if identity not in present
        and owner.trial.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL}
        and (outcome := _parsed_trial_outcome(owner.trial.user_attrs.get(TRIAL_OUTCOME_ATTR)))
        is not None
        and outcome.outcome in _REREAD_OUTCOMES
    ]
    candidates.sort(
        key=lambda owner: (
            0.0 if owner.trial.datetime_start is None else owner.trial.datetime_start.timestamp(),
            owner.study.study_name,
            owner.trial.number,
        ),
        reverse=True,
    )
    return candidates


def _sibling_draw_positions(
    candidates: Sequence[ProducingAttempt],
) -> dict[int, list[ProducingAttempt]]:
    """Group physical candidates by draw position, reading each source history once.

    :param Sequence[ProducingAttempt] candidates: Compatible draws, newest first.
    :return dict[int, list[ProducingAttempt]]: Physical candidates at each logical position.
    """
    source_positions: dict[str, dict[int, int]] = {}
    by_position: dict[int, list[ProducingAttempt]] = {}
    for owner in candidates:
        if owner.study.study_name not in source_positions:
            source_positions[owner.study.study_name] = _draw_positions(owner.rows.values())
        at = source_positions[owner.study.study_name][owner.trial.number]
        by_position.setdefault(at, []).append(owner)
    return by_position


def _planned_sibling_draws(
    experiment: Experiment,
    phase: Phase,
    study: optuna.Study,
    candidates: Sequence[ProducingAttempt],
    *,
    after_gap: bool = False,
) -> tuple[SiblingTrial, ...]:
    """Match this study's prospective draws, stopping when an adaptive result is unknown.

    :param Experiment experiment: Current reading settings and sampler.
    :param Phase phase: Phase whose remaining target and proposals apply.
    :param optuna.Study study: Initialized target, with no pending reruns.
    :param Sequence[ProducingAttempt] candidates: Compatible physical draws, newest first.
    :param bool after_gap: Include later deterministic draws after one needs a launch.
    :return tuple[SiblingTrial, ...]: Kept attempts at their prospective target slots.
    """
    if not candidates:
        return ()
    trials = study.get_trials(deepcopy=False)
    record = _recorded_identity(study)
    assert record is not None
    from phasesweep.engine.ledger import open_preview_study

    verdict = _judge_record(
        record, trials, study.study_name, experiment, phase, _recorded_inheritance(record)
    )
    identity = verdict.record or record
    position = max(_draw_positions(trials).values(), default=-1) + 1
    room = max(0, phase.n_trials - _finished_trial_count(trials))
    grid_slots: list[tuple[int, dict[str, object], list[ProducingAttempt]]] = []
    by_position: dict[int, list[ProducingAttempt]] = {}
    if phase.sampler.type == "grid":
        by_cell: dict[str, list[ProducingAttempt]] = {}
        for owner in candidates:
            by_cell.setdefault(_semantic_payload_digest(owner.trial.params), []).append(owner)
        names = grid_axis_names(list(phase.search_space), phase.repeats)
        order = grid_cell_order(
            identity["search_spaces"],
            seed=phase.sampler.seed,
            repeats=phase.repeats,
            phase_name=study.study_name,
        )
        for at in range(position, min(position + room, len(order))):
            params = dict(zip(names, order[at], strict=True))
            matches = by_cell.get(_semantic_payload_digest(params), [])
            if matches:
                grid_slots.append((at, params, matches))
            elif not after_gap:
                break
        if not grid_slots:
            return ()
    else:
        by_position = {
            at: owners
            for at, owners in _sibling_draw_positions(candidates).items()
            if position <= at < position + room
        }
        if not by_position:
            return ()
        room = min(room, max(by_position) - position + 1)

    projected = open_preview_study(experiment, phase, study_name=study.study_name)
    projected.add_trials(trials)
    for key, value in json_object(study.user_attrs).items():
        projected.set_user_attr(key, value)
    _stamp_identity(projected, identity)
    reads: list[SiblingTrial] = []
    if grid_slots:
        proposed = projected.ask()
        _suggest_all(proposed, phase)
        prototype = projected.get_trials(deepcopy=False)[-1]
        for at, params, matches in grid_slots:
            source = copy.copy(prototype)
            source.number = len(trials) + at - position
            source.params = params
            source.system_attrs = {**prototype.system_attrs, "grid_id": at}
            reads.append(SiblingTrial.draw(source, matches))
        return tuple(reads)
    for offset in range(room):
        proposed = projected.ask()
        params = _suggest_all(proposed, phase)
        source = projected.get_trials(deepcopy=False)[-1]
        matches = [
            owner
            for owner in by_position.get(position + offset, ())
            if _same_value(params, owner.trial.params)
        ]
        if matches:
            reads.append(SiblingTrial.draw(source, matches))
        if phase.sampler.type in {"tpe", "cmaes"} or (not matches and not after_gap):
            break
        projected.tell(proposed, 0.0)
    return tuple(reads)


def pending_sibling_trials(
    experiment: Experiment,
    phase: Phase,
    study: optuna.Study,
    others: Sequence[optuna.Study],
    *,
    after_gap: bool = False,
) -> tuple[SiblingTrial, ...]:
    """Find kept training for pending reruns or the next draws this study would make.

    Grid cells match by their complete parameters. Stochastic draws additionally match their
    position, so a repeated configuration never consumes another replicate's slot. Adaptive
    samplers propose only one draw until its current reading can inform the next proposal.

    :param Experiment experiment: Current trainer and reading settings.
    :param Phase phase: Phase whose next draws and remaining target apply.
    :param optuna.Study study: Initialized target study.
    :param Sequence[optuna.Study] others: Other reading studies of the phase.
    :param bool after_gap: Include later deterministic draws after one needs a launch.
    :return tuple[SiblingTrial, ...]: Kept attempts for the next eligible target slots.
    """
    retries = pending_sibling_retries(phase, study, others)
    if retries or _pending_rerun_count(study, phase):
        return retries
    if phase.n_trials <= _finished_trial_count(study.get_trials(deepcopy=False)):
        return ()
    return _planned_sibling_draws(
        experiment,
        phase,
        study,
        _sibling_draw_candidates(experiment, phase, study, others),
        after_gap=after_gap,
    )


def sibling_launch_batch(
    experiment: Experiment, phase: Phase, study: optuna.Study, others: Sequence[optuna.Study]
) -> int:
    """Bound launches before the next draw a sibling could supply.

    A drained batch lets the ordinary sibling reader run again. Adaptive draws can change
    after every result; their next possible donor position bounds the batch even when the
    current proposal does not match. With no candidate, the ordinary optimize budget stands.

    :param Experiment experiment: Current trainer and reading settings.
    :param Phase phase: Phase whose next draws and remaining target apply.
    :param optuna.Study study: Target about to launch trials.
    :param Sequence[optuna.Study] others: Other reading studies of the phase.
    :return int: Maximum launches before checking sibling evidence again.
    """
    trials = study.get_trials(deepcopy=False)
    remaining = max(0, phase.n_trials - _finished_trial_count(trials))
    candidates = _sibling_draw_candidates(experiment, phase, study, others)
    if not candidates:
        return remaining
    reruns = _pending_rerun_count(study, phase)
    if reruns:
        return min(remaining, reruns)
    if phase.sampler.type != "grid":
        position = max(_draw_positions(trials).values(), default=-1) + 1
        donor_positions = [at for at in _sibling_draw_positions(candidates) if at >= position]
        return (
            min(remaining, max(1, min(donor_positions) - position))
            if donor_positions
            else remaining
        )
    planned = _planned_sibling_draws(experiment, phase, study, candidates, after_gap=True)
    return min(remaining, max(1, planned[0].source.number - len(trials))) if planned else remaining


def reread_sibling_trials(
    experiment: Experiment,
    phase: Phase,
    study: optuna.Study,
    others: Sequence[optuna.Study],
    run_deadline: float | None,
    reader_generation_id: str,
    *,
    recovery_abort: JsonObject | None = None,
) -> int:
    """Read a sibling's finished training under this study's settings before launching.

    The physical attempt comes from the producer. The slot, failure streak and outcome
    sequence belong to this study. Finished attempts are read in newest-first order until
    one measures the requested result. If none measures, an interrupted slot keeps the
    newest verdict and a budgeted retry can still launch. An adopted queued retry is settled
    as unlaunched first so it cannot launch again after the terminal result is appended.

    :param Experiment experiment: Current reading and budget settings.
    :param Phase phase: Phase whose target and failure policy apply.
    :param optuna.Study study: Initialized study accepting its sibling's kept work.
    :param Sequence[optuna.Study] others: The phase's other reading studies.
    :param float | None run_deadline: Absolute invocation deadline.
    :param str reader_generation_id: Generation that owns any tracker reader.
    :param JsonObject | None recovery_abort: Previous abort the run may retire, if any.
    :raises PhaseTimeoutError: The deadline stopped a read before its row was committed.
    :raises StudySchemaMismatchError: A resolved finished attempt lacks its training record.
    :return int: Terminal results appended to this study.
    """
    retries = pending_sibling_trials(experiment, phase, study, others)
    if not retries:
        return 0
    attempts = producing_attempts(
        (study, *(other for other in others if other.study_name != study.study_name))
    )
    launches = directory_launches(attempts)
    producing: dict[str, Experiment | WinnerIntegrityError] = {}
    recorded = 0
    while retries:
        before = recorded
        abort = None
        needs_launch = False
        for retry in retries:
            reread = None
            for donor in retry.producers:
                _require_reread_time(experiment, phase, run_deadline)
                candidate = _reread_trial(
                    experiment,
                    phase,
                    donor.trial,
                    donor.rows,
                    producing,
                    attempts,
                    launches,
                    deadline=run_deadline,
                    reader_generation_id=reader_generation_id,
                )
                _require_reread_time(experiment, phase, run_deadline)
                if candidate is None:
                    if (
                        retry.new_draw
                        and donor.trial.state == optuna.trial.TrialState.FAIL
                        and _trainer_seconds(donor.trial) is None
                    ):
                        # A definite trainer failure spends the draw, as it would in a fill.
                        if reread is None:
                            reread = copy.deepcopy(donor.trial)
                        continue
                    raise StudySchemaMismatchError(
                        f"Finished sibling trial {donor.trial.number} in {donor.study.study_name!r} "
                        "cannot be re-read from its recorded training."
                    )
                if candidate.state == optuna.trial.TrialState.COMPLETE:
                    reread = candidate
                    break
                # A failed gate judged the measured outputs. A donor missing a requested
                # measurement cannot spend a new draw that the target is entitled to launch.
                decisive_failure = (
                    candidate.user_attrs.get(GATES_ATTR) is not None
                    if retry.new_draw
                    else not retry.requires_measurement
                )
                if reread is None and decisive_failure:
                    reread = candidate
            if reread is None:
                if retry.new_draw:
                    # The next draw must occupy this position before any later donor can.
                    needs_launch = True
                    break
                continue
            if recovery_abort is not None and recorded == 0:
                sequence = recovery_abort["completion_sequence"]
                assert type(sequence) is int
                _record_recovery_boundary(study, phase, sequence)
            if _finished_trial_count(study.get_trials(deepcopy=False)) >= _accepted_trial_target(
                study
            ):
                _record_trial_target(study, phase)
            if retry.source.state == optuna.trial.TrialState.WAITING:
                study._storage.set_trial_state_values(
                    retry.source._trial_id, optuna.trial.TrialState.RUNNING
                )
                _record_stale_trial_outcome(
                    study, retry.source, cause="a sibling reading study finished this rerun"
                )
                study.tell(retry.source.number, state=optuna.trial.TrialState.FAIL)
            state = _load_phase_policy_state(study, experiment)
            sequence = _recorded_outcome_sequence(study)
            measured = _parsed_trial_outcome(reread.user_attrs.get(TRIAL_OUTCOME_ATTR))
            assert measured is not None
            abort = _abort_decision(
                phase,
                len(study.get_trials(deepcopy=False)),
                _counted_outcome(measured.outcome, measured.metric, reread, experiment),
                completion_sequence=sequence,
                consecutive_failures=state.consecutive_failures,
                abort_recorded=state.abort is not None,
            )
            if retry.new_draw:
                reread.user_attrs.pop(RERUN_OF_ATTR, None)
            else:
                reread.user_attrs[RERUN_OF_ATTR] = retry.source.number
            reread.user_attrs[TRIAL_OUTCOME_ATTR] = _trial_outcome_payload(
                sequence + 1,
                measured.outcome,
                metric=measured.metric,
                cause=measured.cause,
                abort=abort,
            )
            reread.distributions = retry.source.distributions
            reread.system_attrs = retry.source.system_attrs
            reread.intermediate_values = dict(retry.source.intermediate_values)
            _require_reread_time(experiment, phase, run_deadline)
            study.add_trial(reread)
            recorded += 1
            if abort is not None:
                break
        if needs_launch or abort is not None or recorded == before:
            break
        retries = pending_sibling_trials(experiment, phase, study, others)
    if recorded:
        report.info(
            "%s re-read %s already finished in sibling studies without launching a trainer.",
            phase.name,
            plural(recorded, "trial"),
        )
    return recorded


def describe_changes(changes: Sequence[tuple[str, object, object]]) -> str:
    """Name the reading settings that changed, with both values.

    :param Sequence[tuple[str, object, object]] changes: :attr:`FillSource.changes`.
    :return str: Such as ``metric.extractor.key: "eval_los" -> "eval_loss"``.
    """
    shown = [
        f"{path}: {' -> '.join(_render_changed_values(old, new))}"
        for path, old, new in changes[:_MAX_REPORTED_CHANGES]
    ]
    if len(changes) > _MAX_REPORTED_CHANGES:
        shown.append(f"and {len(changes) - _MAX_REPORTED_CHANGES} more")
    return "; ".join(shown) or "reading settings"


class PendingRereadNotice(PhaseSweepError):
    """A phase not started under the current reading settings, which a run re-reads, not retrains.

    ``status`` and ``status --trials`` print it where they report a phase's refusal, with the
    changed values and the command that fills the study. The status payload, which MCP serves,
    carries :attr:`statement` instead: the same sentence naming only the changed settings,
    since their values are config-authored text such as file names and patterns.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RETRY
    #: The value-free sentence the status payload carries (:func:`_pending_notice`).
    statement: str = ""
    #: Finished trials the fill re-reads (:attr:`FillSource.rereads`).
    finished_trials: int = 0


def _pending_notice(
    source: FillSource, phase: str, unsettled: Sequence[str] = ()
) -> PendingRereadNotice:
    """Build both texts of a :class:`PendingRereadNotice` from one description of the fill.

    A parent that is itself re-read first may select another winner, and then this phase
    inherits other values and runs in another study, so the re-read here is conditional.

    :param FillSource source: The study the fill would read from.
    :param str phase: The phase's name.
    :param Sequence[str] unsettled: Parents whose winner the next run selects again first,
        defaults to none.
    :return PendingRereadNotice: The notice, with its message and step for the CLI and its
        :attr:`PendingRereadNotice.statement` for the status payload.
    """
    opening = f"Phase {phase!r} has not started under the current reading settings"
    trials = plural(source.rereads, "finished trial")
    settings = ", ".join(path for path, _old, _new in source.changes) or "reading settings"
    if unsettled:
        names = " and ".join(repr(name) for name in unsettled)
        keeps = "keeps its winner" if len(unsettled) == 1 else "keep their winners"
        remediation = (
            f"Run the same configuration. If {names} {keeps} after re-reading, this phase "
            f"re-reads {trials} from kept evidence without retraining them; a new winner "
            "starts a new study here."
        )
        statement = (
            f"{opening} ({settings} changed); the next run re-reads {trials} from kept evidence "
            f"if {names} {keeps} after re-reading, and otherwise starts a new study."
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
        f"{opening} ({describe_changes(source.changes)}).", remediation=remediation
    )
    notice.statement = statement
    notice.finished_trials = source.rereads
    return notice


def pending_reread(
    experiment: Experiment,
    phase: Phase,
    study_name: str,
    studies: Sequence[optuna.Study],
    *,
    unsettled: Sequence[str] = (),
) -> PendingRereadNotice | None:
    """Say that a phase's addressed study waits for its fill, for a read that writes nothing.

    The inherited values the address stands for are the ones a study of the phase recorded
    whose values, read under the current settings, give that address.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param str study_name: The study it addresses (:func:`phase_study_name`).
    :param Sequence[optuna.Study] studies: Every study of the phase, oldest first.
    :param Sequence[str] unsettled: Parents whose winner the next run selects again before this
        phase, so the address holds only if they keep it; defaults to none.
    :raises StudyFingerprintMismatchError: A study with these inherited values has another
        training identity (:func:`fill_source`).
    :return PendingRereadNotice | None: The notice, or ``None`` when the addressed study is
        filled or no study holds trials to re-read.
    """
    addressed = next((study for study in studies if study.study_name == study_name), None)
    if addressed is not None and _recorded_identity(addressed) is not None:
        return None
    others = [study for study in studies if study.study_name != study_name]
    pinned = _fill_source_name(addressed)
    for study in _fill_candidates(pinned, others):
        record = _recorded_identity(study)
        if record is None:
            continue
        inherited = _recorded_inheritance(record)
        if phase_study_name(experiment, phase, inherited) == study_name:
            source = fill_source(experiment, phase, inherited, others, target=addressed)
            return None if source is None else _pending_notice(source, phase.name, unsettled)
    if pinned is not None:
        raise StudySchemaMismatchError(
            f"Saved fill source {pinned!r} does not address study {study_name!r}."
        )
    return None


def fill_note(source: FillSource, result: FillResult) -> str:
    """Say what a fill did, for the run's line about the phase.

    :param FillSource source: The study the fill read from.
    :param FillResult result: What it wrote.
    :return str: The sentence.
    """
    unread = result.copied - result.rereads
    copied = f", {plural(unread, 'unfinished trial')} copied as recorded" if unread else ""
    adjustment = ""
    if result.adjusted_target is not None:
        previous, accepted = result.adjusted_target
        adjustment = (
            f" The re-read results reduce the accepted trial target from {previous} to {accepted}."
        )
    return (
        f"reading settings changed ({describe_changes(source.changes)}): re-read "
        f"{plural(result.rereads, 'finished trial')} from kept evidence, {result.changed} "
        f"changed outcome{copied}; nothing was retrained.{adjustment}"
    )
