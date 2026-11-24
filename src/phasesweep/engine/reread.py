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

The fill is resumable. Each trial is appended in one storage write and the identity record is
written last, so a study holding trials but no identity is a fill a killed runner left
part-way; the next run appends the trials still missing, from the same source.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import optuna

from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.attempts import (
    _pending_evidence_refusal,
    _trial_attempt_record,
    read_finished_attempt,
)
from phasesweep.engine.errors import (
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    WinnerIntegrityError,
)
from phasesweep.engine.fingerprints import (
    _MAX_REPORTED_CHANGES,
    READING_KEY,
    _changed_payload_keys,
    _identity_record,
    _judge_record,
    _render_changed_values,
    _same_value,
    _semantic_payload_digest,
    _stamp_identity,
    phase_study_name,
    reading_settings,
)
from phasesweep.engine.paths import _experiment_dir
from phasesweep.engine.provenance import _load_generation_experiment
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
    DURATION_ATTR,
    FAILURE_REASON_ATTR,
    OVERRIDES_ATTR,
    PHASE_IDENTITY_ATTR,
    RETURN_CODE_ATTR,
    STUDY_SCHEMA_ATTR,
    TRIAL_OUTCOME_ATTR,
    PhaseIdentityRecord,
    _parsed_trial_outcome,
    _recorded_identity,
    _required_trial_dir,
    _trial_lifecycle,
    _trial_outcome_payload,
)
from phasesweep.engine.study_policy import _initialize_study_schema
from phasesweep.engine.trial import TrialResult, _failed_trial, record_trial_result
from phasesweep.errors import OperatorAction, PhaseSweepError
from phasesweep.evidence.baseline import read_evidence_baseline
from phasesweep.runtime.json import JsonObject, finite_nonnegative, json_object
from phasesweep.runtime.terminal import plural

#: Study attributes a fill does not copy: the identity is rebuilt under the current reading
#: settings and written last, the opener already bound the artifact root, and the schema stamp
#: is the run's own.
_NOT_COPIED = frozenset({PHASE_IDENTITY_ATTR, ARTIFACT_ROOT_ATTR, STUDY_SCHEMA_ATTR})
#: The recorded outcomes a re-read replaces; an interruption or a fatal outcome stands as it was.
_REREAD_OUTCOMES = frozenset({"success", "failure"})


@dataclass(frozen=True)
class FillSource:
    """The study a new one is filled from, and what its reading settings were."""

    study: optuna.Study
    record: PhaseIdentityRecord
    #: How many of its trials have a finished trainer, so are read again.
    rereads: int
    #: Each reading setting that changed, with its recorded and current value.
    changes: tuple[tuple[str, object, object], ...]


@dataclass(frozen=True)
class FillResult:
    """What a fill wrote: the trials it copied, re-read, and whose outcome the re-read changed."""

    copied: int
    rereads: int
    changed: int


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
        _semantic_payload_digest(payload), payload, list(record["search_spaces"])
    )


def _trainer_seconds(trial: optuna.trial.FrozenTrial) -> float | None:
    """Return a finished trial's trainer time when its trainer exited successfully.

    The trainer's exit is what the ledger recorded when it ended (return code and duration),
    confirmed by the attempt's lifecycle record while that is kept. Only a trial whose recorded
    outcome is a success or a failure is read again; an interruption or a fatal outcome is not
    a reading result.

    :param optuna.trial.FrozenTrial trial: A source study's trial.
    :return float | None: The trainer's seconds, or ``None`` when the trial is copied unread.
    """
    outcome = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    return_code = trial.user_attrs.get(RETURN_CODE_ATTR)
    seconds = finite_nonnegative(trial.user_attrs.get(DURATION_ATTR))
    if (
        trial.state not in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL}
        or outcome is None
        or outcome.outcome not in _REREAD_OUTCOMES
        or type(return_code) is not int
        or return_code != 0
        or seconds is None
    ):
        return None
    try:
        lifecycle = _trial_lifecycle(trial)
    except ValueError:
        lifecycle = None
    if lifecycle is not None and (lifecycle.ended_by != "trainer" or lifecycle.return_code != 0):
        return None
    return float(seconds)


def fill_source(
    experiment: Experiment,
    phase: Phase,
    inherited: Mapping[str, object],
    others: Sequence[optuna.Study],
) -> FillSource | None:
    """Pick the study a phase's new study is filled from, judging without writing anything.

    A candidate holds trials measured under the same inherited values and would continue under
    the current config once its reading settings are replaced: the check is the one a study's
    own identity gets (:func:`phasesweep.engine.fingerprints._judge_record`), so a changed
    trainer, environment or phase is refused as it is in the study it changed. The candidate
    with the most trainer-finished trials wins, the most recently created on a tie.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase whose addressed study records no identity.
    :param Mapping[str, object] inherited: The values the phase inherits now.
    :param Sequence[optuna.Study] others: The phase's other studies, oldest first.
    :raises StudyFingerprintMismatchError: A study with these inherited values exists but its
        training identity differs from the current config; the refusal names what changed.
    :raises StudySchemaMismatchError: A candidate's identity record is malformed.
    :return FillSource | None: The source, or ``None`` when no study holds these values.
    """
    reading = reading_settings(experiment, phase)
    values = json_object(json.loads(json.dumps(dict(inherited), default=str)))
    found: list[FillSource] = []
    refusal: StudyFingerprintMismatchError | None = None
    for study in others:
        record = _recorded_identity(study)
        trials = study.get_trials(deepcopy=False)
        if record is None or not trials:
            continue
        payload = record["payload"]
        if not _same_value(payload.get("inherited_effective_overrides"), values):
            continue
        try:
            _judge_record(
                _reread_record(record, reading),
                trials,
                study.study_name,
                experiment,
                phase,
                values,
            )
        except StudyFingerprintMismatchError as exc:
            refusal = exc
            continue
        recorded = payload.get(READING_KEY)
        changes = _changed_payload_keys(recorded if isinstance(recorded, dict) else {}, reading)
        rereads = sum(_trainer_seconds(trial) is not None for trial in trials)
        found.append(FillSource(study, record, rereads, tuple(changes)))
    if not found:
        if refusal is not None:
            raise refusal
        return None
    return max(reversed(found), key=lambda source: source.rereads)


def _reread_trial(
    experiment: Experiment,
    phase: Phase,
    trial: optuna.trial.FrozenTrial,
    trials: Mapping[int, optuna.trial.FrozenTrial],
    producing: dict[str, Experiment | WinnerIntegrityError],
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
    :raises TrackingTargetUnavailableError: The tracker gave no answer, so the verdict waits for
        a later run; the trials already copied stay and the fill resumes.
    :return optuna.trial.FrozenTrial | None: The trial to write, or ``None`` when the trial is
        copied as it was recorded.
    """
    seconds = _trainer_seconds(trial)
    outcome = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    overrides_json = trial.user_attrs.get(OVERRIDES_ATTR)
    if seconds is None or outcome is None or not isinstance(overrides_json, str):
        return None
    try:
        record = _trial_attempt_record(trial, phase.name, _required_trial_dir(trial))
    except ValueError:
        record = None
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
            trials,
            trial,
            record,
            producing=launched,
            reading=experiment,
            current=experiment,
            trainer_duration=seconds,
            baseline=baseline,
        )
    except ValueError as exc:
        result = _failed_trial(rc=0, duration=seconds, failure_reason=str(exc))
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


def fill_study(
    experiment: Experiment, phase: Phase, target: optuna.Study, source: FillSource
) -> FillResult:
    """Fill a phase's addressed study from its source, re-reading every finished trainer.

    The target is the study a run opened for the phase and records no identity: empty, or left
    part-way by a killed fill, whose trials are then the source's first ones, so the fill
    appends the rest. The identity, the source's with the current reading settings, is written
    last and marks the fill complete.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param optuna.Study target: The addressed study, writable.
    :param FillSource source: The study to fill from (:func:`fill_source`).
    :raises StudySchemaMismatchError: The target holds trials the source does not.
    :raises TrackingTargetUnavailableError: A tracker gave no answer for a finished trial.
    :return FillResult: What the fill wrote in this call.
    """
    trials = source.study.get_trials(deepcopy=False)
    present = target.get_trials(deepcopy=False)
    if [(trial.number, trial.params) for trial in present] != [
        (trial.number, trial.params) for trial in trials[: len(present)]
    ]:
        raise StudySchemaMismatchError(
            f"Study {target.study_name!r} holds trials that study {source.study.study_name!r} "
            "does not, so it cannot be filled from it."
        )
    _initialize_study_schema(target)
    if not present:
        for key, value in source.study.user_attrs.items():
            if key not in _NOT_COPIED:
                target.set_user_attr(key, value)
    by_number = {trial.number: trial for trial in trials}
    producing: dict[str, Experiment | WinnerIntegrityError] = {}
    rereads = changed = 0
    for trial in trials[len(present) :]:
        reread = _reread_trial(experiment, phase, trial, by_number, producing)
        if reread is not None:
            rereads += 1
            changed += (reread.state, reread.value) != (trial.state, trial.value)
        target.add_trial(trial if reread is None else reread)
    _stamp_identity(target, _reread_record(source.record, reading_settings(experiment, phase)))
    return FillResult(copied=len(trials) - len(present), rereads=rereads, changed=changed)


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


def _pending_notice(source: FillSource, phase: str) -> PendingRereadNotice:
    """Build both texts of a :class:`PendingRereadNotice` from one description of the fill.

    :param FillSource source: The study the fill would read from.
    :param str phase: The phase's name.
    :return PendingRereadNotice: The notice, with its message and step for the CLI and its
        :attr:`PendingRereadNotice.statement` for the status payload.
    """
    opening = f"Phase {phase!r} has not started under the current reading settings"
    trials = plural(source.rereads, "finished trial")
    notice = PendingRereadNotice(
        f"{opening} ({describe_changes(source.changes)}).",
        remediation=(
            f"Run `phasesweep run CONFIG`: it re-reads {trials} from kept evidence and retrains "
            "nothing."
        ),
    )
    settings = ", ".join(path for path, _old, _new in source.changes) or "reading settings"
    notice.statement = (
        f"{opening} ({settings} changed); the next run re-reads {trials} from kept evidence and "
        "retrains nothing."
    )
    notice.finished_trials = source.rereads
    return notice


def pending_reread(
    experiment: Experiment, phase: Phase, study_name: str, studies: Sequence[optuna.Study]
) -> PendingRereadNotice | None:
    """Say that a phase's addressed study waits for its fill, for a read that writes nothing.

    The inherited values the address stands for are the ones a study of the phase recorded
    whose values, read under the current settings, give that address.

    :param Experiment experiment: Current config.
    :param Phase phase: The phase.
    :param str study_name: The study it addresses (:func:`phase_study_name`).
    :param Sequence[optuna.Study] studies: Every study of the phase, oldest first.
    :raises StudyFingerprintMismatchError: A study with these inherited values has another
        training identity (:func:`fill_source`).
    :return PendingRereadNotice | None: The notice, or ``None`` when the addressed study is
        filled or no study holds trials to re-read.
    """
    addressed = next((study for study in studies if study.study_name == study_name), None)
    if addressed is not None and _recorded_identity(addressed) is not None:
        return None
    others = [study for study in studies if study.study_name != study_name]
    inherited: dict[str, object] | None = None
    for study in others:
        record = _recorded_identity(study)
        values = None if record is None else record["payload"].get("inherited_effective_overrides")
        if isinstance(values, dict) and phase_study_name(experiment, phase, values) == study_name:
            inherited = dict(values)
            break
    if inherited is None:
        return None
    source = fill_source(experiment, phase, inherited, others)
    if source is None:
        return None
    return _pending_notice(source, phase.name)


def fill_note(source: FillSource, result: FillResult) -> str:
    """Say what a fill did, for the run's line about the phase.

    :param FillSource source: The study the fill read from.
    :param FillResult result: What it wrote.
    :return str: The sentence.
    """
    unread = result.copied - result.rereads
    copied = f", {plural(unread, 'unfinished trial')} copied as recorded" if unread else ""
    return (
        f"reading settings changed ({describe_changes(source.changes)}): re-read "
        f"{plural(result.rereads, 'finished trial')} from kept evidence, {result.changed} "
        f"changed outcome{copied}; nothing was retrained."
    )
