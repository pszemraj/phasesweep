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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import optuna

from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import retry_target_ceiling
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
from phasesweep.engine.evidence import _earlier_attempts
from phasesweep.engine.fingerprints import (
    _MAX_REPORTED_CHANGES,
    READING_KEY,
    _changed_payload_keys,
    _identity_record,
    _inherited_values,
    _judge_record,
    _recorded_inheritance,
    _render_changed_values,
    _same_value,
    _semantic_payload_digest,
    _stamp_identity,
    phase_study_name,
    reading_settings,
)
from phasesweep.engine.optuna import (
    _finished_trial_count,
    _measured_draw_count,
    _measured_trial_count,
)
from phasesweep.engine.paths import _attempt_dir, _experiment_dir, _trial_dir_opener
from phasesweep.engine.provenance import _load_generation_experiment
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
    ATTEMPT_ID_ATTR,
    DURATION_ATTR,
    FAILURE_REASON_ATTR,
    OVERRIDES_ATTR,
    PHASE_DECISION_ATTR,
    PHASE_IDENTITY_ATTR,
    PHASE_RECOVERY_ATTR,
    REREAD_FILL_ATTR,
    RERUN_OF_ATTR,
    RETURN_CODE_ATTR,
    STUDY_SCHEMA_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_TARGET_ATTR,
    PhaseIdentityRecord,
    _parsed_trial_outcome,
    _recorded_identity,
    _recorded_trial_dir,
    _required_trial_dir,
    _trial_lifecycle,
    _trial_outcome_payload,
)
from phasesweep.engine.study_policy import (
    _accepted_trial_target,
    _initialize_study_schema,
    _parse_accepted_partial_decision,
)
from phasesweep.engine.trial import TrialResult, _failed_trial, record_trial_result
from phasesweep.errors import OperatorAction, PhaseSweepError
from phasesweep.evidence.baseline import read_evidence_baseline
from phasesweep.runtime.json import JsonObject, finite_nonnegative, json_object
from phasesweep.runtime.terminal import plural

REREAD_FILL_SCHEMA_VERSION = 3
_FILL_FIELDS = frozenset({"schema_version", "study_name"})
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


def _fill_source_name(target: optuna.Study | None) -> str | None:
    """Read the study name recorded before a fill began.

    :param optuna.Study | None target: The addressed study, if it exists.
    :raises StudySchemaMismatchError: The saved source record is malformed.
    :return str | None: The saved source name, or None for a fresh fill.
    """
    raw = None if target is None else target.user_attrs.get(REREAD_FILL_ATTR)
    if raw is None:
        return None
    assert target is not None
    try:
        record = json_object(raw)
        name = record["study_name"]
        if (
            record.keys() != _FILL_FIELDS
            or type(record.get("schema_version")) is not int
            or record["schema_version"] != REREAD_FILL_SCHEMA_VERSION
            or not isinstance(name, str)
            or not name
        ):
            raise ValueError("invalid source name or schema version")
    except (KeyError, ValueError) as exc:
        raise StudySchemaMismatchError(
            f"Study {target.study_name!r} has a malformed {REREAD_FILL_ATTR!r} record: {exc}."
        ) from exc
    return name


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
    *,
    target: optuna.Study | None = None,
) -> FillSource | None:
    """Pick the study a phase's new study is filled from, judging without writing anything.

    A candidate holds trials measured under the same inherited values and would continue
    under the current config once its reading settings are replaced. The
    candidate with the most trainer-finished trials wins, the most recently created on a tie.
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
            trial.number for trial in trials[copied:] if _trainer_seconds(trial) is not None
        )
        attrs = json_object(
            {key: value for key, value in study.user_attrs.items() if key not in _NOT_COPIED}
        )
        found.append(FillSource(study, record, rereads, tuple(changes), trials, attrs))
    if not found:
        if refusal is not None:
            raise refusal
        return None
    return max(reversed(found), key=lambda source: source.rereads)


def _directory_reuses(
    trials: Iterable[optuna.trial.FrozenTrial],
) -> dict[str, dict[str, float | None]]:
    """Collect the launch epochs of attempts that could write a reused directory.

    A continued directory keeps its basename even if the artifact tree moves. Copies in
    another reading study keep the same attempt ID. The opener needs no launch epoch.

    :param Iterable[optuna.trial.FrozenTrial] trials: Attempts visible in the phase's studies.
    :return dict[str, dict[str, float | None]]: Launch epochs by directory name and attempt ID.
        A missing epoch marks a possible writer whose launch time cannot be established.
    """
    reuses: dict[str, dict[str, float | None]] = {}
    for trial in trials:
        try:
            directory = _recorded_trial_dir(trial)
            opener = None if directory is None else _trial_dir_opener(directory)
        except ValueError:
            continue
        attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
        if (
            directory is None
            or not isinstance(attempt_id, str)
            or not attempt_id
            or attempt_id == opener
        ):
            continue
        launches = reuses.setdefault(directory.name, {})
        if attempt_id in launches:
            continue
        if type(trial.user_attrs.get(RETURN_CODE_ATTR)) is not int:
            try:
                lifecycle = _trial_lifecycle(trial)
            except ValueError:
                lifecycle = None
            if lifecycle is not None and (
                lifecycle.state == "allocated" or lifecycle.ended_by == "not_started"
            ):
                continue
        try:
            launches[attempt_id] = read_evidence_baseline(
                _attempt_dir(directory.parent, attempt_id)
            ).launched_at
        except ValueError:
            launches[attempt_id] = None
    return reuses


def _shared_evidence_cutoff(
    launches: Mapping[str, float | None],
    trials: Mapping[int, optuna.trial.FrozenTrial],
    trial: optuna.trial.FrozenTrial,
    attempt_id: str,
    launched_at: float | None,
) -> float | str | None:
    """Find the next reuse of the directory, after which its evidence belongs elsewhere.

    :param Mapping[str, float | None] launches: Launch epochs of attempts that reused the directory.
    :param Mapping[int, optuna.trial.FrozenTrial] trials: Source study's trials by number.
    :param optuna.trial.FrozenTrial trial: The trial being re-read.
    :param str attempt_id: The trial's attempt ID.
    :param float | None launched_at: The attempt's launch epoch, absent for the opener.
    :return float | str | None: The next launch epoch, a reason shared evidence cannot be attributed,
        or None when there was no later reuse.
    """
    other_launches = {other: start for other, start in launches.items() if other != attempt_id}
    undated = {other for other, start in other_launches.items() if start is None}
    if undated:
        earlier = {earlier_id for _, earlier_id in _earlier_attempts(trials, trial)}
        unknown = min(undated - earlier, default=None)
        if unknown is not None:
            return f"the launch time of attempt {unknown!r} in its shared directory is unavailable"
    return min(
        (
            start
            for start in other_launches.values()
            if start is not None and (launched_at is None or start >= launched_at)
        ),
        default=None,
    )


def _reread_trial(
    experiment: Experiment,
    phase: Phase,
    trial: optuna.trial.FrozenTrial,
    trials: Mapping[int, optuna.trial.FrozenTrial],
    producing: dict[str, Experiment | WinnerIntegrityError],
    directory_reuses: Mapping[str, Mapping[str, float | None]],
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
    :param Mapping[str, Mapping[str, float | None]] directory_reuses: Attempts that reused each
        trainer directory and their launch epochs.
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
        directory = _required_trial_dir(trial)
        record = _trial_attempt_record(trial, phase.name, directory)
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
            trials,
            trial,
            record,
            producing=launched,
            reading=experiment,
            current=experiment,
            trainer_duration=seconds,
            baseline=baseline,
            evidence_before=_shared_evidence_cutoff(
                directory_reuses.get(directory.name, {}),
                trials,
                trial,
                record.attempt_id,
                baseline.launched_at,
            ),
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


def fill_study(
    experiment: Experiment,
    phase: Phase,
    target: optuna.Study,
    source: FillSource,
    *,
    other_studies: Sequence[optuna.Study],
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
    :raises StudySchemaMismatchError: The target holds trials the source does not, or the
        source's accepted partial decision is malformed.
    :raises TrackingTargetUnavailableError: A tracker gave no answer for a finished trial.
    :return FillResult: What the fill wrote in this call.
    """
    end = len(source.trials)
    while end > 0 and (
        (trial := source.trials[end - 1]).state == optuna.trial.TrialState.WAITING
        and RERUN_OF_ATTR in trial.user_attrs
    ):
        end -= 1
    trials = source.trials[:end]
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
    directory_reuses = _directory_reuses(
        trial for study in other_studies for trial in study.get_trials(deepcopy=False)
    )
    producing: dict[str, Experiment | WinnerIntegrityError] = {}
    rereads = changed = 0
    for trial in trials[len(present) :]:
        reread = _reread_trial(experiment, phase, trial, by_number, producing, directory_reuses)
        if reread is not None:
            rereads += 1
            changed += (reread.state, reread.value) != (trial.state, trial.value)
        target.add_trial(trial if reread is None else reread)
    adjusted_target = _copy_study_attrs(phase, target, source)
    _stamp_identity(target, _reread_record(source.record, reading_settings(experiment, phase)))
    return FillResult(len(trials) - len(present), rereads, changed, adjusted_target)


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
