"""Retryable MLflow projections kept separate from scientific evidence."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from phasesweep.engine.errors import (
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrackingSyncRefusedError,
    TrackingTargetUnavailableError,
)
from phasesweep.errors import (
    OperatorAction,
    PhaseSweepError,
    ProcessCleanupUncertainError,
    operator_step,
)
from phasesweep.runtime.json import (
    JsonObject,
    json_number,
    json_object,
)
from phasesweep.runtime.shutdown import PhaseSweepShutdown, signal_handler_scope
from phasesweep.runtime.terminal import REPORT_LOGGER, ReportText, plural
from phasesweep.tracking.records import validate_binding
from phasesweep.tracking.session import (
    BINDING_FILE,
    DELIVERY_DIR,
    GENERATION_BINDING_FILE,
    ORIGINAL_TARGET_STEP,
    SOURCE_FILE,
    SOURCE_SCHEMA_VERSION,
    SUMMARY_RECORD,
    SYNC_TRACKING_STEP,
    DeliveryTally,
    attempt_tracking_dir,
    attempt_tracking_records,
    call_tracking,
    count_text,
    delivered_text,
    delivery_record_path,
    pending_creation_records,
    read_binding,
    read_inherits,
    read_record,
    record_name,
    refused_text,
    require_tracking_target,
    resolve_pending_creations,
    summary_binding_path,
    tracking_root,
    write_record,
)

if TYPE_CHECKING:
    from optuna.trial import FrozenTrial

    from phasesweep.config.models import Experiment
    from phasesweep.engine.generation import _RecordedGeneration
    from phasesweep.engine.ledger import ValidatedLedger
    from phasesweep.tracking.records import DeliveryRecord, Projection, PublishAction, RunBinding

log = logging.getLogger(__name__)
# Lines the console shows as the run's own report, not as warnings.
report = logging.getLogger(REPORT_LOGGER)


def _projection(
    *,
    metrics: dict[str, float],
    params: dict[str, str],
    tags: dict[str, str],
    summary: JsonObject,
) -> Projection:
    """Freeze a retryable result mirror with a stable metric timestamp.

    :param dict[str, float] metrics: Reserved scalar metrics.
    :param dict[str, str] params: Effective override parameters.
    :param dict[str, str] tags: Phase and winner references.
    :param JsonObject summary: Compact artifact contents.
    :return Projection: Retryable projection.
    """
    return {
        "timestamp": int(time.time() * 1000),
        "metrics": metrics,
        "params": params,
        "tags": tags,
        "summary": summary,
    }


def _deliver(
    experiment: Experiment,
    path: Path,
    environment: Mapping[str, str],
    *,
    worker_generation: str | None = None,
    deadline: float | None = None,
) -> None:
    """Upload one persisted projection and mark its record delivered.

    The upload worker is registered under the generation of the run that sends it, so a worker
    whose cleanup that run cannot confirm is attributed to it, whatever generation the record
    belongs to. :func:`_send` decides what a failure means for the record and the pass.

    :param Experiment experiment: Configured target.
    :param Path path: Pending delivery record.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str | None worker_generation: Generation of the run sending another generation's
        record, under which its upload worker is registered; ``None`` registers it under the
        record's own generation, as for a run's own records and for sync-tracking, which is no run.
    :param float | None deadline: Absolute run deadline the upload is cut at, defaults to none.
    :raises ProcessCleanupUncertainError: The upload worker's cleanup is uncertain.
    :raises OSError: The upload or the record's update failed, MLflow's verdict or no answer.
    :raises ValueError: The record is malformed, or the upload failed.
    :raises RuntimeError: The upload failed.
    """
    delivery = _read_delivery(path)
    if delivery.delivered:
        return
    action: PublishAction = {
        "action": "publish",
        "name": path.stem,
        "binding": delivery.binding,
        "projection": delivery.projection,
        "finish": delivery.finish,
    }
    call_tracking(
        experiment, worker_generation or delivery.generation_id, action, environment, deadline
    )
    # A lost acknowledgement resends the same projection on the next sync.
    write_record(path, {**delivery.record, "state": "delivered"})


class _Delivery(NamedTuple):
    """The fields of a delivery record that sending and counting read, each checked."""

    record: JsonObject
    """The whole record, written back with its new state once delivered."""
    delivered: bool
    generation_id: str
    source: JsonObject
    binding: JsonObject
    projection: JsonObject
    finish: str | None


def _read_delivery(path: Path) -> _Delivery:
    """Read a delivery record, refusing one whose fields delivery reads are missing or malformed.

    Every reader of a record's state, target and payload goes through here, so a record that
    lost a field takes its caller's warning path, named by its file, rather than an error.

    :param Path path: Delivery record under ``tracking/<generation>/delivery``.
    :raises OSError: The record cannot be read.
    :raises ValueError: The record is not in the current format, or its state is neither
        ``pending`` nor ``delivered``; the message names the file.
    :return _Delivery: The record and its checked fields.
    """
    record = read_record(path)

    def member(key: str) -> JsonObject:
        """Return the record's ``key`` object, refusing anything else."""
        value = record.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"MLflow delivery record {path} has no {key!r} object.")
        return json_object(value)

    state = record.get("state")
    finish = record.get("finish")
    generation_id = record.get("generation_id")
    if not isinstance(state, str) or state not in {"pending", "delivered"}:
        raise ValueError(f"MLflow delivery record {path} has unknown state {state!r}.")
    if not isinstance(generation_id, str) or not isinstance(finish, str | None):
        raise ValueError(f"MLflow delivery record {path} has a malformed generation or finish.")
    return _Delivery(
        record,
        state == "delivered",
        generation_id,
        member("source"),
        member("binding"),
        member("projection"),
        finish,
    )


def _send(
    experiment: Experiment,
    paths: list[Path],
    environment: Mapping[str, str],
    *,
    worker_generation: str | None,
    tally: DeliveryTally,
    counted: Collection[Path] = (),
    deadline: float | None = None,
) -> None:
    """Send records in order while the delivery is still sending.

    A failure MLflow answered, such as a deleted run, concerns its record alone, so it costs a
    warning and the rest go on. A request MLflow gives no answer to would meet the same tracker
    for every later record, so one such request ends the pass and stops ``tally`` sending,
    leaving the rest pending for the next run or sync-tracking. A failure during a shutdown
    ends the pass too, without a warning, as MLflow was not what failed: on a trial's worker
    thread that is a launch the shutdown refused or a worker it killed, while on the main thread
    the shutdown raises out of the request in flight.

    :param Experiment experiment: Configured target.
    :param list[Path] paths: Delivery records, in sending order.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str | None worker_generation: Generation the upload workers register under, as for
        :func:`_deliver`.
    :param DeliveryTally tally: The delivery this pass belongs to.
    :param Collection[Path] counted: Records whose delivery adds to ``tally.delivered``, counted
        as each is sent so a pass a shutdown interrupts still reports them, defaults to none.
    :param float | None deadline: Absolute run deadline every upload is cut at, defaults to
        none.
    :raises ProcessCleanupUncertainError: An upload worker's cleanup is uncertain.
    """
    for index, path in enumerate(paths):
        if not tally.sending:
            break
        try:
            _deliver(
                experiment,
                path,
                environment,
                worker_generation=worker_generation,
                deadline=deadline,
            )
        except ProcessCleanupUncertainError:
            raise
        except (OSError, ValueError, RuntimeError) as exc:  # every NO_ANSWER error included
            failure = tally.note_failure(experiment, path, exc)
            if failure == "shutdown":
                break
            if failure == "refused again":
                # Warned when first refused in this delivery; the closing line counts it.
                continue
            later = len(paths) - index - 1
            log.warning(
                "MLflow: %s is pending (%s).%s",
                record_name(path),
                str(exc).rstrip("."),
                f" MLflow did not answer, so {plural(later, 'later record')} must wait too."
                if failure == "no answer" and later
                else "",
            )
            if failure == "no answer":
                break
            continue
        tally.refused.discard(path)
        if path in counted:
            tally.delivered += 1


def _records(experiment: Experiment, scope: str | None, *, skip: str | None = None) -> list[Path]:
    """List the delivery records in scope in sending order, each summary run's last.

    A summary run ends last, so a summary run MLflow shows ended had the attempt outcomes and
    winner references it summarizes sent before it. Generations are taken by ID, which says
    nothing of when each ran, so no order among earlier runs is promised; nothing on this path
    reads when a generation started.

    :param Experiment experiment: Configured experiment.
    :param str | None scope: One generation, or ``None`` for every generation.
    :param str | None skip: Generation left out, a live invocation's own.
    :return list[Path]: Records grouped by generation, sorted by generation ID.
    """
    records = tracking_root(experiment).glob(f"{scope or '*'}/{DELIVERY_DIR}/*.json")
    return sorted(
        (path for path in records if path.parent.parent.name != skip),
        key=lambda path: (path.parent.parent.name, path.stem == SUMMARY_RECORD, path.name),
    )


def _pending_records(
    experiment: Experiment, scope: str | None, *, skip: str | None = None
) -> list[Path]:
    """List the delivery records in scope that are still to be sent, in sending order.

    :param Experiment experiment: Configured experiment.
    :param str | None scope: One generation, or ``None`` for every generation.
    :param str | None skip: Generation left out, a live invocation's own.
    :raises OSError: A record cannot be read.
    :raises ValueError: A record is malformed.
    :return list[Path]: Undelivered records, as :func:`_records` orders them.
    """
    return [
        path
        for path in _records(experiment, scope, skip=skip)
        if not _read_delivery(path).delivered
    ]


def _save_intent(
    experiment: Experiment,
    generation_id: str,
    name: str,
    binding: RunBinding,
    projection: Projection,
    *,
    finish: str | None = None,
) -> Path:
    """Save delivery intent before attempting any result upload.

    :param Experiment experiment: Configured target.
    :param str generation_id: Producing generation.
    :param str name: Stable projection identity.
    :param RunBinding binding: Verified target run.
    :param Projection projection: Frozen upload payload.
    :param str | None finish: Authoritative terminal status for a managed run.
    :return Path: The delivery record, unchanged when it already exists.
    """
    assert experiment.mlflow is not None
    path = delivery_record_path(experiment, generation_id, name)
    if not path.exists():
        record: DeliveryRecord = {
            "schema_version": 1,
            "generation_id": generation_id,
            "source": experiment.mlflow.evidence_identity(),
            "binding": binding,
            "projection": projection,
            "finish": finish,
            "state": "pending",
        }
        write_record(path, record)
    return path


def _accepted_projection(
    trial: FrozenTrial, inherits: Mapping[str, str], experiment: Experiment
) -> Projection:
    """Freeze an accepted trial projection from its persisted values and attributes.

    ``phasesweep.feasible`` is the bounds' verdict when the projection is written; PhaseSweep
    never reads it back, and every PhaseSweep read judges the recorded values afresh.

    :param FrozenTrial trial: COMPLETE trial.
    :param Mapping[str, str] inherits: Native run of each parent winner the attempt inherited.
    :param Experiment experiment: Config whose bounds give ``phasesweep.feasible``.
    :return Projection: The attempt's accepted projection.
    """
    from phasesweep.engine.bounds import trial_bound_violations
    from phasesweep.engine.selection import _observed_values
    from phasesweep.engine.state import CONSTRAINT_PREFIX, DURATION_ATTR, OVERRIDES_ATTR

    overrides = json_object(json.loads(trial.user_attrs[OVERRIDES_ATTR]))
    assert trial.value is not None
    constraints = {
        key.removeprefix(CONSTRAINT_PREFIX): json_number(trial.user_attrs, key)
        for key in sorted(trial.user_attrs)
        if key.startswith(CONSTRAINT_PREFIX) and trial.user_attrs[key] is not None
    }
    observed = _observed_values(trial, subject=f"Trial {trial.number}")
    return _projection(
        metrics={
            "phasesweep.objective": float(trial.value),
            "phasesweep.feasible": float(not trial_bound_violations(trial, experiment)),
            "phasesweep.duration_seconds": json_number(trial.user_attrs, DURATION_ATTR),
            **{f"phasesweep.constraint.{k}": v for k, v in constraints.items()},
            **{f"phasesweep.observed.{k}": v for k, v in observed.items()},
        },
        params={
            f"phasesweep.override.{k}": json.dumps(v, sort_keys=True) for k, v in overrides.items()
        },
        tags={
            "phasesweep.trial_number": str(trial.number),
            **{f"phasesweep.inherits.{name}": run_id for name, run_id in inherits.items()},
        },
        summary=json_object(
            {
                "trial_number": trial.number,
                "metric": trial.value,
                "constraints": constraints,
                "observed": observed,
                "effective_overrides": overrides,
            }
        ),
    )


def _unaccepted_projection(binding: RunBinding, status: str) -> Projection:
    """Freeze the end of a managed attempt run that produced no accepted result.

    :param RunBinding binding: The attempt's managed run binding.
    :param str status: Terminal MLflow status from the managed attempt outcome.
    :return Projection: The attempt's outcome projection.
    """
    return _projection(
        metrics={},
        params={},
        tags={"phasesweep.outcome": status},
        summary={
            "phase": binding["phase"],
            "attempt_id": binding["attempt_id"],
            "status": status,
        },
    )


def publish_trial(
    experiment: Experiment,
    generation_id: str,
    trial: FrozenTrial,
    environment: Mapping[str, str],
) -> None:
    """Mirror a terminal trial after confirmed process cleanup.

    COMPLETE trials publish accepted evidence to their bound native run.
    Failed or interrupted trials finish only managed runs; unbound attempts
    and trainer-owned failed runs are left alone. An upload MLflow does not answer stays
    pending for the run's closeout, which retries its own records.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Producing generation.
    :param FrozenTrial trial: Locally terminal trial with confirmed cleanup.
    :param Mapping[str, str] environment: Composed authentication environment.
    :raises ProcessCleanupUncertainError: The upload worker's cleanup is uncertain.
    """
    from optuna.trial import TrialState

    from phasesweep.engine.paths import _experiment_dir
    from phasesweep.engine.state import ATTEMPT_ID_ATTR

    attempt_id: object = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if not isinstance(attempt_id, str) or not attempt_id:
        if trial.state == TrialState.COMPLETE:
            raise ValueError(f"Completed trial {trial.number} records no attempt id.")
        return
    directory = attempt_tracking_dir(_experiment_dir(experiment), generation_id, attempt_id)
    if trial.state != TrialState.COMPLETE and not (directory / BINDING_FILE).is_file():
        return
    binding = read_binding(directory)
    paths = _save_attempt_intents(
        experiment,
        generation_id,
        [(directory, binding)],
        {binding["attempt_id"]: _AttemptTrial(trial, cleanup_confirmed=True)},
        ended=None,
    )
    _send(experiment, paths, environment, worker_generation=None, tally=DeliveryTally())


def _unrecorded_attempts(
    experiment: Experiment, generation_id: str
) -> list[tuple[Path, RunBinding]]:
    """Find a generation's bound attempts that have no delivery record yet.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Producing generation.
    :return list[tuple[Path, RunBinding]]: Tracking directory and binding of each such attempt.
    """
    attempts: list[tuple[Path, RunBinding]] = []
    for path in attempt_tracking_records(experiment, BINDING_FILE, generation_id):
        binding = read_binding(path.parent)
        if binding["generation_id"] != generation_id or binding["attempt_id"] != path.parent.name:
            raise ValueError(f"MLflow binding {path} belongs to a different generation or attempt.")
        if not delivery_record_path(
            experiment, generation_id, f"attempt-{binding['attempt_id']}"
        ).exists():
            attempts.append((path.parent, binding))
    return attempts


class _AttemptTrial(NamedTuple):
    """An attempt's trial and positive cleanup confirmation from its controller or study."""

    trial: FrozenTrial
    cleanup_confirmed: bool


def _attempt_trials(ledger: ValidatedLedger, phases: Iterable[str]) -> dict[str, _AttemptTrial]:
    """Index readable study trials by attempt ID with study-local cleanup confirmation.

    An attempt's delivery projects the row that launched it. Its reread copies can have
    different outcomes and do not own tracker delivery. Unreadable studies leave their
    attempts unresolved.

    :param ValidatedLedger ledger: Ledger the trials were recorded in.
    :param Iterable[str] phases: Phase names whose studies are read.
    :return dict[str, _AttemptTrial]: Trials that carry an attempt id.
    """
    from phasesweep.engine.attempts import _cleanup_recovered_trial_numbers
    from phasesweep.engine.ledger import read_every_phase_study
    from phasesweep.engine.trial_dirs import producing_attempts

    trials: dict[str, _AttemptTrial] = {}
    for phase in sorted(set(phases)):
        try:
            studies = read_every_phase_study(ledger, [phase])[phase]
            owners = producing_attempts(studies)
        except (StudyStorageUnavailableError, StudySchemaMismatchError) as exc:
            log.warning(
                "MLflow attempt outcomes of phase %s are unresolved: its study is unreadable (%s).",
                phase,
                str(exc).rstrip("."),
            )
            continue
        recovered = {study.study_name: _cleanup_recovered_trial_numbers(study) for study in studies}
        for attempt_id, owner in owners.items():
            trial = owner.trial
            trials[attempt_id] = _AttemptTrial(
                trial, trial.number in recovered[owner.study.study_name]
            )
    return trials


def _save_attempt_intents(
    experiment: Experiment,
    generation_id: str,
    attempts: list[tuple[Path, RunBinding]],
    trials: Mapping[str, _AttemptTrial],
    *,
    ended: str | None,
) -> list[Path]:
    """Save unrecorded attempt projections according to their durable Optuna outcomes.

    COMPLETE trials receive accepted projections. Unaccepted managed runs finish
    only after cleanup is confirmed by the trial, live controller, or owning
    study's recovery ledger. Unreadable trials and uncertain cleanup remain unresolved, as
    do RUNNING trials of a persistent study, whose outcome recovery has yet to record, and
    attempts whose trial a persistent ledger does not hold.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Producing generation.
    :param list[tuple[Path, RunBinding]] attempts: From :func:`_unrecorded_attempts`.
    :param Mapping[str, _AttemptTrial] trials: Ledger trials by attempt id, each with
        its controller's cleanup confirmation or its study's recovery.
    :param str | None ended: Status for every unaccepted managed run, terminal or, without
        persistent storage, still RUNNING or no longer held by the ledger, once the invocation
        that owned it ended with confirmed cleanup or was killed outright, or ``None`` to
        finish only trials Optuna failed with confirmed cleanup.
    :return list[Path]: The records saved, in attempt order.
    """
    from optuna.trial import TrialState

    from phasesweep.engine.attempts import _trial_requires_cleanup_recovery
    from phasesweep.engine.paths import _attempts_dir
    from phasesweep.engine.state import _trial_interrupted

    paths: list[Path] = []
    for directory, binding in attempts:
        found = trials.get(binding["attempt_id"])
        name = f"attempt-{binding['attempt_id']}"
        if found is None:
            entry = _attempts_dir(experiment) / f"{binding['attempt_id']}.json"
            if (
                ended is not None
                and not experiment.persistent
                and binding["run_mode"] == "managed"
                and not entry.exists()
            ):
                # An in-memory ledger ended with the invocation that held the trial, so no
                # outcome but that invocation's status can ever reach the run. A registry
                # entry means the attempt's cleanup is unconfirmed and its trainer may still
                # log to the run, so it waits for the run that retires the entry.
                projection = _unaccepted_projection(binding, ended)
                paths.append(
                    _save_intent(experiment, generation_id, name, binding, projection, finish=ended)
                )
            continue
        trial = found.trial
        if trial.state == TrialState.COMPLETE:
            projection = _accepted_projection(trial, read_inherits(directory), experiment)
            paths.append(_save_intent(experiment, generation_id, name, binding, projection))
            continue
        if binding["run_mode"] != "managed":
            continue
        settled = trial.state.is_finished() and (
            found.cleanup_confirmed or not _trial_requires_cleanup_recovery(trial)
        )
        if settled and _trial_interrupted(trial):
            status = "KILLED"
        elif ended is not None and (trial.state.is_finished() or not experiment.persistent):
            # A persistent study's RUNNING trial has no retained outcome yet: recovery may still
            # accept its completed training, so its run stays open rather than reading as a
            # verdict. An in-memory study is gone with the invocation, so nothing can settle it.
            status = ended
        elif settled:
            status = "FAILED"
        else:
            continue
        projection = _unaccepted_projection(binding, status)
        paths.append(
            _save_intent(experiment, generation_id, name, binding, projection, finish=status)
        )
    return paths


def _save_generation_intents(experiment: Experiment, generation_id: str) -> list[Path]:
    """Build winner projections from a generation's validated committed artifacts.

    Its recorded phase order and lineage apply even if the current config differs.
    Build every projection before saving records; preserve records already present.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Published generation that has a summary run.
    :raises PublicationCommitError: The generation summary cannot be read back.
    :raises PublicationAccessError: A manifest artifact cannot be read as the current user.
    :raises PublicationIntegrityError: A manifest-listed artifact fails validation.
    :raises OSError: A delivery record cannot be written.
    :return list[Path]: Winner records in phase order, then the summary run's.
    """
    from phasesweep.engine.generation import _validate_generation_publishable

    parent_path = summary_binding_path(experiment, generation_id)
    parent = read_binding(parent_path.parent, parent_path.name)
    publication, _ = _validate_generation_publishable(experiment, generation_id)
    bindings = {
        name: validate_binding(winner.payload["objective_provenance"]["remote_capture"]["binding"])
        for name, winner in publication.winners.items()
    }
    references = {
        item["name"]: json_object(
            {
                "run_id": bindings[item["name"]]["run_id"],
                "metric": item["metric"],
                "effective_overrides": item["effective_overrides"],
            }
        )
        for item in publication.summary["phases"]
    }
    winners = {
        phase.name: _projection(
            metrics={},
            params={},
            tags={
                "phasesweep.winner": "true",
                f"phasesweep.winner_generation.{generation_id}": phase.name,
                **{
                    f"phasesweep.inherits.{name}": bindings[name]["run_id"]
                    for name in publication.inherits[phase.name]
                },
            },
            summary={"phase": phase.name, **references[phase.name]},
        )
        for phase in publication.phase_plan
    }
    paths = [
        _save_intent(experiment, generation_id, f"winner-{name}", bindings[name], projection)
        for name, projection in winners.items()
    ]
    paths.append(
        _save_intent(
            experiment,
            generation_id,
            SUMMARY_RECORD,
            parent,
            _projection(
                metrics={},
                params={},
                tags={f"phasesweep.winner.{name}": bindings[name]["run_id"] for name in references},
                summary=json_object({"generation_id": generation_id, "phases": references}),
            ),
            finish="FINISHED",
        )
    )
    return paths


def _save_published(experiment: Experiment, generation_id: str) -> list[Path]:
    """Save a committed generation's winner references to its MLflow runs, sending nothing.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Committed generation.
    :raises PublicationIntegrityError: The committed artifacts no longer validate.
    :raises OSError: A delivery record cannot be read or written.
    :return list[Path]: Every record of the generation still pending, attempt outcomes an
        earlier upload left behind included; none when the generation bound no summary run, as
        a no-work replay sends nothing of its own.
    """
    if not summary_binding_path(experiment, generation_id).is_file():
        return []
    _save_generation_intents(experiment, generation_id)
    return _pending_records(experiment, generation_id)


def _generation_publication_record(
    experiment: Experiment, generation_id: str
) -> _RecordedGeneration:
    """Resolve the generation's state using the pointer first and its lifecycle record otherwise.

    :param Experiment experiment: Experiment that owns the generation.
    :param str generation_id: Generation to check.
    :return _RecordedGeneration: The state and error class, or a reason they could not be read.
    """
    from phasesweep.engine.generation import _recorded_generation, _RecordedGeneration
    from phasesweep.engine.paths import _generation_record_path, _last_successful_generation_path
    from phasesweep.engine.publication_validation import _read_pointer_target

    # The pointer commits first; the lifecycle record after it is best-effort.
    target = _read_pointer_target(
        _last_successful_generation_path(experiment), experiment_name=experiment.experiment
    )
    if target is not None and target.generation_id == generation_id:
        return _RecordedGeneration("published", None)
    return _recorded_generation(_generation_record_path(experiment, generation_id))


def _published(experiment: Experiment, generation_id: str) -> bool:
    """Report whether the lifecycle record or last-success pointer names a generation published.

    :param Experiment experiment: Experiment that owns the generation.
    :param str generation_id: Generation to check.
    :return bool: Whether the generation is published.
    """
    return _generation_publication_record(experiment, generation_id).state == "published"


def _failure_status(error_class: str | None) -> str:
    """Name the MLflow status that ends a failed invocation's summary run.

    Shutdown and keyboard interruption end it KILLED and any other failure FAILED. The class is
    matched by name, which is all a generation's lifecycle record keeps of it.

    :param str | None error_class: Class name of the invocation's failure, if it recorded one.
    :return str: ``KILLED`` or ``FAILED``.
    """
    return (
        "KILLED"
        if error_class in {PhaseSweepShutdown.__name__, KeyboardInterrupt.__name__}
        else "FAILED"
    )


def _save_failure_summary(
    experiment: Experiment, generation_id: str, status: str, error_class: str | None
) -> Path:
    """Save the record that ends an unpublished generation's summary run with ``status``.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Generation whose summary run is bound.
    :param str status: ``FAILED`` or ``KILLED``.
    :param str | None error_class: Class name of the invocation's failure, or ``None`` when it
        recorded none.
    :return Path: The summary run's delivery record, unchanged when it already exists.
    """
    parent_path = summary_binding_path(experiment, generation_id)
    return _save_intent(
        experiment,
        generation_id,
        SUMMARY_RECORD,
        read_binding(parent_path.parent, parent_path.name),
        _projection(
            metrics={},
            params={},
            tags={"phasesweep.outcome": status},
            summary={"generation_id": generation_id, "status": status, "error_class": error_class},
        ),
        finish=status,
    )


def _save_failed(
    experiment: Experiment,
    generation_id: str,
    error: BaseException,
    ledger: ValidatedLedger,
) -> list[Path]:
    """Save the records that finish a failed invocation's owned runs, sending nothing.

    Accepted trials keep accepted projections; other finished managed attempts, and without
    persistent storage RUNNING ones too, finish with the invocation's status. Committed
    generations remain untouched.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Failed invocation identity, whose cleanup is confirmed.
    :param BaseException error: Authoritative engine failure, without its private message.
    :param ValidatedLedger ledger: The invocation's ledger, holding its trials.
    :return list[Path]: Every record of the generation still pending, attempt outcomes an
        earlier upload left behind included; none when it bound no summary run or is published.
    """
    if not summary_binding_path(experiment, generation_id).is_file() or _published(
        experiment, generation_id
    ):
        return []
    error_class = type(error).__name__
    status = _failure_status(error_class)
    attempts = _unrecorded_attempts(experiment, generation_id)
    trials = _attempt_trials(ledger, {binding["phase"] for _, binding in attempts})
    _save_attempt_intents(experiment, generation_id, attempts, trials, ended=status)
    _save_failure_summary(experiment, generation_id, status, error_class)
    return _pending_records(experiment, generation_id)


def _rebuild_attempt_intents(
    experiment: Experiment,
    ledger: ValidatedLedger,
    generation_ids: Iterable[str],
    *,
    unresolved: dict[str, TrackingSyncRefusedError] | None = None,
) -> None:
    """Save the attempt records an interrupted or failed invocation never saved.

    Without persistent storage the ledger that held an ended generation's trials is gone, so
    each managed attempt of an unpublished one ends with the status that ends its summary run.

    :param Experiment experiment: Configured experiment.
    :param ValidatedLedger ledger: The experiment's validated ledger, holding the attempts' trials.
    :param Iterable[str] generation_ids: Ended generations whose attempts are checked; never a
        live invocation's.
    :param dict[str, TrackingSyncRefusedError] | None unresolved: Collect per-generation lifecycle
        errors when rebuilding across generations. Raise on the first such error by default.
    """
    unrecorded = {
        generation_id: attempts
        for generation_id in generation_ids
        if (attempts := _unrecorded_attempts(experiment, generation_id))
    }
    if not unrecorded:
        return
    phases = {binding["phase"] for attempts in unrecorded.values() for _, binding in attempts}
    trials = _attempt_trials(ledger, phases)
    for generation_id, attempts in unrecorded.items():
        # A persistent ledger still holds the trials, and a published generation is not ended
        # by its lifecycle record.
        ended = (
            None
            if experiment.persistent
            else _ended_status(experiment, generation_id, unresolved=unresolved)
        )
        _save_attempt_intents(
            experiment, generation_id, attempts, trials, ended=None if ended is None else ended[0]
        )


def _ended_status(
    experiment: Experiment,
    generation_id: str,
    *,
    unresolved: dict[str, TrackingSyncRefusedError] | None = None,
) -> tuple[str, str | None] | None:
    """Name the status and failure that end a generation's runs, if unpublished.

    It is the status the failure in the generation's lifecycle record maps to, or KILLED
    when no terminal record was written, as when the invocation was killed outright.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Ended generation.
    :param dict[str, TrackingSyncRefusedError] | None unresolved: Collect unreadable or malformed
        records so the caller can finish other generations. Each broken record is read once.
    :raises TrackingSyncRefusedError: The record cannot supply a status and no collector is set.
    :return tuple[str, str | None] | None: ``FAILED`` or ``KILLED`` and the failure class when
        recorded. ``None`` for a published or unresolved generation.
    """
    from phasesweep.engine.paths import _generation_record_path

    if unresolved is not None and generation_id in unresolved:
        return None
    recorded = _generation_publication_record(experiment, generation_id)
    if recorded.state == "published":
        return None
    if recorded.problem is not None:
        path = _generation_record_path(experiment, generation_id)
        error = TrackingSyncRefusedError(
            f"MLflow terminal outcome is unresolved: generation record {path} "
            f"is {recorded.problem}.",
            action=OperatorAction.RESTORE_TREE,
            remediation=f"Restore the generation record, then run {SYNC_TRACKING_STEP}.",
        )
        if unresolved is None:
            raise error
        unresolved[generation_id] = error
        return None
    if recorded.state in {"failed", "publication_failed"}:
        return _failure_status(recorded.error_class), recorded.error_class
    return "KILLED", None


def _save_ended_summaries(
    experiment: Experiment,
    generation_ids: Iterable[str],
    *,
    unresolved: dict[str, TrackingSyncRefusedError] | None = None,
) -> list[Path]:
    """Save the summary-run record each ended, unpublished generation's invocation never saved.

    An invocation killed outright saves none, so its summary run would stay open. It ends with
    the status :func:`_ended_status` names. A published generation's record is built from its
    committed artifacts instead, and a record already saved is kept as it is. Callers hold the
    experiment lock and pass no live invocation's generation: a run never passes its own, and
    recover-run passes the run whose runner it proved dead.

    :param Experiment experiment: Configured experiment.
    :param Iterable[str] generation_ids: Ended generations to check.
    :param dict[str, TrackingSyncRefusedError] | None unresolved: Collect per-generation lifecycle
        errors when rebuilding across generations. Raise on the first such error by default.
    :raises OSError: A record or binding cannot be written or the binding cannot be read.
    :raises TrackingSyncRefusedError: The generation's lifecycle record is unreadable or malformed.
    :return list[Path]: The summary-run record of each of those generations that is unpublished
        and has a bound summary run, whether saved here or before.
    """
    paths = []
    for generation_id in sorted(set(generation_ids)):
        if not summary_binding_path(experiment, generation_id).is_file():
            continue
        path = delivery_record_path(experiment, generation_id, SUMMARY_RECORD)
        if path.exists():
            if _published(experiment, generation_id):
                continue
        else:
            ended = _ended_status(experiment, generation_id, unresolved=unresolved)
            if ended is None:
                continue
            _save_failure_summary(experiment, generation_id, *ended)
        paths.append(path)
    return paths


def _target_recorded(experiment: Experiment) -> bool:
    """Report whether the configured MLflow target is the one the tree's bound runs live in.

    Searched runs and rebuilt records take the configured target, so a config naming another
    one would stamp an earlier sweep's records with a target its runs are not in. A tree
    without ``source.json`` has bound no run.

    :param Experiment experiment: MLflow-backed experiment.
    :raises OSError: ``source.json`` cannot be read.
    :raises ValueError: ``source.json`` does not use the current format.
    :return bool: Whether ``source.json`` is absent or records the configured target.
    """
    assert experiment.mlflow is not None
    source = tracking_root(experiment) / SOURCE_FILE
    if not source.exists():
        return True
    recorded = read_record(source, schema_version=SOURCE_SCHEMA_VERSION).get("source")
    if not isinstance(recorded, dict):
        raise ValueError(f"MLflow tracking record {source} has no 'source' object.")
    return recorded == experiment.mlflow.evidence_identity()


def _bound_generations(experiment: Experiment, scope: str | None, skip: str | None) -> list[str]:
    """Name the generations in scope whose summary run is bound, sorted by generation ID.

    :param Experiment experiment: Configured experiment.
    :param str | None scope: One generation, or ``None`` for every generation.
    :param str | None skip: Generation left out, a live invocation's own.
    :return list[str]: Generation IDs, without ``skip``.
    """
    parents = sorted(tracking_root(experiment).glob(f"{scope or '*'}/{GENERATION_BINDING_FILE}"))
    return [parent.parent.name for parent in parents if parent.parent.name != skip]


def _stage_outstanding(
    experiment: Experiment,
    ledger: ValidatedLedger,
    generation_ids: list[str],
    *,
    unresolved: dict[str, TrackingSyncRefusedError],
) -> None:
    """Save every record ended generations still owe their MLflow runs, sending nothing.

    Missing attempt records are rebuilt from the ledger, the summary-run record of each ended,
    unpublished generation from its lifecycle record, and a published generation's winner and
    summary records from its committed artifacts. Records already saved are kept as they are.

    :param Experiment experiment: Configured experiment whose target the tree recorded.
    :param ValidatedLedger ledger: The experiment's validated ledger, holding the attempts' trials.
    :param list[str] generation_ids: Ended generations with a bound summary run.
    :param dict[str, TrackingSyncRefusedError] unresolved: Collects each generation whose
        lifecycle record is unreadable or malformed; its outcomes stay unsaved, and a generation
        already in it is not read again.
    :raises OSError: A record cannot be read or written.
    :raises PublicationIntegrityError: A published generation's artifacts no longer validate.
    """
    _rebuild_attempt_intents(experiment, ledger, generation_ids, unresolved=unresolved)
    _save_ended_summaries(experiment, generation_ids, unresolved=unresolved)
    for generation_id in generation_ids:
        if (
            generation_id in unresolved
            or delivery_record_path(experiment, generation_id, SUMMARY_RECORD).exists()
            or not _published(experiment, generation_id)
        ):
            continue
        _save_generation_intents(experiment, generation_id)


def _deliver_outstanding(
    experiment: Experiment,
    ledger: ValidatedLedger,
    *,
    scope: str | None,
    skip: str | None,
    worker_generation: str | None,
    tally: DeliveryTally,
    errors: list[TrackingSyncRefusedError],
    resolve: bool = False,
    own: list[Path] | None = None,
    deadline: float | None = None,
) -> None:
    """Deliver what every ended generation in scope still owes its MLflow runs.

    This is sync-tracking's work, shared with a run's two passes, after its first target check
    and at closeout, and with recover-run. The caller holds the
    experiment lock, so every generation in scope but ``skip`` has ended. Missing records are
    staged first, locally, so nothing is requested when nothing is pending or ``tally`` has
    stopped sending. Then the target is resolved if asked, pending creations are searched, so a
    run a timed-out creation made late is bound and its records rebuilt, and every pending record
    is sent, a closing run's own after the ended generations' ones, so the outcomes an earlier
    run left go first and one MLflow refuses holds back none of the run's own. The first request
    MLflow gives no answer to ends the pass, search or upload, so an outage costs the pass one
    request's wait; what is left stays pending for the next run or sync-tracking.

    :param Experiment experiment: Configured MLflow-backed experiment.
    :param ValidatedLedger ledger: The experiment's validated ledger, holding the attempts' trials.
    :param str | None scope: One generation, or ``None`` for every generation.
    :param str | None skip: Generation left out, a live invocation's own, whose records arrive
        as ``own``.
    :param str | None worker_generation: Generation the search and upload workers register
        under, or ``None`` for each record's own.
    :param DeliveryTally tally: The delivery this pass belongs to; records of ended generations
        delivered here are added to its count.
    :param list[TrackingSyncRefusedError] errors: Collects local creation record failures and
        unreadable lifecycle records, whose outcomes stay pending.
    :param bool resolve: Resolve the target under ``worker_generation`` before the first request,
        for a caller whose process made no target check, defaults to False.
    :param list[Path] | None own: A closing run's own records, already saved, sent last.
    :param float | None deadline: Absolute run deadline every request is cut at, defaults to
        none.
    :raises TrackingSyncRefusedError: The configured target, or a saved record's target or
        generation, is not the one the tree recorded; nothing is sent.
    :raises TrackingTargetUnavailableError: The target did not resolve; nothing else is sent.
    :raises ProcessCleanupUncertainError: A search or upload worker's cleanup is uncertain.
    :raises PublicationIntegrityError: A published generation's artifacts no longer validate.
    """
    from phasesweep.engine.trial import _environment_identity

    assert experiment.mlflow is not None
    different = "Tracking delivery belongs to a different target or generation."
    if not _target_recorded(experiment):
        raise TrackingSyncRefusedError(different, remediation=ORIGINAL_TARGET_STEP)
    unresolved: dict[str, TrackingSyncRefusedError] = {}
    try:
        _stage_outstanding(
            experiment, ledger, _bound_generations(experiment, scope, skip), unresolved=unresolved
        )
        for path in _records(experiment, scope, skip=skip):
            delivery = _read_delivery(path)
            if (
                delivery.source != experiment.mlflow.evidence_identity()
                or delivery.generation_id != path.parent.parent.name
            ):
                raise TrackingSyncRefusedError(different, remediation=ORIGINAL_TARGET_STEP)
        creations = pending_creation_records(experiment, scope, skip=skip)
        own_pending = [path for path in own or [] if not _read_delivery(path).delivered]
        pending = _pending_records(experiment, scope, skip=skip)
        if not tally.sending or not (creations or pending or own_pending):
            return
        environment = _environment_identity(experiment, experiment.phases[0].name).values
        if resolve:
            assert worker_generation is not None
            require_tracking_target(
                experiment,
                generation_id=worker_generation,
                environment=environment,
                deadline=deadline,
                resolve_only=True,
            )
        # Report lines, so the console shows what the seconds before the next lines are for,
        # printed once the target answered, as only then is something sent. A search can bind a
        # run whose record is then rebuilt, so the records are counted for their line only once
        # the searches are done.
        if creations:
            report.info(
                ReportText(
                    f"MLflow: checking {count_text(0, len(creations))} left by earlier runs."
                )
            )
        else:
            _report_sending(pending, own_pending)
        if creations:
            resolve_pending_creations(
                experiment,
                environment,
                scope,
                skip=skip,
                worker_generation=worker_generation,
                errors=errors,
                tally=tally,
                deadline=deadline,
            )
            # A run a search bound gets its records now rather than on the next pass.
            _stage_outstanding(
                experiment,
                ledger,
                _bound_generations(experiment, scope, skip),
                unresolved=unresolved,
            )
            pending = _pending_records(experiment, scope, skip=skip)
            if tally.sending and (pending or own_pending):
                _report_sending(pending, own_pending)
        _send(
            experiment,
            pending + own_pending,
            environment,
            worker_generation=worker_generation,
            tally=tally,
            counted=set(pending),
            deadline=deadline,
        )
    finally:
        errors.extend(unresolved.values())


def _report_sending(pending: list[Path], own: list[Path]) -> None:
    """Log the report line that says how many records a pass is about to send.

    :param list[Path] pending: Ended generations' records about to be sent.
    :param list[Path] own: A closing run's own records about to be sent after them.
    """
    total = len(pending) + len(own)
    if not pending:
        from_earlier = ""
    elif not own:
        from_earlier = " left pending by earlier runs"
    else:
        from_earlier = f", {len(pending)} of them left pending by earlier runs"
    report.info(ReportText(f"MLflow: sending {count_text(total, 0)}{from_earlier}."))


def _owed(experiment: Experiment, skip: str | None) -> bool:
    """Report whether any generation but ``skip`` has something left to save or send.

    That is a pending record or creation, or a bound summary run with no delivered record, as
    an invocation killed outright leaves.

    :param Experiment experiment: Configured experiment.
    :param str | None skip: Generation left out, a live invocation's own.
    :raises OSError: A record cannot be read.
    :raises ValueError: A record does not use the current format.
    :return bool: Whether an ended generation still owes MLflow anything.
    """
    return bool(
        _pending_records(experiment, None, skip=skip)
        or pending_creation_records(experiment, skip=skip)
        or any(
            not delivery_record_path(experiment, generation_id, SUMMARY_RECORD).exists()
            for generation_id in _bound_generations(experiment, None, skip)
        )
    )


def deliver_ended_generations(
    experiment: Experiment,
    ledger: ValidatedLedger,
    generation_id: str,
    tally: DeliveryTally,
    *,
    recovering: bool = False,
    own: list[Path] | None = None,
    deadline: float | None = None,
) -> None:
    """Finish what sync-tracking would for every generation that has ended.

    A run with trials to run calls it twice: right after its first target check, before its
    first trial, so MLflow shows what an interrupted earlier run left as soon as the next run
    reaches the tracker, and at its closeout, for what is still pending and its own records; a
    run with none to run, such as a replay, calls it only at closeout. Its scope is every
    generation but a run's own, whose saved records come at closeout as ``own`` and go last; for
    recover-run it is every generation. The core stages first and requests nothing when nothing
    is pending or ``tally`` has stopped sending. Its first request follows a target check: the
    one the calling run made when it bound its own summary run, or else a resolve the core
    makes, which binds none. Workers register under ``generation_id`` because this invocation
    is accountable for their cleanup. A configured target other than the recorded one stages
    and sends nothing, with a warning when an earlier run left something to send. Every failure
    but an uncertain worker cleanup or a shutdown costs a warning and leaves the records for the
    next run or sync-tracking, never the caller's outcome, so a silent tracker costs each call
    at most one request's ``timeout_seconds``.

    :param Experiment experiment: MLflow-backed experiment.
    :param ValidatedLedger ledger: The caller's ledger, holding the trials the records describe.
    :param str generation_id: The calling run's generation, or for ``recover-run`` the run it
        recovers.
    :param DeliveryTally tally: The caller's delivery, which says whether anything is sent and
        counts what is.
    :param bool recovering: The caller is ``recover-run``, so ``generation_id`` is the run it
        recovers rather than its own invocation: that run's records are sent with the rest, and
        the target is resolved first, as no check in this process covered it.
    :param list[Path] | None own: The calling run's own pending records, saved already.
    :param float | None deadline: Absolute run deadline every request is cut at, so a pass
        before a run's first trial never outlasts ``timeout_seconds_per_run``, defaults to none.
    :raises ProcessCleanupUncertainError: A search or upload worker's cleanup is uncertain.
    """
    skip = None if recovering else generation_id
    checked = not recovering and summary_binding_path(experiment, generation_id).is_file()
    errors: list[TrackingSyncRefusedError] = []
    try:
        if not _target_recorded(experiment):
            if _owed(experiment, skip):
                log.warning(
                    "MLflow: records of earlier runs were neither saved nor sent: this config's "
                    "MLflow target is not the one those runs used. %s",
                    ORIGINAL_TARGET_STEP,
                )
            return
        _deliver_outstanding(
            experiment,
            ledger,
            scope=None,
            skip=skip,
            worker_generation=generation_id,
            tally=tally,
            errors=errors,
            resolve=not checked,
            own=own,
            deadline=deadline,
        )
    except ProcessCleanupUncertainError:
        raise
    except TrackingTargetUnavailableError as exc:
        tally.sending = False
        log.warning("%s", exc)
    except (OSError, ValueError, PhaseSweepError) as exc:
        tally.sending = False
        log.warning(
            "MLflow: records were not all sent (%s). Once that is resolved, run %s.",
            str(exc).rstrip("."),
            SYNC_TRACKING_STEP,
        )
    finally:
        for error in errors:
            log.warning("%s %s", error, operator_step(error))


def deliver_closeout(
    experiment: Experiment,
    ledger: ValidatedLedger,
    generation_id: str,
    *,
    error: BaseException | None = None,
    published: bool = False,
    tally: DeliveryTally | None = None,
) -> None:
    """Save a closing run's own MLflow records, send what is pending, and log the closing lines.

    The run's own records are saved before anything is sent, so however the delivery ends they
    are counted; then every ended generation's records still pending go, the run's own last,
    through the core sync-tracking runs. A run that is shutting down, or that just found its
    tracker silent, only saves: a shutdown refuses every worker launch. So does a run that
    failed before its target check, which would otherwise wait on a tracker it never reached
    before reporting its own error. Otherwise the closeout sends whatever the run's earlier
    pass met, as a tracker silent then may answer now. The closing lines are logged however the
    delivery ends, a shutdown during it included, and count the earlier pass too.

    :param Experiment experiment: MLflow-backed experiment.
    :param ValidatedLedger ledger: The run's ledger, holding its trials and those it settled.
    :param str generation_id: The closing run's generation.
    :param BaseException | None error: The run's primary failure, or ``None`` for a run that ended
        normally.
    :param bool published: Whether a run that ended normally published ``generation_id``; a
        replay of an existing publication did not.
    :param DeliveryTally | None tally: The run's delivery, which its pass after its first target
        check already counts into; a fresh one when omitted.
    :raises PublicationIntegrityError: The run's committed artifacts no longer validate.
    :raises OSError: A record of the run's own cannot be read or written.
    :raises ProcessCleanupUncertainError: An upload worker's cleanup is uncertain.
    """
    # A run binds its summary run in its first target check.
    checked = summary_binding_path(experiment, generation_id).is_file()
    tally = DeliveryTally() if tally is None else tally
    tally.sending = error is None or (
        checked
        and not isinstance(
            error, (TrackingTargetUnavailableError, PhaseSweepShutdown, KeyboardInterrupt)
        )
    )
    try:
        if error is not None:
            own = _save_failed(experiment, generation_id, error, ledger)
        else:
            own = _save_published(experiment, generation_id) if published else []
        deliver_ended_generations(experiment, ledger, generation_id, tally, own=own)
    finally:
        report_delivery(experiment, generation_id, tally)


def report_delivery(experiment: Experiment, generation_id: str, tally: DeliveryTally) -> None:
    """Log the lines that close a run's MLflow delivery.

    What was delivered and settled for earlier runs is counted when there is any, then what
    stays pending, the run's own and earlier runs', with the step that sends it and, when MLflow
    refused some, what keeps those pending. Together they account for everything the sending
    lines named: records and run creations alike are either counted here as done or pending.

    :param Experiment experiment: MLflow-backed experiment.
    :param str generation_id: The run's own generation.
    :param DeliveryTally tally: The run's delivery, its pass after its first target check and
        its closeout.
    """
    try:
        counts = delivery_counts(experiment)
        earlier = counts.pending - delivery_counts(experiment, generation_id).pending
    except (OSError, ValueError) as exc:
        log.warning("MLflow: pending records could not be counted (%s).", str(exc).rstrip("."))
        return
    done = delivered_text(tally.delivered, tally.settled)
    if done:
        # A report line, so the console shows that an earlier interruption was made good.
        report.info(ReportText(f"MLflow: {done} left pending by earlier runs."))
    if counts.pending:
        log.warning(
            "MLflow: %s pending%s; results are saved locally. Send %s with %s.%s",
            count_text(counts.records, counts.creations),
            f" ({earlier} from earlier runs)" if earlier else "",
            "it" if counts.pending == 1 else "them",
            SYNC_TRACKING_STEP,
            refused_text(tally.refused),
        )


class DeliveryCounts(NamedTuple):
    """What a scope still owes MLflow, records and creations apart, and what it delivered."""

    records: int
    """Delivery records still to be sent."""
    creations: int
    """Unconfirmed run creations still to be settled by a search."""
    delivered: int
    """Delivery records MLflow accepted."""

    @property
    def pending(self) -> int:
        """Return everything still owed: the pending records and creations together."""
        return self.records + self.creations


def delivery_counts(experiment: Experiment, generation_id: str | None = None) -> DeliveryCounts:
    """Count pending run creations and delivery records without importing the tracking SDK.

    :param Experiment experiment: Local namespace.
    :param str | None generation_id: Optional generation filter.
    :raises OSError: A record cannot be read.
    :raises ValueError: A record is malformed.
    :return DeliveryCounts: Pending records and creations, and delivered records.
    """
    records = delivered = 0
    for path in _records(experiment, generation_id):
        if _read_delivery(path).delivered:
            delivered += 1
        else:
            records += 1
    creations = len(pending_creation_records(experiment, generation_id))
    return DeliveryCounts(records, creations, delivered)


def delivery_status(experiment: Experiment, generation_id: str | None = None) -> dict[str, int]:
    """Count what is pending and delivered, for the status surfaces that report both.

    :param Experiment experiment: Local namespace.
    :param str | None generation_id: Optional generation filter.
    :raises OSError: A record cannot be read.
    :raises ValueError: A record is malformed.
    :return dict[str, int]: Path-free pending and delivered counts.
    """
    counts = delivery_counts(experiment, generation_id)
    return {"pending": counts.pending, "delivered": counts.delivered}


def sync_tracking(
    experiment: Experiment,
    generation_id: str | None = None,
    *,
    tally: DeliveryTally | None = None,
) -> dict[str, int]:
    """Retry pending result projections without selecting or training.

    Check the artifact-root binding and ledger format as every writer does, and confirm
    cleanup of registered tracking workers across generations. Then rebuild missing attempt
    records from Optuna, summary-run records from ended generations' lifecycle records, and
    winner records from committed generation artifacts. Reconcile unresolved creations and
    send what is pending until MLflow stops answering through the core run closeout shares.
    Report local creation or lifecycle record errors after delivering independent records
    in the selected scope.

    :param Experiment experiment: Matching source experiment configuration.
    :param str | None generation_id: Generation whose deliveries should be
        retried, or ``None`` for every generation's retained deliveries.
    :param DeliveryTally | None tally: Receives what the requests met, for a caller that words
        the outcome; a fresh one when omitted.
    :raises ValueError: The generation name is unsafe, or a delivery record is malformed.
    :raises TrackingSyncRefusedError: The config has no ``mlflow`` block, no deliveries
        exist, the target or generation does not match the saved records, or a generation's
        terminal lifecycle record is unreadable or malformed.
    :raises ArtifactRootConflictError: The tree records another owner or another format;
        nothing is written.
    :raises StudySchemaMismatchError: The ledger holds unsupported study state; nothing is written.
    :raises StudyStorageUnavailableError: An unbound tree's ledger cannot be inspected.
    :raises ProcessCleanupUncertainError: A tracking worker's cleanup is uncertain.
    :raises PhaseSweepError: The MLflow SDK is not installed, or a published
        generation's artifacts no longer validate.
    :return dict[str, int]: Remaining pending and delivered counts in that scope.
    """
    from phasesweep.config.common import _validate_safe_name
    from phasesweep.engine.attempts import _preflight_active_attempts, _PreflightCleanupReport
    from phasesweep.engine.ledger import validate_ledger
    from phasesweep.engine.locking import _experiment_lock
    from phasesweep.tracking.mlflow import require_mlflow_sdk

    if generation_id is not None:
        _validate_safe_name("generation", generation_id)
    if experiment.mlflow is None:
        raise TrackingSyncRefusedError(
            "sync-tracking needs an `mlflow` block, and this config has none.",
            remediation=(
                "Pass the config the sweep ran with; `phasesweep explain mlflow` describes the block."
            ),
        )
    require_mlflow_sdk()
    with signal_handler_scope(), _experiment_lock(experiment):
        # Refuse a tree this checkout cannot own before anything here writes to it.
        ledger = validate_ledger(experiment)
        # A later run registers its own workers even when they send an earlier run's records.
        _preflight_active_attempts(
            experiment,
            _PreflightCleanupReport(),
            confirm=True,
            tracking_only=True,
        )
        root = tracking_root(experiment)
        had_creations = bool(pending_creation_records(experiment, generation_id))
        errors: list[TrackingSyncRefusedError] = []
        # The lock rules out a live invocation, so every unpublished generation here has ended.
        _deliver_outstanding(
            experiment,
            ledger,
            scope=generation_id,
            skip=None,
            worker_generation=None,
            tally=DeliveryTally() if tally is None else tally,
            errors=errors,
        )
        if (
            not errors
            and not had_creations
            and not _records(experiment, generation_id)
            and not any(root.glob(f"{generation_id or '*'}/operations"))
        ):
            raise TrackingSyncRefusedError(
                "No tracking deliveries exist for this scope.",
                remediation=(
                    f"Check --generation {generation_id!r} and pass the `--set` overrides the run "
                    "used; an MCP run's ID is its generation ID."
                    if generation_id is not None
                    else "Pass the config and `--set` overrides the sweep ran with; no run of this "
                    "experiment has saved an MLflow record yet."
                ),
            )
        counts = delivery_status(experiment, generation_id)
        if errors:
            detail = "\n".join(str(error) for error in errors)
            raise TrackingSyncRefusedError.rewrap(
                errors[0], f"{detail}\nDelivered records in scope: {counts['delivered']}."
            ) from errors[0]
        return counts
