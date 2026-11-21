"""Retryable MLflow projections kept separate from scientific evidence."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from phasesweep.engine.errors import TrackingSyncRefusedError
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
    json_optional_text,
    json_text,
)
from phasesweep.runtime.shutdown import PhaseSweepShutdown, servicing_shutdown_signal
from phasesweep.runtime.terminal import REPORT_LOGGER, ReportText, plural
from phasesweep.tracking.mlflow import OperationRejectedError
from phasesweep.tracking.records import validate_binding
from phasesweep.tracking.session import (
    BINDING_FILE,
    GENERATION_BINDING_FILE,
    NO_ANSWER,
    DeliveryTally,
    call_tracking,
    pending_creation_records,
    read_binding,
    read_inherits,
    read_record,
    record_name,
    require_tracking_target,
    resolve_pending_creations,
    tracking_root,
    unready_target_text,
    write_record,
)

if TYPE_CHECKING:
    from optuna.trial import FrozenTrial

    from phasesweep.config.models import Experiment
    from phasesweep.engine.generation import _RecordedGeneration
    from phasesweep.engine.ledger import ValidatedLedger
    from phasesweep.tracking.records import DeliveryRecord, Projection, PublishAction, RunBinding

log = logging.getLogger(__name__)


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
    :raises ProcessCleanupUncertainError: The upload worker's cleanup is uncertain.
    :raises OSError: The upload or the record's update failed, MLflow's verdict or no answer.
    :raises ValueError: The upload failed.
    :raises RuntimeError: The upload failed.
    """
    record = read_record(path)
    if record["state"] == "delivered":
        return
    action: PublishAction = {
        "action": "publish",
        "name": path.stem,
        "binding": json_object(record["binding"]),
        "projection": json_object(record["projection"]),
        "finish": json_optional_text(record, "finish"),
    }
    generation_id = json_text(record, "generation_id")
    call_tracking(experiment, worker_generation or generation_id, action, environment)
    # A lost acknowledgement resends the same projection on the next sync.
    write_record(path, {**record, "state": "delivered"})


def _send(
    experiment: Experiment,
    paths: list[Path],
    environment: Mapping[str, str],
    *,
    worker_generation: str | None,
    tally: DeliveryTally,
    counted: Collection[Path] = (),
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
    :raises ProcessCleanupUncertainError: An upload worker's cleanup is uncertain.
    """
    for index, path in enumerate(paths):
        if not tally.sending:
            break
        try:
            _deliver(experiment, path, environment, worker_generation=worker_generation)
        except ProcessCleanupUncertainError:
            raise
        except (OSError, ValueError, RuntimeError) as exc:  # every NO_ANSWER error included
            if servicing_shutdown_signal() is not None:
                # A shutdown refuses every worker launch and kills running workers, so this
                # failure says nothing about MLflow.
                tally.sending = False
                break
            unanswered = isinstance(exc, NO_ANSWER)
            if isinstance(exc, OperationRejectedError) and not unanswered:
                tally.refused.add(path)
            later = len(paths) - index - 1
            log.warning(
                "MLflow: %s is pending (%s).%s",
                record_name(path),
                exc,
                f" MLflow did not answer, so {later} later record(s) wait too."
                if unanswered and later
                else "",
            )
            if unanswered:
                tally.sending = False
                tally.unanswered = unready_target_text(experiment, exc)
                break
            continue
        tally.refused.discard(path)
        if path in counted:
            tally.delivered += 1


def _records(experiment: Experiment, scope: str | None, *, skip: str | None = None) -> list[Path]:
    """List the delivery records in scope in sending order, each summary run's last.

    A summary run ends last, so a summary run MLflow shows ended had the attempt outcomes and
    winner references it summarizes sent before it.

    :param Experiment experiment: Configured experiment.
    :param str | None scope: One generation, or ``None`` for every generation.
    :param str | None skip: Generation left out, a live invocation's own.
    :return list[Path]: Records in generation order.
    """
    records = tracking_root(experiment).glob(f"{scope or '*'}/delivery/*.json")
    return sorted(
        (path for path in records if path.parent.parent.name != skip),
        key=lambda path: (path.parent.parent.name, path.name == "generation.json", path.name),
    )


def _pending_records(
    experiment: Experiment, scope: str | None, *, skip: str | None = None
) -> list[Path]:
    """List the delivery records in scope that are still to be sent, in sending order.

    :param Experiment experiment: Configured experiment.
    :param str | None scope: One generation, or ``None`` for every generation.
    :param str | None skip: Generation left out, a live invocation's own.
    :raises OSError: A record cannot be read.
    :raises ValueError: A record does not use the current format.
    :return list[Path]: Undelivered records, as :func:`_records` orders them.
    """
    return [
        path
        for path in _records(experiment, scope, skip=skip)
        if read_record(path)["state"] != "delivered"
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
    path = tracking_root(experiment) / generation_id / "delivery" / f"{name}.json"
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


def _accepted_projection(trial: FrozenTrial, inherits: Mapping[str, str]) -> Projection:
    """Freeze an accepted trial projection from its persisted values and attributes.

    :param FrozenTrial trial: COMPLETE trial.
    :param Mapping[str, str] inherits: Native run of each parent winner the attempt inherited.
    :return Projection: The attempt's accepted projection.
    """
    from phasesweep.engine.selection import _observed_values
    from phasesweep.engine.state import (
        CONSTRAINT_PREFIX,
        DURATION_ATTR,
        FEASIBLE_ATTR,
        OVERRIDES_ATTR,
    )

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
            "phasesweep.feasible": float(trial.user_attrs[FEASIBLE_ATTR]),
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

    from phasesweep.engine.state import TRIAL_DIR_ATTR

    if trial.state == TrialState.COMPLETE:
        directory = Path(trial.user_attrs[TRIAL_DIR_ATTR])
    else:
        stored = trial.user_attrs.get(TRIAL_DIR_ATTR)
        if not isinstance(stored, str) or not (Path(stored) / BINDING_FILE).is_file():
            return
        directory = Path(stored)
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
    :return list[tuple[Path, RunBinding]]: Trial directory and binding of each such attempt.
    """
    from phasesweep.engine.paths import _experiment_dir

    delivery = tracking_root(experiment) / generation_id / "delivery"
    # Trial directories are named by _trial_dir_for.
    pattern = f"*/trial_*__generation_{generation_id}__attempt_*/{BINDING_FILE}"
    attempts = [
        (path.parent, read_binding(path.parent))
        for path in sorted(_experiment_dir(experiment).glob(pattern))
    ]
    return [
        (directory, binding)
        for directory, binding in attempts
        if not (delivery / f"attempt-{binding['attempt_id']}.json").exists()
    ]


class _AttemptTrial(NamedTuple):
    """An attempt's trial and positive cleanup confirmation from its controller or study."""

    trial: FrozenTrial
    cleanup_confirmed: bool


def _attempt_trials(ledger: ValidatedLedger, phases: Iterable[str]) -> dict[str, _AttemptTrial]:
    """Index readable study trials by attempt ID with study-local cleanup confirmation.

    Unreadable studies contribute no trials, leaving their attempts unresolved.

    :param ValidatedLedger ledger: Ledger the trials were recorded in.
    :param Iterable[str] phases: Phase names whose studies are read.
    :return dict[str, _AttemptTrial]: Trials that carry an attempt id.
    """
    from phasesweep.engine.errors import StudySchemaMismatchError, StudyStorageUnavailableError
    from phasesweep.engine.ledger import read_phase_trials
    from phasesweep.engine.state import ATTEMPT_ID_ATTR

    trials: dict[str, _AttemptTrial] = {}
    for phase in sorted(set(phases)):
        try:
            phase_trials = read_phase_trials(ledger, phase)
        except (StudyStorageUnavailableError, StudySchemaMismatchError) as exc:
            log.warning(
                "MLflow attempt outcomes of phase %s are unresolved: its study is unreadable (%s).",
                phase,
                exc,
            )
            continue
        if phase_trials is None:
            continue
        for trial in phase_trials.trials:
            attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
            if isinstance(attempt_id, str):
                trials[attempt_id] = _AttemptTrial(
                    trial, trial.number in phase_trials.cleanup_recovered
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
            projection = _accepted_projection(trial, read_inherits(directory))
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

    parent = read_binding(tracking_root(experiment) / generation_id, GENERATION_BINDING_FILE)
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
            "generation",
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
    if not (tracking_root(experiment) / generation_id / GENERATION_BINDING_FILE).exists():
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
    from phasesweep.runtime.shutdown import PhaseSweepShutdown

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
    return _save_intent(
        experiment,
        generation_id,
        "generation",
        read_binding(tracking_root(experiment) / generation_id, GENERATION_BINDING_FILE),
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
    parent_path = tracking_root(experiment) / generation_id / GENERATION_BINDING_FILE
    if not parent_path.exists() or _published(experiment, generation_id):
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
            remediation=(
                "Restore the generation record, then run phasesweep sync-tracking with "
                "the same config."
            ),
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
        root = tracking_root(experiment) / generation_id
        if not (root / GENERATION_BINDING_FILE).is_file():
            continue
        path = root / "delivery" / "generation.json"
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
    source = tracking_root(experiment) / "source.json"
    return (
        not source.exists()
        or read_record(source)["source"] == experiment.mlflow.evidence_identity()
    )


def _bound_generations(experiment: Experiment, scope: str | None, skip: str | None) -> list[str]:
    """Name the generations in scope whose summary run is bound, in generation order.

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
    root = tracking_root(experiment)
    for generation_id in generation_ids:
        if (
            generation_id in unresolved
            or (root / generation_id / "delivery" / "generation.json").exists()
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
) -> None:
    """Deliver what every ended generation in scope still owes its MLflow runs.

    This is sync-tracking's work, shared with a run's two passes, after its first target check
    and at closeout, and with recover-run. The caller holds the
    experiment lock, so every generation in scope but ``skip`` has ended. Missing records are
    staged first, locally, so nothing is requested when nothing is pending or ``tally`` has
    stopped sending. Then the target is resolved if asked, pending creations are searched, so a
    run a timed-out creation made late is bound and its records rebuilt, and every pending record
    is sent, a closing run's own after the ended generations' ones, so MLflow never shows a run's
    winner before the outcomes an earlier run left for it. The first request MLflow gives no
    answer to ends the pass, search or upload, so an outage costs one failed request; what is
    left stays pending for the next run or sync-tracking.

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
        raise TrackingSyncRefusedError(different)
    unresolved: dict[str, TrackingSyncRefusedError] = {}
    try:
        _stage_outstanding(
            experiment, ledger, _bound_generations(experiment, scope, skip), unresolved=unresolved
        )
        for path in _records(experiment, scope, skip=skip):
            record = read_record(path)
            if (
                record["source"] != experiment.mlflow.evidence_identity()
                or record["generation_id"] != path.parent.parent.name
            ):
                raise TrackingSyncRefusedError(different)
        creations = _pending_creations(experiment, scope, skip)
        own_pending = [path for path in own or [] if read_record(path)["state"] != "delivered"]
        pending = _pending_records(experiment, scope, skip=skip)
        if not tally.sending or not (creations or pending or own_pending):
            return
        # Report lines, so the console shows what the seconds before the next lines are for. A
        # search can bind a run whose record is then rebuilt, so the records are counted for
        # their line only once the searches are done.
        if creations:
            logging.getLogger(REPORT_LOGGER).info(
                ReportText(
                    f"MLflow: checking {count_text(0, len(creations))} left by earlier runs."
                )
            )
        else:
            _report_sending(pending, own_pending)
        environment = _environment_identity(experiment, experiment.phases[0].name).values
        if resolve:
            assert worker_generation is not None
            require_tracking_target(
                experiment,
                generation_id=worker_generation,
                environment=environment,
                deadline=None,
                resolve_only=True,
            )
        if creations:
            resolve_pending_creations(
                experiment,
                environment,
                scope,
                skip=skip,
                worker_generation=worker_generation,
                errors=errors,
                tally=tally,
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
    logging.getLogger(REPORT_LOGGER).info(
        ReportText(f"MLflow: sending {count_text(total, 0)}{from_earlier}.")
    )


def _pending_creations(experiment: Experiment, scope: str | None, skip: str | None) -> list[Path]:
    """List the pending creation intents in scope, but ``skip``'s.

    :param Experiment experiment: Configured experiment.
    :param str | None scope: One generation, or ``None`` for every generation.
    :param str | None skip: Generation left out, a live invocation's own.
    :return list[Path]: Pending summary and attempt creation records.
    """
    skipped = set() if skip is None else set(pending_creation_records(experiment, skip))
    return [path for path in pending_creation_records(experiment, scope) if path not in skipped]


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
    root = tracking_root(experiment)
    return bool(
        _pending_records(experiment, None, skip=skip)
        or _pending_creations(experiment, None, skip)
        or any(
            not (root / generation_id / "delivery" / "generation.json").exists()
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
) -> None:
    """Finish what sync-tracking would for every generation that has ended.

    A run calls it twice: right after its first target check, before its first trial, so MLflow
    shows what an interrupted earlier run left as soon as the next run reaches the tracker, and
    at its closeout, for what is still pending and its own records. Its scope is every
    generation but a run's own, whose saved records come at closeout as ``own`` and go last; for
    recover-run it is every generation. The core stages first and requests nothing when nothing
    is pending or ``tally`` has stopped sending. Its first request follows a target check: the
    one the calling run made when it bound its own summary run, or else a resolve the core
    makes, which binds none. Workers register under ``generation_id`` because this invocation
    is accountable for their cleanup. A configured target other than the recorded one stages
    and sends nothing, with a warning when an earlier run left something to send. Every failure
    but an uncertain worker cleanup or a shutdown costs a warning and leaves the records for the
    next run or sync-tracking, never the caller's outcome, so a silent tracker costs the caller
    one request's ``timeout_seconds``.

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
    :raises ProcessCleanupUncertainError: A search or upload worker's cleanup is uncertain.
    """
    from phasesweep.engine.errors import TrackingTargetUnavailableError

    skip = None if recovering else generation_id
    checked = (
        not recovering
        and (tracking_root(experiment) / generation_id / GENERATION_BINDING_FILE).is_file()
    )
    errors: list[TrackingSyncRefusedError] = []
    try:
        if not _target_recorded(experiment):
            if _owed(experiment, skip):
                log.warning(
                    "MLflow: records of earlier runs were neither saved nor sent: this config's "
                    "MLflow target is not the one those runs used. A run or `phasesweep "
                    "sync-tracking CONFIG` with the config the sweep ran with sends what they "
                    "left pending."
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
        )
    except ProcessCleanupUncertainError:
        raise
    except TrackingTargetUnavailableError as exc:
        tally.sending = False
        log.warning("%s", exc)
    except (OSError, ValueError, PhaseSweepError) as exc:
        tally.sending = False
        log.warning(
            "MLflow: records were not all sent (%s). The next run or `phasesweep sync-tracking "
            "CONFIG` rebuilds and sends them.",
            exc,
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
    from phasesweep.engine.errors import TrackingTargetUnavailableError

    # A run binds its summary run in its first target check.
    checked = (tracking_root(experiment) / generation_id / GENERATION_BINDING_FILE).is_file()
    tally = DeliveryTally() if tally is None else tally
    tally.sending = error is None or (
        checked
        and not isinstance(
            error, (TrackingTargetUnavailableError, PhaseSweepShutdown, KeyboardInterrupt)
        )
    )
    tally.unanswered = None
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
        pending = delivery_status(experiment)["pending"]
        earlier = pending - delivery_status(experiment, generation_id)["pending"]
        creations = len(pending_creation_records(experiment))
    except (OSError, ValueError) as exc:
        log.warning("MLflow: pending records could not be counted (%s).", exc)
        return
    done = delivered_text(tally.delivered, tally.settled)
    if done:
        # A report line, so the console shows that an earlier interruption was made good.
        logging.getLogger(REPORT_LOGGER).info(
            ReportText(f"MLflow: {done} left pending by earlier runs.")
        )
    if pending:
        log.warning(
            "MLflow: %s pending%s; results are saved locally. The next run or "
            "`phasesweep sync-tracking CONFIG` sends them.%s",
            count_text(pending - creations, creations),
            f" ({earlier} from earlier runs)" if earlier else "",
            refused_text(len(tally.refused)),
        )


def count_text(records: int, creations: int) -> str:
    """Count delivery records and unconfirmed run creations in the words every MLflow line uses.

    A run creation whose answer was lost is searched for, never sent again, so it is counted
    apart from the records that are uploaded.

    :param int records: Delivery records.
    :param int creations: Unconfirmed run creations, the intents a search settles.
    :return str: For example ``2 records``, ``1 unconfirmed run creation`` or ``2 records and
        1 unconfirmed run creation``; ``0 records`` when both are zero.
    """
    parts = [plural(records, "record")] if records or not creations else []
    if creations:
        parts.append(plural(creations, "unconfirmed run creation"))
    return " and ".join(parts)


def delivered_text(delivered: int, settled: int) -> str:
    """Say what a delivery delivered and settled, in the words of :func:`count_text`.

    :param int delivered: Delivery records delivered.
    :param int settled: Unconfirmed run creations settled, bound to their run or retired.
    :return str: For example ``delivered 2 records and settled 1 unconfirmed run creation``,
        or an empty string when there was neither.
    """
    parts = [f"delivered {plural(delivered, 'record')}"] if delivered else []
    if settled:
        parts.append(f"settled {plural(settled, 'unconfirmed run creation')}")
    return " and ".join(parts)


def refused_text(refused: int) -> str:
    """Say how many pending records MLflow refused, and what keeps them pending.

    :param int refused: Records MLflow refused in this delivery.
    :return str: A sentence with a leading space, or an empty string when none were refused.
    """
    if not refused:
        return ""
    return f" MLflow refused {plural(refused, 'record')}; a refused record {refusal_rule()}."


def refusal_rule(where: str = "") -> str:
    """Say what keeps a record MLflow refused pending, in the words every surface uses.

    A refused record is retried like any other, so a retry alone meets the same refusal; only
    fixing its cause in MLflow helps.

    :param str where: Where the record's warning is, with a leading space, for a reader who
        never saw the console, defaults to the console's own warning.
    :return str: A clause that follows the record it is about, without a final period.
    """
    return (
        f"stays pending until the cause given in its warning{where} is fixed in MLflow, for "
        "example by restoring a deleted run or its experiment"
    )


def delivery_status(experiment: Experiment, generation_id: str | None = None) -> dict[str, int]:
    """Count pending run creations and delivery records without importing the tracking SDK.

    :param Experiment experiment: Local namespace.
    :param str | None generation_id: Optional generation filter.
    :return dict[str, int]: Path-free pending and delivered counts.
    """
    counts = {"pending": len(pending_creation_records(experiment, generation_id)), "delivered": 0}
    for path in tracking_root(experiment).glob(f"{generation_id or '*'}/delivery/*.json"):
        state = json_text(read_record(path), "state")
        if state not in counts:
            raise ValueError(f"Tracking delivery record {path.name!r} has unknown state {state!r}.")
        counts[state] += 1
    return counts


def sync_tracking(
    experiment: Experiment,
    generation_id: str | None = None,
    *,
    tally: DeliveryTally | None = None,
) -> dict[str, int]:
    """Retry pending result projections without selecting or training.

    Check the artifact-root binding and ledger format as every writer does, then
    rebuild missing attempt records from Optuna, the summary-run record of each ended
    unpublished generation from its lifecycle record, and winner records from committed
    generation artifacts, reconcile unresolved creations, and send what is pending
    until MLflow stops answering, through the core run closeout shares.
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
    :raises PhaseSweepError: The MLflow SDK is not installed, or a published
        generation's artifacts no longer validate.
    :return dict[str, int]: Remaining pending and delivered counts in that scope.
    """
    from phasesweep.config.common import _validate_safe_name
    from phasesweep.engine.attempts import _preflight_active_attempts, _PreflightCleanupReport
    from phasesweep.engine.ledger import validate_ledger
    from phasesweep.engine.locking import _experiment_lock
    from phasesweep.runtime.shutdown import signal_handler_scope
    from phasesweep.tracking.mlflow import require_mlflow_sdk

    if generation_id is not None:
        _validate_safe_name("generation", generation_id)
    if experiment.mlflow is None:
        raise TrackingSyncRefusedError(
            "sync-tracking needs an `mlflow` block, and this config has none.",
            remediation=(
                "Pass the config the sweep ran with, which has its `mlflow` block; "
                "`phasesweep explain mlflow` describes the block."
            ),
        )
    require_mlflow_sdk()
    with signal_handler_scope(), _experiment_lock(experiment):
        # Refuse a tree this checkout cannot own before anything here writes to it.
        ledger = validate_ledger(experiment)
        _preflight_active_attempts(
            experiment, _PreflightCleanupReport(), confirm=True, tracking_only=True
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
            and not any(root.glob(f"{generation_id or '*'}/delivery/*.json"))
            and not any(root.glob(f"{generation_id or '*'}/operations"))
        ):
            raise TrackingSyncRefusedError(
                "No tracking deliveries exist for this scope.",
                remediation=(
                    f"Check the generation ID {generation_id!r}, such as an MCP run ID, or omit "
                    "--generation to sync every generation of this experiment."
                    if generation_id is not None
                    else "Pass the config the sweep ran with; no run of this experiment has "
                    "saved an MLflow record yet."
                ),
            )
        counts = delivery_status(experiment, generation_id)
        if errors:
            detail = "\n".join(str(error) for error in errors)
            raise TrackingSyncRefusedError.rewrap(
                errors[0], f"{detail}\nDelivered records in scope: {counts['delivered']}."
            ) from errors[0]
        return counts
