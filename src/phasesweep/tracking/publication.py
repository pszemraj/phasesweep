"""Retryable MLflow projections kept separate from scientific evidence."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from phasesweep.engine.errors import TrackingSyncRefusedError
from phasesweep.errors import (
    OperatorAction,
    PhaseSweepError,
    ProcessCleanupUncertainError,
    operator_step,
)
from phasesweep.runtime.files import UnsafePrivatePathError
from phasesweep.runtime.json import (
    JsonObject,
    json_number,
    json_object,
    json_optional_text,
    json_text,
)
from phasesweep.runtime.terminal import REPORT_LOGGER, ReportText
from phasesweep.tracking.records import validate_binding
from phasesweep.tracking.session import (
    BINDING_FILE,
    GENERATION_BINDING_FILE,
    NO_ANSWER,
    PENDING_GENERATION_BINDING_FILE,
    call_tracking,
    pending_creation_records,
    read_binding,
    read_inherits,
    read_record,
    require_tracking_target,
    resolve_pending_creations,
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
) -> bool:
    """Try one persisted projection, keeping it pending when the upload fails.

    A failure MLflow answered, such as a deleted run, concerns this record alone, so it costs
    a warning and the caller's other uploads go on. The upload worker is registered under the
    generation of the run that sends it, so a worker whose cleanup that run cannot confirm is
    attributed to it, whatever generation the record belongs to.

    :param Experiment experiment: Configured target.
    :param Path path: Pending delivery record.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str | None worker_generation: Generation of the run sending another generation's
        record, under which its upload worker is registered; ``None`` registers it under the
        record's own generation, as for a run's own records and for sync-tracking, which is no run.
    :raises ProcessCleanupUncertainError: The upload worker's cleanup is uncertain.
    :raises TimeoutError: MLflow gave no answer in time, as for every member of
        :data:`~phasesweep.tracking.session.NO_ANSWER`, which the caller handles for the pass.
    :return bool: Whether delivery has succeeded.
    """
    record = read_record(path)
    if record["state"] == "delivered":
        return True
    generation_id = json_text(record, "generation_id")
    action: PublishAction = {
        "action": "publish",
        "name": path.stem,
        "binding": json_object(record["binding"]),
        "projection": json_object(record["projection"]),
        "finish": json_optional_text(record, "finish"),
    }
    try:
        call_tracking(experiment, worker_generation or generation_id, action, environment)
        # A lost acknowledgement resends the same projection on the next sync.
        write_record(path, {**record, "state": "delivered"})
    except ProcessCleanupUncertainError:
        raise
    except NO_ANSWER:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        log.warning(
            "MLflow upload %s of generation %s is pending (%s).", path.stem, generation_id, exc
        )
        return False
    return True


def _send(
    experiment: Experiment,
    paths: list[Path],
    environment: Mapping[str, str],
    *,
    worker_generation: str | None,
) -> bool:
    """Send records in order until MLflow stops answering.

    A rejection of one record leaves it pending and the rest go on. A request MLflow gives no
    answer to would meet the same tracker for every later record, so one such request ends the
    pass and leaves the rest pending for the next run or sync-tracking.

    :param Experiment experiment: Configured target.
    :param list[Path] paths: Delivery records, in sending order.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str | None worker_generation: Generation the upload workers register under, as for
        :func:`_deliver`.
    :raises ProcessCleanupUncertainError: An upload worker's cleanup is uncertain.
    :return bool: Whether MLflow answered every request sent.
    """
    for index, path in enumerate(paths):
        try:
            _deliver(experiment, path, environment, worker_generation=worker_generation)
        except NO_ANSWER as exc:
            later = len(paths) - index - 1
            log.warning(
                "MLflow upload %s of generation %s is pending (%s).%s",
                path.stem,
                path.parent.parent.name,
                exc,
                f" MLflow did not answer, so {later} later upload(s) wait too." if later else "",
            )
            return False
    return True


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
    _send(experiment, paths, environment, worker_generation=None)


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


def publish_generation(experiment: Experiment, generation_id: str) -> bool:
    """Mirror a committed generation's winner references to its MLflow runs.

    Every record is saved before the first upload, then every record of the generation still
    pending is sent, including attempt outcomes an earlier upload left behind, so an outage
    leaves all of them pending for the next run or sync-tracking.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Committed generation.
    :raises PublicationIntegrityError: The committed artifacts no longer validate.
    :raises OSError: A delivery record cannot be read or written.
    :raises ProcessCleanupUncertainError: An upload worker's cleanup is uncertain.
    :return bool: Whether MLflow answered every upload sent; True when the generation bound no
        summary run, as a no-work replay sends nothing of its own.
    """
    from phasesweep.engine.trial import _environment_identity

    if not (tracking_root(experiment) / generation_id / GENERATION_BINDING_FILE).exists():
        return True  # No-work replay never imports the SDK or creates remote runs.
    environment = _environment_identity(experiment, experiment.phases[0].name).values
    _save_generation_intents(experiment, generation_id)
    return _send(
        experiment, _pending_records(experiment, generation_id), environment, worker_generation=None
    )


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


def publish_failed_generation(
    experiment: Experiment,
    generation_id: str,
    error: BaseException,
    ledger: ValidatedLedger,
) -> bool:
    """Finish a failed invocation's owned runs after confirmed cleanup.

    Accepted trials keep accepted projections; other finished managed attempts, and without
    persistent storage RUNNING ones too, finish with the invocation's status. Committed
    generations remain untouched. Every record of the generation still pending is sent,
    including attempt outcomes an earlier upload left behind. A shutdown being serviced refuses
    each upload, so an interrupted invocation only saves its records for the next run.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Failed invocation identity.
    :param BaseException error: Authoritative engine failure, without its private message.
    :param ValidatedLedger ledger: The invocation's ledger, holding its trials.
    :raises ProcessCleanupUncertainError: An upload worker's cleanup is uncertain.
    :return bool: Whether MLflow answered every upload sent; True when nothing was sent.
    """
    from phasesweep.engine.trial import _environment_identity

    parent_path = tracking_root(experiment) / generation_id / GENERATION_BINDING_FILE
    if not parent_path.exists():
        return True
    if _published(experiment, generation_id):
        return True
    error_class = type(error).__name__
    status = _failure_status(error_class)
    environment = _environment_identity(experiment, experiment.phases[0].name).values
    attempts = _unrecorded_attempts(experiment, generation_id)
    trials = _attempt_trials(ledger, {binding["phase"] for _, binding in attempts})
    _save_attempt_intents(experiment, generation_id, attempts, trials, ended=status)
    _save_failure_summary(experiment, generation_id, status, error_class)
    return _send(
        experiment, _pending_records(experiment, generation_id), environment, worker_generation=None
    )


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


def generations_left_open(experiment: Experiment) -> set[str]:
    """Name unpublished generations whose summary creation or terminal delivery is missing.

    A published generation is left out, as its record is built from its committed artifacts.

    :param Experiment experiment: Configured experiment.
    :return set[str]: Unpublished generations with a summary creation or binding and no
        summary-run delivery record.
    """
    root = tracking_root(experiment)
    parents = {path.parent for path in root.glob(f"*/{GENERATION_BINDING_FILE}")}
    parents.update(path.parent for path in root.glob(f"*/{PENDING_GENERATION_BINDING_FILE}"))
    return {
        parent.name
        for parent in parents
        if not (parent / "delivery" / "generation.json").exists()
        and not _published(experiment, parent.name)
    }


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


def save_recovered_attempts(
    experiment: Experiment, ledger: ValidatedLedger, generation_ids: Iterable[str]
) -> None:
    """Save the records of attempts whose outcomes recovery committed, sending nothing.

    As for sync-tracking, only a generation whose summary run is bound has attempt runs to
    finish, a record already saved is kept as it is, and nothing is saved unless the configured
    target is the one the tree recorded: a run with a changed target is refused only after this
    save. A generation that is not published also gets the record that ends its summary run,
    as its invocation is over. The outcomes are already in the ledger and lifecycle records,
    from which closeout and sync-tracking rebuild any record left unsaved, so another target or
    a record that cannot be written or read costs a warning, never the caller's run or recovery.

    :param Experiment experiment: Configured experiment.
    :param ValidatedLedger ledger: The experiment's validated ledger, holding the attempts' trials.
    :param Iterable[str] generation_ids: Ended generations: those whose trials recovery settled,
        those left open by an invocation killed outright, and for recover-run the run it
        recovers; never the caller's own.
    """
    root = tracking_root(experiment)
    generations = sorted(
        generation_id
        for generation_id in set(generation_ids)
        if (root / generation_id / GENERATION_BINDING_FILE).is_file()
    )
    try:
        if generations and not _target_recorded(experiment):
            log.warning(
                "The MLflow outcomes of recovered attempts were not saved: this config's MLflow "
                "target is not the one the sweep's runs were created in. They wait for a run or "
                "`phasesweep sync-tracking` with the config the sweep ran with."
            )
            return
        unresolved: dict[str, TrackingSyncRefusedError] = {}
        _rebuild_attempt_intents(experiment, ledger, generations, unresolved=unresolved)
        _save_ended_summaries(experiment, generations, unresolved=unresolved)
        for error in unresolved.values():
            log.warning("%s %s", error, error.remediation)
    except (OSError, ValueError, UnsafePrivatePathError, TrackingSyncRefusedError) as exc:
        log.warning(
            "The MLflow outcomes of recovered attempts were not all saved (%s); the next run or "
            "`phasesweep sync-tracking` with the same config rebuilds them from the ledger and "
            "delivers them.",
            exc,
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
    errors: list[TrackingSyncRefusedError],
) -> None:
    """Save every record ended generations still owe their MLflow runs, sending nothing.

    Missing attempt records are rebuilt from the ledger, the summary-run record of each ended,
    unpublished generation from its lifecycle record, and a published generation's winner and
    summary records from its committed artifacts. Records already saved are kept as they are.

    :param Experiment experiment: Configured experiment whose target the tree recorded.
    :param ValidatedLedger ledger: The experiment's validated ledger, holding the attempts' trials.
    :param list[str] generation_ids: Ended generations with a bound summary run.
    :param list[TrackingSyncRefusedError] errors: Collects each generation whose lifecycle record
        is unreadable or malformed; its outcomes stay unsaved.
    :raises OSError: A record cannot be read or written.
    :raises PublicationIntegrityError: A published generation's artifacts no longer validate.
    """
    unresolved: dict[str, TrackingSyncRefusedError] = {}
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
    errors.extend(unresolved.values())


def _deliver_outstanding(
    experiment: Experiment,
    ledger: ValidatedLedger,
    environment: Mapping[str, str],
    *,
    scope: str | None,
    skip: str | None,
    worker_generation: str | None,
    errors: list[TrackingSyncRefusedError],
) -> list[Path]:
    """Deliver what every ended generation in scope still owes its MLflow runs.

    This is sync-tracking's work, shared with run closeout and recover-run. The caller holds the
    experiment lock, so every generation in scope but ``skip`` has ended. Pending creations are
    searched first, so a run a timed-out creation made late is bound before its records are
    rebuilt; then missing records are staged and every pending record is sent. The first request
    MLflow gives no answer to ends the pass, search or upload, so an outage costs one failed
    request; records are still staged, and stay pending for the next run or sync-tracking.

    :param Experiment experiment: Configured MLflow-backed experiment.
    :param ValidatedLedger ledger: The experiment's validated ledger, holding the attempts' trials.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str | None scope: One generation, or ``None`` for every generation.
    :param str | None skip: Generation left out, a live invocation's own, whose records its
        terminal delivery sends.
    :param str | None worker_generation: Generation the search and upload workers register
        under, or ``None`` for each record's own.
    :param list[TrackingSyncRefusedError] errors: Collects local creation record failures and
        unreadable lifecycle records, whose outcomes stay pending.
    :raises TrackingSyncRefusedError: The configured target, or a saved record's target or
        generation, is not the one the tree recorded; nothing is sent.
    :raises ProcessCleanupUncertainError: A search or upload worker's cleanup is uncertain.
    :raises PublicationIntegrityError: A published generation's artifacts no longer validate.
    :return list[Path]: The records sent, in sending order, or none after an unanswered search.
    """
    assert experiment.mlflow is not None
    different = "Tracking delivery belongs to a different target or generation."
    if not _target_recorded(experiment):
        raise TrackingSyncRefusedError(different)
    answered = resolve_pending_creations(
        experiment,
        environment,
        scope,
        skip=skip,
        worker_generation=worker_generation,
        errors=errors,
    )
    _stage_outstanding(
        experiment, ledger, _bound_generations(experiment, scope, skip), errors=errors
    )
    for path in _records(experiment, scope, skip=skip):
        record = read_record(path)
        if (
            record["source"] != experiment.mlflow.evidence_identity()
            or record["generation_id"] != path.parent.parent.name
        ):
            raise TrackingSyncRefusedError(different)
    if not answered:
        # The unanswered search already warned that the uploads wait too.
        return []
    paths = _pending_records(experiment, scope, skip=skip)
    _send(experiment, paths, environment, worker_generation=worker_generation)
    return paths


def _pending_creations(experiment: Experiment, skip: str | None) -> list[Path]:
    """List the pending creation intents of every generation but ``skip``.

    :param Experiment experiment: Configured experiment.
    :param str | None skip: Generation left out, a live invocation's own.
    :return list[Path]: Pending summary and attempt creation records.
    """
    skipped = set() if skip is None else set(pending_creation_records(experiment, skip))
    return [path for path in pending_creation_records(experiment) if path not in skipped]


def deliver_ended_generations(
    experiment: Experiment,
    ledger: ValidatedLedger,
    generation_id: str,
    *,
    recovering: bool = False,
    tracker_unreachable: bool = False,
) -> int:
    """Finish, at a run's end, what sync-tracking would for every generation that has ended.

    That is every generation but a run's own, whose records are its terminal delivery's, which
    knows how the run ended; for recover-run it is every generation. Records are staged locally
    first, so nothing is requested when nothing is pending. The rest goes only after a target
    check: the one the calling run made when it bound its own summary run, or else a resolve
    made here, which binds none. Workers register under ``generation_id`` because this
    invocation is accountable for their cleanup. A configured target other than the recorded
    one stages and sends nothing. Every failure but an uncertain worker cleanup or a shutdown
    costs a warning and leaves the records for the next run or sync-tracking, never the
    caller's outcome.

    :param Experiment experiment: MLflow-backed experiment.
    :param ValidatedLedger ledger: The caller's ledger, holding the trials the records describe.
    :param str generation_id: The calling run's generation, or for ``recover-run`` the run it
        recovers.
    :param bool recovering: The caller is ``recover-run``, so ``generation_id`` is the run it
        recovers rather than its own invocation: that run's records are sent with the rest, and
        the target is resolved here, as no check in this process covered it.
    :param bool tracker_unreachable: The caller found MLflow silent, so records are only staged.
    :raises ProcessCleanupUncertainError: A search or upload worker's cleanup is uncertain.
    :return int: Records of those generations delivered here.
    """
    from phasesweep.engine.errors import TrackingTargetUnavailableError
    from phasesweep.engine.trial import _environment_identity

    skip = None if recovering else generation_id
    errors: list[TrackingSyncRefusedError] = []
    try:
        generations = _bound_generations(experiment, None, skip)
        if not _target_recorded(experiment):
            if generations or _pending_creations(experiment, skip):
                log.warning(
                    "MLflow: records of earlier runs were neither saved nor sent: this config's "
                    "MLflow target is not the one those runs used. A run or `phasesweep "
                    "sync-tracking CONFIG` with the config the sweep ran with delivers what they "
                    "left pending."
                )
            return 0
        _stage_outstanding(experiment, ledger, generations, errors=errors)
        pending = _pending_records(experiment, None, skip=skip) or _pending_creations(
            experiment, skip
        )
        if not pending or tracker_unreachable:
            return 0
        environment = _environment_identity(experiment, experiment.phases[0].name).values
        checked = (
            not recovering
            and (tracking_root(experiment) / generation_id / GENERATION_BINDING_FILE).is_file()
        )
        if not checked:
            require_tracking_target(
                experiment,
                generation_id=generation_id,
                environment=environment,
                deadline=None,
                resolve_only=True,
            )
        # The core stages again after its searches, so its errors replace the first staging's.
        errors = []
        sent = _deliver_outstanding(
            experiment,
            ledger,
            environment,
            scope=None,
            skip=skip,
            worker_generation=generation_id,
            errors=errors,
        )
        return sum(1 for path in sent if read_record(path)["state"] == "delivered")
    except ProcessCleanupUncertainError:
        raise
    except TrackingTargetUnavailableError as exc:
        log.warning("%s", exc)
        return 0
    except (OSError, ValueError, PhaseSweepError) as exc:
        log.warning(
            "MLflow: records left by earlier runs were not all delivered (%s). The next run or "
            "`phasesweep sync-tracking CONFIG` rebuilds and delivers them.",
            exc,
        )
        return 0
    finally:
        for error in errors:
            log.warning("%s %s", error, operator_step(error))


def report_delivery(experiment: Experiment, generation_id: str, delivered: int) -> None:
    """Log the lines that close a run's MLflow delivery.

    Records delivered for earlier runs are counted when there are any, then the records still
    pending, the run's own and earlier runs', with the step that delivers them.

    :param Experiment experiment: MLflow-backed experiment.
    :param str generation_id: The run's own generation.
    :param int delivered: Records of earlier generations this run delivered.
    """
    try:
        pending = delivery_status(experiment)["pending"]
        earlier = pending - delivery_status(experiment, generation_id)["pending"]
    except (OSError, ValueError) as exc:
        log.warning("MLflow: pending uploads could not be counted (%s).", exc)
        return
    if delivered:
        # A report line, so the console shows that an earlier interruption was made good.
        logging.getLogger(REPORT_LOGGER).info(
            ReportText(f"MLflow: delivered {delivered} record(s) left pending by earlier runs.")
        )
    if pending:
        log.warning(
            "MLflow: %d upload(s) pending%s; results are saved locally. The next run or "
            "`phasesweep sync-tracking CONFIG` delivers them.",
            pending,
            f" ({earlier} from earlier runs)" if earlier else "",
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


def sync_tracking(experiment: Experiment, generation_id: str | None = None) -> dict[str, int]:
    """Retry pending result projections without selecting or training.

    Check the artifact-root binding and ledger format as every writer does, then
    reconcile unresolved creations, rebuild missing attempt records from Optuna,
    the summary-run record of each ended unpublished generation from its lifecycle
    record, and winner records from committed generation artifacts, and send what is pending
    until MLflow stops answering, through the core run closeout shares.
    Report local creation or lifecycle record errors after delivering independent records
    in the selected scope.

    :param Experiment experiment: Matching source experiment configuration.
    :param str | None generation_id: Generation whose deliveries should be
        retried, or ``None`` for every generation's retained deliveries.
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
    from phasesweep.engine.trial import _environment_identity
    from phasesweep.runtime.shutdown import signal_handler_scope
    from phasesweep.tracking.mlflow import require_mlflow_sdk

    if generation_id is not None:
        _validate_safe_name("generation", generation_id)
    if experiment.mlflow is None:
        raise TrackingSyncRefusedError(
            "sync-tracking needs an `mlflow` block, and this config has none. Use the config "
            "the sweep ran with, or see `phasesweep explain mlflow`."
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
        environment = _environment_identity(experiment, experiment.phases[0].name).values
        errors: list[TrackingSyncRefusedError] = []
        # The lock rules out a live invocation, so every unpublished generation here has ended.
        _deliver_outstanding(
            experiment,
            ledger,
            environment,
            scope=generation_id,
            skip=None,
            worker_generation=None,
            errors=errors,
        )
        if (
            not errors
            and not had_creations
            and not any(root.glob(f"{generation_id or '*'}/delivery/*.json"))
            and not any(root.glob(f"{generation_id or '*'}/operations"))
        ):
            raise TrackingSyncRefusedError("No tracking deliveries exist for this scope.")
        counts = delivery_status(experiment, generation_id)
        if errors:
            detail = "\n".join(str(error) for error in errors)
            raise TrackingSyncRefusedError.rewrap(
                errors[0], f"{detail}\nDelivered records in scope: {counts['delivered']}."
            ) from errors[0]
        return counts
