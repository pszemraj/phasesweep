"""Retryable MLflow projections kept separate from scientific evidence."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from phasesweep.errors import UnsafeProcessCleanupError
from phasesweep.runtime.json import (
    JsonObject,
    json_number,
    json_object,
    json_optional_text,
    json_text,
)
from phasesweep.tracking.records import validate_binding
from phasesweep.tracking.session import (
    BINDING_FILE,
    call_tracking,
    read_binding,
    read_inherits,
    read_record,
    resolve_pending_creations,
    tracking_root,
    write_record,
)

if TYPE_CHECKING:
    from optuna.trial import FrozenTrial

    from phasesweep.config.models import Experiment
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


def _deliver(experiment: Experiment, path: Path, environment: Mapping[str, str]) -> bool:
    """Try one persisted projection and retain pending state after an outage.

    :param Experiment experiment: Configured target.
    :param Path path: Pending delivery record.
    :param Mapping[str, str] environment: Composed credentials and transport.
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
        call_tracking(experiment, generation_id, action, environment)
        # A lost acknowledgement resends the same projection on the next sync.
        write_record(path, {**record, "state": "delivered"})
    except (OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, UnsafeProcessCleanupError):
            raise
        log.warning(
            "MLflow result upload pending for generation %s (%s). Retry with sync-tracking.",
            generation_id,
            str(exc),
        )
        return False
    return True


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
    :param str | None finish: Terminal status for a still-running managed run.
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


def _enqueue(
    experiment: Experiment,
    generation_id: str,
    name: str,
    binding: RunBinding,
    projection: Projection,
    environment: Mapping[str, str],
    *,
    finish: str | None = None,
) -> None:
    """Save delivery intent, then try the upload once.

    :param Experiment experiment: Configured target.
    :param str generation_id: Producing generation.
    :param str name: Stable projection identity.
    :param RunBinding binding: Verified target run.
    :param Projection projection: Frozen upload payload.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str | None finish: Terminal status for a still-running managed run.
    """
    path = _save_intent(experiment, generation_id, name, binding, projection, finish=finish)
    _deliver(experiment, path, environment)


def _accepted_projection(trial: FrozenTrial, inherits: Mapping[str, str]) -> Projection:
    """Freeze one accepted trial's result from its durable Optuna record.

    Constraints and observed values come from the trial's own attributes rather
    than the current config, so the projection can be rebuilt for any accepted
    trial later.

    :param FrozenTrial trial: COMPLETE trial.
    :param Mapping[str, str] inherits: Native run of each parent winner the attempt inherited.
    :return Projection: The attempt's accepted projection.
    """
    from phasesweep.engine.state import (
        CONSTRAINT_PREFIX,
        DURATION_ATTR,
        FEASIBLE_ATTR,
        OBSERVED_PREFIX,
        OVERRIDES_ATTR,
    )

    overrides = json_object(json.loads(trial.user_attrs[OVERRIDES_ATTR]))
    assert trial.value is not None
    constraints = {
        key.removeprefix(CONSTRAINT_PREFIX): json_number(trial.user_attrs, key)
        for key in sorted(trial.user_attrs)
        if key.startswith(CONSTRAINT_PREFIX)
    }
    observed = {
        key.removeprefix(OBSERVED_PREFIX): json_number(trial.user_attrs, key)
        for key in sorted(trial.user_attrs)
        if key.startswith(OBSERVED_PREFIX)
    }
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
    :param str status: Terminal MLflow status for a still-running run.
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
    """Mirror an accepted Optuna trial to its native bound MLflow run.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Producing generation.
    :param FrozenTrial trial: Locally terminal accepted trial.
    :param Mapping[str, str] environment: Composed authentication environment.
    """
    from phasesweep.engine.state import TRIAL_DIR_ATTR

    directory = Path(trial.user_attrs[TRIAL_DIR_ATTR])
    binding = read_binding(directory)
    _enqueue(
        experiment,
        generation_id,
        f"attempt-{binding['attempt_id']}",
        binding,
        _accepted_projection(trial, read_inherits(directory)),
        environment,
    )


def _finish_attempt(
    experiment: Experiment,
    generation_id: str,
    binding: RunBinding,
    status: str,
    environment: Mapping[str, str],
) -> None:
    """Queue the end of a managed attempt run that produced no accepted result.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Producing generation.
    :param RunBinding binding: The attempt's managed run binding.
    :param str status: Terminal MLflow status for a still-running run.
    :param Mapping[str, str] environment: Composed authentication environment.
    """
    _enqueue(
        experiment,
        generation_id,
        f"attempt-{binding['attempt_id']}",
        binding,
        _unaccepted_projection(binding, status),
        environment,
        finish=status,
    )


def publish_unaccepted_trial(
    experiment: Experiment,
    generation_id: str,
    trial: FrozenTrial,
    environment: Mapping[str, str],
) -> None:
    """Finish the managed run of a failed or pruned trial after confirmed cleanup.

    No evidence capture ran to finish that run, so it would otherwise stay
    RUNNING. Trainer-owned runs and attempts that never created a run are
    left alone.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Producing generation.
    :param FrozenTrial trial: Locally terminal FAIL or PRUNED trial.
    :param Mapping[str, str] environment: Composed authentication environment.
    """
    from optuna.trial import TrialState

    from phasesweep.engine.state import TRIAL_DIR_ATTR

    directory = trial.user_attrs.get(TRIAL_DIR_ATTR)
    if not isinstance(directory, str) or not (Path(directory) / BINDING_FILE).is_file():
        return
    binding = read_binding(Path(directory))
    if binding["run_mode"] == "managed":
        status = "KILLED" if trial.state == TrialState.PRUNED else "FAILED"
        _finish_attempt(experiment, generation_id, binding, status, environment)


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
    """An attempt's Optuna trial and whether its own study recorded the trial's cleanup recovery."""

    trial: FrozenTrial
    cleanup_recovered: bool


def _attempt_trials(ledger: ValidatedLedger, phases: Iterable[str]) -> dict[str, _AttemptTrial]:
    """Index the ledger's trials of some phases by attempt id.

    A phase whose study cannot be read contributes nothing, so its attempts
    stay unresolved instead of being guessed. Trial numbers are study-local,
    so each trial is paired with the recovery ledger of the study it came from.

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
    """Save the delivery intent of each unrecorded attempt whose Optuna outcome settles it.

    The trial's state decides what an attempt reports, never the absence of a
    record: a COMPLETE trial whose callback saved nothing, because a write
    failed or the controller died after Optuna committed it, gets its accepted
    projection. A managed run of a trial that was not accepted is finished
    once its cleanup is established, either by the trial's own positive
    evidence or by a recovery its study recorded; cleanup recovery never
    rewrites the trial's attributes, so the study's ledger is the only place
    the second form is visible. An attempt without a readable trial stays
    unresolved, as does one whose cleanup is still uncertain.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Producing generation.
    :param list[tuple[Path, RunBinding]] attempts: From :func:`_unrecorded_attempts`.
    :param Mapping[str, _AttemptTrial] trials: Ledger trials by attempt id, each with
        its study's recovery.
    :param str | None ended: Status for every unaccepted managed run once the
        invocation that owned it ended with confirmed cleanup, or ``None`` to
        finish only trials Optuna failed or pruned with confirmed cleanup.
    :return list[Path]: The records saved, in attempt order.
    """
    from optuna.trial import TrialState

    from phasesweep.engine.attempts import _trial_requires_cleanup_recovery

    paths: list[Path] = []
    for directory, binding in attempts:
        found = trials.get(binding["attempt_id"])
        if found is None:
            continue
        trial = found.trial
        name = f"attempt-{binding['attempt_id']}"
        if trial.state == TrialState.COMPLETE:
            projection = _accepted_projection(trial, read_inherits(directory))
            paths.append(_save_intent(experiment, generation_id, name, binding, projection))
            continue
        if binding["run_mode"] != "managed":
            continue
        settled = trial.state.is_finished() and (
            found.cleanup_recovered or not _trial_requires_cleanup_recovery(trial)
        )
        if settled and trial.state == TrialState.PRUNED:
            status = "KILLED"
        elif ended is not None:
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
    """Save a published generation's winner references from its committed artifacts.

    Reading the validated artifacts instead of in-memory winners lets
    sync-tracking rebuild records that a failed write after publication never
    saved. Phase membership, order, and lineage come from the generation's own
    validated config snapshot, so a phase appended or rewired since cannot
    change what an older generation reports. Every projection is built before
    the first record is saved. Records that already exist are kept.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Published generation that has a summary run.
    :raises PublicationCommitError: The generation summary cannot be read back.
    :raises PublicationAccessError: A manifest artifact cannot be read as the current user.
    :raises PublicationIntegrityError: A manifest-listed artifact fails validation.
    :raises OSError: A delivery record cannot be written.
    :return list[Path]: Winner records in phase order, then the summary run's.
    """
    from phasesweep.engine.generation import _validate_generation_publishable

    parent = read_binding(tracking_root(experiment) / generation_id, "generation-binding.json")
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


def publish_generation(experiment: Experiment, generation_id: str) -> None:
    """Mirror a committed generation's winner references to its MLflow runs.

    Every record is saved before the first upload, so an outage leaves all of
    them pending for sync-tracking.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Committed generation.
    :raises PublicationIntegrityError: The committed artifacts no longer validate.
    :raises OSError: A delivery record cannot be read or written.
    :raises UnsafeProcessCleanupError: An upload worker's cleanup is uncertain.
    """
    from phasesweep.engine.trial import _trainer_environment

    if not (tracking_root(experiment) / generation_id / "generation-binding.json").exists():
        return  # No-work replay never imports the SDK or creates remote runs.
    environment = _trainer_environment(experiment, experiment.phases[0].name)
    for path in _save_generation_intents(experiment, generation_id):
        _deliver(experiment, path, environment)


def _published(experiment: Experiment, generation_id: str) -> bool:
    """Report whether a generation committed as a successful result.

    :param Experiment experiment: Experiment that owns the generation.
    :param str generation_id: Generation to check.
    :return bool: Whether its lifecycle record or the last-success pointer names it published.
    """
    from phasesweep.engine.generation import _recorded_generation_state
    from phasesweep.engine.paths import _generation_record_path, _last_successful_generation_path
    from phasesweep.engine.publication_validation import _read_pointer_target

    # The pointer commits first; the lifecycle record after it is best-effort.
    target = _read_pointer_target(
        _last_successful_generation_path(experiment), experiment_name=experiment.experiment
    )
    return (target is not None and target.generation_id == generation_id) or (
        _recorded_generation_state(_generation_record_path(experiment, generation_id))
        == "published"
    )


def publish_failed_generation(
    experiment: Experiment,
    generation_id: str,
    error: BaseException,
    ledger: ValidatedLedger,
) -> None:
    """Finish a failed invocation's owned runs only after the engine confirms cleanup.

    Covers the generation summary and every attempt whose post-trial callback
    saved no record. Each attempt's Optuna outcome decides what it reports:
    an accepted trial gets its accepted projection, and the managed run of
    any other attempt is finished with the invocation's status.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Failed invocation identity.
    :param BaseException error: Authoritative engine failure, without its private message.
    :param ValidatedLedger ledger: The invocation's ledger, holding its trials.
    """
    from phasesweep.engine.paths import _last_successful_generation_path
    from phasesweep.engine.publication_validation import _read_pointer_target
    from phasesweep.engine.trial import _trainer_environment
    from phasesweep.runtime.shutdown import PhaseSweepShutdown

    parent_path = tracking_root(experiment) / generation_id / "generation-binding.json"
    if not parent_path.exists():
        return
    # The pointer commits before the optional lifecycle record. A secondary
    # error after that boundary must never turn committed success into failure.
    target = _read_pointer_target(
        _last_successful_generation_path(experiment), experiment_name=experiment.experiment
    )
    if target is not None and target.generation_id == generation_id:
        return
    status = "KILLED" if isinstance(error, (PhaseSweepShutdown, KeyboardInterrupt)) else "FAILED"
    environment = _trainer_environment(experiment, experiment.phases[0].name)
    attempts = _unrecorded_attempts(experiment, generation_id)
    trials = _attempt_trials(ledger, {binding["phase"] for _, binding in attempts})
    paths = _save_attempt_intents(experiment, generation_id, attempts, trials, ended=status)
    paths.append(
        _save_intent(
            experiment,
            generation_id,
            "generation",
            read_binding(parent_path.parent, parent_path.name),
            _projection(
                metrics={},
                params={},
                tags={"phasesweep.outcome": status},
                summary={
                    "generation_id": generation_id,
                    "status": status,
                    "error_class": type(error).__name__,
                },
            ),
            finish=status,
        )
    )
    for path in paths:
        _deliver(experiment, path, environment)


def _rebuild_attempt_intents(experiment: Experiment, generation_ids: Iterable[str]) -> None:
    """Save the attempt records an interrupted or failed invocation never saved.

    :param Experiment experiment: Configured experiment.
    :param Iterable[str] generation_ids: Generations whose attempts are checked.
    """
    from phasesweep.engine.ledger import validate_ledger
    from phasesweep.errors import PhaseSweepError

    unrecorded = {
        generation_id: attempts
        for generation_id in generation_ids
        if (attempts := _unrecorded_attempts(experiment, generation_id))
    }
    if not unrecorded:
        return
    try:
        ledger = validate_ledger(experiment)
    except PhaseSweepError as exc:
        log.warning("Unsaved MLflow attempt results stay unresolved: %s", exc)
        return
    phases = {binding["phase"] for attempts in unrecorded.values() for _, binding in attempts}
    trials = _attempt_trials(ledger, phases)
    for generation_id, attempts in unrecorded.items():
        _save_attempt_intents(experiment, generation_id, attempts, trials, ended=None)


def delivery_status(experiment: Experiment, generation_id: str | None = None) -> dict[str, int]:
    """Read mutable delivery counts without importing the tracking SDK.

    :param Experiment experiment: Local namespace.
    :param str | None generation_id: Optional generation filter.
    :return dict[str, int]: Path-free pending and delivered counts.
    """
    counts = {"pending": 0, "delivered": 0}
    for path in tracking_root(experiment).glob(_delivery_pattern(generation_id)):
        state = json_text(read_record(path), "state")
        if state not in counts:
            raise ValueError(f"Tracking delivery record {path.name!r} has unknown state {state!r}.")
        counts[state] += 1
    return counts


def _delivery_pattern(generation_id: str | None) -> str:
    """Select one generation's delivery records, or every generation's.

    :param str | None generation_id: Generation filter, or ``None`` for all.
    :return str: Glob relative to the tracking root.
    """
    return f"{generation_id or '*'}/delivery/*.json"


def sync_tracking(experiment: Experiment, generation_id: str | None = None) -> dict[str, int]:
    """Retry result projections, without selecting or training.

    A run whose creation timed out is bound first if it has since appeared.
    Records an invocation never saved, because a write failed or the
    controller died, are then rebuilt: an attempt's from its trial in the
    Optuna ledger, and a published generation's winner records from its
    committed artifacts.

    :param Experiment experiment: Matching source experiment configuration.
    :param str | None generation_id: Generation whose deliveries should be
        retried, or ``None`` for every generation's retained deliveries.
    :raises ValueError: No deliveries exist, or the target or generation does
        not match the saved records.
    :raises PhaseSweepError: The MLflow SDK is not installed, or a published
        generation's artifacts no longer validate.
    :return dict[str, int]: Remaining pending and delivered counts in that scope.
    """
    from phasesweep.config.common import _validate_safe_name
    from phasesweep.engine.attempts import _preflight_active_attempts, _PreflightCleanupReport
    from phasesweep.engine.locking import _experiment_lock
    from phasesweep.engine.trial import _trainer_environment
    from phasesweep.runtime.shutdown import signal_handler_scope
    from phasesweep.tracking.mlflow import require_mlflow_sdk

    if generation_id is not None:
        _validate_safe_name("generation", generation_id)
    if experiment.mlflow is None:
        raise ValueError("sync-tracking requires MLflow settings.")
    require_mlflow_sdk()
    with signal_handler_scope(), _experiment_lock(experiment):
        _preflight_active_attempts(
            experiment, _PreflightCleanupReport(), confirm=True, tracking_only=True
        )
        root = tracking_root(experiment)
        source = root / "source.json"
        # Searched runs and rebuilt records take the configured target, so it
        # must be the recorded one.
        if (
            source.exists()
            and read_record(source)["source"] != experiment.mlflow.evidence_identity()
        ):
            raise ValueError("Tracking delivery belongs to a different target or generation.")
        environment = _trainer_environment(experiment, experiment.phases[0].name)
        # A run a timed-out creation made late is bound before its records are rebuilt.
        resolve_pending_creations(experiment, environment, generation_id)
        parents = sorted(root.glob(f"{generation_id or '*'}/generation-binding.json"))
        _rebuild_attempt_intents(experiment, [parent.parent.name for parent in parents])
        for parent in parents:
            if (parent.parent / "delivery" / "generation.json").exists() or not _published(
                experiment, parent.parent.name
            ):
                continue
            _save_generation_intents(experiment, parent.parent.name)
        paths = sorted(root.glob(_delivery_pattern(generation_id)))
        if not paths:
            raise ValueError("No tracking deliveries exist for this scope.")
        for path in paths:
            record = read_record(path)
            if (
                record["source"] != experiment.mlflow.evidence_identity()
                or record["generation_id"] != path.parent.parent.name
            ):
                raise ValueError("Tracking delivery belongs to a different target or generation.")
        for path in paths:
            _deliver(experiment, path, environment)
    return delivery_status(experiment, generation_id)
