"""Retryable MLflow projections kept separate from scientific evidence."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from phasesweep.errors import UnsafeProcessCleanupError
from phasesweep.runtime.files import UnsafePrivatePathError
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
    require_tracking_target,
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


def _deliver(
    experiment: Experiment,
    path: Path,
    environment: Mapping[str, str],
    *,
    worker_generation: str | None = None,
) -> bool:
    """Try one persisted projection and retain pending state after an outage.

    The upload worker is registered under the generation of the run that sends it, so a worker
    whose cleanup that run cannot confirm is attributed to it, whatever generation the record
    belongs to.

    :param Experiment experiment: Configured target.
    :param Path path: Pending delivery record.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str | None worker_generation: Generation of the run sending another generation's
        record, under which its upload worker is registered; ``None`` registers it under the
        record's own generation, as for a run's own records and for sync-tracking, which is no run.
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
    and trainer-owned failed runs are left alone.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Producing generation.
    :param FrozenTrial trial: Locally terminal trial with confirmed cleanup.
    :param Mapping[str, str] environment: Composed authentication environment.
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
    for path in paths:
        _deliver(experiment, path, environment)


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
    from phasesweep.engine.trial import _environment_identity

    if not (tracking_root(experiment) / generation_id / "generation-binding.json").exists():
        return  # No-work replay never imports the SDK or creates remote runs.
    environment = _environment_identity(experiment, experiment.phases[0].name).values
    for path in _save_generation_intents(experiment, generation_id):
        _deliver(experiment, path, environment)
    pending = delivery_status(experiment, generation_id)["pending"]
    if pending:
        log.warning(
            "MLflow uploads PENDING generation=%s pending=%d; published result saved locally. "
            "Run phasesweep sync-tracking with the same config to finish delivery.",
            generation_id,
            pending,
        )


def _published(experiment: Experiment, generation_id: str) -> bool:
    """Report whether a generation committed as a successful result.

    :param Experiment experiment: Experiment that owns the generation.
    :param str generation_id: Generation to check.
    :return bool: Whether its lifecycle record or the last-success pointer names it published.
    """
    from phasesweep.engine.generation import _recorded_generation
    from phasesweep.engine.paths import _generation_record_path, _last_successful_generation_path
    from phasesweep.engine.publication_validation import _read_pointer_target

    # The pointer commits first; the lifecycle record after it is best-effort.
    target = _read_pointer_target(
        _last_successful_generation_path(experiment), experiment_name=experiment.experiment
    )
    return (target is not None and target.generation_id == generation_id) or (
        _recorded_generation(_generation_record_path(experiment, generation_id)).state
        == "published"
    )


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
        read_binding(tracking_root(experiment) / generation_id, "generation-binding.json"),
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
) -> None:
    """Finish a failed invocation's owned runs after confirmed cleanup.

    Accepted trials keep accepted projections; other finished managed attempts, and without
    persistent storage RUNNING ones too, finish with the invocation's status. Committed
    generations remain untouched.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Failed invocation identity.
    :param BaseException error: Authoritative engine failure, without its private message.
    :param ValidatedLedger ledger: The invocation's ledger, holding its trials.
    """
    from phasesweep.engine.trial import _environment_identity

    parent_path = tracking_root(experiment) / generation_id / "generation-binding.json"
    if not parent_path.exists():
        return
    if _published(experiment, generation_id):
        return
    error_class = type(error).__name__
    status = _failure_status(error_class)
    environment = _environment_identity(experiment, experiment.phases[0].name).values
    attempts = _unrecorded_attempts(experiment, generation_id)
    trials = _attempt_trials(ledger, {binding["phase"] for _, binding in attempts})
    paths = _save_attempt_intents(experiment, generation_id, attempts, trials, ended=status)
    paths.append(_save_failure_summary(experiment, generation_id, status, error_class))
    for path in paths:
        _deliver(experiment, path, environment)


def _rebuild_attempt_intents(
    experiment: Experiment, ledger: ValidatedLedger, generation_ids: Iterable[str]
) -> None:
    """Save the attempt records an interrupted or failed invocation never saved.

    Without persistent storage the ledger that held an ended generation's trials is gone, so
    each managed attempt of an unpublished one ends with the status that ends its summary run.

    :param Experiment experiment: Configured experiment.
    :param ValidatedLedger ledger: The experiment's validated ledger, holding the attempts' trials.
    :param Iterable[str] generation_ids: Ended generations whose attempts are checked; never a
        live invocation's.
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
        untouched = experiment.persistent or _published(experiment, generation_id)
        ended = None if untouched else _ended_status(experiment, generation_id)[0]
        _save_attempt_intents(experiment, generation_id, attempts, trials, ended=ended)


def _ended_status(experiment: Experiment, generation_id: str) -> tuple[str, str | None]:
    """Name the status that ends an ended, unpublished generation's runs, and its failure.

    It is the status the failure in the generation's lifecycle record maps to, or KILLED when
    the invocation, as one killed outright does, recorded no failure.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Ended, unpublished generation.
    :raises OSError: The lifecycle record cannot be read.
    :return tuple[str, str | None]: ``FAILED`` or ``KILLED``, and the class name of the recorded
        failure, or ``None`` when none was recorded.
    """
    from phasesweep.engine.generation import _recorded_generation
    from phasesweep.engine.paths import _generation_record_path

    recorded = _recorded_generation(_generation_record_path(experiment, generation_id))
    if recorded.state in {"failed", "publication_failed"}:
        return _failure_status(recorded.error_class), recorded.error_class
    return "KILLED", None


def _save_ended_summaries(experiment: Experiment, generation_ids: Iterable[str]) -> list[Path]:
    """Save the summary-run record each ended, unpublished generation's invocation never saved.

    An invocation killed outright saves none, so its summary run would stay open. It ends with
    the status :func:`_ended_status` names. A published generation's record is built from its
    committed artifacts instead, and a record already saved is kept as it is. Callers hold the
    experiment lock and pass no live invocation's generation: a run never passes its own, and
    recover-run passes the run whose runner it proved dead.

    :param Experiment experiment: Configured experiment.
    :param Iterable[str] generation_ids: Ended generations to check.
    :raises OSError: A record or binding cannot be read or written.
    :return list[Path]: The summary-run record of each of those generations that is unpublished
        and has a bound summary run, whether saved here or before.
    """
    paths = []
    for generation_id in sorted(set(generation_ids)):
        root = tracking_root(experiment) / generation_id
        if not (root / "generation-binding.json").is_file() or _published(
            experiment, generation_id
        ):
            continue
        path = root / "delivery" / "generation.json"
        if not path.exists():
            _save_failure_summary(
                experiment, generation_id, *_ended_status(experiment, generation_id)
            )
        paths.append(path)
    return paths


def generations_left_open(experiment: Experiment) -> set[str]:
    """Name the generations whose summary run is bound but no saved record ends it.

    A published generation is left out, as its record is built from its committed artifacts.

    :param Experiment experiment: Configured experiment.
    :return set[str]: Each bound, unpublished generation without a summary-run record.
    """
    return {
        parent.parent.name
        for parent in tracking_root(experiment).glob("*/generation-binding.json")
        if not (parent.parent / "delivery" / "generation.json").exists()
        and not _published(experiment, parent.parent.name)
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
) -> list[Path]:
    """Save the records of attempts whose outcomes recovery committed, sending nothing.

    As for sync-tracking, only a generation whose summary run is bound has attempt runs to
    finish, a record already saved is kept as it is, and nothing is saved unless the configured
    target is the one the tree recorded: a run with a changed target is refused only after this
    save. A generation that is not published also gets the record that ends its summary run,
    as its invocation is over. The outcomes are already in the ledger and lifecycle records,
    from which sync-tracking rebuilds any record left unsaved, so another target or a record
    that cannot be written or read costs a warning, never the caller's run or recovery.

    :param Experiment experiment: Configured experiment.
    :param ValidatedLedger ledger: The experiment's validated ledger, holding the attempts' trials.
    :param Iterable[str] generation_ids: Ended generations: those whose trials recovery settled,
        those left open by an invocation killed outright, and for recover-run the run it
        recovers; never the caller's own.
    :return list[Path]: Those generations' attempt and summary-run records not yet delivered, in
        generation order, or none when they could not all be saved and read or the target is
        not the recorded one.
    """
    root = tracking_root(experiment)
    generations = sorted(
        generation_id
        for generation_id in set(generation_ids)
        if (root / generation_id / "generation-binding.json").is_file()
    )
    try:
        if generations and not _target_recorded(experiment):
            log.warning(
                "The MLflow outcomes of recovered attempts were not saved: this config's MLflow "
                "target is not the one the sweep's runs were created in. They wait for a run or "
                "`phasesweep sync-tracking` with the config the sweep ran with."
            )
            return []
        _rebuild_attempt_intents(experiment, ledger, generations)
        summaries = set(_save_ended_summaries(experiment, generations))
        records: list[Path] = []
        for generation_id in generations:
            delivery = root / generation_id / "delivery"
            records += sorted(delivery.glob("attempt-*.json"))
            records += [path for path in [delivery / "generation.json"] if path in summaries]
        return [path for path in records if read_record(path)["state"] != "delivered"]
    except (OSError, ValueError, UnsafePrivatePathError) as exc:
        log.warning(
            "The MLflow outcomes of recovered attempts were not all saved (%s); run phasesweep "
            "sync-tracking with the same config to rebuild them from the ledger and deliver them.",
            exc,
        )
        return []


def deliver_recovered_attempts(
    experiment: Experiment,
    ledger: ValidatedLedger,
    generation_id: str,
    generation_ids: Iterable[str],
    *,
    recovering: bool = False,
    tracker_unreachable: bool = False,
) -> dict[str, int]:
    """Send the MLflow outcomes a recovery committed for earlier generations' attempts and runs.

    Each such generation's records are its settled attempts' outcomes and, when it is not
    published, the end of its summary run, from :func:`save_recovered_attempts`. Recovery
    itself sends nothing, so an unreachable tracker costs it no request. The records
    go only after a target check: the one :func:`require_tracking_target` made when the calling
    run bound its own summary run, or else a resolve made here, which binds no summary run. A
    failed check or upload leaves the rest pending for sync-tracking. Every worker this starts
    is registered under ``generation_id``, so one whose cleanup is unconfirmed belongs to the run
    that is accountable for it, not to the generation whose record it was sending.

    :param Experiment experiment: Configured experiment.
    :param ValidatedLedger ledger: The ledger holding the settled trials.
    :param str generation_id: The calling run's generation, whose own attempts its terminal
        delivery sends, or for ``recover-run`` the run it recovers.
    :param Iterable[str] generation_ids: Ended generations, as for
        :func:`save_recovered_attempts`.
    :param bool recovering: The caller is ``recover-run``, so ``generation_id`` is the run it
        recovers rather than its own invocation: that run's records are sent with the rest, and
        the target is resolved here, as no check in this process covered it.
    :param bool tracker_unreachable: The caller found a tracker it could not read, so the
        records are only saved.
    :raises UnsafeProcessCleanupError: An upload worker's cleanup is uncertain.
    :return dict[str, int]: How many of those records were ``delivered`` here and how many stay
        ``pending``.
    """
    from phasesweep.engine.errors import TrackingTargetUnavailableError
    from phasesweep.engine.trial import _environment_identity

    # A run's own records are its terminal delivery's, which knows how the run ended.
    own = set() if recovering else {generation_id}
    paths = save_recovered_attempts(experiment, ledger, set(generation_ids) - own)
    unsent = {"delivered": 0, "pending": len(paths)}
    if not paths:
        return unsent
    pending = (
        f"{len(paths)} MLflow outcome(s) of the recovered trials and generations stay pending; "
        "run phasesweep sync-tracking with the same config to deliver them."
    )
    if tracker_unreachable:
        log.warning(pending)
        return unsent
    environment = _environment_identity(experiment, experiment.phases[0].name).values
    checked = (
        not recovering
        and (tracking_root(experiment) / generation_id / "generation-binding.json").is_file()
    )
    if not checked:
        try:
            require_tracking_target(
                experiment,
                generation_id=generation_id,
                environment=environment,
                deadline=None,
                resolve_only=True,
            )
        except TrackingTargetUnavailableError as exc:
            log.warning("%s %s", exc, pending)
            return unsent
    delivered = 0
    for path in paths:
        if not _deliver(experiment, path, environment, worker_generation=generation_id):
            break
        delivered += 1
    return {"delivered": delivered, "pending": len(paths) - delivered}


def delivery_status(experiment: Experiment, generation_id: str | None = None) -> dict[str, int]:
    """Read mutable delivery counts without importing the tracking SDK.

    :param Experiment experiment: Local namespace.
    :param str | None generation_id: Optional generation filter.
    :return dict[str, int]: Path-free pending and delivered counts.
    """
    counts = {"pending": 0, "delivered": 0}
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
    record, and winner records from committed generation artifacts.

    :param Experiment experiment: Matching source experiment configuration.
    :param str | None generation_id: Generation whose deliveries should be
        retried, or ``None`` for every generation's retained deliveries.
    :raises ValueError: The generation name is unsafe, or a delivery record is malformed.
    :raises TrackingSyncRefusedError: The config has no ``mlflow`` block, no deliveries
        exist, or the target or generation does not match the saved records.
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
    from phasesweep.engine.errors import TrackingSyncRefusedError
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
        if not _target_recorded(experiment):
            raise TrackingSyncRefusedError(
                "Tracking delivery belongs to a different target or generation."
            )
        environment = _environment_identity(experiment, experiment.phases[0].name).values
        # A run a timed-out creation made late is bound before its records are rebuilt.
        resolve_pending_creations(experiment, environment, generation_id)
        parents = sorted(root.glob(f"{generation_id or '*'}/generation-binding.json"))
        # The lock rules out a live invocation, so every unpublished generation here has ended.
        generations = [parent.parent.name for parent in parents]
        _rebuild_attempt_intents(experiment, ledger, generations)
        _save_ended_summaries(experiment, generations)
        for parent in parents:
            if (parent.parent / "delivery" / "generation.json").exists() or not _published(
                experiment, parent.parent.name
            ):
                continue
            _save_generation_intents(experiment, parent.parent.name)
        paths = sorted(root.glob(f"{generation_id or '*'}/delivery/*.json"))
        if not paths:
            raise TrackingSyncRefusedError("No tracking deliveries exist for this scope.")
        for path in paths:
            record = read_record(path)
            if (
                record["source"] != experiment.mlflow.evidence_identity()
                or record["generation_id"] != path.parent.parent.name
            ):
                raise TrackingSyncRefusedError(
                    "Tracking delivery belongs to a different target or generation."
                )
        for path in paths:
            _deliver(experiment, path, environment)
    return delivery_status(experiment, generation_id)
