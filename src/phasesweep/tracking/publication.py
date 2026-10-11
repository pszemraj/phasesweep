"""Retryable MLflow projections kept separate from scientific evidence."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from phasesweep.errors import UnsafeProcessCleanupError
from phasesweep.tracking.session import (
    BINDING_FILE,
    call_tracking,
    read_binding,
    read_record,
    tracking_root,
    write_record,
)

if TYPE_CHECKING:
    from optuna.trial import FrozenTrial

    from phasesweep.config.models import Experiment
    from phasesweep.engine.state import Winner

log = logging.getLogger(__name__)


def _projection(
    *,
    metrics: dict[str, float],
    params: dict[str, str],
    tags: dict[str, str],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Freeze a retryable result mirror with a stable metric timestamp.

    :param dict[str, float] metrics: Reserved scalar metrics.
    :param dict[str, str] params: Effective override parameters.
    :param dict[str, str] tags: Phase and winner references.
    :param dict[str, Any] summary: Compact artifact contents.
    :return dict[str, Any]: Retryable projection.
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
    try:
        call_tracking(
            experiment,
            record["generation_id"],
            {
                "action": "publish",
                "name": path.stem,
                "binding": record["binding"],
                "projection": record["projection"],
                "finish": record["finish"],
            },
            environment,
        )
        # A lost acknowledgement resends the same projection on the next sync.
        write_record(path, {**record, "state": "delivered"})
    except (OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, UnsafeProcessCleanupError):
            raise
        log.warning(
            "MLflow result upload pending for generation %s (%s). Retry with sync-tracking.",
            record["generation_id"],
            str(exc),
        )
        return False
    return True


def _save_intent(
    experiment: Experiment,
    generation_id: str,
    name: str,
    binding: dict[str, Any],
    projection: dict[str, Any],
    *,
    finish: str | None = None,
) -> Path:
    """Save delivery intent before attempting any result upload.

    :param Experiment experiment: Configured target.
    :param str generation_id: Producing generation.
    :param str name: Stable projection identity.
    :param dict[str, Any] binding: Verified target run.
    :param dict[str, Any] projection: Frozen upload payload.
    :param str | None finish: Terminal status for a still-running managed run.
    :return Path: The delivery record, unchanged when it already exists.
    """
    assert experiment.mlflow is not None
    path = tracking_root(experiment) / generation_id / "delivery" / f"{name}.json"
    if not path.exists():
        write_record(
            path,
            {
                "schema_version": 1,
                "generation_id": generation_id,
                "source": experiment.mlflow.evidence_identity(),
                "binding": binding,
                "projection": projection,
                "finish": finish,
                "state": "pending",
            },
        )
    return path


def _enqueue(
    experiment: Experiment,
    generation_id: str,
    name: str,
    binding: dict[str, Any],
    projection: dict[str, Any],
    environment: Mapping[str, str],
    *,
    finish: str | None = None,
) -> None:
    """Save delivery intent, then try the upload once.

    :param Experiment experiment: Configured target.
    :param str generation_id: Producing generation.
    :param str name: Stable projection identity.
    :param dict[str, Any] binding: Verified target run.
    :param dict[str, Any] projection: Frozen upload payload.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str | None finish: Terminal status for a still-running managed run.
    """
    path = _save_intent(experiment, generation_id, name, binding, projection, finish=finish)
    _deliver(experiment, path, environment)


def publish_trial(
    experiment: Experiment,
    generation_id: str,
    trial: FrozenTrial,
    environment: Mapping[str, str],
    inherited: Mapping[str, Winner],
) -> None:
    """Mirror an accepted Optuna trial to its native bound MLflow run.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Producing generation.
    :param FrozenTrial trial: Locally terminal accepted trial.
    :param Mapping[str, str] environment: Composed authentication environment.
    :param Mapping[str, Winner] inherited: Source winners fixed into this attempt.
    """
    from phasesweep.engine.state import (
        DURATION_ATTR,
        FEASIBLE_ATTR,
        OVERRIDES_ATTR,
        TRIAL_DIR_ATTR,
        constraint_attr,
    )

    directory = Path(trial.user_attrs[TRIAL_DIR_ATTR])
    binding = read_binding(directory)
    overrides = json.loads(trial.user_attrs[OVERRIDES_ATTR])
    assert trial.value is not None
    constraints = {
        c.name: trial.user_attrs[constraint_attr(c.name)] for c in experiment.constraints
    }
    metrics = {
        "phasesweep.objective": float(trial.value),
        "phasesweep.feasible": float(trial.user_attrs[FEASIBLE_ATTR]),
        "phasesweep.duration_seconds": trial.user_attrs[DURATION_ATTR],
        **{f"phasesweep.constraint.{k}": v for k, v in constraints.items()},
    }
    projection = _projection(
        metrics=metrics,
        params={
            f"phasesweep.override.{k}": json.dumps(v, sort_keys=True) for k, v in overrides.items()
        },
        tags={
            "phasesweep.trial_number": str(trial.number),
            **{
                f"phasesweep.inherits.{name}": winner.objective_provenance["remote_capture"][
                    "binding"
                ]["run_id"]
                for name, winner in inherited.items()
                if winner.objective_provenance is not None
            },
        },
        summary={
            "trial_number": trial.number,
            "metric": trial.value,
            "constraints": constraints,
            "effective_overrides": overrides,
        },
    )
    _enqueue(
        experiment,
        generation_id,
        f"attempt-{binding['attempt_id']}",
        binding,
        projection,
        environment,
    )


def _finish_attempt(
    experiment: Experiment,
    generation_id: str,
    binding: dict[str, Any],
    status: str,
    environment: Mapping[str, str],
) -> None:
    """Queue the end of a managed attempt run that produced no accepted result.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Producing generation.
    :param dict[str, Any] binding: The attempt's managed run binding.
    :param str status: Terminal MLflow status for a still-running run.
    :param Mapping[str, str] environment: Composed authentication environment.
    """
    _enqueue(
        experiment,
        generation_id,
        f"attempt-{binding['attempt_id']}",
        binding,
        _projection(
            metrics={},
            params={},
            tags={"phasesweep.outcome": status},
            summary={
                "phase": binding["phase"],
                "attempt_id": binding["attempt_id"],
                "status": status,
            },
        ),
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


def _save_generation_intents(experiment: Experiment, generation_id: str) -> list[Path]:
    """Save a published generation's winner references from its committed artifacts.

    Reading the validated artifacts instead of in-memory winners lets
    sync-tracking rebuild records that a failed write after publication never
    saved. Records that already exist are kept.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Published generation that has a summary run.
    :raises PublicationCommitError: The generation summary cannot be read back.
    :raises PublicationAccessError: A manifest artifact cannot be read as the current user.
    :raises PublicationIntegrityError: A manifest-listed artifact fails validation.
    :raises OSError: A delivery record cannot be written.
    :return list[Path]: Winner records in phase order, then the summary run's.
    """
    from phasesweep.engine.generation import _validate_generation_publishable

    parent = read_record(tracking_root(experiment) / generation_id / "generation-binding.json")
    publication, _ = _validate_generation_publishable(experiment, generation_id)
    bindings = {
        name: winner.payload["objective_provenance"]["remote_capture"]["binding"]
        for name, winner in publication.winners.items()
    }
    references = {
        item["name"]: {
            "run_id": bindings[item["name"]]["run_id"],
            "metric": item["metric"],
            "effective_overrides": item["effective_overrides"],
        }
        for item in publication.summary["phases"]
    }
    paths = [
        _save_intent(
            experiment,
            generation_id,
            f"winner-{phase.name}",
            bindings[phase.name],
            _projection(
                metrics={},
                params={},
                tags={
                    "phasesweep.winner": "true",
                    f"phasesweep.winner_generation.{generation_id}": phase.name,
                    **{
                        f"phasesweep.inherits.{name}": references[name]["run_id"]
                        for name in phase.inherits
                    },
                },
                summary={"phase": phase.name, **references[phase.name]},
            ),
        )
        for phase in experiment.phases
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
                tags={f"phasesweep.winner.{name}": r["run_id"] for name, r in references.items()},
                summary={"generation_id": generation_id, "phases": references},
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
    experiment: Experiment, generation_id: str, error: BaseException
) -> None:
    """Finish a failed invocation's owned runs only after the engine confirms cleanup.

    Covers the generation summary and every managed attempt run whose trial
    outcome was never recorded.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Failed invocation identity.
    :param BaseException error: Authoritative engine failure, without its private message.
    """
    from phasesweep.engine.paths import _experiment_dir, _last_successful_generation_path
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
    # Attempts interrupted before their post-trial callback still own running
    # runs. Trial directories are named by _trial_dir_for.
    delivery = tracking_root(experiment) / generation_id / "delivery"
    pattern = f"*/trial_*__generation_{generation_id}__attempt_*/{BINDING_FILE}"
    for path in sorted(_experiment_dir(experiment).glob(pattern)):
        binding = read_binding(path.parent)
        if (
            binding["run_mode"] == "managed"
            and not (delivery / f"attempt-{binding['attempt_id']}.json").exists()
        ):
            _finish_attempt(experiment, generation_id, binding, status, environment)
    _enqueue(
        experiment,
        generation_id,
        "generation",
        read_record(parent_path),
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
        environment,
        finish=status,
    )


def delivery_status(experiment: Experiment, generation_id: str | None = None) -> dict[str, int]:
    """Read mutable delivery counts without importing the tracking SDK.

    :param Experiment experiment: Local namespace.
    :param str | None generation_id: Optional generation filter.
    :return dict[str, int]: Path-free pending and delivered counts.
    """
    counts = {"pending": 0, "delivered": 0}
    for path in tracking_root(experiment).glob(_delivery_pattern(generation_id)):
        counts[read_record(path)["state"]] += 1
    return counts


def _delivery_pattern(generation_id: str | None) -> str:
    """Select one generation's delivery records, or every generation's.

    :param str | None generation_id: Generation filter, or ``None`` for all.
    :return str: Glob relative to the tracking root.
    """
    return f"{generation_id or '*'}/delivery/*.json"


def sync_tracking(experiment: Experiment, generation_id: str | None = None) -> dict[str, int]:
    """Retry result projections, without selecting or training.

    A published generation whose winner records were never saved, because a
    write failed or the controller died after publication, has them rebuilt
    from its committed artifacts first.

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
        for parent in sorted(root.glob(f"{generation_id or '*'}/generation-binding.json")):
            if (parent.parent / "delivery" / "generation.json").exists() or not _published(
                experiment, parent.parent.name
            ):
                continue
            if read_record(root / "source.json")["source"] != experiment.mlflow.evidence_identity():
                raise ValueError("Tracking delivery belongs to a different target or generation.")
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
        environment = _trainer_environment(experiment, experiment.phases[0].name)
        for path in paths:
            _deliver(experiment, path, environment)
    return delivery_status(experiment, generation_id)
