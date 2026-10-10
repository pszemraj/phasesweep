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
                "binding": record["binding"],
                "projection": record["projection"],
                "finish": record["finish"],
            },
            environment,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, UnsafeProcessCleanupError):
            raise
        log.warning(
            "MLflow result upload pending for generation %s (%s). Retry with sync-tracking.",
            record["generation_id"],
            str(exc),
        )
        return False
    write_record(path, {**record, "state": "delivered"})
    return True


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
    """Save delivery intent before attempting any result upload.

    :param Experiment experiment: Configured target.
    :param str generation_id: Producing generation.
    :param str name: Stable projection identity.
    :param dict[str, Any] binding: Verified target run.
    :param dict[str, Any] projection: Frozen upload payload.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str | None finish: Optional generation-summary terminal status.
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


def publish_generation(
    experiment: Experiment, generation_id: str, winners: dict[str, Winner]
) -> None:
    """Publish winner references after the local generation has committed.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Committed generation.
    :param dict[str, Winner] winners: Accepted phase winners.
    """
    from phasesweep.engine.trial import _trainer_environment

    parent_path = tracking_root(experiment) / generation_id / "generation-binding.json"
    if not parent_path.exists():
        return  # No-work replay never imports the SDK or creates remote runs.
    parent = read_record(parent_path)
    environment = _trainer_environment(experiment, experiment.phases[0].name)
    references = {
        name: {
            "run_id": winner.objective_provenance["remote_capture"]["binding"]["run_id"],
            "metric": winner.metric,
            "effective_overrides": winner.effective_overrides,
        }
        for name, winner in winners.items()
        if winner.objective_provenance is not None
    }
    for phase in experiment.phases:
        winner = winners[phase.name]
        assert winner.objective_provenance is not None
        binding = winner.objective_provenance["remote_capture"]["binding"]
        tags = {
            "phasesweep.winner": "true",
            f"phasesweep.winner_generation.{generation_id}": phase.name,
            **{
                f"phasesweep.inherits.{name}": references[name]["run_id"] for name in phase.inherits
            },
        }
        _enqueue(
            experiment,
            generation_id,
            f"winner-{phase.name}",
            binding,
            _projection(
                metrics={},
                params={},
                tags=tags,
                summary={"phase": phase.name, **references[phase.name]},
            ),
            environment,
        )
    _enqueue(
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
        environment,
        finish="FINISHED",
    )


def publish_failed_generation(
    experiment: Experiment, generation_id: str, error: BaseException
) -> None:
    """Mirror a failed invocation only after the engine confirms process cleanup.

    :param Experiment experiment: Configured experiment.
    :param str generation_id: Failed invocation identity.
    :param BaseException error: Authoritative engine failure, without its private message.
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
        _trainer_environment(experiment, experiment.phases[0].name),
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
    """Retry only retained result projections, without selecting or training.

    :param Experiment experiment: Matching source experiment configuration.
    :param str | None generation_id: Generation whose deliveries should be
        retried, or ``None`` for every generation's retained deliveries.
    :raises ValueError: No deliveries exist, or the target or generation does
        not match the saved records.
    :raises PhaseSweepError: The MLflow SDK is not installed.
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
        paths = sorted(tracking_root(experiment).glob(_delivery_pattern(generation_id)))
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
