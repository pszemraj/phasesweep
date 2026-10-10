"""Native MLflow queries, deterministic history selection, and offline evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from phasesweep.evidence.models import (
    Gate,
    MlflowExtractor,
    MlflowRequiredGate,
    WandbExtractor,
    WandbSummaryRequiredGate,
)

if TYPE_CHECKING:
    from phasesweep.config.models import Experiment


def observation_id(spec: MlflowExtractor) -> str:
    """Name an exact metric selection within the shared capture.

    :param MlflowExtractor spec: Metric key and selector.
    :return str: Canonical selection identity.
    """
    return json.dumps([spec.metric_key, spec.select, spec.step], separators=(",", ":"))


def compile_query(experiment: Experiment, gates: list[Gate]) -> dict[str, Any] | None:
    """Compile one MLflow capture and enforce one tracking backend per sweep.

    :param Experiment experiment: Parsed experiment, without live SDK state.
    :param list[Gate] gates: Gates for the current phase.
    :raises ValueError: Targets or evidence backends conflict.
    :return dict[str, Any] | None: Shared query, or no MLflow integration.
    """
    objective = experiment.metric.extractor
    consumers = [objective, *(c.extractor for c in experiment.constraints), *gates]
    has_mlflow = any(isinstance(c, MlflowExtractor | MlflowRequiredGate) for c in consumers)
    if experiment.mlflow is None:
        if has_mlflow:
            raise ValueError("MLflow evidence requires the top-level mlflow settings.")
        return None
    if not isinstance(objective, MlflowExtractor):
        raise ValueError(
            "mlflow settings require an MLflow objective; standalone or cross-backend publishing is unsupported."
        )
    configured = experiment.mlflow
    expected_environment = {
        "MLFLOW_TRACKING_URI": configured.tracking_uri,
        "MLFLOW_EXPERIMENT_ID": configured.experiment_id,
        "MLFLOW_EXPERIMENT_NAME": configured.experiment_name,
    }
    for key, value in experiment.env.items():
        if key == "MLFLOW_RUN_ID" or (
            key in expected_environment and value != expected_environment[key]
        ):
            raise ValueError(f"env.{key} conflicts with PhaseSweep's MLflow run binding.")
    if any(isinstance(c, WandbExtractor | WandbSummaryRequiredGate) for c in consumers):
        raise ValueError(
            "A sweep must use one remote evidence backend; W&B and MLflow cannot be mixed."
        )
    constraints = {
        c.name: c.extractor.model_dump(mode="json")
        for c in experiment.constraints
        if isinstance(c.extractor, MlflowExtractor)
    }
    required = [g.model_dump(mode="json") for g in gates if isinstance(g, MlflowRequiredGate)]
    selections = [
        objective,
        *(c.extractor for c in experiment.constraints if isinstance(c.extractor, MlflowExtractor)),
    ]
    selections.extend(
        MlflowExtractor(type="mlflow", metric_key=k, select="last")
        for g in required
        for k in g["metrics"]
    )
    return {
        "revision": 1,
        "objective": objective.model_dump(mode="json"),
        "constraints": constraints,
        "gates": required,
        "selections": {observation_id(s): s.model_dump(mode="json") for s in selections},
    }


def select_observation(
    history: Sequence[Mapping[str, Any]], spec: MlflowExtractor
) -> dict[str, Any]:
    """Reduce an unordered, complete native history without using summary ordering.

    :param Sequence[Mapping[str, Any]] history: Public MLflow metric records.
    :param MlflowExtractor spec: Explicit within-run selection policy.
    :raises ValueError: Missing, ambiguous, scoped, or invalid observations.
    :return dict[str, Any]: Selected finite value, native step, and timestamp.
    """
    points: dict[tuple[int, int], dict[str, Any]] = {}
    for record in history:
        if any(record.get(k) is not None for k in ("model_id", "dataset_name", "dataset_digest")):
            raise ValueError("Model/dataset-scoped MLflow metrics are unsupported.")
        step, timestamp, value = record.get("step"), record.get("timestamp"), record.get("value")
        if type(step) is not int or step < 0 or type(timestamp) is not int or timestamp < 0:
            raise ValueError("MLflow history requires nonnegative integer steps and timestamps.")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("MLflow metric values must be numbers, not booleans.")
        position = (step, timestamp)
        if position in points and points[position]["value"] != value:
            raise ValueError("MLflow history has conflicting values at the same step/timestamp.")
        points[position] = {"value": float(value), "step": step, "timestamp": timestamp}
    candidates = list(points.values())
    if spec.select == "step":
        candidates = [p for p in candidates if p["step"] == spec.step]
    if not candidates:
        raise ValueError(f"MLflow metric {spec.metric_key!r} has no requested observation.")
    if spec.select in {"last", "step"}:
        selected = max(candidates, key=lambda p: (p["step"], p["timestamp"]))
    else:
        if any(not math.isfinite(p["value"]) for p in candidates):
            raise ValueError("MLflow min/max history contains non-finite values.")
        selected = min(
            candidates,
            key=lambda p: (
                p["value"] if spec.select == "min" else -p["value"],
                p["step"],
                p["timestamp"],
            ),
        )
    if not math.isfinite(selected["value"]):
        raise ValueError("Selected MLflow observation is non-finite.")
    return selected


def validate_capture(capture: Mapping[str, Any]) -> None:
    """Validate the retained MLflow subset without constructing a client.

    :param Mapping[str, Any] capture: Accepted remote evidence.
    :raises ValueError: Capture structure or required evidence disagrees.
    """
    from phasesweep.tracking.session import validate_binding

    if capture.get("kind") != "mlflow" or capture.get("run_state") != "FINISHED":
        raise ValueError("MLflow capture must name a finished run.")
    binding = capture.get("binding")
    validate_binding(binding)
    query, observations = capture.get("query"), capture.get("observations")
    if (
        not isinstance(query, dict)
        or query.get("revision") != 1
        or not isinstance(observations, dict)
    ):
        raise ValueError("Invalid MLflow query or selected observations.")
    if set(observations) != set(query["selections"]):
        raise ValueError("MLflow selections differ from the captured query.")
    for identity, specification in query["selections"].items():
        spec = MlflowExtractor.model_validate(specification)
        point = observations[identity]
        if identity != observation_id(spec) or select_observation([point], spec) != point:
            raise ValueError("Invalid selected MLflow observation.")
    if not isinstance(capture.get("retrieved_at"), str) or not capture["retrieved_at"]:
        raise ValueError("Missing MLflow retrieval timestamp.")
    for namespace in ("params", "tags"):
        metadata = capture.get(namespace)
        if not isinstance(metadata, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()
        ):
            raise ValueError("Invalid MLflow metadata subset.")
        if any(key not in metadata for gate in query["gates"] for key in gate[namespace]):
            raise ValueError("MLflow capture lacks required metadata.")
