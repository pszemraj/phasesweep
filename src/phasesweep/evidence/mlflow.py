"""Native MLflow queries, deterministic history selection, and offline evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from phasesweep.evidence.models import (
    Gate,
    MlflowExtractor,
    MlflowRequiredGate,
    WandbExtractor,
    WandbSummaryRequiredGate,
)
from phasesweep.runtime.json import dump_json, json_object, json_text_map

if TYPE_CHECKING:
    from phasesweep.config.models import Experiment
    from phasesweep.tracking.records import MlflowCapture, MlflowQuery, Observation


def observation_id(spec: MlflowExtractor) -> str:
    """Name an exact metric selection within the shared capture.

    :param MlflowExtractor spec: Metric key and selector.
    :return str: Canonical selection identity.
    """
    return json.dumps([spec.metric_key, spec.select, spec.step], separators=(",", ":"))


def compile_query(experiment: Experiment, gates: list[Gate]) -> MlflowQuery | None:
    """Compile one MLflow capture and enforce one tracking backend per sweep.

    :param Experiment experiment: Parsed experiment, without live SDK state.
    :param list[Gate] gates: Gates for the current phase.
    :raises ValueError: Targets or evidence backends conflict.
    :return MlflowQuery | None: Shared query, or no MLflow integration.
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
        c.name: dump_json(c.extractor)
        for c in experiment.constraints
        if isinstance(c.extractor, MlflowExtractor)
    }
    required = [g for g in gates if isinstance(g, MlflowRequiredGate)]
    selections = [
        objective,
        *(c.extractor for c in experiment.constraints if isinstance(c.extractor, MlflowExtractor)),
    ]
    selections.extend(
        MlflowExtractor(type="mlflow", metric_key=k, select="last")
        for g in required
        for k in g.metrics
    )
    return {
        "revision": 1,
        "objective": dump_json(objective),
        "constraints": constraints,
        "gates": [dump_json(g) for g in required],
        "selections": {observation_id(s): dump_json(s) for s in selections},
    }


def select_observation(
    history: Sequence[Mapping[str, object]], spec: MlflowExtractor
) -> Observation:
    """Reduce an unordered, complete native history without using summary ordering.

    :param Sequence[Mapping[str, object]] history: Public MLflow metric records.
    :param MlflowExtractor spec: Explicit within-run selection policy.
    :raises ValueError: Missing, ambiguous, scoped, or invalid observations.
    :return Observation: Selected finite value, native step, and timestamp.
    """
    points: dict[tuple[int, int], Observation] = {}
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


def parse_query(value: object) -> MlflowQuery:
    """Narrow a decoded query to the shape :func:`compile_query` produces.

    :param object value: Decoded shared query.
    :raises KeyError: A query member is absent.
    :raises ValueError: The query is not a current-revision object, or a member has the
        wrong type.
    :return MlflowQuery: The narrowed query.
    """
    if not isinstance(value, dict) or value.get("revision") != 1:
        raise ValueError("Invalid MLflow query or selected observations.")
    query = json_object(value)
    gates = query["gates"]
    if not isinstance(gates, list):
        raise ValueError("Invalid MLflow query gates.")
    return {
        "revision": 1,
        "objective": json_object(query["objective"]),
        "constraints": {k: json_object(v) for k, v in json_object(query["constraints"]).items()},
        "gates": [json_object(gate) for gate in gates],
        "selections": {k: json_object(v) for k, v in json_object(query["selections"]).items()},
    }


def validate_capture(capture: Mapping[str, object]) -> MlflowCapture:
    """Validate the retained MLflow subset without constructing a client.

    :param Mapping[str, object] capture: Accepted remote evidence.
    :raises ValueError: Capture structure or required evidence disagrees.
    :return MlflowCapture: The narrowed evidence.
    """
    from phasesweep.tracking.records import validate_binding

    if capture.get("kind") != "mlflow" or capture.get("run_state") != "FINISHED":
        raise ValueError("MLflow capture must name a finished run.")
    binding = validate_binding(capture.get("binding"))
    query = parse_query(capture.get("query"))
    raw_observations = capture.get("observations")
    if not isinstance(raw_observations, dict):
        raise ValueError("Invalid MLflow query or selected observations.")
    if set(raw_observations) != set(query["selections"]):
        raise ValueError("MLflow selections differ from the captured query.")
    observations: dict[str, Observation] = {}
    for identity, specification in query["selections"].items():
        spec = MlflowExtractor.model_validate(specification)
        point = raw_observations[identity]
        if identity != observation_id(spec) or not isinstance(point, dict):
            raise ValueError("Invalid selected MLflow observation.")
        selected = select_observation([point], spec)
        if selected != point:
            raise ValueError("Invalid selected MLflow observation.")
        observations[identity] = selected
    retrieved_at = capture.get("retrieved_at")
    if not isinstance(retrieved_at, str) or not retrieved_at:
        raise ValueError("Missing MLflow retrieval timestamp.")
    params, tags = json_text_map(capture.get("params")), json_text_map(capture.get("tags"))
    if params is None or tags is None:
        raise ValueError("Invalid MLflow metadata subset.")
    for gate in (MlflowRequiredGate.model_validate(g) for g in query["gates"]):
        if any(key not in params for key in gate.params) or any(
            key not in tags for key in gate.tags
        ):
            raise ValueError("MLflow capture lacks required metadata.")
    return {
        "kind": "mlflow",
        "binding": binding,
        "run_state": "FINISHED",
        "query": query,
        "observations": observations,
        "retrieved_at": retrieved_at,
        "params": params,
        "tags": tags,
    }
