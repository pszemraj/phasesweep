"""Supervised public-client operations for native MLflow tracking."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from phasesweep.runtime.files import private_atomic_write_text
from phasesweep.runtime.remote_worker import run_worker
from phasesweep.runtime.time import utc_now_iso


def request_operation(
    request: dict[str, Any],
    *,
    operation_id: str,
    directory: Path,
    environment: Mapping[str, str],
    deadline: float,
) -> dict[str, Any]:
    """Execute a bounded SDK operation in a registered process slot.

    :param dict[str, Any] request: Trusted operation and payload.
    :param str operation_id: PhaseSweep process identity, never the MLflow ID.
    :param Path directory: Registered operation or attempt directory.
    :param Mapping[str, str] environment: Composed authentication environment.
    :param float deadline: Absolute monotonic deadline.
    :raises RuntimeError: The worker failed, timed out, or rejected the request.
    :return dict[str, Any]: Successful public-client response.
    """
    result, response, _diagnostic = run_worker(
        request,
        Path(__file__).resolve(),
        "mlflow",
        statuses=frozenset({"ok", "error"}),
        run_id=operation_id,
        trial_dir=directory,
        environment=environment,
        deadline=deadline,
    )
    if result.timed_out:
        raise TimeoutError("MLflow operation exceeded its deadline.")
    if result.return_code != 0 or response.get("status") not in {"ok", "error"}:
        raise RuntimeError("MLflow worker failed; inspect its private diagnostic.")
    if response["status"] == "error":
        raise RuntimeError(response["message"])
    return dict(response["result"])


def identity_tags(binding: Mapping[str, Any]) -> dict[str, str]:
    """Return the reserved identity tags shared with trainer-created runs.

    :param Mapping[str, Any] binding: Durable PhaseSweep run identity.
    :return dict[str, str]: Reserved remote identity, without credentials.
    """
    return {
        f"phasesweep.{key}": str(binding[key])
        for key in ("experiment", "phase", "generation_id", "attempt_id", "overrides_sha256")
    }


def verify_remote(run: Any, binding: Mapping[str, Any]) -> None:
    """Require a remote run to retain its exact experiment and attempt tags.

    :param Any run: Public MLflow Run entity.
    :param Mapping[str, Any] binding: Locally recorded identity.
    :raises ValueError: Remote identity contradicts the binding.
    """
    if (
        run.info.run_id != binding["run_id"]
        or run.info.experiment_id != binding["experiment_id"]
        or run.info.lifecycle_stage != "active"
    ):
        raise ValueError("MLflow run does not match the bound experiment/run identity.")
    if any(run.data.tags.get(k) != v for k, v in identity_tags(binding).items()):
        raise ValueError("MLflow run has conflicting PhaseSweep identity tags.")


def _resolve(client: Any, settings: dict[str, Any]) -> str:
    """Resolve an existing, active experiment through the public client.

    :param Any client: Public MLflow client in the supervised worker.
    :param dict[str, Any] settings: Trusted target or operation payload.
    :return str: Existing experiment ID.
    """
    experiment = (
        client.get_experiment(settings["experiment_id"])
        if settings.get("experiment_id")
        else client.get_experiment_by_name(settings["experiment_name"])
    )
    if experiment is None or experiment.lifecycle_stage != "active":
        raise ValueError(
            "MLflow experiment is missing or deleted; provision an existing experiment first."
        )
    return str(experiment.experiment_id)


def _create(client: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Create once or reconcile an uncertain acknowledgment by exact identity.

    :param Any client: Public MLflow client in the supervised worker.
    :param dict[str, Any] request: Trusted target or operation payload.
    :return dict[str, Any]: Verified remote run binding.
    """
    binding = dict(request["binding"])
    binding["experiment_id"] = _resolve(client, request["settings"])
    tags = identity_tags(binding)
    query = " AND ".join(
        f"tags.`{key}` = '{tags[key]}'"
        for key in ("phasesweep.attempt_id", "phasesweep.generation_id")
    )
    matches = []
    token = None
    while True:
        page = client.search_runs([binding["experiment_id"]], filter_string=query, page_token=token)
        matches.extend(page)
        token = page.token
        if not token:
            break
    if len(matches) > 1:
        raise ValueError("Multiple MLflow runs match the creation intent; binding is ambiguous.")
    if matches:
        run = matches[0]
    elif request["allow_create"]:
        if binding.get("parent_run_id"):
            tags["mlflow.parentRunId"] = binding["parent_run_id"]
        run = client.create_run(binding["experiment_id"], tags=tags, run_name=request["run_name"])
    else:
        raise ValueError(
            "MLflow creation acknowledgment is unresolved; no matching run was found. Refusing a duplicate create."
        )
    binding["run_id"] = run.info.run_id
    verify_remote(run, binding)
    return binding


def _capture(client: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Fetch finished evidence, sharing complete histories across selectors.

    :param Any client: Public MLflow client in the supervised worker.
    :param dict[str, Any] request: Trusted target or operation payload.
    :return dict[str, Any]: Frozen shared native evidence.
    """
    from phasesweep.evidence.mlflow import select_observation
    from phasesweep.evidence.models import MlflowExtractor

    binding, query = request["binding"], request["query"]
    stop = time.monotonic() + request["settings"]["timeout_seconds"]
    while True:
        run = client.get_run(binding["run_id"])
        verify_remote(run, binding)
        if run.info.status in {"FAILED", "KILLED"}:
            raise ValueError("MLflow run ended unsuccessfully; remote metrics cannot rescue it.")
        if run.info.status == "RUNNING" and binding["run_mode"] == "managed":
            client.set_terminated(binding["run_id"], status="FINISHED")
            run = client.get_run(binding["run_id"])
        if run.info.status == "FINISHED":
            keys = sorted({s["metric_key"] for s in query["selections"].values()})
            if all(key in run.data.metrics for key in keys) and all(
                key in getattr(run.data, namespace)
                for g in query["gates"]
                for namespace in ("params", "tags")
                for key in g[namespace]
            ):
                observations: dict[str, Any] = {}
                for key in keys:
                    history = client.get_metric_history(binding["run_id"], key)
                    if not history:
                        break
                    records = [
                        {
                            "value": m.value,
                            "step": m.step,
                            "timestamp": m.timestamp,
                            "model_id": getattr(m, "model_id", None),
                            "dataset_name": getattr(m, "dataset_name", None),
                            "dataset_digest": getattr(m, "dataset_digest", None),
                        }
                        for m in history
                    ]
                    for identity, specification in query["selections"].items():
                        if specification["metric_key"] == key:
                            observations[identity] = select_observation(
                                records, MlflowExtractor.model_validate(specification)
                            )
                else:
                    final = client.get_run(binding["run_id"])
                    verify_remote(final, binding)
                    if final.info.status != "FINISHED":
                        raise ValueError("MLflow run restarted during evidence capture.")
                    metadata = {
                        namespace: {
                            key: getattr(final.data, namespace)[key]
                            for gate in query["gates"]
                            for key in gate[namespace]
                        }
                        for namespace in ("params", "tags")
                    }
                    return {
                        "kind": "mlflow",
                        "binding": binding,
                        "run_state": "FINISHED",
                        "query": query,
                        "observations": observations,
                        "retrieved_at": utc_now_iso(),
                        **metadata,
                    }
        if time.monotonic() >= stop:
            raise TimeoutError(
                "MLflow finished-run evidence is missing or delayed beyond its timeout."
            )
        time.sleep(min(request["settings"]["poll_seconds"], max(0, stop - time.monotonic())))


def _publish(client: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Mirror one retained projection without fluent run reopening.

    :param Any client: Public MLflow client in the supervised worker.
    :param dict[str, Any] request: Trusted target or operation payload.
    :return dict[str, Any]: Empty success response.
    """
    from mlflow.entities import Metric, Param, RunTag

    binding, projection = request["binding"], request["projection"]
    run = client.get_run(binding["run_id"])
    verify_remote(run, binding)
    if request.get("finish") and run.info.status in {"FAILED", "KILLED"}:
        raise ValueError(
            "MLflow generation run ended unsuccessfully; refusing to overwrite its status."
        )
    client.log_batch(
        binding["run_id"],
        metrics=[
            Metric(k, v, projection["timestamp"], 0) for k, v in projection["metrics"].items()
        ],
        params=[Param(k, v) for k, v in projection["params"].items()],
        tags=[RunTag(k, v) for k, v in projection["tags"].items()],
        synchronous=True,
    )
    with TemporaryDirectory(prefix="phasesweep-mlflow-summary-") as directory:
        path = Path(directory) / "summary.json"
        path.write_text(
            json.dumps(projection["summary"], sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        client.log_artifact(binding["run_id"], str(path), artifact_path="phasesweep")
    if request.get("finish") and run.info.status != "FINISHED":
        client.set_terminated(binding["run_id"], status=request["finish"])
    return {}


def _dispatch(request: dict[str, Any]) -> dict[str, Any]:
    """Dispatch trusted controller requests using public MLflow APIs.

    :param dict[str, Any] request: Trusted controller operation.
    :return dict[str, Any]: Successful operation payload.
    """
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=request["settings"]["tracking_uri"])
    action = request["action"]
    if action == "resolve":
        return {"experiment_id": _resolve(client, request["settings"])}
    if action == "create":
        return _create(client, request)
    if action == "capture":
        return _capture(client, request)
    if action == "publish":
        return _publish(client, request)
    raise ValueError("Unknown internal MLflow operation.")


def main(argv: list[str] | None = None) -> int:
    """Run an internal request without leaking SDK error details.

    :param list[str] | None argv: Private request/response paths.
    :return int: Zero after a structured response, 130 on interruption.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    # A timed-out POST can have succeeded. Reconciliation, not HTTP replay,
    # decides what to do with run creation.
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
    try:
        response = {"status": "ok", "result": _dispatch(request)}
    except ImportError:
        response = {
            "status": "error",
            "message": "MLflow requires phasesweep[mlflow] in the controller environment.",
        }
    except (ValueError, TimeoutError) as exc:
        response = {"status": "error", "message": str(exc)}
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        from mlflow.exceptions import MlflowException
        from requests.exceptions import RequestException

        if not isinstance(exc, MlflowException | RequestException | OSError):
            raise
        response = {
            "status": "error",
            "message": f"MLflow {request['action']} failed ({type(exc).__name__}); check tracking access and credentials.",
        }
    private_atomic_write_text(args.response, json.dumps(response, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
