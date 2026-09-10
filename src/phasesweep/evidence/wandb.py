"""W&B polling helpers."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast


@dataclass(frozen=True)
class WandbPollTimeout(TimeoutError):
    """Raised when a W&B run summary is not ready before the poll deadline."""

    run_id: str
    timeout_seconds: float
    last_error: Exception | None = None


@dataclass(frozen=True)
class WandbRunTerminalError(RuntimeError):
    """Raised when the attempt's W&B run terminates unsuccessfully."""

    run_id: str
    state: str


@dataclass(frozen=True)
class WandbSetupError(RuntimeError):
    """Raised for permanent W&B setup or authentication failures.

    A non-transport failure to construct the first client is classified this
    way because bad credentials or broken settings will not improve by retrying.
    W&B verifies credentials over the network during construction, so connection
    and timeout failures — including ones wrapped by ``AuthenticationError`` —
    remain polling errors and are retried within the existing deadline.
    HTTP 401 and 403 run-lookup failures are also setup errors.
    """

    run_id: str
    cause: str


def _is_transient_transport_error(exc: Exception) -> bool:
    """Return whether the active exception chain contains a transport failure.

    Follow explicit causes and unsuppressed implicit contexts. W&B suppresses
    a sidecar timeout when it raises ``WandbApiFailedError``, so recognize its
    no-response and timeout status markers directly.

    :param Exception exc: W&B client-construction failure to classify.
    :return bool: Whether the failure is a connection or timeout error.
    """
    try:
        from requests.exceptions import ConnectionError as RequestsConnectionError
        from requests.exceptions import Timeout as RequestsTimeout
    except ImportError:  # pragma: no cover - installed W&B depends on requests
        request_errors: tuple[type[BaseException], ...] = ()
    else:
        request_errors = (RequestsConnectionError, RequestsTimeout)

    try:
        from wandb.errors import CommError
    except ImportError:  # pragma: no cover - W&B is optional
        comm_errors: tuple[type[BaseException], ...] = ()
    else:
        comm_errors = (CommError,)

    try:
        from wandb.sdk.lib.service.service_connection import WandbApiFailedError
    except ImportError:  # Older supported SDKs use requests errors inside CommError.
        service_api_errors: tuple[type[BaseException], ...] = ()
    else:
        service_api_errors = (WandbApiFailedError,)

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (ConnectionError, TimeoutError, *request_errors),
        ):
            return True
        if isinstance(current, service_api_errors):
            response = cast(Any, current).response
            status = response.http_status if response is not None else None
            if status in (None, 0, 408, 504):
                return True
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
        elif isinstance(current, comm_errors):
            current = cast(Any, current).exc
        else:
            current = None
    return False


def _is_nonretryable_authorization_error(exc: Exception) -> bool:
    """Return whether a W&B API error reports an HTTP authentication denial.

    Public API calls wrap the underlying service or requests error in ``CommError``.
    Status 404 remains retriable because a newly created run may not be visible yet.

    :param Exception exc: W&B public-API lookup failure.
    :return bool: Whether the error has a 401 or 403 response status.
    """
    try:
        from requests.exceptions import HTTPError
        from wandb.errors import CommError
    except ImportError:  # pragma: no cover - installed W&B depends on requests
        return False

    try:
        from wandb.sdk.lib.service.service_connection import WandbApiFailedError
    except ImportError:  # W&B 0.16 has CommError but no service API error class.
        service_api_errors: tuple[type[BaseException], ...] = ()
    else:
        service_api_errors = (WandbApiFailedError,)

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status: int | None = None
        if isinstance(current, service_api_errors):
            response = cast(Any, current).response
            status = response.http_status if response is not None else None
        elif isinstance(current, HTTPError):
            response = current.response
            status = response.status_code if response is not None else None
        if status in (401, 403):
            return True
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
        elif isinstance(current, CommError) and current.exc is not None:
            current = current.exc
        else:
            current = None
    return False


def poll_wandb_summary(
    *,
    base_url: str,
    entity: str,
    project: str,
    run_id: str,
    trial_dir: Path,
    poll_seconds: float,
    timeout_seconds: float,
    required_keys: Iterable[str] = (),
    wait_for_keys: bool = True,
) -> dict[str, Any]:
    """Poll W&B after the trial's previous subprocess has been confirmed gone.

    The worker takes over the trial's durable process identity and lifecycle,
    so normal attempt recovery can find it if polling is interrupted.

    :param str base_url: Explicit W&B API endpoint for this evidence source.
    :param str entity: W&B entity or team name.
    :param str project: W&B project name.
    :param str run_id: Immutable W&B run id assigned to this trial attempt.
    :param Path trial_dir: Existing trial directory holding recovery records.
    :param float poll_seconds: Delay between polling attempts.
    :param float timeout_seconds: Maximum budget for worker startup, SDK
        initialization, requests, and retries. Process cleanup may finish afterward.
    :param Iterable[str] required_keys: Summary keys that must be present.
    :param bool wait_for_keys: Whether to wait for all required keys before returning.
    :raises WandbSetupError: If the first API client fails for a non-transport
        reason such as bad credentials or settings, or a run lookup reports HTTP
        401 or 403. Connection and timeout failures are retried within the polling
        budget.
    :raises WandbRunTerminalError: If the run crashes, fails, is killed, or is preempted.
    :raises WandbPollTimeout: If the run summary is not ready before timeout.
    :raises UnsafeProcessCleanupError: If the worker's cleanup is uncertain.
    :raises RuntimeError: If the polling worker fails unexpectedly.
    :return dict[str, Any]: Terminal run summary values.
    """
    from phasesweep.errors import UnsafeProcessCleanupError
    from phasesweep.runtime.process import PROCESS_IDENTITY_FILE, run_supervised

    deadline = time.monotonic() + timeout_seconds
    if timeout_seconds <= 0:
        raise WandbPollTimeout(run_id, timeout_seconds)
    with TemporaryDirectory(prefix="phasesweep-wandb-") as directory:
        worker_dir = Path(directory)
        request_path = worker_dir / "request.json"
        response_path = worker_dir / "response.json"
        request_path.write_text(
            json.dumps(
                {
                    "base_url": base_url,
                    "entity": entity,
                    "project": project,
                    "run_id": run_id,
                    "poll_seconds": poll_seconds,
                    "timeout_seconds": timeout_seconds,
                    "required_keys": list(required_keys),
                    "wait_for_keys": wait_for_keys,
                    "deadline": deadline,
                }
            ),
            encoding="utf-8",
        )
        with (
            (worker_dir / "stdout.log").open("w", encoding="utf-8") as stdout,
            (worker_dir / "stderr.log").open("w", encoding="utf-8") as stderr,
        ):
            # The preceding subprocess is already gone. Remove its identity
            # before launching so a failed worker-identity write cannot leave
            # recovery pointing at that earlier, safely exited process.
            (trial_dir / PROCESS_IDENTITY_FILE).unlink(missing_ok=True)
            # -P keeps this file's directory from shadowing the installed wandb package.
            result = run_supervised(
                shlex.join(
                    [
                        sys.executable,
                        "-P",
                        str(Path(__file__).resolve()),
                        str(request_path),
                        str(response_path),
                    ]
                ),
                env=dict(os.environ),
                stdout=stdout,
                stderr=stderr,
                timeout=max(0.0, deadline - time.monotonic()),
                trial_dir=trial_dir,
                attempt_id=run_id,
            )
        if not result.cleanup_confirmed:
            raise UnsafeProcessCleanupError(
                f"W&B polling worker cleanup could not be confirmed for {run_id!r}. "
                f"Recovery records remain in {trial_dir}."
            )
        if result.timed_out:
            stderr_text = (worker_dir / "stderr.log").read_text(encoding="utf-8").strip()
            response_diagnostic = ""
            if response_path.is_file():
                response_text = response_path.read_text(encoding="utf-8").strip()
                if response_text:
                    try:
                        response = json.loads(response_text)
                    except json.JSONDecodeError:
                        response_diagnostic = response_text
                    else:
                        cause = response.get("cause")
                        response_diagnostic = cause if isinstance(cause, str) else ""
            diagnostic = response_diagnostic or stderr_text
            last_error = RuntimeError(diagnostic) if diagnostic else None
            raise WandbPollTimeout(run_id, timeout_seconds, last_error)
        if result.return_code != 0 or result.failure_reason is not None:
            diagnostic = (worker_dir / "stderr.log").read_text(encoding="utf-8").strip()
            raise RuntimeError(
                f"W&B polling worker failed for {run_id!r}: {result.failure_reason or diagnostic}"
            )
        response = json.loads(response_path.read_text(encoding="utf-8"))

    if response["status"] == "import_error":
        raise ImportError(response["cause"])
    if response["status"] == "setup_error":
        raise WandbSetupError(run_id, response["cause"])
    if response["status"] == "terminal_error":
        raise WandbRunTerminalError(run_id, response["state"])
    if response["status"] == "timeout":
        last_error = RuntimeError(response["cause"]) if response["cause"] is not None else None
        raise WandbPollTimeout(run_id, timeout_seconds, last_error)
    if time.monotonic() >= deadline:
        raise WandbPollTimeout(run_id, timeout_seconds)
    return response["summary"]


def _poll_wandb_summary(
    *,
    base_url: str,
    entity: str,
    project: str,
    run_id: str,
    poll_seconds: float,
    timeout_seconds: float,
    required_keys: Iterable[str] = (),
    wait_for_keys: bool = True,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Run the polling loop inside the cancellable worker.

    :param str base_url: Explicit W&B API endpoint.
    :param str entity: W&B entity or team name.
    :param str project: W&B project name.
    :param str run_id: Immutable W&B run id for this attempt.
    :param float poll_seconds: Delay between polling attempts.
    :param float timeout_seconds: Original budget, used in timeout diagnostics.
    :param Iterable[str] required_keys: Summary keys required before returning.
    :param bool wait_for_keys: Whether to wait for every required key.
    :param float | None deadline: Parent's monotonic deadline; defaults to a new budget.
    :return dict[str, Any]: Finished run summary received before the deadline.
    """
    from wandb.apis.public import Api  # type: ignore[import-not-found]

    path = f"{entity}/{project}/{run_id}"
    if deadline is None:
        deadline = time.monotonic() + timeout_seconds
    last_err: Exception | None = None
    required = tuple(required_keys)
    api_constructed = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        # W&B's integer HTTP timeout does not bound SDK retries. The parent
        # supervises the whole worker against the deadline. Constructing per iteration
        # means construction failures happen mid-poll too; the SDK's
        # constructor performs a network round-trip, so connection and timeout
        # failures must consume the polling budget even on the first attempt.
        # Other first-construction failures still identify deterministic setup
        # problems; after one successful construction, every failure is retried.
        try:
            api = Api(
                overrides={"base_url": base_url},
                timeout=max(1, ceil(remaining)),
            )
        except Exception as exc:  # noqa: BLE001 - classified into the typed error model
            if not api_constructed and not _is_transient_transport_error(exc):
                raise WandbSetupError(run_id, str(exc)) from exc
            last_err = exc
            # The parent may terminate this worker before ``main`` serializes ``last_err``.
            print(str(exc), file=sys.stderr, flush=True)
        else:
            api_constructed = True
            if time.monotonic() >= deadline:
                break
            try:
                run = api.run(path)
                if run.state in {"crashed", "failed", "killed", "preempted"}:
                    raise WandbRunTerminalError(run_id, run.state)
                if run.state == "finished":
                    summary = run.summary_metrics
                    if not wait_for_keys or all(key in summary for key in required):
                        if time.monotonic() >= deadline:
                            break
                        return summary
            except WandbRunTerminalError:
                raise
            except Exception as exc:  # noqa: BLE001
                if _is_nonretryable_authorization_error(exc):
                    raise WandbSetupError(run_id, str(exc)) from exc
                last_err = exc
                # The parent may terminate this worker before ``main`` serializes ``last_err``.
                print(str(exc), file=sys.stderr, flush=True)
        sleep_seconds = max(0.0, deadline - time.monotonic())
        time.sleep(min(poll_seconds, sleep_seconds))
    raise WandbPollTimeout(run_id, timeout_seconds, last_err)


def main(argv: list[str] | None = None) -> int:
    """Run one internal polling request and write its result for the parent.

    :param list[str] | None argv: Arguments, defaulting to the process arguments.
    :return int: Zero after writing a summary or an expected polling error.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("request", type=Path, help="Internal polling request JSON")
    parser.add_argument("response", type=Path, help="Internal polling response JSON")
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    response: dict[str, Any]
    try:
        response = {"status": "summary", "summary": _poll_wandb_summary(**request)}
    except ImportError as exc:
        response = {"status": "import_error", "cause": str(exc)}
    except WandbSetupError as exc:
        response = {"status": "setup_error", "cause": exc.cause}
    except WandbRunTerminalError as exc:
        response = {"status": "terminal_error", "state": exc.state}
    except WandbPollTimeout as exc:
        response = {
            "status": "timeout",
            "cause": str(exc.last_error) if exc.last_error is not None else None,
        }
    args.response.write_text(json.dumps(response), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
