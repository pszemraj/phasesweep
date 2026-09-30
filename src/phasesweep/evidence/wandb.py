"""Read finished W&B evidence in the existing supervised attempt slot.

A trial's summary capture is mandatory evidence; locating its accepted
objective in run history is advisory and runs as a separate request.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shlex
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from phasesweep.runtime.process import ProcessResult

_HistoryRead = Literal["settled", "lagging", "retry"]
"""How one history read ended: an answer, history behind the summary, or a
transient failure worth another read."""

_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_WORKER_STDERR_LOG = "wandb-worker.stderr.log"
# How long an accepted objective waits for run history to catch up with its
# summary before recording no position; the position is advisory, so waiting
# must not hold every trial for the whole timeout. Reading history that has
# caught up may use the rest of the read's deadline.
_HISTORY_SETTLE_SECONDS = 30.0
# Longest wait between history reads, however slowly the summary is polled:
# the partial reads just after a run finishes cleared within about a second.
_HISTORY_RETRY_SECONDS = 2.0
# The server's silent cap on the rows of one history request, both a sampled
# request and, until the run's history is exported, a scan page: a longer
# answer comes back as a sample. A scan page of this many steps holds at most
# this many rows, so it is never sampled.
_HISTORY_SAMPLES = 10_000


# Not frozen: unwinding sets an exception's __traceback__ (contextlib does so
# explicitly), which a frozen dataclass refuses with FrozenInstanceError.
# eq=False keeps the identity equality and hashing every exception has.
@dataclass(eq=False)
class WandbPollTimeout(TimeoutError):
    """The complete worker exhausted its summary visibility budget."""

    run_id: str
    timeout_seconds: float
    last_error: Exception | None = None


@dataclass(eq=False)
class WandbRunTerminalError(RuntimeError):
    """The expected run terminated unsuccessfully."""

    run_id: str
    state: str


@dataclass(eq=False)
class WandbSetupError(RuntimeError):
    """W&B rejected authentication or client configuration permanently."""

    run_id: str
    cause: str


def require_wandb_sdk() -> None:
    """Check the optional SDK without authenticating or making a network request.

    :raises PhaseSweepError: The W&B public API cannot be imported.
    """
    try:
        importlib.import_module("wandb.apis.public")
    except ImportError as exc:
        from phasesweep.errors import OperatorAction, PhaseSweepError

        raise PhaseSweepError(
            "W&B evidence requires the optional SDK in the PhaseSweep environment. "
            'Install this distribution with its wandb extra: python -m pip install "phasesweep[wandb]".',
            action=OperatorAction.FIX_CONFIG,
        ) from exc


def _error_chain(exc: BaseException) -> Iterable[BaseException]:
    """Walk active SDK causes, including the public CommError wrapper.

    :param BaseException exc: Outermost error raised by the SDK or transport.
    :return Iterable[BaseException]: ``exc`` and each distinct cause, context, or
        wrapped ``exc`` attribute behind it, outermost first.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
        else:
            wrapped = getattr(current, "exc", None)
            current = wrapped if isinstance(wrapped, BaseException) else None


def _http_status(exc: BaseException) -> int | None:
    """Read requests or SDK transport response status without private imports.

    :param BaseException exc: Error that may carry a transport ``response``.
    :return int | None: HTTP status code, or ``None`` when none is attached.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(response, "http_status", None)
    return status if isinstance(status, int) else None


def _is_nonretryable_authorization_error(exc: Exception) -> bool:
    """Recognize permanent HTTP authorization denial through SDK wrappers.

    :param Exception exc: Error raised while creating the client or reading the run.
    :return bool: Whether any error in the chain carries HTTP 401 or 403.
    """
    return any(_http_status(error) in {401, 403} for error in _error_chain(exc))


def _is_retryable_setup_error(exc: Exception) -> bool:
    """Recognize temporary transport failures using public SDK/request errors.

    :param Exception exc: Error raised while creating the client, reading the
        run, or reading its history.
    :return bool: Whether the failure is transient and worth another attempt.
    """
    import requests.exceptions as request_errors

    if _is_nonretryable_authorization_error(exc):
        return False
    for error in _error_chain(exc):
        if _http_status(error) in _RETRYABLE_HTTP_STATUSES or isinstance(
            error,
            (ConnectionError, TimeoutError, request_errors.ConnectionError, request_errors.Timeout),
        ):
            return True
        # The supported SDK can suppress the original sidecar transport cause.
        # Inspect its response protocol rather than importing its private class.
        if (
            type(error).__module__.startswith("wandb.")
            and hasattr(error, "response")
            and _http_status(error) in {None, 0}
        ):
            return True
    return False


def _error_detail(exc: Exception) -> str:
    """Describe error type/status without copying secret-bearing SDK messages.

    :param Exception exc: Error to describe.
    :return str: Exception type name plus any HTTP statuses found in its chain.
    """
    statuses = [str(status) for error in _error_chain(exc) if (status := _http_status(error))]
    suffix = f" (HTTP {', '.join(statuses)})" if statuses else ""
    return f"{type(exc).__name__}{suffix}"


def _poll_worker_environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    """Retain composed credentials and the orchestrator's Python bootstrap.

    :param Mapping[str, str] | None environment: Composed trainer environment, or
        ``None`` for the orchestrator's own environment.
    :return dict[str, str]: Worker environment whose ``PYTHON*`` variables match
        the orchestrator's exactly.
    """
    worker_environment = dict(os.environ if environment is None else environment)
    for name in set(worker_environment) | set(os.environ):
        if name.startswith("PYTHON"):
            if name in os.environ:
                worker_environment[name] = os.environ[name]
            else:
                worker_environment.pop(name, None)
    return worker_environment


def _run_worker(
    request: Mapping[str, Any],
    *,
    statuses: frozenset[str],
    run_id: str,
    trial_dir: Path,
    environment: Mapping[str, str] | None,
    deadline: float,
) -> tuple[ProcessResult, dict[str, Any], Path | None]:
    """Run one supervised worker request in the attempt's process slot.

    :param Mapping[str, Any] request: Worker request, including its ``task``.
    :param frozenset[str] statuses: Response statuses the task can report; any
        other outcome preserves the worker's stderr for diagnosis.
    :param str run_id: Immutable attempt ID, also used by process recovery.
    :param Path trial_dir: Existing attempt directory and durable process slot.
    :param Mapping[str, str] | None environment: Actual composed trainer environment.
    :param float deadline: Absolute ``time.monotonic()`` time the worker is stopped at.
    :return tuple[ProcessResult, dict[str, Any], Path | None]: The worker's
        process result, its response (empty when it wrote none), and the
        preserved stderr path, if any.
    :raises UnsafeProcessCleanupError: Worker cleanup is uncertain.
    """
    from phasesweep.errors import UnsafeProcessCleanupError
    from phasesweep.runtime.files import private_atomic_write_text
    from phasesweep.runtime.process import run_supervised
    from phasesweep.runtime.reaper import PROCESS_IDENTITY_FILE

    diagnostic_path: Path | None = None
    with TemporaryDirectory(prefix="phasesweep-wandb-") as directory:
        worker_dir = Path(directory)
        request_path = worker_dir / "request.json"
        response_path = worker_dir / "response.json"
        stderr_path = worker_dir / "stderr.log"
        private_atomic_write_text(request_path, json.dumps(dict(request)))
        with (
            (worker_dir / "stdout.log").open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            # Never let a previous safe trainer identity hide a failed worker launch.
            (trial_dir / PROCESS_IDENTITY_FILE).unlink(missing_ok=True)
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
                env=_poll_worker_environment(environment),
                stdout=stdout,
                stderr=stderr,
                timeout=max(0.0, deadline - time.monotonic()),
                wallclock_deadline=deadline,
                trial_dir=trial_dir,
                attempt_id=run_id,
            )
        if not result.cleanup_confirmed:
            raise UnsafeProcessCleanupError(
                f"W&B worker cleanup is uncertain for attempt {run_id!r}; recover {trial_dir}."
            )
        response = (
            json.loads(response_path.read_text(encoding="utf-8")) if response_path.is_file() else {}
        )
        if not result.timed_out and (
            result.return_code != 0
            or result.failure_reason is not None
            or response.get("status") not in statuses
        ):
            diagnostic = stderr_path.read_text(encoding="utf-8", errors="replace")
            if diagnostic:
                diagnostic_path = trial_dir / _WORKER_STDERR_LOG
                private_atomic_write_text(
                    diagnostic_path,
                    diagnostic,
                    require_private_dir=False,
                )
    return result, response, diagnostic_path


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
    presence_keys: Iterable[str] = (),
    environment: Mapping[str, str] | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Capture requested evidence after confirmed trainer cleanup.

    :param str base_url: Normalized W&B API endpoint.
    :param str entity: W&B entity that owns the project.
    :param str project: W&B project containing the run.
    :param str run_id: Immutable attempt ID, also used by process recovery.
    :param Path trial_dir: Existing attempt directory and durable process slot.
    :param float poll_seconds: Delay between summary polls.
    :param float timeout_seconds: Visibility budget for the whole capture.
    :param Iterable[str] required_keys: Numeric keys required before accepting a capture.
    :param Iterable[str] presence_keys: Keys whose presence is recorded without their values.
    :param Mapping[str, str] | None environment: Actual composed trainer environment.
    :param float | None deadline: Absolute deadline including worker startup.
    :return dict[str, Any]: Numeric values, present keys, and retrieval time.
    :raises WandbPollTimeout: Startup, SDK work, or visibility exceeded the budget.
    :raises WandbSetupError: Authentication or setup was denied.
    :raises WandbRunTerminalError: The expected run terminated unsuccessfully.
    :raises ValueError: A requested scalar is invalid.
    :raises UnsafeProcessCleanupError: Worker cleanup is uncertain.
    """
    deadline = min(
        time.monotonic() + timeout_seconds, deadline if deadline is not None else math.inf
    )
    if time.monotonic() >= deadline:
        raise WandbPollTimeout(run_id, timeout_seconds)
    result, response, diagnostic_path = _run_worker(
        {
            "task": "summary",
            "base_url": base_url,
            "entity": entity,
            "project": project,
            "run_id": run_id,
            "poll_seconds": poll_seconds,
            "timeout_seconds": timeout_seconds,
            "required_keys": list(required_keys),
            "presence_keys": list(presence_keys),
            "deadline": deadline,
        },
        statuses=frozenset(
            {
                "summary",
                "setup_error",
                "terminal_error",
                "invalid_evidence",
                "import_error",
                "timeout",
            }
        ),
        run_id=run_id,
        trial_dir=trial_dir,
        environment=environment,
        deadline=deadline,
    )
    # Definite operational causes remain definite even if cleanup crossed a deadline.
    if response.get("status") == "setup_error":
        raise WandbSetupError(run_id, response["cause"])
    if response.get("status") == "terminal_error":
        raise WandbRunTerminalError(run_id, response["state"])
    if response.get("status") == "invalid_evidence":
        raise ValueError(response["cause"])
    if response.get("status") == "import_error":
        raise ImportError(
            "W&B SDK is unavailable in the supervised worker; install phasesweep[wandb]."
        )
    # The worker writes a summary response whole, and only before the
    # deadline, so one that exists was captured in time even if the worker
    # was stopped while exiting.
    if response.get("status") != "summary":
        if result.timed_out or response.get("status") == "timeout" or time.monotonic() >= deadline:
            detail = response.get("cause")
            raise WandbPollTimeout(
                run_id, timeout_seconds, RuntimeError(detail) if detail else None
            )
        diagnostic = f" Diagnostic preserved at {diagnostic_path}." if diagnostic_path else ""
        raise RuntimeError(
            f"W&B evidence worker failed for attempt {run_id!r} (exit {result.return_code})."
            f"{diagnostic}"
        )
    capture = response["capture"]
    if not isinstance(capture, dict):
        raise RuntimeError(
            f"W&B evidence worker returned a malformed capture for attempt {run_id!r}."
        )
    return capture


def locate_wandb_position(
    *,
    base_url: str,
    entity: str,
    project: str,
    run_id: str,
    trial_dir: Path,
    poll_seconds: float,
    timeout_seconds: float,
    metric_key: str,
    axis: str,
    value: float,
    environment: Mapping[str, str] | None = None,
    deadline: float | None = None,
) -> int | float | None:
    """Locate where on ``axis`` an accepted objective was logged in run history.

    Callers run this only once the trial's result is accepted: the position
    is advisory, so reading it must never cost the trial its objective or
    the budget its acceptance needs. Every failure, and a worker stopped at
    the deadline, means "no position".

    :param str base_url: Normalized W&B API endpoint.
    :param str entity: W&B entity that owns the project.
    :param str project: W&B project containing the run.
    :param str run_id: Immutable attempt ID, also used by process recovery.
    :param Path trial_dir: Existing attempt directory and durable process slot.
    :param float poll_seconds: Summary poll interval, the delay between
        history reads up to ``_HISTORY_RETRY_SECONDS``.
    :param float timeout_seconds: Budget for the whole read.
    :param str metric_key: Objective summary key.
    :param str axis: History key marking where the metric was measured.
    :param float value: Captured objective value.
    :param Mapping[str, str] | None environment: Actual composed trainer environment.
    :param float | None deadline: Absolute phase/run deadline, if any.
    :return int | float | None: The position, or ``None`` when history never
        settled, no row produced the value, or the read failed.
    :raises UnsafeProcessCleanupError: Worker cleanup is uncertain.
    """
    from phasesweep.evidence.evaluation import progress_position

    deadline = min(
        time.monotonic() + timeout_seconds, deadline if deadline is not None else math.inf
    )
    if time.monotonic() >= deadline:
        return None
    result, response, _ = _run_worker(
        {
            "task": "position",
            "base_url": base_url,
            "entity": entity,
            "project": project,
            "run_id": run_id,
            "poll_seconds": poll_seconds,
            "metric_key": metric_key,
            "axis": axis,
            "value": value,
            "deadline": deadline,
        },
        statuses=frozenset({"position", "import_error"}),
        run_id=run_id,
        trial_dir=trial_dir,
        environment=environment,
        deadline=deadline,
    )
    if result.timed_out or response.get("status") != "position":
        return None
    return progress_position(response.get("value"))


def _history_position(
    run: Any,
    summary: Mapping[str, Any],
    *,
    metric_key: str,
    axis: str,
    value: float,
    deadline: float,
) -> tuple[_HistoryRead, int | float | None]:
    """Locate how far along ``axis`` a finished run's objective was measured.

    A plain summary key holds its series' last logged value: the latest
    history row with that value is where the objective was measured, and a
    value assigned straight to the summary matches no row and has no position.
    The server names an aggregate by its series and aggregation
    (``eval/loss.min`` summarizes the ``eval/loss`` series), so a key the
    history index does not list is looked up as its ``head`` series. An
    aggregate is measured over the whole series, so its position is where the
    series ends: how far the trial trained, not where its best value fell,
    which differs between comparable trials by nature.

    The run's history index says how far history reaches and how many rows
    logged each key, but for a while after a run finishes a read can return
    part of a series without error: only its newest rows, or a sample. History
    is ``lagging`` until the index reaches the summary's own ``_step`` and a
    read returns every row the index counts for the series. The position is
    advisory evidence, so a failed history read never fails anything: a
    transient transport failure or a read still running at ``deadline`` is
    worth a ``retry``, and any other failure is ``settled`` as "no position".
    A ``define_metric`` step metric the trainer logs in a separate ``log()``
    call is filled from its previous value, and an axis missing from some of
    the series' rows keeps the read short of the count, so the axis must be
    logged with the metric.

    :param Any run: Finished public-API run.
    :param Mapping[str, Any] summary: Its decoded summary.
    :param str metric_key: Configured objective summary key.
    :param str axis: History key marking where the metric was measured.
    :param float value: Captured objective value.
    :param float deadline: ``time.monotonic()`` time at which a read gives up.
    :return tuple[_HistoryRead, int | float | None]: How the read ended, and
        the position once ``settled`` (``None`` when no row produced a plain
        key's value).
    """
    from phasesweep.evidence.evaluation import json_float, progress_position

    last_step = progress_position(summary.get("_step"))
    if last_step is None:
        return "settled", None
    head, dot, _ = metric_key.rpartition(".")
    position: int | float | None = None
    try:
        run.load(force=True)
        index = run.history_keys
        if index["lastStep"] < last_step:
            return "lagging", None
        logged_keys = index["keys"]
        if metric_key in logged_keys:
            series = metric_key
        elif dot and head in logged_keys:
            series = head
        else:
            return "settled", None
        count = sum(entry["count"] for entry in logged_keys[series]["typeCounts"])
        # A series under the cap, such as any evaluation series, comes back from
        # one sampled request however long the run; a longer one is scanned in
        # pages of the cap's size in steps, too few rows for the server to
        # sample even before export.
        keys = [series, axis]
        if count < _HISTORY_SAMPLES:
            rows = run.history(keys=keys, samples=_HISTORY_SAMPLES, pandas=False)
        else:
            rows = run.scan_history(keys=keys, page_size=_HISTORY_SAMPLES, use_cache=False)
        read = 0
        for row in rows:
            if time.monotonic() >= deadline:
                return "retry", None
            if series not in row:
                continue
            read += 1
            if series != metric_key:
                position = progress_position(row.get(axis))
                continue
            try:
                logged = json_float(row[series], label=series)
            except ValueError:
                continue
            if logged == value:
                position = progress_position(row.get(axis))
        if read < count:
            return "lagging", None
    except Exception as exc:
        return ("retry" if _is_retryable_setup_error(exc) else "settled"), None
    return "settled", position


def _poll_wandb_summary(
    *,
    base_url: str,
    entity: str,
    project: str,
    run_id: str,
    poll_seconds: float,
    timeout_seconds: float,
    required_keys: Iterable[str] = (),
    presence_keys: Iterable[str] = (),
    deadline: float | None = None,
) -> dict[str, Any]:
    """Poll an exact run using refreshed SDK clients within the parent's deadline.

    :param str base_url: Normalized W&B API endpoint.
    :param str entity: W&B entity that owns the project.
    :param str project: W&B project containing the run.
    :param str run_id: Exact W&B run ID to read.
    :param float poll_seconds: Delay between summary polls.
    :param float timeout_seconds: Budget used when no ``deadline`` is given.
    :param Iterable[str] required_keys: Numeric keys required before accepting a capture.
    :param Iterable[str] presence_keys: Keys whose presence is recorded without their values.
    :param float | None deadline: Absolute ``time.monotonic()`` deadline from the parent.
    :return dict[str, Any]: Numeric values, present keys, and retrieval time.
    :raises WandbPollTimeout: The run did not finish with the required keys in time.
    :raises WandbSetupError: Authentication or setup was denied.
    :raises WandbRunTerminalError: The expected run terminated unsuccessfully.
    :raises ValueError: A requested scalar is invalid or non-finite.
    """
    from wandb.apis.public import Api
    from wandb.errors import AuthenticationError, UsageError

    from phasesweep.evidence.evaluation import json_float
    from phasesweep.runtime.time import utc_now_iso

    deadline = time.monotonic() + timeout_seconds if deadline is None else deadline
    last_error: Exception | None = None
    required = tuple(required_keys)
    while time.monotonic() < deadline:
        try:
            api = Api(
                overrides={"base_url": base_url},
                timeout=max(1, math.ceil(deadline - time.monotonic())),
            )
        except Exception as exc:
            if not _is_retryable_setup_error(exc):
                raise WandbSetupError(run_id, _error_detail(exc)) from exc
            last_error = RuntimeError(_error_detail(exc))
        else:
            if time.monotonic() >= deadline:
                break
            try:
                run = api.run(f"{entity}/{project}/{run_id}")
            except Exception as exc:
                if any(
                    _http_status(error) in {400, 401, 403, 422} for error in _error_chain(exc)
                ) or (
                    isinstance(exc, (AuthenticationError, UsageError))
                    and not _is_retryable_setup_error(exc)
                ):
                    raise WandbSetupError(run_id, _error_detail(exc)) from exc
                last_error = RuntimeError(_error_detail(exc))
            else:
                if time.monotonic() >= deadline:
                    break
                if run.state in {"failed", "crashed", "killed", "preempted"}:
                    raise WandbRunTerminalError(run_id, run.state)
                if run.state == "finished":
                    summary = run.summary_metrics
                    if all(key in summary for key in required):
                        values = {}
                        for key in required:
                            value = json_float(summary[key], label=key)
                            if not math.isfinite(value):
                                raise ValueError(f"W&B metric {key!r} is non-finite.")
                            values[key] = value
                        capture: dict[str, Any] = {
                            "values": values,
                            "present_keys": sorted(key for key in presence_keys if key in summary),
                            "retrieved_at": utc_now_iso(timespec="microseconds"),
                        }
                        if time.monotonic() >= deadline:
                            break
                        return capture
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    raise WandbPollTimeout(run_id, timeout_seconds, last_error)


def _locate_wandb_position(
    *,
    base_url: str,
    entity: str,
    project: str,
    run_id: str,
    poll_seconds: float,
    metric_key: str,
    axis: str,
    value: float,
    deadline: float,
) -> int | float | None:
    """Read where a finished run's objective was logged, within the parent's deadline.

    History gets at most ``_HISTORY_SETTLE_SECONDS`` to reach the summary's
    last step; reading it, and retrying a transient failure, may use the rest
    of the deadline. Any other failure means "no position".

    :param str base_url: Normalized W&B API endpoint.
    :param str entity: W&B entity that owns the project.
    :param str project: W&B project containing the run.
    :param str run_id: Exact W&B run ID to read.
    :param float poll_seconds: Summary poll interval, the delay between
        history reads up to ``_HISTORY_RETRY_SECONDS``.
    :param str metric_key: Objective summary key.
    :param str axis: History key marking where the metric was measured.
    :param float value: Captured objective value.
    :param float deadline: Absolute ``time.monotonic()`` deadline from the parent.
    :return int | float | None: The position, or ``None`` when history never
        settled, no row produced the value, or the read failed.
    """
    from wandb.apis.public import Api

    settle_by = min(deadline, time.monotonic() + _HISTORY_SETTLE_SECONDS)
    interval = min(poll_seconds, _HISTORY_RETRY_SECONDS)
    run: Any = None
    while True:
        read: _HistoryRead = "retry"
        try:
            if run is None:
                api = Api(
                    overrides={"base_url": base_url},
                    timeout=max(1, math.ceil(deadline - time.monotonic())),
                )
                run = api.run(f"{entity}/{project}/{run_id}")
        except Exception as exc:
            if not _is_retryable_setup_error(exc):
                return None
        else:
            read, position = _history_position(
                run,
                run.summary_metrics,
                metric_key=metric_key,
                axis=axis,
                value=value,
                deadline=deadline,
            )
            if read == "settled":
                return position
        retry_by = settle_by if read == "lagging" else deadline
        if time.monotonic() + interval >= retry_by:
            return None
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    """Run an internal request and return typed, bounded evidence to its owner.

    :param list[str] | None argv: Request and response paths; ``None`` reads
        ``sys.argv``.
    :return int: Process exit status, ``0`` once the response file is written.
    """
    from phasesweep.runtime.files import private_atomic_write_text

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("request", type=Path, help="Private polling request")
    parser.add_argument("response", type=Path, help="Private polling response")
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    task = request.pop("task")

    response: dict[str, Any]
    try:
        if task == "position":
            response = {"status": "position", "value": _locate_wandb_position(**request)}
        else:
            response = {"status": "summary", "capture": _poll_wandb_summary(**request)}
    except ImportError:
        response = {"status": "import_error"}
    except WandbSetupError as exc:
        response = {"status": "setup_error", "cause": exc.cause}
    except WandbRunTerminalError as exc:
        response = {"status": "terminal_error", "state": exc.state}
    except WandbPollTimeout as exc:
        response = {"status": "timeout", "cause": str(exc.last_error) if exc.last_error else None}
    except ValueError:
        response = {
            "status": "invalid_evidence",
            "cause": "Requested W&B summary evidence is not a finite JSON number.",
        }
    private_atomic_write_text(args.response, json.dumps(response, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
