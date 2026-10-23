"""Read finished W&B evidence in the existing supervised attempt slot.

A trial's summary capture is mandatory evidence; locating its accepted
objective in run history is advisory and runs as a separate request.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from phasesweep.runtime.json import json_text, json_text_list, progress_position
from phasesweep.runtime.remote_worker import run_worker

_HistoryRead = Literal["settled", "lagging", "retry"]
"""How one history read ended: an answer, history behind the summary, or a
transient failure worth another read."""

_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
# How long the public API may lag a run it already reports finished: a summary
# still missing required keys, or history behind the summary. Complete history
# reads may use the remaining deadline.
_FINISH_SETTLE_SECONDS = 30.0
_HISTORY_RETRY_SECONDS = 2.0
# Server row cap for unsampled history requests and scan pages.
_HISTORY_SAMPLES = 10_000


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


@dataclass(eq=False)
class WandbMissingKeysError(RuntimeError):
    """The finished run's summary still lacked required keys after the settle window."""

    run_id: str
    missing: list[str]
    available: list[str]


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
            'Install this distribution with its wandb extra: pip install "phasesweep[wandb]".',
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
        # SDK transport wrappers may expose status without preserving the cause.
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
    :raises WandbMissingKeysError: The finished run's summary still lacked
        required keys once the settle window passed.
    :raises ValueError: A requested scalar is invalid.
    :raises UnsafeProcessCleanupError: Worker cleanup is uncertain.
    """
    deadline = min(
        time.monotonic() + timeout_seconds, deadline if deadline is not None else math.inf
    )
    if time.monotonic() >= deadline:
        raise WandbPollTimeout(run_id, timeout_seconds)
    result, response, diagnostic_path = run_worker(
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
        Path(__file__).resolve(),
        "wandb",
        statuses=frozenset(
            {
                "summary",
                "setup_error",
                "terminal_error",
                "missing_keys",
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
        raise WandbSetupError(run_id, json_text(response, "cause"))
    if response.get("status") == "terminal_error":
        raise WandbRunTerminalError(run_id, json_text(response, "state"))
    if response.get("status") == "missing_keys":
        missing = json_text_list(response.get("missing"))
        available = json_text_list(response.get("available"))
        if missing is None or available is None:
            raise RuntimeError(
                f"W&B evidence worker returned a malformed missing-keys report for attempt {run_id!r}."
            )
        raise WandbMissingKeysError(run_id, missing, available)
    if response.get("status") == "invalid_evidence":
        raise ValueError(response["cause"])
    if response.get("status") == "import_error":
        raise ImportError(
            "W&B SDK is unavailable in the supervised worker; install phasesweep[wandb]."
        )
    # A complete response was captured before the deadline, even if worker exit timed out.
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
    """Read an accepted objective's advisory history position on the requested axis.

    Read failures and deadlines return no position without discarding the objective.

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
    deadline = min(
        time.monotonic() + timeout_seconds, deadline if deadline is not None else math.inf
    )
    if time.monotonic() >= deadline:
        return None
    result, response, _ = run_worker(
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
        Path(__file__).resolve(),
        "wandb",
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
    """Locate an accepted objective's advisory evaluation position in complete history.

    A plain key uses its latest exact-value match. Without one, a single adjacent
    binary64 value qualifies only when it matches a valid final row. Ambiguous
    neighbors or a malformed final row disqualify this fallback. Recognized
    aggregate suffixes use the underlying series' final position.

    History is lagging until its index reaches the summary step and all counted
    series rows are read. The axis must be logged alongside the metric. Transient
    read failures and deadline expiry request retry; other failures settle with
    no position. The captured objective is never changed.

    :param Any run: Finished public-API run.
    :param Mapping[str, Any] summary: Its decoded summary.
    :param str metric_key: Configured objective summary key.
    :param str axis: History key marking where the metric was measured.
    :param float value: Captured objective value.
    :param float deadline: ``time.monotonic()`` time at which a read gives up.
    :return tuple[_HistoryRead, int | float | None]: How the read ended, and
        the position once ``settled`` (``None`` when a plain key has neither
        an exact match nor an unambiguous neighboring final observation).
    """
    from phasesweep.evidence.evaluation import json_float

    last_step = progress_position(summary.get("_step"))
    if last_step is None:
        return "settled", None
    head, dot, aggregation = metric_key.rpartition(".")
    position: int | float | None = None
    try:
        run.load(force=True)
        index = run.history_keys
        if index["lastStep"] < last_step:
            return "lagging", None
        logged_keys = index["keys"]
        if metric_key in logged_keys:
            series = metric_key
        elif dot and aggregation in {"min", "max", "mean", "last", "first"} and head in logged_keys:
            series = head
        else:
            return "settled", None
        count = sum(entry["count"] for entry in logged_keys[series]["typeCounts"])
        # Below-cap requests return the complete series; longer series require scan pages.
        keys = [series, axis]
        if count < _HISTORY_SAMPLES:
            rows = run.history(keys=keys, samples=_HISTORY_SAMPLES, pandas=False)
        else:
            rows = run.scan_history(keys=keys, page_size=_HISTORY_SAMPLES, use_cache=False)
        read = 0
        exact_match = False
        neighbors: set[float] = set()
        final_value: float | None = None
        final_position: int | float | None = None
        for row in rows:
            if time.monotonic() >= deadline:
                return "retry", None
            if series not in row:
                continue
            read += 1
            if series != metric_key:
                position = progress_position(row.get(axis))
                continue
            final_value = None
            final_position = progress_position(row.get(axis))
            try:
                logged = json_float(row[series], label=series)
            except ValueError:
                continue
            if not math.isfinite(logged):
                continue
            final_value = logged
            if logged == value:
                exact_match = True
                position = final_position
            elif math.nextafter(value, logged) == logged:
                neighbors.add(logged)
        if read < count:
            return "lagging", None
        if not exact_match and len(neighbors) == 1 and final_value in neighbors:
            position = final_position
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

    A finished summary may still gain values while the public API catches up,
    so missing required keys are polled for ``_FINISH_SETTLE_SECONDS`` from the
    first finished read that lacked them; once that window passes before the
    deadline, the latest finished summary's missing and available keys are
    reported instead of waiting out the deadline.

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
    :raises WandbMissingKeysError: The finished run's summary still lacked
        required keys once the settle window passed.
    :raises ValueError: A requested scalar is invalid or non-finite.
    """
    from wandb.apis.public import Api
    from wandb.errors import AuthenticationError, UsageError

    from phasesweep.evidence.evaluation import json_float
    from phasesweep.runtime.time import utc_now_iso

    deadline = time.monotonic() + timeout_seconds if deadline is None else deadline
    last_error: Exception | None = None
    missing_keys: WandbMissingKeysError | None = None
    settle_by = math.inf
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
                    missing = sorted(key for key in required if key not in summary)
                    available = sorted(summary)
                    missing_keys = WandbMissingKeysError(run_id, missing, available)
                    last_error = RuntimeError(
                        f"Missing required summary keys: {missing!r}; "
                        f"available summary keys: {available!r}."
                    )
                    settle_by = min(settle_by, time.monotonic() + _FINISH_SETTLE_SECONDS)
        # A window still open at the deadline ends as a timeout, like any late read.
        if missing_keys is not None and settle_by <= time.monotonic() < deadline:
            raise missing_keys
        wake_by = min(deadline, settle_by)
        time.sleep(min(poll_seconds, max(0.0, wake_by - time.monotonic())))
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
    """Read a finished run's objective position within the parent's deadline.

    History has _FINISH_SETTLE_SECONDS to catch up; reads and transient retries
    may use the rest of the deadline. Other failures return no position.

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

    settle_by = min(deadline, time.monotonic() + _FINISH_SETTLE_SECONDS)
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
            # Leave time to report missing keys before the supervisor's deadline.
            remaining = max(0.0, request["deadline"] - time.monotonic())
            request["deadline"] -= min(0.1, remaining / 2)
            response = {"status": "summary", "capture": _poll_wandb_summary(**request)}
    except ImportError:
        response = {"status": "import_error"}
    except WandbSetupError as exc:
        response = {"status": "setup_error", "cause": exc.cause}
    except WandbRunTerminalError as exc:
        response = {"status": "terminal_error", "state": exc.state}
    except WandbMissingKeysError as exc:
        response = {"status": "missing_keys", "missing": exc.missing, "available": exc.available}
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
