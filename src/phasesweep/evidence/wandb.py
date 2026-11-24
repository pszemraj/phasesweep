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
from bisect import bisect_left
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from phasesweep.runtime.json import finite_nonnegative, json_text, json_text_list
from phasesweep.runtime.remote_worker import report_reserve, run_worker

_HistoryRead = Literal["settled", "lagging", "retry"]
"""How one history read ended: an answer, history behind the summary, or a
transient failure worth another read."""

_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_HISTORY_RETRY_SECONDS = 2.0
# Server row cap for unsampled history requests and scan pages.
_HISTORY_SAMPLES = 10_000
# A summary poll stops this long before its deadline. Each lookup's client is made with the
# polling time left, rounded up to whole seconds and at least one, as its timeout, so a lookup
# still in flight at the polling deadline ends within about a second of it, and the report
# follows in what remains.
_REPORT_SECONDS = 2.0


@dataclass(eq=False)
class WandbPollTimeout(TimeoutError):
    """The complete worker exhausted its summary visibility budget.

    ``reached`` says whether the last poll got a usable answer about the run. A transport
    failure or a finished run whose summary is still being ingested leaves the read pending.
    """

    run_id: str
    timeout_seconds: float
    reached: bool
    last_error: Exception | None = None


@dataclass(eq=False)
class WandbRunMissingError(RuntimeError):
    """The final run lookup at the visibility deadline found no matching run."""

    run_id: str


@dataclass(eq=False)
class WandbRunTerminalError(RuntimeError):
    """The expected run terminated unsuccessfully."""

    run_id: str
    state: str


@dataclass(eq=False)
class WandbRunStaleError(RuntimeError):
    """The run's last logged row cannot be attributed to the attempt being read."""

    run_id: str
    waited_seconds: float
    superseded: bool = False


@dataclass(eq=False)
class WandbSetupError(RuntimeError):
    """W&B rejected authentication or client configuration permanently."""

    run_id: str
    cause: str


@dataclass(eq=False)
class WandbMissingKeysError(RuntimeError):
    """The finished run's summary still lacked required keys at the deadline."""

    run_id: str
    missing: list[str]
    available: list[str]
    waited_seconds: float


class _NoRetainedObjectiveError(ValueError):
    """An observed rollback discarded every finite objective observation."""


WANDB_SDK_MISSING = 'W&B evidence requires the optional SDK: pip install "phasesweep[wandb]".'


def require_wandb_sdk() -> None:
    """Check the optional SDK without authenticating or making a network request.

    :raises PhaseSweepError: The W&B public API cannot be imported.
    """
    try:
        importlib.import_module("wandb.apis.public")
    except ImportError as exc:
        from phasesweep.errors import OperatorAction, PhaseSweepError

        raise PhaseSweepError(WANDB_SDK_MISSING, action=OperatorAction.FIX_CONFIG) from exc


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


def _is_missing_run_error(exc: Exception) -> bool:
    """Recognize the public API's absent-run answer by its message.

    The SDK raises a ``ValueError`` and its error wrapper re-raises it as a ``CommError``
    with the same message, keeping the original as the cause; either one in the chain
    is the answer.

    :param Exception exc: Error raised by an exact ``Api.run`` lookup.
    :return bool: Whether W&B reported that the requested run could not be found.
    """
    from wandb.errors import CommError

    return any(
        isinstance(error, (ValueError, CommError)) and str(error).startswith("Could not find run ")
        for error in _error_chain(exc)
    )


def poll_wandb_summary(
    *,
    base_url: str,
    entity: str,
    project: str,
    run_id: str,
    attempt_id: str,
    attempt_dir: Path,
    poll_seconds: float,
    timeout_seconds: float,
    required_keys: Iterable[str] = (),
    presence_keys: Iterable[str] = (),
    logged_after: float | None = None,
    logged_before: float | None = None,
    earlier_launches: tuple[float, ...] = (),
    resume_metric: tuple[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Capture requested evidence after confirmed trainer cleanup.

    :param str base_url: Normalized W&B API endpoint.
    :param str entity: W&B entity that owns the project.
    :param str project: W&B project containing the run.
    :param str run_id: The trial's W&B run, which every attempt of the trial logs to.
    :param str attempt_id: Attempt whose process slot runs the reader.
    :param Path attempt_dir: Existing attempt directory and durable process slot.
    :param float poll_seconds: Delay between summary polls.
    :param float timeout_seconds: Visibility budget for the whole capture.
    :param Iterable[str] required_keys: Numeric keys required before accepting a capture.
    :param Iterable[str] presence_keys: Keys whose presence is recorded without their values.
    :param float | None logged_after: Unix time a rerun launched at, which the summary's
        ``_timestamp`` must pass before the run counts as this attempt's; ``None`` for an
        attempt whose run is its own, defaults to None.
    :param float | None logged_before: Unix time the summary must predate when a later
        attempt reused the run, defaults to None.
    :param tuple[float, ...] earlier_launches: Launch times of earlier reruns in this run.
    :param tuple[str, str] | None resume_metric: Objective key and its trainer progress axis
        for detecting repeated training in a native min/max aggregate.
    :param Mapping[str, str] | None environment: Actual composed trainer environment.
    :param float | None deadline: Absolute deadline including worker startup.
    :return dict[str, Any]: Numeric values, present keys, and retrieval time.
    :raises WandbPollTimeout: Startup, SDK work, or visibility exceeded the budget; it was
        reached only when the worker reports that its last poll got a usable answer.
    :raises WandbRunMissingError: The final lookup at the deadline found no matching run.
    :raises WandbRunStaleError: By the deadline the rerun's run never showed a row logged
        after it launched, or a newer attempt replaced the retained summary.
    :raises WandbSetupError: Authentication or setup was denied.
    :raises WandbRunTerminalError: The expected run terminated unsuccessfully.
    :raises WandbMissingKeysError: At the deadline the finished run's summary still lacked
        required keys.
    :raises ValueError: A requested scalar is invalid or the native objective has no retained row.
    :raises UnsafeProcessCleanupError: Worker cleanup is uncertain.
    """
    deadline = min(
        time.monotonic() + timeout_seconds, deadline if deadline is not None else math.inf
    )
    if time.monotonic() >= deadline:
        raise WandbPollTimeout(run_id, timeout_seconds, reached=False)
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
            "logged_after": logged_after,
            "logged_before": logged_before,
            "earlier_launches": earlier_launches,
            "resume_metric": resume_metric,
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
                "run_missing",
                "stale",
            }
        ),
        run_id=attempt_id,
        attempt_dir=attempt_dir,
        environment=environment,
        deadline=deadline,
    )
    # Definite operational causes remain definite even if cleanup crossed a deadline.
    if response.get("status") == "setup_error":
        raise WandbSetupError(run_id, json_text(response, "cause"))
    if response.get("status") == "terminal_error":
        raise WandbRunTerminalError(run_id, json_text(response, "state"))
    if response.get("status") == "run_missing":
        raise WandbRunMissingError(run_id)
    waited = finite_nonnegative(response.get("waited_seconds"))
    if response.get("status") == "stale":
        if waited is None:
            raise RuntimeError(
                f"W&B evidence worker returned a malformed stale-run report for attempt {attempt_id!r}."
            )
        raise WandbRunStaleError(run_id, float(waited), response.get("superseded") is True)
    if response.get("status") == "missing_keys":
        missing = json_text_list(response.get("missing"))
        available = json_text_list(response.get("available"))
        if missing is None or available is None or waited is None:
            raise RuntimeError(
                f"W&B evidence worker returned a malformed missing-keys report for attempt {attempt_id!r}."
            )
        raise WandbMissingKeysError(run_id, missing, available, float(waited))
    if response.get("status") == "invalid_evidence":
        raise ValueError(response["cause"])
    if response.get("status") == "import_error":
        raise ImportError(WANDB_SDK_MISSING)
    # A complete response was captured before the deadline, even if worker exit timed out.
    if response.get("status") != "summary":
        if result.timed_out or response.get("status") == "timeout" or time.monotonic() >= deadline:
            detail = response.get("cause")
            # A worker stopped before it reported never got an answer this side can use.
            raise WandbPollTimeout(
                run_id,
                timeout_seconds,
                reached=response.get("status") == "timeout" and response.get("reached") is True,
                last_error=RuntimeError(detail) if detail else None,
            )
        diagnostic = f" Diagnostic preserved at {diagnostic_path}." if diagnostic_path else ""
        raise RuntimeError(
            f"W&B evidence worker failed for attempt {attempt_id!r} (exit {result.return_code})."
            f"{diagnostic}"
        )
    capture = response["capture"]
    if not isinstance(capture, dict):
        raise RuntimeError(
            f"W&B evidence worker returned a malformed capture for attempt {attempt_id!r}."
        )
    return capture


def locate_wandb_position(
    *,
    base_url: str,
    entity: str,
    project: str,
    run_id: str,
    attempt_id: str,
    attempt_dir: Path,
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
    :param str run_id: The trial's W&B run, which every attempt of the trial logs to.
    :param str attempt_id: Attempt whose process slot runs the reader.
    :param Path attempt_dir: Existing attempt directory and durable process slot.
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
        run_id=attempt_id,
        attempt_dir=attempt_dir,
        environment=environment,
        deadline=deadline,
    )
    if result.timed_out or response.get("status") != "position":
        return None
    return finite_nonnegative(response.get("value"))


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

    The rows decide, not the index: W&B's index lists an aggregate summary key such as
    ``eval/loss.min`` that no row carries, and its per-key counts are not row counts. So
    rows are checked for ``metric_key`` first, independently of the axis when the key
    looks like an aggregate. Use the head of an aggregate only when no row carries
    the literal key. Positions come from rows that carry the series with its axis.
    History is lagging only while
    its index is behind the summary's ``_step``, which the summary capture has normally
    already waited out; an axis the index does not list, or a series no row carries,
    settles with no position at once. Transient read failures and deadline expiry
    request retry; other failures settle with no position. The captured objective is
    never changed.

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

    last_step = finite_nonnegative(summary.get("_step"))
    if last_step is None:
        return "settled", None
    head, dot, aggregation = metric_key.rpartition(".")
    position: int | float | None = None
    try:
        run.load(force=True)
        indexed = _last_history_step(run)
        if indexed is None or indexed < last_step:
            return "lagging", None
        logged_keys = run.history_keys["keys"]
        if axis not in logged_keys:
            return "settled", None
        candidates = [metric_key]
        if dot and aggregation in {"min", "max", "mean", "last", "first"}:
            candidates.append(head)
        series = ""
        rows: list[Mapping[str, object]] = []
        for candidate in candidates:
            if candidate not in logged_keys:
                continue
            literal: bool | None = False
            if candidate == metric_key and len(candidates) > 1 and axis != "_step":
                literal = _contains_series(run, candidate, last_step=last_step, deadline=deadline)
                if literal is None:
                    return "retry", None
                if not literal:
                    continue
            read = _read_series(run, [candidate, axis], last_step=last_step, deadline=deadline)
            if read is None:
                return "retry", None
            if read or literal:
                series, rows = candidate, read
                break
        if not rows:
            return "settled", None
        exact_match = False
        neighbors: set[float] = set()
        final_value: float | None = None
        final_position: int | float | None = None
        for row in rows:
            if series != metric_key:
                position = finite_nonnegative(row.get(axis))
                continue
            final_value = None
            final_position = finite_nonnegative(row.get(axis))
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
        if not exact_match and len(neighbors) == 1 and final_value in neighbors:
            position = final_position
    except Exception as exc:
        return ("retry" if _is_retryable_setup_error(exc) else "settled"), None
    return "settled", position


class _HistoryReader(Protocol):
    """The public-API run's two history reads."""

    def history(
        self, *, keys: list[str], samples: int, pandas: bool
    ) -> Iterable[Mapping[str, object]]:
        """Return a sample of rows logging every key in ``keys``."""
        ...

    def scan_history(
        self, *, keys: list[str], page_size: int, use_cache: bool
    ) -> Iterable[Mapping[str, object]]:
        """Return every row logging every key in ``keys``, in pages."""
        ...


def _read_series(
    run: _HistoryReader,
    keys: list[str],
    *,
    last_step: int | float,
    deadline: float,
    scan: bool = False,
) -> list[Mapping[str, object]] | None:
    """Read the history rows that log the requested keys.

    A run has at most ``last_step + 1`` rows, so below the server's sample cap the
    sampled call returns them all and longer histories are scanned in pages. W&B has
    served scanned rows before sampled ones, so an empty sampled read is scanned once.

    :param _HistoryReader run: Finished public-API run.
    :param list[str] keys: History keys to read, metric series first.
    :param int | float last_step: The summary's ``_step``, the last row's step.
    :param float deadline: ``time.monotonic()`` time at which the read gives up.
    :param bool scan: Use the unsampled read for scoring evidence.
    :return list[Mapping[str, object]] | None: The rows carrying the series, or ``None``
        when the deadline passed mid-read.
    """
    if scan and time.monotonic() >= deadline:
        return None
    series = keys[0]
    rows: list[Mapping[str, object]] = []
    if not scan and last_step + 1 < _HISTORY_SAMPLES:
        for row in run.history(keys=keys, samples=_HISTORY_SAMPLES, pandas=False):
            if time.monotonic() >= deadline:
                return None
            if series in row:
                rows.append(row)
        if rows:
            return rows
    for row in run.scan_history(keys=keys, page_size=_HISTORY_SAMPLES, use_cache=False):
        if time.monotonic() >= deadline:
            return None
        if series in row and (not scan or row[series] is not None):
            rows.append(row)
    return None if scan and time.monotonic() >= deadline else rows


def _contains_series(
    run: _HistoryReader, key: str, *, last_step: int | float, deadline: float, scan: bool = False
) -> bool | None:
    """Check the exact series without requiring an evaluation axis on the same row.

    :param _HistoryReader run: Finished public-API run.
    :param str key: Literal history key, potentially also a summary key.
    :param int | float last_step: The summary's last W&B history step.
    :param float deadline: Monotonic deadline for history queries.
    :param bool scan: Use the unsampled read for scoring evidence.
    :return bool | None: Whether a row carries the series, or ``None`` after deadline.
    """
    rows = _read_series(run, [key, "_step"], last_step=last_step, deadline=deadline, scan=scan)
    return None if rows is None else any(row.get(key) is not None for row in rows)


def _resumed_extremum(
    run: _HistoryReader,
    key: str,
    value: float,
    *,
    axis: str,
    launches: tuple[float, ...],
    last_step: int | float,
    deadline: float,
) -> float:
    """Reduce a native min/max over the retained rows when a rerun re-logs progress.

    The first progress reported by each rerun replaces the earlier suffix starting at that
    progress. Positions between a checkpoint and the first re-logged evaluation are unknown.
    A literal dotted metric, an incomplete axis, and a separately supplied summary scalar keep
    their summary value.

    :param _HistoryReader run: Finished public-API run.
    :param str key: Exact summary key.
    :param float value: Finite scalar captured from the summary.
    :param str axis: Trainer's progress key, shared by the history rows being reduced.
    :param tuple[float, ...] launches: Chronological launch times of all reruns.
    :param int | float last_step: The summary's last W&B history step.
    :param float deadline: Monotonic deadline for history queries.
    :return float: The corrected extremum, or the original summary scalar.
    :raises TimeoutError: History could not be read before the deadline.
    :raises _NoRetainedObjectiveError: The native objective has no retained finite row.
    """
    from phasesweep.evidence.evaluation import json_float

    series, _, operation = key.rpartition(".")
    if not series or operation not in {"min", "max"}:
        return value
    literal = _contains_series(run, key, last_step=last_step, deadline=deadline, scan=True)
    if literal is None:
        raise TimeoutError("W&B history read expired.")
    if literal:
        return value
    axis_keys = list(dict.fromkeys((axis, "_step", "_timestamp")))
    rows = _read_series(run, axis_keys, last_step=last_step, deadline=deadline, scan=True)
    if rows is None:
        raise TimeoutError("W&B history read expired.")

    points: list[tuple[int | float, int, int | float]] = []
    for row in rows:
        step = finite_nonnegative(row.get("_step"))
        timestamp = finite_nonnegative(row.get("_timestamp"))
        progress = finite_nonnegative(row.get(axis))
        if step is None or timestamp is None or progress is None:
            return value
        points.append((step, bisect_left(launches, timestamp), progress))

    retained: list[tuple[int | float, float]] = []
    attempt = 0
    overlap = False
    for step, segment, progress in sorted(points):
        if retained and progress < retained[-1][1] and segment == attempt:
            return value
        if retained and segment != attempt and progress <= retained[-1][1]:
            retained = [(index, position) for index, position in retained if position < progress]
            overlap = True
        retained.append((step, progress))
        attempt = segment
    if not overlap:
        return value

    full_series = _read_series(
        run, [series, "_step"], last_step=last_step, deadline=deadline, scan=True
    )
    if full_series is None:
        raise TimeoutError("W&B history read expired.")
    axis_steps = {step for step, _, _ in points}
    retained_steps = {step for step, _ in retained}
    all_values: list[float] = []
    kept_values: list[float] = []
    for row in full_series:
        step = finite_nonnegative(row.get("_step"))
        if step not in axis_steps:
            return value
        try:
            metric = json_float(row[series], label=series)
        except ValueError:
            return value
        if math.isfinite(metric):
            all_values.append(metric)
            if step in retained_steps:
                kept_values.append(metric)
    reduce = min if operation == "min" else max
    if not all_values or reduce(all_values) != value:
        return value
    if not kept_values:
        raise _NoRetainedObjectiveError(
            f"W&B objective {key!r} has no finite observation in the retained history "
            "after checkpoint rollback."
        )
    return reduce(kept_values)


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
    logged_after: float | None = None,
    logged_before: float | None = None,
    earlier_launches: tuple[float, ...] = (),
    resume_metric: tuple[str, str] | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Poll an exact run using refreshed SDK clients within the parent's deadline.

    No lookup starts after the deadline, but the answer to one in flight is used. Every poll
    builds a new client, because a client keeps the run it first fetched.

    The public API can report a run ``finished`` before it serves the history index
    or the summary. The summary may take minutes to arrive: first empty, or
    for a resumed run still the earlier attempt's. So nothing short of the deadline
    establishes that a finished summary lacks a key. A summary counts only once it is as
    far as the history the same answer reports (:func:`_summary_matches_history`), so a
    half-ingested one is never captured.

    A rerun resumes the run an earlier attempt of its trial left, in whatever state that
    attempt's end left it. Until the summary's ``_timestamp``, the time of the last logged
    history row, passes ``logged_after``, the state and summary are the earlier attempt's,
    so neither a terminal state nor a finished summary decides anything; a run that still
    shows nothing newer at the deadline was not written by this attempt. A trainer axis can
    show where a rerun re-logged progress, letting min/max exclude the replaced history.

    :param str base_url: Normalized W&B API endpoint.
    :param str entity: W&B entity that owns the project.
    :param str project: W&B project containing the run.
    :param str run_id: Exact W&B run ID to read.
    :param float poll_seconds: Delay between summary polls.
    :param float timeout_seconds: Budget used when no ``deadline`` is given.
    :param Iterable[str] required_keys: Numeric keys required before accepting a capture.
    :param Iterable[str] presence_keys: Keys whose presence is recorded without their values.
    :param float | None logged_after: Unix time a rerun launched at; ``None`` for an
        attempt whose run is its own, defaults to None.
    :param float | None logged_before: Unix time the summary must predate when a later
        attempt reused the run, defaults to None.
    :param tuple[float, ...] earlier_launches: Launch times of earlier reruns in this run.
    :param tuple[str, str] | None resume_metric: Objective key and its trainer progress axis
        for detecting repeated training in a native min/max aggregate.
    :param float | None deadline: Absolute ``time.monotonic()`` deadline from the parent.
    :return dict[str, Any]: Numeric values, present keys, and retrieval time.
    :raises WandbPollTimeout: The run did not finish with the required keys in time,
        reached when the last poll got a usable answer about the run.
    :raises WandbRunMissingError: The final lookup at the deadline found no matching run.
    :raises WandbRunStaleError: At the deadline the run still showed no row logged after
        ``logged_after``, or a newer attempt replaced the retained summary.
    :raises WandbSetupError: Authentication or setup was denied.
    :raises WandbRunTerminalError: The expected run terminated unsuccessfully.
    :raises WandbMissingKeysError: At the deadline the finished run's summary still lacked
        required keys.
    :raises ValueError: A requested scalar is invalid or the native objective has no retained row.
    """
    from wandb.apis.public import Api
    from wandb.errors import AuthenticationError, UsageError

    from phasesweep.evidence.evaluation import json_float
    from phasesweep.runtime.time import utc_now_iso

    started = time.monotonic()
    deadline = started + timeout_seconds if deadline is None else deadline
    last_error: Exception | None = None
    # Whether the latest poll got an answer that can settle the trial at the deadline.
    reached = False
    # Whether the latest answered poll still showed only an earlier attempt's run.
    stale = False
    # The keys the latest answered poll's finished summary lacked, and the keys it had.
    missing: list[str] = []
    available: list[str] = []
    required = tuple(required_keys)
    launches = () if logged_after is None else tuple(sorted((*earlier_launches, logged_after)))
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
            reached = False
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
                # "No such run" answers the lookup; any other failure leaves it unanswered.
                reached = _is_missing_run_error(exc)
                last_error = (
                    WandbRunMissingError(run_id) if reached else RuntimeError(_error_detail(exc))
                )
            else:
                # An answer in hand is used even when it arrived after the polling deadline:
                # the reserve before the supervisor's deadline exists to report it.
                last_error = None
                reached = True
                missing = []
                summary = run.summary_metrics
                stale = logged_after is not None and not _logged_since(summary, logged_after)
                if run.state == "finished" and not _summary_matches_history(run, summary):
                    reached = False
                    last_error = RuntimeError(
                        "The finished run's summary is still empty."
                        if not summary
                        else f"The summary's _step {summary.get('_step')!r} does not match the "
                        f"history's last step {_last_history_step(run)!r}."
                    )
                elif stale:
                    last_error = RuntimeError(
                        "The summary records no row logged since the attempt launched."
                    )
                elif run.state in {"failed", "crashed", "killed", "preempted"}:
                    raise WandbRunTerminalError(run_id, run.state)
                elif run.state == "finished":
                    if logged_before is not None:
                        stamp = finite_nonnegative(summary.get("_timestamp"))
                        if stamp is None or stamp >= logged_before:
                            raise WandbRunStaleError(
                                run_id, time.monotonic() - started, superseded=True
                            )
                    if all(key in summary for key in required):
                        values = {}
                        for key in required:
                            value = json_float(summary[key], label=key)
                            if not math.isfinite(value):
                                raise ValueError(f"W&B metric {key!r} is non-finite.")
                            values[key] = value
                        last_step = _last_history_step(run)
                        try:
                            if launches and resume_metric is not None and last_step is not None:
                                key, axis = resume_metric
                                if key in values and axis != "_step":
                                    values[key] = _resumed_extremum(
                                        run,
                                        key,
                                        values[key],
                                        axis=axis,
                                        launches=launches,
                                        last_step=last_step,
                                        deadline=deadline,
                                    )
                        except Exception as exc:
                            if not _is_retryable_setup_error(exc):
                                raise
                            last_error = RuntimeError(_error_detail(exc))
                            reached = False
                        else:
                            return {
                                "values": values,
                                "present_keys": sorted(
                                    key for key in presence_keys if key in summary
                                ),
                                "retrieved_at": utc_now_iso(timespec="microseconds"),
                            }
                    else:
                        missing = sorted(key for key in required if key not in summary)
                        available = sorted(summary)
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    waited = time.monotonic() - started
    if isinstance(last_error, WandbRunMissingError):
        raise last_error
    if reached and stale:
        raise WandbRunStaleError(run_id, waited)
    if reached and missing:
        raise WandbMissingKeysError(run_id, missing, available, waited)
    raise WandbPollTimeout(run_id, timeout_seconds, reached=reached, last_error=last_error)


def _last_history_step(run: object) -> int | float | None:
    """Return the last step of the history index in the same answer as the run's summary.

    ``Api.run`` fetches the run's ``historyKeys`` index with its state and summary, so the
    two describe one moment, unlike a separate ``lastHistoryStep`` query. W&B serves the
    index after the summary, as ``null``, a ``lastStep`` of -1 or no keys, so any of
    those reads as no indexed history.

    :param object run: Public-API run.
    :return int | float | None: The step, or ``None`` when no history is indexed.
    """
    index = getattr(run, "history_keys", None)
    if not isinstance(index, Mapping):
        return None
    keys = index.get("keys")
    if not isinstance(keys, Mapping) or not keys:
        return None
    return finite_nonnegative(index.get("lastStep"))


def _summary_matches_history(run: object, summary: Mapping[str, object]) -> bool:
    """Say whether a finished run's summary and history index are both fully served.

    W&B can serve an empty summary before either the summary or the index is visible.
    A populated summary can also arrive before or after its matching history index.

    :param object run: Public-API run.
    :param Mapping[str, object] summary: Its decoded summary.
    :return bool: Whether the summary is populated and its step matches the index.
        A summary written directly can have values without a history step.
    """
    return bool(summary) and finite_nonnegative(summary.get("_step")) == _last_history_step(run)


def _logged_since(summary: Mapping[str, object], logged_after: float) -> bool:
    """Say whether a run's summary records a history row logged after a rerun launched.

    W&B stamps every logged row with the logging host's clock and keeps the last stamp as
    the summary's ``_timestamp``; the trainer and PhaseSweep share that host's clock.

    :param Mapping[str, object] summary: The run's decoded summary.
    :param float logged_after: Unix time the rerun launched at.
    :return bool: Whether ``_timestamp`` is a finite number later than ``logged_after``.
    """
    stamp = finite_nonnegative(summary.get("_timestamp"))
    return stamp is not None and stamp > logged_after


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

    Lagging history and transient failures are read again until the deadline; other
    failures return no position. The summary capture already waited for the history
    index, so history normally settles on the first read.

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
        if time.monotonic() + interval >= deadline:
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
            # Stop polling early enough that a lookup in flight and the report of a missing run,
            # missing keys or a timeout finish before the supervisor's deadline.
            remaining = max(0.0, request["deadline"] - time.monotonic())
            request["deadline"] -= report_reserve(remaining, _REPORT_SECONDS)
            response = {"status": "summary", "capture": _poll_wandb_summary(**request)}
    except ImportError:
        response = {"status": "import_error"}
    except WandbSetupError as exc:
        response = {"status": "setup_error", "cause": exc.cause}
    except WandbRunTerminalError as exc:
        response = {"status": "terminal_error", "state": exc.state}
    except WandbMissingKeysError as exc:
        response = {
            "status": "missing_keys",
            "missing": exc.missing,
            "available": exc.available,
            "waited_seconds": exc.waited_seconds,
        }
    except WandbRunMissingError:
        response = {"status": "run_missing"}
    except WandbRunStaleError as exc:
        response = {
            "status": "stale",
            "waited_seconds": exc.waited_seconds,
            "superseded": exc.superseded,
        }
    except WandbPollTimeout as exc:
        response = {
            "status": "timeout",
            "cause": str(exc.last_error) if exc.last_error else None,
            "reached": exc.reached,
        }
    except _NoRetainedObjectiveError as exc:
        response = {"status": "invalid_evidence", "cause": str(exc)}
    except ValueError:
        response = {
            "status": "invalid_evidence",
            "cause": "Requested W&B summary evidence is not a finite JSON number.",
        }
    private_atomic_write_text(args.response, json.dumps(response, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
