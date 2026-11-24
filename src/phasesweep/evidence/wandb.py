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

from phasesweep.runtime.json import finite_nonnegative, json_text, json_text_list
from phasesweep.runtime.remote_worker import report_reserve, run_worker

_HistoryRead = Literal["settled", "lagging", "retry"]
"""How one history read ended: an answer, history behind the summary, or a
transient failure worth another read."""

_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
# How long history may lag the summary of a run whose capture was accepted. Complete
# history reads may use the remaining deadline.
_FINISH_SETTLE_SECONDS = 30.0
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

    ``reached`` says whether the last poll got an answer about the run: a run seen but
    never finished, or finished without its keys, is a verdict on the trial, while polls
    that all failed in transport or setup say nothing about it.
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
    """A rerun's run still showed no history row logged after the rerun launched at the deadline."""

    run_id: str
    waited_seconds: float


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
    """Recognize the public API's absent-run ValueError through SDK wrappers.

    :param Exception exc: Error raised by an exact ``Api.run`` lookup.
    :return bool: Whether W&B reported that the requested run could not be found.
    """
    return any(
        isinstance(error, ValueError) and str(error).startswith("Could not find run ")
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
    :param Mapping[str, str] | None environment: Actual composed trainer environment.
    :param float | None deadline: Absolute deadline including worker startup.
    :return dict[str, Any]: Numeric values, present keys, and retrieval time.
    :raises WandbPollTimeout: Startup, SDK work, or visibility exceeded the budget; it was
        reached only when the worker reports that its last poll got an answer about the run.
    :raises WandbRunMissingError: The final lookup at the deadline found no matching run.
    :raises WandbRunStaleError: By the deadline the rerun's run never showed a row logged
        after it launched.
    :raises WandbSetupError: Authentication or setup was denied.
    :raises WandbRunTerminalError: The expected run terminated unsuccessfully.
    :raises WandbMissingKeysError: At the deadline the finished run's summary still lacked
        required keys.
    :raises ValueError: A requested scalar is invalid.
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
        raise WandbRunStaleError(run_id, float(waited))
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

    last_step = finite_nonnegative(summary.get("_step"))
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
    logged_after: float | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Poll an exact run using refreshed SDK clients within the parent's deadline.

    No lookup starts after the deadline, but the answer to one in flight is used. Every poll
    builds a new client, because a client keeps the run it first fetched.

    The public API reports a run ``finished`` with its full history count as soon as
    ``wandb.finish()`` returns, but may serve the summary minutes later: first empty, or
    for a resumed run still the earlier attempt's. So nothing short of the deadline
    establishes that a finished summary lacks a key. A summary counts only once it is as
    far as the history the same answer reports (:func:`_summary_matches_history`), so a
    half-ingested one is never captured.

    A rerun resumes the run an earlier attempt of its trial left, in whatever state that
    attempt's end left it. Until the summary's ``_timestamp``, the time of the last logged
    history row, passes ``logged_after``, the state and summary are the earlier attempt's,
    so neither a terminal state nor a finished summary decides anything; a run that still
    shows nothing newer at the deadline was not written by this attempt.

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
    :param float | None deadline: Absolute ``time.monotonic()`` deadline from the parent.
    :return dict[str, Any]: Numeric values, present keys, and retrieval time.
    :raises WandbPollTimeout: The run did not finish with the required keys in time,
        reached when the last poll got an answer about the run.
    :raises WandbRunMissingError: The final lookup at the deadline found no matching run.
    :raises WandbRunStaleError: At the deadline the run still showed no row logged after
        ``logged_after``.
    :raises WandbSetupError: Authentication or setup was denied.
    :raises WandbRunTerminalError: The expected run terminated unsuccessfully.
    :raises WandbMissingKeysError: At the deadline the finished run's summary still lacked
        required keys.
    :raises ValueError: A requested scalar is invalid or non-finite.
    """
    from wandb.apis.public import Api
    from wandb.errors import AuthenticationError, UsageError

    from phasesweep.evidence.evaluation import json_float
    from phasesweep.runtime.time import utc_now_iso

    started = time.monotonic()
    deadline = started + timeout_seconds if deadline is None else deadline
    last_error: Exception | None = None
    # Whether the latest poll got an answer about the run rather than failing in transport.
    reached = False
    # Whether the latest answered poll still showed only an earlier attempt's run.
    stale = False
    # The keys the latest answered poll's finished summary lacked, and the keys it had.
    missing: list[str] = []
    available: list[str] = []
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
                if stale:
                    last_error = RuntimeError(
                        "The summary records no row logged since the attempt launched."
                    )
                elif run.state in {"failed", "crashed", "killed", "preempted"}:
                    raise WandbRunTerminalError(run_id, run.state)
                elif run.state == "finished" and not _summary_matches_history(run, summary):
                    last_error = RuntimeError(
                        f"The summary's _step {summary.get('_step')!r} does not match the "
                        f"history's last step {_last_history_step(run)!r}."
                    )
                elif run.state == "finished":
                    if all(key in summary for key in required):
                        values = {}
                        for key in required:
                            value = json_float(summary[key], label=key)
                            if not math.isfinite(value):
                                raise ValueError(f"W&B metric {key!r} is non-finite.")
                            values[key] = value
                        return {
                            "values": values,
                            "present_keys": sorted(key for key in presence_keys if key in summary),
                            "retrieved_at": utc_now_iso(timespec="microseconds"),
                        }
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
    """Return the last history step from the same answer as the run's state and summary.

    ``Api.run`` fetches the run's ``historyKeys`` index with its state and summary, so the
    two describe one moment, unlike a separate ``lastHistoryStep`` query.

    :param object run: Public-API run.
    :return int | float | None: The step, or ``None`` when the run has no history row or
        the answer carried no index.
    """
    index = getattr(run, "history_keys", None)
    return finite_nonnegative(index.get("lastStep")) if isinstance(index, Mapping) else None


def _summary_matches_history(run: object, summary: Mapping[str, object]) -> bool:
    """Say whether a finished run's summary is as far as its history.

    The summary's ``_step`` is the step of the last logged row, so a summary behind the
    history, or empty while history exists, is still being ingested. An answer without
    a history index leaves nothing to compare, so its summary is taken as it is.

    :param object run: Public-API run.
    :param Mapping[str, object] summary: Its decoded summary.
    :return bool: Whether the summary's ``_step`` equals the history's last step, both
        absent counting as equal, or the answer carried no index.
    """
    if not isinstance(getattr(run, "history_keys", None), Mapping):
        return True
    return finite_nonnegative(summary.get("_step")) == _last_history_step(run)


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
        response = {"status": "stale", "waited_seconds": exc.waited_seconds}
    except WandbPollTimeout as exc:
        response = {
            "status": "timeout",
            "cause": str(exc.last_error) if exc.last_error else None,
            "reached": exc.reached,
        }
    except ValueError:
        response = {
            "status": "invalid_evidence",
            "cause": "Requested W&B summary evidence is not a finite JSON number.",
        }
    private_atomic_write_text(args.response, json.dumps(response, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
