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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, Protocol, TypedDict

from phasesweep.evidence.history import merge_attempt_histories
from phasesweep.runtime.json import JsonObject, finite_nonnegative, json_text, json_text_list
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
# States only a closed session leaves, so the run holds every row the session logged.
_CLOSED_STATES = frozenset({"finished", "failed", "killed", "preempted"})


class WandbRunAddress(TypedDict):
    """An earlier attempt's W&B run: the destination that attempt recorded, and its ID."""

    base_url: str
    entity: str
    project: str
    run_id: str


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
class WandbNothingToScoreError(RuntimeError):
    """At the poll deadline a rerun's run and its trial's earlier runs leave no result to score.

    ``cause`` is a predicate for the rerun's run, saying what it and the earlier runs show.
    """

    run_id: str
    cause: str


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


class _RetainedHistoryError(ValueError):
    """A run's series rows cannot be placed on the trainer's evaluation axis."""


class _NoAnswer(RuntimeError):
    """A lookup failed transiently, so the poll got no answer about the run."""


class _PastDeadline(Exception):
    """The polling deadline passed before a lookup could start."""


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
    objective_key: str | None = None,
    earlier: Sequence[WandbRunAddress] = (),
    evaluation_axis: str = "_step",
    environment: Mapping[str, str] | None = None,
    deadline: float | None = None,
) -> JsonObject:
    """Capture requested evidence after confirmed trainer cleanup.

    The capture is assembled from the attempt's run and its trial's earlier runs by the rules
    of :func:`_poll_wandb_summary`. It carries ``deadline_verdict`` when it was decided at the
    poll's deadline, and ``earlier_attempt``, the run ID, when an earlier run scored it.

    :param str base_url: Normalized W&B API endpoint.
    :param str entity: W&B entity that owns the project.
    :param str project: W&B project containing the run.
    :param str run_id: The attempt's own W&B run.
    :param str attempt_id: Attempt whose process slot runs the reader.
    :param Path attempt_dir: Existing attempt directory and durable process slot.
    :param float poll_seconds: Delay between summary polls.
    :param float timeout_seconds: Visibility budget for the whole capture.
    :param Iterable[str] required_keys: Numeric keys required before accepting a capture.
    :param Iterable[str] presence_keys: Keys whose presence is recorded without their values.
    :param str | None objective_key: The objective's summary key when W&B supplies the
        objective, defaults to None.
    :param Sequence[WandbRunAddress] earlier: The runs of the earlier attempts that trained
        in the attempt's directory, oldest first; empty for a first attempt.
    :param str evaluation_axis: Trainer progress key shared by native min/max aggregates.
    :param Mapping[str, str] | None environment: Actual composed trainer environment.
    :param float | None deadline: Absolute deadline including worker startup.
    :return JsonObject: Numeric values, present keys, and retrieval time.
    :raises WandbPollTimeout: Startup, SDK work, or visibility exceeded the budget; it was
        reached only when the worker reports that its last poll got a usable answer.
    :raises WandbRunMissingError: The final lookup at the deadline found no matching run.
    :raises WandbNothingToScoreError: At the deadline a rerun's run and its trial's earlier
        runs leave no result to score.
    :raises WandbSetupError: Authentication or setup was denied.
    :raises WandbRunTerminalError: The expected run terminated unsuccessfully.
    :raises WandbMissingKeysError: At the deadline the finished run's summary still lacked
        required keys.
    :raises ValueError: A requested scalar is invalid or a native aggregate has no retained row.
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
            "objective_key": objective_key,
            "earlier": [dict(address) for address in earlier],
            "evaluation_axis": evaluation_axis,
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
                "nothing_to_score",
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
    if response.get("status") == "nothing_to_score":
        raise WandbNothingToScoreError(run_id, json_text(response, "cause"))
    waited = finite_nonnegative(response.get("waited_seconds"))
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
    :param str run_id: The W&B run whose summary supplied the objective.
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
    run: _LoadableRun,
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
    The summary and the history index are read from the same fresh load of the run on
    every call, so a retry never reuses an earlier answer. History is lagging while
    that summary is still empty or its index is behind the summary's ``_step``, which the
    summary capture has normally already waited out; a populated summary without a
    ``_step``, an axis the index does not list, or a series no row carries, settles with
    no position at once. Transient read failures and deadline expiry request retry;
    other failures settle with no position. The captured objective is never changed.

    :param _LoadableRun run: Finished public-API run.
    :param str metric_key: Configured objective summary key.
    :param str axis: History key marking where the metric was measured.
    :param float value: Captured objective value.
    :param float deadline: ``time.monotonic()`` time at which a read gives up.
    :return tuple[_HistoryRead, int | float | None]: How the read ended, and
        the position once ``settled`` (``None`` when a plain key has neither
        an exact match nor an unambiguous neighboring final observation).
    """
    from phasesweep.evidence.evaluation import json_float

    head, dot, aggregation = metric_key.rpartition(".")
    position: int | float | None = None
    try:
        run.load(force=True)
        summary = run.summary_metrics
        if not summary:
            return "lagging", None
        last_step = finite_nonnegative(summary.get("_step"))
        if last_step is None:
            return "settled", None
        indexed = _last_history_step(run)
        if indexed is None or indexed < last_step:
            return "lagging", None
        # _last_history_step found an index whose keys are a mapping.
        index = getattr(run, "history_keys", None)
        logged_keys = index.get("keys") if isinstance(index, Mapping) else None
        if not isinstance(logged_keys, Mapping) or axis not in logged_keys:
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


class _Run(_HistoryReader, Protocol):
    """The public-API run fields the summary reader uses."""

    @property
    def state(self) -> str:
        """Return the run's state, such as ``running`` or ``finished``."""
        ...

    @property
    def summary_metrics(self) -> Mapping[str, object]:
        """Return the run's decoded summary."""
        ...


class _LoadableRun(_Run, Protocol):
    """A public-API run that can fetch its state, summary and history index again."""

    def load(self, force: bool = False) -> object:
        """Fetch the run's attributes from the service again."""
        ...


class _Api(Protocol):
    """The public-API client's exact run lookup."""

    def run(self, path: str) -> _Run:
        """Return the run at ``entity/project/run_id``."""
        ...


class _Scored(NamedTuple):
    """What a scoring run and the trial's runs before it supply."""

    values: dict[str, float]
    present: list[str]
    # Required keys no run holds, and keys taken from an earlier run's summary.
    missing: list[str]
    borrowed: list[str]
    # Whether the scoring run's own history has yet to reproduce its summary.
    unsettled: bool


def _summary_number(run: _Run, key: str) -> float | None:
    """Return a run's finite summary value for a key, if it holds one.

    :param _Run run: Public-API run.
    :param str key: Summary key.
    :return float | None: The value, or ``None`` when the summary lacks it or it is not finite.
    """
    from phasesweep.evidence.evaluation import json_float

    if key not in run.summary_metrics:
        return None
    try:
        value = json_float(run.summary_metrics[key], label=key)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _attempt_series(
    run: _HistoryReader, series: str, *, axis: str, deadline: float
) -> list[tuple[int | float, float]]:
    """Read one run's measurements of a series at the trainer progress each was logged at.

    A row without the axis sits at the progress the run logged last before it, where W&B's
    ``define_metric(step_sync=True)`` records it, or first after it when the run had logged
    none yet. Reading the series with and without the axis finds those rows, and the run's
    whole axis is read only when there are some.

    :param _HistoryReader run: Public-API run, in any state.
    :param str series: History key underlying a native min or max.
    :param str axis: Trainer's progress key.
    :param float deadline: Monotonic deadline for history queries.
    :return list[tuple[int | float, float]]: Progress and value of each measurement, in
        logging order.
    :raises TimeoutError: History could not be read before the deadline.
    :raises _RetainedHistoryError: The run logged the series but never the axis.
    """
    from phasesweep.evidence.evaluation import json_float

    def scan(keys: list[str]) -> list[Mapping[str, object]]:
        rows = _read_series(run, keys, last_step=0, deadline=deadline, scan=True)
        if rows is None:
            raise TimeoutError("W&B history read expired.")
        return rows

    def measurement(row: Mapping[str, object]) -> tuple[int | float, float] | None:
        step = finite_nonnegative(row.get("_step"))
        try:
            metric = json_float(row[series], label=series)
        except ValueError:
            return None
        return None if step is None else (step, metric)

    points: dict[int | float, tuple[int | float, float]] = {}
    for row in scan([series, axis, "_step"]):
        found, progress = measurement(row), finite_nonnegative(row.get(axis))
        if found is not None and progress is not None:
            points[found[0]] = (progress, found[1])
    unplaced: dict[int | float, float] = {}
    for row in scan([series, "_step"]):
        found = measurement(row)
        if found is not None and found[0] not in points:
            unplaced[found[0]] = found[1]
    if unplaced:
        track: list[tuple[int | float, int | float]] = []
        for row in scan([axis, "_step"]):
            step, progress = finite_nonnegative(row.get("_step")), finite_nonnegative(row.get(axis))
            if step is not None and progress is not None:
                track.append((step, progress))
        if not track:
            raise _RetainedHistoryError(
                f"W&B metric {series!r} has rows in a run that never logged {axis!r}, so the "
                "attempts' histories cannot be merged. Set metric.extractor.evaluation_axis to "
                "a key every attempt logs."
            )
        track.sort()
        for step, metric in unplaced.items():
            index = bisect_left(track, step, key=lambda entry: entry[0])
            points[step] = (track[max(index - 1, 0)][1], metric)
    return [points[step] for step in sorted(points)]


def _merged_extrema(
    runs: Sequence[_Run],
    sources: Mapping[str, int],
    values: Mapping[str, float],
    *,
    axis: str,
    deadline: float,
) -> tuple[dict[str, float], bool]:
    """Reduce required native aggregates over a trial's runs, sharing reads of each series.

    With an evaluation axis, each run's rows are its attempt's history, and the reduction runs
    over the series :func:`merge_attempt_histories` retains from them, the rule MLflow
    histories use. W&B's own ``_step`` restarts in every run, so without an axis the reduction
    runs over the runs' summary values. A literal dotted series, and a summary value the
    scoring run's own rows do not reproduce, keep their summary value.

    W&B serves a run's keyed history up to a minute after its summary, so with an axis a run
    whose summary holds the aggregate but whose served rows do not reproduce it may not be
    fully served yet. A closed earlier run in that state leaves the read pending. An earlier
    run that is not closed may never receive its last rows and is read as W&B holds it.

    :param Sequence[_Run] runs: The trial's runs through the scoring run, oldest first.
    :param Mapping[str, int] sources: Index of the newest run holding each required value.
    :param Mapping[str, float] values: Finite summary values from those runs.
    :param str axis: Trainer's progress key, or ``_step`` for none.
    :param float deadline: Monotonic deadline for history queries.
    :return tuple[dict[str, float], bool]: Reduced extrema keyed by their exact summary
        names, and whether a run's served rows did not reproduce its summary aggregate.
    :raises TimeoutError: History could not be read before the deadline.
    :raises _RetainedHistoryError: A run's series rows cannot be placed on the axis or the
        retained series has no finite observation.
    :raises _NoAnswer: A closed earlier run's keyed history does not reproduce its aggregate.
    """
    operations: dict[str, list[str]] = {}
    for key in values:
        series, _, operation = key.rpartition(".")
        if not series or operation not in {"min", "max"}:
            continue
        source = runs[sources[key]]
        literal = _contains_series(
            source, key, last_step=_last_history_step(source) or 0, deadline=deadline, scan=True
        )
        if literal is None:
            raise TimeoutError("W&B history read expired.")
        if not literal:
            operations.setdefault(series, []).append(operation)

    def reproduces(rows: Sequence[float], number: float, reduce: Callable[..., float]) -> bool:
        # W&B can serialize the same value one binary64 step apart in summary and history.
        return bool(rows) and math.nextafter(reduce(rows), number) == number

    reduced: dict[str, float] = {}
    unsettled = False
    for series, kinds in operations.items():
        histories = (
            []
            if axis == "_step"
            else [_attempt_series(run, series, axis=axis, deadline=deadline) for run in runs]
        )
        finite = [
            [metric for _, metric in history if math.isfinite(metric)] for history in histories
        ]
        retained = [
            metric for metric in merge_attempt_histories(histories) if math.isfinite(metric)
        ]
        for operation in kinds:
            key = f"{series}.{operation}"
            reduce = min if operation == "min" else max
            source_index = sources[key]
            if axis == "_step":
                numbers = (_summary_number(run, key) for run in runs[:source_index])
                reduced[key] = reduce([values[key], *(n for n in numbers if n is not None)])
                continue
            for run, rows in zip(runs[:-1], finite[:-1], strict=True):
                number = _summary_number(run, key)
                if (
                    run.state in _CLOSED_STATES
                    and number is not None
                    and not reproduces(rows, number, reduce)
                ):
                    raise _NoAnswer(
                        f"A closed earlier run's history does not yet reproduce {key!r}."
                    )
            if reproduces(finite[source_index], values[key], reduce):
                if not retained:
                    raise _RetainedHistoryError(
                        f"W&B metric {key!r} has no finite observation in the retained history."
                    )
                reduced[key] = reduce(retained)
            elif source_index == len(runs) - 1:
                unsettled = True
    return reduced, unsettled


def _score(
    scoring: _Run,
    older: Sequence[_Run],
    *,
    required: tuple[str, ...],
    presence: tuple[str, ...],
    axis: str,
    deadline: float,
) -> _Scored:
    """Read the required values from a scoring run's summary and the trial's runs before it.

    A required key the scoring run's summary lacks comes from the newest earlier summary that
    holds it, whatever that run's state: a trainer may log a value only on a fresh start. The
    read stays pending when a closed earlier run's summary does not match its
    history index, as W&B serves a signalled session's first row as its summary for a minute,
    or when :func:`_merged_extrema` finds history not yet served. An earlier run that is not
    closed may never settle, as a session killed outright stays ``crashed`` with its first-row
    summary and none of the rows it had not uploaded, so it is read as W&B holds it.

    :param _Run scoring: The run whose summary holds the objective.
    :param Sequence[_Run] older: The trial's runs before it, oldest first.
    :param tuple[str, ...] required: Numeric keys the capture requires.
    :param tuple[str, ...] presence: Keys whose presence is recorded.
    :param str axis: Trainer's progress key, or ``_step`` for none.
    :param float deadline: Monotonic deadline for history queries.
    :return _Scored: The values, present keys, the keys missing or borrowed, and whether the
        scoring is unsettled.
    :raises ValueError: A required value is not a finite number, or a run's series rows
        cannot be placed on the axis.
    :raises TimeoutError: History could not be read before the deadline.
    :raises _NoAnswer: A closed earlier run's summary or history is still being served.
    """
    from phasesweep.evidence.evaluation import json_float

    for run in older:
        if (
            run.state in _CLOSED_STATES
            and not _logged_nothing(run)
            and not _summary_matches_history(run, run.summary_metrics)
        ):
            raise _NoAnswer("A closed earlier run's summary does not yet match its history index.")
    runs = (*older, scoring)
    values: dict[str, float] = {}
    sources: dict[str, int] = {}
    missing: list[str] = []
    borrowed: list[str] = []
    for key in required:
        source_index = next(
            (i for i in reversed(range(len(runs))) if key in runs[i].summary_metrics), None
        )
        if source_index is None:
            missing.append(key)
            continue
        value = json_float(runs[source_index].summary_metrics[key], label=key)
        if not math.isfinite(value):
            raise ValueError(f"W&B metric {key!r} is non-finite.")
        values[key] = value
        sources[key] = source_index
        if source_index < len(older):
            borrowed.append(key)
    unsettled = False
    if older and not missing:
        reduced, unsettled = _merged_extrema(runs, sources, values, axis=axis, deadline=deadline)
        values.update(reduced)
    present = sorted(key for key in presence if any(key in run.summary_metrics for run in runs))
    return _Scored(values, present, missing, borrowed, unsettled)


def _lookup(clients: dict[str, _Api], address: WandbRunAddress, deadline: float) -> _Run | None:
    """Look a run up with this poll's client for its endpoint.

    Every poll builds new clients, because a client keeps the run it first fetched.

    :param dict[str, _Api] clients: This poll's clients by endpoint, extended in place.
    :param WandbRunAddress address: The run's endpoint, entity, project and ID.
    :param float deadline: ``time.monotonic()`` time after which no lookup starts.
    :raises _PastDeadline: The deadline passed before the lookup could start.
    :raises WandbSetupError: W&B refused authentication or the client configuration.
    :raises _NoAnswer: Building the client or the lookup failed transiently.
    :return _Run | None: The run, or ``None`` when W&B answered that it does not exist.
    """
    from wandb.apis.public import Api
    from wandb.errors import AuthenticationError, UsageError

    run_id = address["run_id"]
    api = clients.get(address["base_url"])
    if api is None:
        if time.monotonic() >= deadline:
            raise _PastDeadline
        try:
            api = Api(
                overrides={"base_url": address["base_url"]},
                timeout=max(1, math.ceil(deadline - time.monotonic())),
            )
        except Exception as exc:
            if not _is_retryable_setup_error(exc):
                raise WandbSetupError(run_id, _error_detail(exc)) from exc
            raise _NoAnswer(_error_detail(exc)) from exc
        clients[address["base_url"]] = api
    if time.monotonic() >= deadline:
        raise _PastDeadline
    try:
        return api.run(f"{address['entity']}/{address['project']}/{run_id}")
    except Exception as exc:
        if any(_http_status(error) in {400, 401, 403, 422} for error in _error_chain(exc)) or (
            isinstance(exc, (AuthenticationError, UsageError))
            and not _is_retryable_setup_error(exc)
        ):
            raise WandbSetupError(run_id, _error_detail(exc)) from exc
        # "No such run" answers the lookup; any other failure leaves it unanswered.
        if _is_missing_run_error(exc):
            return None
        raise _NoAnswer(_error_detail(exc)) from exc


def _answer_key(runs: Sequence[_Run]) -> str:
    """Fingerprint what lookups answered about runs, to reuse a scoring of the same answers.

    :param Sequence[_Run] runs: Runs in a scoring.
    :return str: Their states, row counts, indexed steps and summaries, as JSON.
    """
    return json.dumps(
        [
            [run.state, _row_count(run), _last_history_step(run), dict(run.summary_metrics)]
            for run in runs
        ],
        sort_keys=True,
        default=str,
    )


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
    objective_key: str | None = None,
    earlier: Sequence[WandbRunAddress] = (),
    evaluation_axis: str = "_step",
    deadline: float | None = None,
) -> dict[str, object]:
    """Poll an attempt's run, and its trial's earlier runs, within the parent's deadline.

    No lookup starts after the deadline, but the answer to one in flight is used. The public
    API can report a run ``finished`` before it serves the history index or the summary, and
    the summary may take minutes to arrive, so nothing short of the deadline establishes that
    a finished summary lacks a key. A summary counts only once it is as far as the history the
    same answer reports (:func:`_summary_matches_history`), so a half-ingested one is never
    captured.

    Every attempt logs to a run of its own. A rerun continues its trial's earlier attempts
    from their checkpoint in the same directory, so its result is assembled from its run and
    their runs (``earlier``, oldest first, skipping any W&B does not hold):

    1. The rerun's finished run, its summary matching its history and holding the objective,
       scores the trial. A required key that summary still lacks at the deadline comes from
       the newest earlier summary that holds it (:func:`_score`).
    2. A native ``.min`` or ``.max`` key is reduced over the trial's runs
       (:func:`_merged_extrema`).
    3. A rerun's run that is missing at the deadline, or finished with zero rows and
       no user summary key, logged nothing: its trainer found the training finished. The newest earlier
       run then scores the trial by rules 1 and 2 over the earlier runs alone, if its session
       closed and its summary matches its history and holds the objective, and the capture
       carries ``earlier_attempt``. Otherwise nothing is scored, saying why. An earlier run
       that closed with zero rows and no user summary key is skipped as a
       missing one is. A row-less run that is ``running`` or ``crashed`` is not skipped: W&B
       never receives what a session killed outright had not uploaded, so it cannot say
       whether that session trained.
    4. A rerun's finished run whose summary matches its history but lacks the objective at the
       deadline ran without reporting it, so no earlier run scores it.

    Rules 3 and 4, a key taken from an earlier run, and a scoring run whose own history does
    not reproduce its summary aggregate, are verdicts the latest
    answered poll decides, returned or raised after the deadline as a missing key is; such a
    capture carries ``deadline_verdict``. A closed earlier run whose summary or history
    is unsettled leaves the read pending. An earlier run that is not closed is read as W&B holds it,
    since a session killed outright stays ``crashed`` with its first-row summary and never
    receives the rows it had not uploaded. A rerun whose own summary holds every required key and no native aggregate
    reads no earlier run and returns at once. A first attempt has no earlier runs and only
    rule 1 applies.

    :param str base_url: Normalized W&B API endpoint.
    :param str entity: W&B entity that owns the project.
    :param str project: W&B project containing the run.
    :param str run_id: The attempt's own W&B run.
    :param float poll_seconds: Delay between summary polls.
    :param float timeout_seconds: Budget used when no ``deadline`` is given.
    :param Iterable[str] required_keys: Numeric keys required before accepting a capture.
    :param Iterable[str] presence_keys: Keys whose presence is recorded without their values.
    :param str | None objective_key: The objective's summary key when W&B supplies the
        objective, defaults to None.
    :param Sequence[WandbRunAddress] earlier: The trial's earlier runs, oldest first.
    :param str evaluation_axis: Trainer progress key shared by native min/max aggregates.
    :param float | None deadline: Absolute ``time.monotonic()`` deadline from the parent.
    :return dict[str, object]: Numeric values, present keys, and retrieval time.
    :raises WandbPollTimeout: The run did not finish with the required keys in time,
        reached when the last poll got a usable answer about the run.
    :raises WandbRunMissingError: The final lookup at the deadline found no matching run.
    :raises WandbNothingToScoreError: At the deadline a rerun's run and its trial's earlier
        runs leave no result to score.
    :raises WandbSetupError: Authentication or setup was denied.
    :raises WandbRunTerminalError: The attempt's run terminated unsuccessfully.
    :raises WandbMissingKeysError: At the deadline no run's summary held a required key.
    :raises ValueError: A requested scalar is invalid or a series cannot be placed on the axis.
    """
    from phasesweep.runtime.time import utc_now_iso

    started = time.monotonic()
    deadline = started + timeout_seconds if deadline is None else deadline
    last_error: Exception | None = None
    # Whether the latest poll got an answer that can settle the trial at the deadline.
    reached = False
    # What the latest answered poll decides at the deadline: the keys no run held and the
    # keys the scoring summary had, or a capture or the reason nothing is scored.
    missing: list[str] = []
    available: list[str] = []
    verdict: dict[str, object] | str | None = None
    required = tuple(required_keys)
    presence = tuple(presence_keys)
    own: WandbRunAddress = {
        "base_url": base_url,
        "entity": entity,
        "project": project,
        "run_id": run_id,
    }
    # The latest settled scoring and the answers it read, so unchanged runs waited on to the
    # deadline are not read through their histories again on every poll. An unsettled one
    # is read again, since history W&B serves late changes no answer the key sees.
    cache: list[tuple[str, _Scored]] = []

    def score(scoring: _Run, older: Sequence[_Run]) -> _Scored:
        key = _answer_key((*older, scoring))
        if cache and cache[0][0] == key:
            return cache[0][1]
        scored = _score(
            scoring,
            older,
            required=required,
            presence=presence,
            axis=evaluation_axis,
            deadline=deadline,
        )
        cache[:] = [] if scored.unsettled else [(key, scored)]
        return scored

    def capture(scored: _Scored) -> dict[str, object]:
        return {
            "values": dict(scored.values),
            "present_keys": list(scored.present),
            "retrieved_at": utc_now_iso(timespec="microseconds"),
        }

    def earlier_runs(clients: dict[str, _Api]) -> list[tuple[WandbRunAddress, _Run]]:
        found = ((address, _lookup(clients, address, deadline)) for address in earlier)
        return [(address, prior) for address, prior in found if prior is not None]

    def judge_earlier(
        clients: dict[str, _Api], subject: str
    ) -> tuple[dict[str, object] | str | None, list[str], list[str]]:
        runs = [
            (address, run)
            for address, run in earlier_runs(clients)
            if not (run.state in _CLOSED_STATES and _logged_nothing(run))
        ]
        if not runs:
            return f"{subject}, and no earlier attempt of its trial logged to a W&B run", [], []
        address, newest = runs[-1]
        name = f"{address['entity']}/{address['project']}/{address['run_id']}"
        summary = newest.summary_metrics
        if newest.state not in _CLOSED_STATES:
            return (
                f"{subject}, and its trial's newest earlier run {name} is {newest.state!r}, so "
                "it may lack its session's last row",
                [],
                [],
            )
        if not _summary_matches_history(newest, summary):
            raise _NoAnswer(
                f"The closed earlier run {name}'s summary does not yet match its history index."
            )
        if objective_key is not None and objective_key not in summary:
            return (
                f"{subject}, and its trial's newest earlier run {name} does not hold "
                f"{objective_key!r}",
                [],
                [],
            )
        scored = score(newest, [prior for _, prior in runs[:-1]])
        if scored.unsettled:
            raise _NoAnswer(
                f"The closed earlier run {name}'s history does not yet reproduce its summary aggregate."
            )
        if scored.missing:
            return None, scored.missing, sorted(summary)
        found = {**capture(scored), "deadline_verdict": True, "earlier_attempt": address["run_id"]}
        return found, [], []

    while time.monotonic() < deadline:
        clients: dict[str, _Api] = {}
        poll_error: Exception | None = None
        poll_verdict: dict[str, object] | str | None = None
        poll_missing: list[str] = []
        poll_available: list[str] = []
        try:
            run = _lookup(clients, own, deadline)
            if run is None:
                poll_error = WandbRunMissingError(run_id)
                if earlier:
                    poll_verdict, poll_missing, poll_available = judge_earlier(
                        clients, "was not found"
                    )
            else:
                summary = run.summary_metrics
                finished = run.state == "finished"
                unreported = objective_key is not None and objective_key not in summary
                if earlier and finished and unreported and _logged_nothing(run):
                    poll_verdict, poll_missing, poll_available = judge_earlier(
                        clients, "finished without logging any rows or user summary keys"
                    )
                elif finished and not _summary_matches_history(run, summary):
                    raise _NoAnswer(
                        "The finished run's summary is still empty."
                        if not summary
                        else "W&B has not served a consistent summary and history index "
                        f"(summary _step={summary.get('_step')!r}, indexed step={_last_history_step(run)!r}, "
                        f"history rows={_row_count(run)!r})."
                    )
                elif run.state in {"failed", "crashed", "killed", "preempted"}:
                    raise WandbRunTerminalError(run_id, run.state)
                elif finished and earlier and unreported:
                    poll_verdict = (
                        f"finished with logged data but without {objective_key!r}, so its "
                        "trial's earlier runs do not score it"
                    )
                elif finished:
                    needs_earlier = any(
                        key not in summary or key.endswith((".min", ".max")) for key in required
                    ) or any(key not in summary for key in presence)
                    older = [prior for _, prior in earlier_runs(clients)] if needs_earlier else []
                    scored = score(run, older)
                    if scored.missing:
                        poll_missing, poll_available = scored.missing, sorted(summary)
                    elif not scored.borrowed and not scored.unsettled:
                        return capture(scored)
                    else:
                        poll_verdict = {**capture(scored), "deadline_verdict": True}
        except _PastDeadline:
            # The latest answered poll still decides.
            break
        except (WandbSetupError, WandbRunTerminalError):
            raise
        except Exception as exc:
            if not isinstance(exc, _NoAnswer) and not _is_retryable_setup_error(exc):
                raise
            reached, verdict, missing = False, None, []
            last_error = RuntimeError(
                str(exc) if isinstance(exc, _NoAnswer) else _error_detail(exc)
            )
        else:
            # An answer in hand is used even when it arrived after the polling deadline: the
            # reserve before the supervisor's deadline exists to report it.
            reached, last_error, verdict = True, poll_error, poll_verdict
            missing, available = poll_missing, poll_available
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    waited = time.monotonic() - started
    if isinstance(verdict, dict):
        return verdict
    if isinstance(verdict, str):
        raise WandbNothingToScoreError(run_id, verdict)
    if reached and missing:
        raise WandbMissingKeysError(run_id, missing, available, waited)
    if isinstance(last_error, WandbRunMissingError):
        raise last_error
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
    A positive row count guarantees there is a history step to wait for, even when both
    the summary step and the index are still absent.

    :param object run: Public-API run.
    :param Mapping[str, object] summary: Its decoded summary.
    :return bool: Whether the summary is populated and its step matches the index.
        A summary written directly can have values without a history step.
    """
    last_step = _last_history_step(run)
    if last_step is None and _row_count(run):
        return False
    return bool(summary) and finite_nonnegative(summary.get("_step")) == last_step


def _row_count(run: object) -> int | None:
    """Return the number of history rows the service holds for a run.

    ``historyLineCount`` arrives in the same ``Api.run`` answer as the state and summary and
    is exact within a second of a trainer's exit, while the summary and history index can
    take half a minute, so it tells a run that logged no row from one still being ingested.

    :param object run: Public-API run.
    :return int | None: The count, or ``None`` when it is not a non-negative integer.
    """
    rows = getattr(run, "historyLineCount", None)
    return rows if type(rows) is int and rows >= 0 else None


def _logged_nothing(run: _Run) -> bool:
    """Return whether W&B reports zero rows and only its own underscore-prefixed summary keys.

    :param _Run run: The run as served in this poll.
    :return bool: Whether the trainer logged neither history nor a summary value.
    """
    return _row_count(run) == 0 and all(key.startswith("_") for key in run.summary_metrics)


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
    run: _LoadableRun | None = None
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

    response: dict[str, object]
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
    except WandbNothingToScoreError as exc:
        response = {"status": "nothing_to_score", "cause": exc.cause}
    except WandbPollTimeout as exc:
        response = {
            "status": "timeout",
            "cause": str(exc.last_error) if exc.last_error else None,
            "reached": exc.reached,
        }
    except _RetainedHistoryError as exc:
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
