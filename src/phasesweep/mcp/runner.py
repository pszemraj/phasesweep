"""Run one detached sweep and persist its terminal status."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from phasesweep.config.io import load_experiment_snapshot
from phasesweep.config.models import Experiment
from phasesweep.engine.errors import (
    ActiveAttemptPersistenceError,
    ArtifactRootConflictError,
    ExperimentLockBusyError,
    PublishedStudyMissingError,
    StudyContextConflictError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrackingTargetUnavailableError,
    TrialTargetRefusedError,
)
from phasesweep.engine.run import PublicationHook, TerminalReport, run_experiment
from phasesweep.engine.selection import (
    PhaseAbortRefusedError,
    PhaseTrialsFailedError,
    PhaseTrialsInfeasibleError,
    ReplayWithoutFeasibleTrialError,
)
from phasesweep.engine.state import Winner
from phasesweep.errors import (
    OPERATOR_STEPS,
    OperatorAction,
    PhaseSweepError,
    ProcessCleanupUncertainError,
    operator_step,
)
from phasesweep.mcp.runs import (
    LAUNCH_ACK_BYTE,
    LAUNCH_READY_BYTE,
    RUNNER_HEARTBEAT_INTERVAL_SECONDS,
    RunHandle,
    RunStore,
    write_status_file,
)
from phasesweep.mcp.snapshots import (
    capture_pre_generation_result_snapshot,
    capture_result_snapshot,
    finalize_result_snapshot,
    mark_result_snapshot_published,
)
from phasesweep.runtime.files import UnsafeLockPathError, open_private_text
from phasesweep.runtime.reaper import read_boot_id, read_pid_namespace, read_proc_starttime
from phasesweep.runtime.shutdown import (
    PhaseSweepShutdown,
    absorb_shutdown_signals,
    defer_shutdown_signals,
    install_signal_handlers,
)
from phasesweep.runtime.time import utc_now_iso

FailureCode: TypeAlias = Literal[
    "fingerprint_mismatch",
    "artifact_root_conflict",
    "study_schema_mismatch",
    "storage_unavailable",
    "tracking_unavailable",
    "published_study_missing",
    "trial_target_refused",
    "experiment_busy",
    "trials_failed",
    "trials_infeasible",
    "replay_without_feasible_trial",
    "timeout",
    "cleanup_uncertain",
    "cancelled",
    "result_snapshot_unavailable",
    "publication_not_committed",
    "internal_error",
]

FailureStage: TypeAlias = Literal["preflight", "execution", "cleanup"]
FailureActor: TypeAlias = Literal["agent", "operator"]


class FailureCausePayload(BaseModel):
    """Safe secondary cause retained beneath an actionable terminal failure."""

    model_config = ConfigDict(extra="forbid")

    code: FailureCode
    stage: FailureStage
    retryable: bool
    actor: FailureActor
    remediation: str


class FailurePayload(FailureCausePayload):
    """Path-free terminal failure category and recovery policy."""

    cause: FailureCausePayload | None = None


_SELF_CONTAINED_STEPS = frozenset({OperatorAction.RUN_RECOVER_RUN, OperatorAction.INSPECT_LOGS})
_RUN_LOG_DETAILS = "The error in the PhaseSweep run log gives the details."


def _operator_remediation(steps: Sequence[OperatorAction], *, before: str = "") -> str:
    """Compose operator steps into one remediation, in the order they must happen.

    :param Sequence[OperatorAction] steps: Non-empty operator steps, none of them ``RETRY``.
    :param str before: Clause naming what every step must precede, if any.
    :return str: The remediation, pointing at the run log when a step needs
        the error's details.
    """
    phrases = [
        "run phasesweep mcp recover-run"
        if step is OperatorAction.RUN_RECOVER_RUN
        else OPERATOR_STEPS[step]
        for step in steps
    ]
    joined = phrases[0]
    if len(phrases) > 1:
        joined = f"{', and '.join(phrases[:-1])}, then {phrases[-1]}"
    remediation = f"Ask the operator to {joined}{before}."
    if not _SELF_CONTAINED_STEPS.issuperset(steps):
        remediation = f"{remediation} {_RUN_LOG_DETAILS}"
    return remediation


def _routed_next_steps(error: PhaseSweepError) -> dict[str, object]:
    """Return the retryability, actor, and remediation a raise's routed action calls for.

    :param PhaseSweepError error: Failure carrying the routed action and its concrete detail.
    :return dict[str, object]: ``retryable``, ``actor``, and ``remediation``
        payload fields: the agent's retry when the action is ``RETRY``, and
        otherwise the operator's step.
    """
    if error.action is OperatorAction.RETRY:
        return {"retryable": True, "actor": "agent", "remediation": operator_step(error)}
    remediation = (
        f"Ask the operator: {operator_step(error)}"
        if error.remediation is not None
        else _operator_remediation([error.action])
    )
    return {"retryable": False, "actor": "operator", "remediation": remediation}


def _cleanup_uncertain_payload(actions: Sequence[OperatorAction]) -> dict[str, object]:
    """Build a cleanup-uncertain failure with required repairs followed by recover-run.

    :param Sequence[OperatorAction] actions: Steps the cleanup routes, in order.
    :return dict[str, object]: The ``"cleanup_uncertain"`` payload.
    """
    last = OperatorAction.RUN_RECOVER_RUN
    repairs = [action for action in actions if action not in (OperatorAction.RETRY, last)]
    return {
        "code": "cleanup_uncertain",
        "stage": "cleanup",
        "retryable": False,
        "actor": "operator",
        "remediation": _operator_remediation([*repairs, last], before=" before another launch"),
    }


# First matching exception type determines the durable failure code.
_FAILURE_CODES: tuple[tuple[type[PhaseSweepError], FailureCode], ...] = (
    (ProcessCleanupUncertainError, "cleanup_uncertain"),
    (ExperimentLockBusyError, "experiment_busy"),
    (ArtifactRootConflictError, "artifact_root_conflict"),
    (StudyFingerprintMismatchError, "fingerprint_mismatch"),
    (StudyContextConflictError, "fingerprint_mismatch"),
    (StudySchemaMismatchError, "study_schema_mismatch"),
    (PublishedStudyMissingError, "published_study_missing"),
    (StudyStorageUnavailableError, "storage_unavailable"),
    (UnsafeLockPathError, "storage_unavailable"),
    (TrackingTargetUnavailableError, "tracking_unavailable"),
    (ActiveAttemptPersistenceError, "storage_unavailable"),
    (TrialTargetRefusedError, "trial_target_refused"),
    (PhaseAbortRefusedError, "trial_target_refused"),
    (PhaseTrialsFailedError, "trials_failed"),
    (PhaseTrialsInfeasibleError, "trials_infeasible"),
    (ReplayWithoutFeasibleTrialError, "replay_without_feasible_trial"),
)


def _unrouted_failure_payload(error: BaseException, *, stage: str) -> dict[str, object]:
    """Map an exception that carries no operator action to its fixed failure.

    :param BaseException error: Exception that is not a :class:`PhaseSweepError`.
    :param str stage: Stage the failure occurred at.
    :return dict[str, object]: A ``"timeout"`` or ``"cancelled"`` payload, or
        ``"internal_error"`` for any other exception.
    """
    payload: dict[str, object] = {
        "code": "internal_error",
        "stage": stage,
        "retryable": False,
        "actor": "operator",
        "remediation": "Ask the operator to inspect the PhaseSweep run log before retrying.",
    }
    if isinstance(error, TimeoutError):
        payload.update(
            code="timeout",
            remediation=(
                "Ask the operator to review the configured wallclock budget before retrying."
            ),
        )
    elif isinstance(error, PhaseSweepShutdown):
        payload.update(
            code="cancelled",
            retryable=True,
            actor="agent",
            remediation="Start a new run only if the user still wants the sweep to continue.",
        )
    return payload


def _base_failure_payload(
    error: BaseException,
    *,
    stage: str | None,
) -> dict[str, object]:
    """Map an operator-facing exception to one stable, path-free agent failure.

    The error's type selects ``code``. For a :class:`PhaseSweepError`, ``retryable``,
    ``actor``, and ``remediation`` come from the action and concrete detail its raise routes; an
    exception without a routed action keeps its type's fixed text.

    :param BaseException error: Exception whose type selects the failure code.
    :param str | None stage: Failure stage to report; clamped to one of
        ``"preflight"``, ``"execution"``, or ``"cleanup"``, falling back to
        ``"execution"`` for any other value.
        :class:`ProcessCleanupUncertainError` always reports ``"cleanup"``.
    :return dict[str, object]: Payload with ``code``, ``stage``, ``retryable``,
        ``actor``, and ``remediation`` keys; falls back to ``"internal_error"``
        for any exception type not otherwise recognized.
    """
    failure_stage = (
        stage if stage is not None and stage in {"preflight", "cleanup"} else "execution"
    )
    if not isinstance(error, PhaseSweepError):
        return _unrouted_failure_payload(error, stage=failure_stage)
    code: FailureCode = next(
        (code for error_type, code in _FAILURE_CODES if isinstance(error, error_type)),
        "internal_error",
    )
    if code == "cleanup_uncertain":
        return _cleanup_uncertain_payload([error.action])
    return {"code": code, "stage": failure_stage, **_routed_next_steps(error)}


def _cleanup_steps(
    primary: BaseException, cleanup_error: BaseException | None
) -> list[OperatorAction]:
    """Return the steps an unconfirmed cleanup routes, ahead of recover-run.

    A cleanup refusal routes its own repair. Otherwise the cleanup error the
    engine recorded supplies it, and a storage primary adds the ledger that
    recover-run must read first.

    :param BaseException primary: Terminal error observed before cleanup was
        found to be uncertain.
    :param BaseException | None cleanup_error: The engine's recorded cleanup failure.
    :return list[OperatorAction]: De-duplicated steps in order.
    """
    if isinstance(primary, ProcessCleanupUncertainError):
        return [primary.action]
    steps: list[OperatorAction] = []
    if isinstance(primary, StudyStorageUnavailableError):
        steps.append(OperatorAction.RESTORE_LEDGER)
    if isinstance(cleanup_error, ProcessCleanupUncertainError):
        steps.append(cleanup_error.action)
    else:
        steps.append(ProcessCleanupUncertainError.default_action)
    return list(dict.fromkeys(steps))


def _cleanup_failure_payload(
    primary: BaseException,
    *,
    cause_stage: str | None,
    cleanup_error: BaseException | None = None,
) -> dict[str, object]:
    """Make cleanup uncertainty actionable while retaining its safe primary cause.

    :param BaseException primary: Terminal error observed before cleanup was
        found to be uncertain.
    :param str | None cause_stage: Stage at which ``primary`` occurred, used
        when classifying it as the nested cause.
    :param BaseException | None cleanup_error: The engine's recorded cleanup
        failure, whose repairs the payload keeps.
    :return dict[str, object]: A ``"cleanup_uncertain"`` failure payload with
        ``primary`` nested under ``"cause"``; a ``ProcessCleanupUncertainError``
        primary nests only its ``StudyStorageUnavailableError`` cause, when it has one.
        For an existing cleanup error, retain its explicit storage failure when that
        prevents inspection.
    """
    payload = _cleanup_uncertain_payload(_cleanup_steps(primary, cleanup_error))
    # Preserve the primary cause's policy beneath the authoritative cleanup refusal.
    if isinstance(primary, ProcessCleanupUncertainError):
        if isinstance(primary.__cause__, StudyStorageUnavailableError):
            payload["cause"] = _base_failure_payload(primary.__cause__, stage=cause_stage)
    else:
        payload["cause"] = _base_failure_payload(primary, stage=cause_stage)
    return FailurePayload.model_validate(payload).model_dump(mode="json", exclude_none=True)


def _terminal_error(
    report: TerminalReport | None,
    fallback: BaseException,
) -> tuple[BaseException, str | None]:
    """Return the engine's primary error and stage, or the caller's fallback.

    The engine delivers a report only from the block that can claim its generation,
    so a failure with no report came before any claim and is a preflight failure.

    :param TerminalReport | None report: Engine terminal report, when one was delivered.
    :param BaseException fallback: Exception caught by the runner.
    :return tuple[BaseException, str | None]: Authoritative error and failure stage.
    """
    if report is None:
        return fallback, "preflight"
    return report.primary_error or fallback, report.failure_stage


def _recorded_cleanup_error(report: TerminalReport | None) -> BaseException | None:
    """Return the cleanup failure an engine report recorded, if one was delivered.

    :param TerminalReport | None report: Engine terminal report, when one was delivered.
    :return BaseException | None: The report's cleanup error.
    """
    return report.cleanup_error if report is not None else None


def _terminal_failure_payload(
    error: BaseException,
    *,
    stage: str | None,
    cleanup_confirmed: bool,
    cleanup_error: BaseException | None = None,
) -> dict[str, object]:
    """Classify one terminal error, making cleanup uncertainty authoritative.

    :param BaseException error: Primary terminal error.
    :param str | None stage: Stage at which the primary error occurred.
    :param bool cleanup_confirmed: Whether child-process cleanup was confirmed.
    :param BaseException | None cleanup_error: The engine's recorded cleanup
        failure, when cleanup is unconfirmed.
    :return dict[str, object]: Validated, path-free terminal failure payload.
    """
    if not cleanup_confirmed:
        return _cleanup_failure_payload(error, cause_stage=stage, cleanup_error=cleanup_error)
    payload = _base_failure_payload(error, stage=stage)
    return FailurePayload.model_validate(payload).model_dump(mode="json", exclude_none=True)


# Terminal persistence retries use a sub-second backoff budget.
_STATUS_WRITE_BACKOFF_SECONDS: tuple[float, ...] = (0.05, 0.25)


def _write_status_file_with_retry(
    status_path: Path,
    payload: dict[str, object],
    *,
    attempts: int = 3,
) -> None:
    """Write terminal status, retrying only :class:`OSError`.

    Serialization errors propagate immediately for the caller to record a reduced payload.

    :param Path status_path: Destination ``status.json`` path for the run.
    :param dict[str, object] payload: JSON-serializable terminal status payload.
    :param int attempts: Total attempts, including the first; must be at least one.
    :raises OSError: The final attempt's persistence failure, once the retry
        budget is exhausted.
    :raises Exception: Any non-:class:`OSError` failure, from the first attempt
        and without retrying.
    """
    log = logging.getLogger("phasesweep.mcp.runner")
    for attempt in range(attempts):
        try:
            write_status_file(status_path, payload)
        except OSError:
            if attempt >= attempts - 1:
                raise
            backoff = _STATUS_WRITE_BACKOFF_SECONDS[
                min(attempt, len(_STATUS_WRITE_BACKOFF_SECONDS) - 1)
            ]
            log.warning(
                "failed to persist terminal status (attempt %d of %d); retrying in %.2fs",
                attempt + 1,
                attempts,
                backoff,
                exc_info=True,
            )
            time.sleep(backoff)
        else:
            return


class _RunnerPublicationHook(PublicationHook):
    """Durably bridge the engine publication pointer to one MCP run snapshot."""

    def __init__(self, status_path: Path, status: dict[str, object]) -> None:
        """Create an empty publication transaction for one detached run.

        :param Path status_path: Durable MCP status file for the run.
        :param dict[str, object] status: Shared terminal-status facts.
        """
        self._status_path = status_path
        self._status = status
        self.snapshot: dict[str, object] | None = None
        self.state: Literal["prepared", "committed"] | None = None
        self.generation_id: str | None = None
        self.summary: Mapping[str, object] | None = None

    def prepare(
        self,
        *,
        experiment: Experiment,
        generation_id: str,
        winners: Mapping[str, Winner],
        summary: Mapping[str, object],
    ) -> None:
        """Persist the exact run result before the last-success pointer advances.

        :param Experiment experiment: Executed frozen experiment configuration.
        :param str generation_id: Generation ready for publication.
        :param Mapping[str, Winner] winners: Engine-selected generation winners.
        :param Mapping[str, object] summary: Validated generation summary awaiting publication.
        :raises Exception: If capture, validation, or durable persistence fails.
        """
        snapshot = capture_result_snapshot(
            experiment,
            generation_id=generation_id,
            engine_winners=winners,
        )
        prepared = {
            **self._status,
            "result_snapshot_state": "pending",
            "result_snapshot": snapshot,
            "result_publication_state": "prepared",
            "result_publication_generation_id": generation_id,
        }
        _write_status_file_with_retry(self._status_path, prepared)
        self.snapshot = snapshot
        self.state = "prepared"
        self.generation_id = generation_id
        self.summary = summary
        self._status["result_publication_state"] = "prepared"
        self._status["result_publication_generation_id"] = generation_id

    def committed(self, *, generation_id: str) -> None:
        """Record the committed publication in memory before persisting it.

        The terminal write retries failures; after a hard exit, recovery compares the durable
        prepared generation with the validated last-success pointer.

        :param str generation_id: Generation whose pointer commit completed.
        :raises RuntimeError: If no matching prepared snapshot exists.
        :raises OSError: If the committed receipt cannot be persisted.
        """
        if (
            self.state != "prepared"
            or self.snapshot is None
            or self.generation_id != generation_id
            or self.summary is None
        ):
            raise RuntimeError("publication commit has no matching prepared run snapshot")
        committed = mark_result_snapshot_published(
            self.snapshot,
            generation_id=generation_id,
            published_summary=self.summary,
        )
        self.snapshot = committed
        self.state = "committed"
        self._status["result_publication_state"] = "committed"
        self._status["result_publication_generation_id"] = generation_id
        _write_status_file_with_retry(
            self._status_path,
            {
                **self._status,
                "result_snapshot_state": "pending",
                "result_snapshot": committed,
            },
        )

    def clear_uncommitted_marker(self) -> None:
        """Remove a normally reported preparation whose pointer never committed."""
        if self.state != "prepared":
            return
        self._status.pop("result_publication_state", None)
        self._status.pop("result_publication_generation_id", None)
        self.state = None
        self.generation_id = None
        self.summary = None


def _write_status(
    status_path: Path,
    payload: dict[str, object],
    *,
    result_snapshot: dict[str, object] | None,
    result_snapshot_error: str | None,
) -> bool:
    """Persist terminal evidence and its captured snapshot without masking the engine failure.

    Return whether any terminal record reached disk. Failed complete writes retain the
    durable pending snapshot for operator finalization. Serialization or capture defects
    record a failed snapshot while preserving the terminal cause.

    Shutdown signals remain deferred until persistence finishes, then propagate.

    :param Path status_path: JSON file where terminal cause should be recorded.
    :param dict[str, object] payload: Status payload containing run id, return code, and error class.
    :param dict[str, object] | None result_snapshot: Raw snapshot captured under the experiment lock.
    :param str | None result_snapshot_error: Capture error class when no snapshot exists.
    :return bool: Whether durable terminal evidence exists, i.e. some record
        (``pending``, ``complete``, or ``failed``) reached disk. ``False`` only
        when nothing could be persisted at all.
    """
    log = logging.getLogger("phasesweep.mcp.runner")
    with defer_shutdown_signals():
        terminal = {
            **payload,
            "ended_at": utc_now_iso(),
            "result_snapshot_state": "pending",
        }
        if result_snapshot is not None:
            terminal["result_snapshot"] = result_snapshot
        try:
            _write_status_file_with_retry(status_path, terminal)
        except OSError:
            log.exception(
                "failed to persist terminal status.json after every retry; the sweep's own "
                "outcome is unaffected but this run has no terminal MCP evidence"
            )
            return False
        except Exception as exc:  # terminal evidence must not mask the run's exit
            log.exception("failed to write status.json with the captured result snapshot")
            terminal.pop("result_snapshot", None)
            terminal["result_snapshot_state"] = "failed"
            terminal["result_snapshot_error"] = result_snapshot_error or type(exc).__name__
            try:
                _write_status_file_with_retry(status_path, terminal)
            except Exception:  # no further persistence fallback is available
                log.exception("failed to write status.json without the result snapshot")
                return False
            return True

        try:
            if result_snapshot is None:
                raise RuntimeError(result_snapshot_error or "terminal snapshot was not captured")
            terminal["result_snapshot"] = finalize_result_snapshot(
                result_snapshot,
            )
        except Exception as exc:  # minimal terminal evidence is already durable
            terminal.pop("result_snapshot", None)
            terminal["result_snapshot_state"] = "failed"
            terminal["result_snapshot_error"] = result_snapshot_error or type(exc).__name__
            log.exception("failed to finalize terminal result snapshot")
        else:
            terminal["result_snapshot_state"] = "complete"

        try:
            _write_status_file_with_retry(status_path, terminal)
        except OSError:
            log.exception(
                "failed to persist the finalized result snapshot state after every retry; "
                "the durable pending record still holds the captured snapshot and remains "
                "recoverable with `phasesweep mcp recover-run`"
            )
            return True
        except Exception as exc:  # preserve a serializable failed state
            log.exception("failed to finalize result snapshot state in status.json")
            terminal.pop("result_snapshot", None)
            terminal["result_snapshot_state"] = "failed"
            terminal["result_snapshot_error"] = type(exc).__name__
            try:
                _write_status_file_with_retry(status_path, terminal)
            except Exception:  # no further persistence fallback is available
                log.exception("failed to persist result snapshot finalization failure")
        return True


class _RunnerHeartbeat:
    """Refresh one run's heartbeat file from a daemon thread until stopped.

    A reader that cannot observe this runner's PID namespace takes a stale heartbeat
    as a sign that the runner has probably stopped. A refresh only advances the file's
    modification time: it writes no data and syncs nothing.
    """

    def __init__(self, path: Path, interval_seconds: float) -> None:
        """Create the private heartbeat file and start refreshing it.

        :param Path path: The run's heartbeat path in the MCP state directory.
        :param float interval_seconds: Seconds between refreshes.
        :raises OSError: If the heartbeat file cannot be created.
        :raises UnsafePrivatePathError: If the heartbeat path or its directory is unsafe.
        """
        self._handle = open_private_text(path, "w")
        self._interval_seconds = interval_seconds
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._refresh, name="phasesweep-runner-heartbeat", daemon=True
        )
        self._thread.start()

    def _refresh(self) -> None:
        """Advance the heartbeat's modification time once per interval until stopped."""
        failing = False
        while not self._stopped.wait(self._interval_seconds):
            try:
                os.utime(self._handle.fileno())
            except OSError:
                if not failing:
                    logging.getLogger("phasesweep.mcp.runner").warning(
                        "could not refresh the runner heartbeat; a reader in another PID "
                        "namespace may report that this runner has probably stopped",
                        exc_info=True,
                    )
                failing = True
            else:
                failing = False

    def stop(self) -> None:
        """Stop refreshing and close the heartbeat file, which stays with the run's records."""
        self._stopped.set()
        self._thread.join()
        self._handle.close()


def _persist_spawned_handle(
    *,
    state_dir: Path,
    run_id: str,
    experiment_id: str,
    config_sha256: str,
    started_at: str,
    allow_cancel: bool,
) -> RunHandle:
    """Persist this runner's process identity before it launches any training work.

    :param Path state_dir: Server state dir whose ``runs/`` receives the handle.
    :param str run_id: Run id minted by the launching server.
    :param str experiment_id: Catalog id this run belongs to.
    :param str config_sha256: Hash of the config snapshot this runner executes.
    :param str started_at: ISO-8601 UTC launch timestamp recorded by the server.
    :param bool allow_cancel: Cancel permission frozen at launch time.
    :raises RuntimeError: If kernel process start time, boot id, or PID namespace
        is unavailable, so the handle could not be made PID-reuse safe, or the
        server never created a pending handle for ``run_id``.
    :return RunHandle: The spawned handle now on disk.
    """
    store = RunStore(state_dir)
    pid = os.getpid()
    pgid = os.getpgrp()
    pid_starttime = read_proc_starttime(pid)
    if pid_starttime is None:
        raise RuntimeError(
            "cannot persist a PID-reuse-safe MCP runner handle because process "
            "start time is unavailable"
        )
    boot_id = read_boot_id()
    if boot_id is None:
        raise RuntimeError(
            "cannot persist a PID-reuse-safe MCP runner handle because boot id is unavailable"
        )
    pid_namespace = read_pid_namespace()
    if pid_namespace is None:
        raise RuntimeError(
            "cannot persist a PID-reuse-safe MCP runner handle because the host PID "
            "namespace is unavailable"
        )
    pending = store.get(run_id)
    if pending is None:
        raise RuntimeError(f"cannot persist runner identity for unknown run {run_id!r}")
    handle = replace(
        pending,
        experiment_id=experiment_id,
        config_sha256=config_sha256,
        pid=pid,
        pgid=pgid,
        pid_starttime=pid_starttime,
        started_at=started_at,
        launch_state="spawned",
        allow_cancel=allow_cancel,
        boot_id=boot_id,
        pid_namespace=pid_namespace,
    )
    store.update(handle)
    return handle


def _complete_launch_handshake(ready_fd: int, ack_fd: int, lease_fd: int) -> None:
    """Report a durable process receipt and block until the server acknowledges it.

    :param int ready_fd: Pipe descriptor receiving the runner-ready byte.
    :param int ack_fd: Pipe descriptor supplying the server acknowledgement.
    :param int lease_fd: Inherited lease proving a child exists before its receipt.
    :raises RuntimeError: If either half of the launch handshake is unavailable.
    """
    try:
        if os.write(ready_fd, LAUNCH_READY_BYTE) != len(LAUNCH_READY_BYTE):
            raise RuntimeError("could not report the durable MCP launch receipt")
    except OSError as exc:
        raise RuntimeError("could not report the durable MCP launch receipt") from exc
    finally:
        with contextlib.suppress(OSError):
            os.close(ready_fd)
        with contextlib.suppress(OSError):
            os.close(lease_fd)
    try:
        acknowledgement = os.read(ack_fd, len(LAUNCH_ACK_BYTE))
    except OSError as exc:
        raise RuntimeError("could not receive the MCP launch acknowledgement") from exc
    finally:
        with contextlib.suppress(OSError):
            os.close(ack_fd)
    if acknowledgement != LAUNCH_ACK_BYTE:
        raise RuntimeError("MCP server exited before acknowledging the runner launch")


def main(argv: list[str] | None = None) -> int:
    """Run one config to completion and record its terminal cause in status.json.

    :param list[str] | None argv: Argument vector; defaults to ``sys.argv``.
    :return int: Zero for a completed sweep with durable terminal evidence; one if the
        sweep completed but status.json could not be persisted.
    :raises RuntimeError: The config snapshot is unreadable or its hash mismatches.
    :raises PhaseSweepShutdown: Cancellation, after recording its cause. A signal absorbed
        during successful terminal capture preserves that successful status and exits
        with the POSIX signalled code.
    :raises ProcessCleanupUncertainError: Cleanup is unconfirmed, after recording uncertainty.
    :raises BaseException: Handle-write or engine failure, after recording its terminal cause.
    """
    parser = argparse.ArgumentParser(prog="phasesweep mcp runner")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True, type=Path)  # snapshot, server-supplied
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--status-path", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--launch-ready-fd", type=int)
    parser.add_argument("--launch-ack-fd", type=int)
    parser.add_argument("--launch-lease-fd", type=int)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--allow-cancel", action="store_true")
    parser.add_argument("--from-phase", default=None)
    args = parser.parse_args(argv)
    launch_fds = (args.launch_ready_fd, args.launch_ack_fd, args.launch_lease_fd)
    if any(fd is None for fd in launch_fds) and any(fd is not None for fd in launch_fds):
        parser.error(
            "--launch-ready-fd, --launch-ack-fd, and --launch-lease-fd must be supplied together"
        )

    project_cwd = args.cwd.expanduser()
    config_path = args.config
    status_path = args.status_path
    state_dir = args.state_dir

    # stderr is the server-captured run log; stdout stays free of runner diagnostics.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname).1s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Config loading and identity persistence must also produce cancellation status.
    install_signal_handlers()

    status: dict[str, object] = {
        "run_id": args.run_id,
        "from_phase": args.from_phase,
        "returncode": 0,
        "error_class": None,
        "cleanup_confirmed": True,
        "failure": None,
    }
    result_snapshot: dict[str, object] | None = None
    result_snapshot_error: str | None = None
    terminal_report: TerminalReport | None = None
    config: Experiment | None = None
    status_persisted = False
    publication_hook = _RunnerPublicationHook(status_path, status)
    heartbeat: _RunnerHeartbeat | None = None
    try:
        # The heartbeat starts before the identity is durable, so no reader finds a
        # spawned runner without one, and stops only once terminal status is durable.
        heartbeat = _RunnerHeartbeat(
            RunStore(state_dir).heartbeat_path(args.run_id), RUNNER_HEARTBEAT_INTERVAL_SECONDS
        )
        # Persist identity even if the spawning server has exited.
        handle = _persist_spawned_handle(
            state_dir=state_dir,
            run_id=args.run_id,
            experiment_id=args.experiment_id,
            config_sha256=args.config_sha256,
            started_at=args.started_at,
            allow_cancel=args.allow_cancel,
        )
        if (
            args.launch_ready_fd is not None
            and args.launch_ack_fd is not None
            and args.launch_lease_fd is not None
        ):
            try:
                _complete_launch_handshake(
                    args.launch_ready_fd,
                    args.launch_ack_fd,
                    args.launch_lease_fd,
                )
            finally:
                RunStore(state_dir).remove_launch_lease(args.run_id)
        os.chdir(project_cwd)
        try:
            # Resolve relative config paths against the recorded launch directory.
            config = load_experiment_snapshot(
                config_path,
                args.config_sha256,
                source=f"run snapshot {args.run_id}",
                config_dir=Path(handle.config_dir),
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc

        def capture_terminal(report: TerminalReport) -> None:
            """Capture terminal cleanup facts and unpublished results under the experiment lock.

            Successful publication reuses the prepared snapshot. Shutdown signals remain pending
            across the engine's diagnostic callback boundary until terminal persistence.

            :param TerminalReport report: Engine outcome and cleanup evidence.
            """
            nonlocal result_snapshot, result_snapshot_error, terminal_report
            with absorb_shutdown_signals() as absorbed:
                terminal_report = report
                status["cleanup_confirmed"] = report.cleanup_confirmed
                status["recovered_attempt_ids"] = sorted(report.recovered_attempt_ids)
                status["recovered_attempt_generations"] = dict(
                    sorted(report.recovered_attempt_generations.items())
                )
                status["uncertain_attempt_ids"] = sorted(report.uncertain_attempt_ids)
                if report.primary_error is not None:
                    status["failure"] = _terminal_failure_payload(
                        report.primary_error,
                        stage=report.failure_stage,
                        cleanup_confirmed=report.cleanup_confirmed,
                        cleanup_error=report.cleanup_error,
                    )
                if publication_hook.snapshot is not None:
                    result_snapshot = dict(publication_hook.snapshot)
                else:
                    try:
                        # A replay that wrote no generation represents the one it matched;
                        # a failure before the claim represents none.
                        result_snapshot = capture_result_snapshot(
                            config,
                            generation_id=report.replayed_generation_id or report.generation_id,
                            trial_generation_id=report.generation_id,
                            generation_claimed=report.failure_stage != "preflight",
                            engine_winners=report.winners,
                        )
                    except Exception as exc:  # preserve engine terminal cause
                        result_snapshot_error = type(exc).__name__
                        logging.getLogger("phasesweep.mcp.runner").exception(
                            "failed to capture terminal result snapshot under the experiment lock"
                        )
            if absorbed.signum is not None:
                logging.getLogger("phasesweep.mcp.runner").warning(
                    "shutdown signal %d arrived while the terminal result snapshot was being "
                    "captured; it was held until the snapshot was safe and will be honored "
                    "once terminal status is durable",
                    absorbed.signum,
                )

        run_experiment(
            config,
            from_phase=args.from_phase,
            dry_run=False,
            terminal_callback=capture_terminal,
            generation_id=args.run_id,
            publication_hook=publication_hook,
        )
    except PhaseSweepShutdown as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        cleanup_confirmed = (
            terminal_report.cleanup_confirmed
            if terminal_report is not None
            else exc.report.cleanup_confirmed
        )
        status["cleanup_confirmed"] = cleanup_confirmed
        if terminal_report is None or terminal_report.primary_error is not None:
            status["returncode"] = code
            status["error_class"] = "cancelled"
            primary, failure_stage = _terminal_error(terminal_report, exc)
            status["failure"] = _terminal_failure_payload(
                primary,
                stage=failure_stage,
                cleanup_confirmed=cleanup_confirmed,
                cleanup_error=_recorded_cleanup_error(terminal_report),
            )
        raise
    except BaseException as exc:  # record every terminal cause, then re-raise
        status["returncode"] = 1
        primary, failure_stage = _terminal_error(terminal_report, exc)
        cleanup_uncertain = isinstance(exc, ProcessCleanupUncertainError)
        status["error_class"] = type(exc if cleanup_uncertain else primary).__name__
        cleanup_confirmed = (
            terminal_report.cleanup_confirmed
            if terminal_report is not None
            else not cleanup_uncertain
        )
        status["cleanup_confirmed"] = cleanup_confirmed
        status["failure"] = _terminal_failure_payload(
            primary,
            stage=failure_stage,
            cleanup_confirmed=False if cleanup_uncertain else cleanup_confirmed,
            cleanup_error=_recorded_cleanup_error(terminal_report),
        )
        raise
    finally:
        try:
            if publication_hook.snapshot is not None and result_snapshot is None:
                result_snapshot = dict(publication_hook.snapshot)
            if publication_hook.state == "prepared" and status["returncode"] != 0:
                # Reported non-commit clears the marker; a hard exit leaves it for recovery.
                publication_hook.clear_uncommitted_marker()
            if result_snapshot is None and terminal_report is None and config is not None:
                try:
                    result_snapshot = capture_pre_generation_result_snapshot(config).model_dump(
                        mode="json"
                    )
                except Exception as exc:  # preserve the terminal engine failure
                    result_snapshot_error = type(exc).__name__
                else:
                    status["generation_unavailable_reason"] = "engine_generation_not_claimed"
            status_persisted = _write_status(
                status_path,
                status,
                result_snapshot=result_snapshot,
                result_snapshot_error=result_snapshot_error,
            )
        finally:
            if heartbeat is not None:
                heartbeat.stop()
    # Missing terminal evidence makes an otherwise successful run exit nonzero.
    return 0 if status_persisted else 1


if __name__ == "__main__":
    raise SystemExit(main())
