"""Detached orchestrator entrypoint for one sweep.

Spawned by the MCP server in its own session (``start_new_session=True``) with
stdout/stderr redirected to a per-run log. Runs ``run_config`` and records the
terminal cause in ``status.json`` so the server can report succeeded / failed /
cancelled without scraping logs. The path it runs is supplied by the server
from the frozen registry; it is never agent input.

The server starts this process in its own state directory, not the
experiment's, so that nothing in the project tree can execute during
interpreter startup - before any identity exists to clean up (review v0.5.17 /
blocker 9). The project directory arrives as ``--cwd`` and is entered here,
after every import has run and this runner's identity is durable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from phasesweep.config import Experiment
from phasesweep.engine import NoFeasibleTrialError, TerminalReport, run_experiment
from phasesweep.engine.errors import (
    ArtifactRootConflictError,
    ExperimentLockBusyError,
    SamplerContinuationUnsupportedError,
    StudyContextConflictError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrialTargetRegressionError,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError
from phasesweep.mcp.config_snapshot import load_experiment_snapshot
from phasesweep.mcp.runs import RunHandle, RunStore, write_status_file
from phasesweep.mcp.snapshots import (
    capture_pre_generation_result_snapshot,
    capture_result_snapshot,
    finalize_result_snapshot,
)
from phasesweep.mcp.time import utc_now_iso
from phasesweep.runtime.process import (
    PhaseSweepShutdown,
    defer_shutdown_signals,
    install_signal_handlers,
    read_boot_id,
    read_proc_starttime,
)

FailureCode: TypeAlias = Literal[
    "fingerprint_mismatch",
    "artifact_root_conflict",
    "study_schema_mismatch",
    "storage_unavailable",
    "sampler_continuation_unsupported",
    "trial_target_regression",
    "experiment_busy",
    "trainer_failed",
    "timeout",
    "cleanup_uncertain",
    "cancelled",
    "result_snapshot_unavailable",
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


def _base_failure_payload(
    error: BaseException,
    *,
    stage: str | None,
) -> dict[str, object]:
    """Map an operator-facing exception to one stable, path-free agent failure.

    :param BaseException error: Exception whose type selects the failure code.
    :param str | None stage: Failure stage to report for most error types;
        clamped to one of ``"preflight"``, ``"execution"``, or ``"cleanup"``,
        falling back to ``"execution"`` for any other value. Two error types
        override this with a fixed stage regardless of input:
        :class:`ExperimentLockBusyError` always reports ``"preflight"`` and
        :class:`ProcessCleanupUncertainError` always reports ``"cleanup"``.
    :return dict[str, object]: Payload with ``code``, ``stage``, ``retryable``,
        ``actor``, and ``remediation`` keys; falls back to ``"internal_error"``
        for any exception type not otherwise recognized.
    """
    failure_stage = stage if stage in {"preflight", "execution", "cleanup"} else "execution"
    if isinstance(error, ExperimentLockBusyError):
        return {
            "code": "experiment_busy",
            "stage": "preflight",
            "retryable": True,
            "actor": "agent",
            "remediation": (
                "Wait briefly, then start a new run; another orchestrator currently owns "
                "this experiment's consistency lock."
            ),
        }
    if isinstance(error, ArtifactRootConflictError):
        # Not a fingerprint problem: the study is healthy but is not provably
        # this workdir's, and the fingerprint remediation (new experiment name
        # / archive the study) would destroy the binding's value. The wording
        # covers the subclass too: a study that predates the binding records no
        # root at all, so "the workdir its studies are bound to" would name
        # nothing (re-review v0.5.19 / blocker B1).
        return {
            "code": "artifact_root_conflict",
            "stage": failure_stage,
            "retryable": False,
            "actor": "operator",
            "remediation": (
                "Ask the operator to run this experiment from the workdir that owns its "
                "artifact tree, or to relocate/restore that tree and run "
                "`phasesweep rebind-workdir` against it before retrying."
            ),
        }
    if isinstance(error, (StudyFingerprintMismatchError, StudyContextConflictError)):
        return {
            "code": "fingerprint_mismatch",
            "stage": failure_stage,
            "retryable": False,
            "actor": "operator",
            "remediation": (
                "Use a new experiment name, or ask the operator to archive the "
                "incompatible persistent study before retrying."
            ),
        }
    if isinstance(error, StudySchemaMismatchError):
        return {
            "code": "study_schema_mismatch",
            "stage": failure_stage,
            "retryable": False,
            "actor": "operator",
            "remediation": (
                "Use a new experiment name, or ask the operator to archive the "
                "unsupported persistent study before retrying."
            ),
        }
    if isinstance(error, StudyStorageUnavailableError):
        return {
            "code": "storage_unavailable",
            "stage": failure_stage,
            "retryable": True,
            "actor": "operator",
            "remediation": (
                "Ask the operator to restore the configured study storage, then start a new run."
            ),
        }
    if isinstance(error, SamplerContinuationUnsupportedError):
        return {
            "code": "sampler_continuation_unsupported",
            "stage": failure_stage,
            "retryable": False,
            "actor": "operator",
            "remediation": (
                "Use a new experiment name for this TPE/CMA-ES extension, or run the "
                "full target in one invocation."
            ),
        }
    if isinstance(error, TrialTargetRegressionError):
        return {
            "code": "trial_target_regression",
            "stage": failure_stage,
            "retryable": False,
            "actor": "operator",
            "remediation": (
                "Restore the study's prior trial target, or use a new experiment name."
            ),
        }
    if isinstance(error, ProcessCleanupUncertainError):
        return {
            "code": "cleanup_uncertain",
            "stage": "cleanup",
            "retryable": False,
            "actor": "operator",
            "remediation": (
                "Ask the operator to run phasesweep mcp recover-run before another launch."
            ),
        }
    if isinstance(error, NoFeasibleTrialError):
        return {
            "code": "trainer_failed",
            "stage": failure_stage,
            "retryable": False,
            "actor": "operator",
            "remediation": (
                "Ask the operator to inspect trainer and evidence logs before starting a new run."
            ),
        }
    if isinstance(error, TimeoutError):
        return {
            "code": "timeout",
            "stage": failure_stage,
            "retryable": False,
            "actor": "operator",
            "remediation": (
                "Ask the operator to review the configured wallclock budget before retrying."
            ),
        }
    if isinstance(error, PhaseSweepShutdown):
        return {
            "code": "cancelled",
            "stage": failure_stage,
            "retryable": True,
            "actor": "agent",
            "remediation": "Start a new run only if the user still wants the sweep to continue.",
        }
    return {
        "code": "internal_error",
        "stage": failure_stage,
        "retryable": False,
        "actor": "operator",
        "remediation": "Ask the operator to inspect the PhaseSweep run log before retrying.",
    }


def _safe_failure_payload(
    error: BaseException,
    *,
    stage: str | None,
) -> dict[str, object]:
    """Map an error to one stable agent failure.

    :param BaseException error: Primary exception to classify.
    :param str | None stage: Stage forwarded to :func:`_base_failure_payload`
        for ``error``.
    :return dict[str, object]: The validated primary failure payload.
    """
    payload = _base_failure_payload(error, stage=stage)
    return FailurePayload.model_validate(payload).model_dump(mode="json", exclude_none=True)


def _cleanup_failure_payload(
    primary: BaseException,
    *,
    cause_stage: str | None,
) -> dict[str, object]:
    """Make cleanup uncertainty actionable while retaining its safe primary cause.

    :param BaseException primary: Terminal error observed before cleanup was
        found to be uncertain.
    :param str | None cause_stage: Stage at which ``primary`` occurred, used
        when classifying it as the nested cause.
    :return dict[str, object]: A ``"cleanup_uncertain"`` failure payload, with
        ``primary`` nested under ``"cause"`` unless ``primary`` is itself a
        :class:`ProcessCleanupUncertainError`.
    """
    cleanup = ProcessCleanupUncertainError("trainer process cleanup could not be confirmed")
    payload = _base_failure_payload(cleanup, stage="cleanup")
    if not isinstance(primary, ProcessCleanupUncertainError):
        payload["cause"] = _base_failure_payload(primary, stage=cause_stage)
    return FailurePayload.model_validate(payload).model_dump(mode="json", exclude_none=True)


def _terminal_error(
    report: TerminalReport | None,
    fallback: BaseException,
) -> tuple[BaseException, str | None]:
    """Return the engine's primary error and stage, or the caller's fallback.

    :param TerminalReport | None report: Engine terminal report, when one was delivered.
    :param BaseException fallback: Exception caught by the runner.
    :return tuple[BaseException, str | None]: Authoritative error and failure stage.
    """
    if report is None:
        return fallback, "execution"
    return report.primary_error or fallback, report.failure_stage


def _terminal_failure_payload(
    error: BaseException,
    *,
    stage: str | None,
    cleanup_confirmed: bool,
) -> dict[str, object]:
    """Classify one terminal error, making cleanup uncertainty authoritative.

    :param BaseException error: Primary terminal error.
    :param str | None stage: Stage at which the primary error occurred.
    :param bool cleanup_confirmed: Whether child-process cleanup was confirmed.
    :return dict[str, object]: Validated, path-free terminal failure payload.
    """
    if not cleanup_confirmed:
        return _cleanup_failure_payload(error, cause_stage=stage)
    return _safe_failure_payload(error, stage=stage)


def _write_status(
    status_path: Path,
    payload: dict,
    *,
    result_snapshot: dict | None,
    result_snapshot_error: str | None,
) -> None:
    """Persist terminal evidence and its already-captured result snapshot.

    :param Path status_path: JSON file where terminal cause should be recorded.
    :param dict payload: Status payload containing run id, return code, and error class.
    :param dict | None result_snapshot: Raw snapshot captured under the experiment lock.
    :param str | None result_snapshot_error: Capture error class when no snapshot exists.
    :raises RuntimeError: Raised and handled in-process when no snapshot was
        captured; it never reaches the caller, because a missing snapshot is
        recorded as ``result_snapshot_state="failed"`` instead of failing the
        already-durable terminal evidence.
    """
    # A catchable shutdown may arrive after the durable pending write. Defer it
    # until the complete/failed replacement is durable so cancellation cannot
    # strand an otherwise terminal run in the intermediate state.
    with defer_shutdown_signals():
        terminal = {
            **payload,
            "ended_at": utc_now_iso(),
            "result_snapshot_state": "pending",
        }
        if result_snapshot is not None:
            terminal["result_snapshot"] = result_snapshot
        try:
            write_status_file(status_path, terminal)
        except Exception:  # noqa: BLE001 - terminal evidence must not mask the run's exit
            logging.getLogger("phasesweep.mcp.runner").exception("failed to write status.json")
            return

        try:
            if result_snapshot is None:
                raise RuntimeError(result_snapshot_error or "terminal snapshot was not captured")
            terminal["result_snapshot"] = finalize_result_snapshot(
                result_snapshot,
            )
        except Exception as exc:  # noqa: BLE001 - minimal terminal evidence is already durable
            terminal.pop("result_snapshot", None)
            terminal["result_snapshot_state"] = "failed"
            terminal["result_snapshot_error"] = result_snapshot_error or type(exc).__name__
            logging.getLogger("phasesweep.mcp.runner").exception(
                "failed to finalize terminal result snapshot"
            )
        else:
            terminal["result_snapshot_state"] = "complete"

        try:
            write_status_file(status_path, terminal)
        except Exception as exc:  # noqa: BLE001 - preserve a serializable failed state
            logging.getLogger("phasesweep.mcp.runner").exception(
                "failed to finalize result snapshot state in status.json"
            )
            terminal.pop("result_snapshot", None)
            terminal["result_snapshot_state"] = "failed"
            terminal["result_snapshot_error"] = type(exc).__name__
            try:
                write_status_file(status_path, terminal)
            except Exception:  # noqa: BLE001 - no further persistence fallback is available
                logging.getLogger("phasesweep.mcp.runner").exception(
                    "failed to persist result snapshot finalization failure"
                )


def _persist_spawned_handle(
    *,
    state_dir: Path,
    run_id: str,
    experiment_id: str,
    config_sha256: str,
    started_at: str,
    allow_cancel: bool,
) -> None:
    """Persist this runner's process identity before it launches any training work.

    :param Path state_dir: Server state dir whose ``runs/`` receives the handle.
    :param str run_id: Run id minted by the launching server.
    :param str experiment_id: Catalog id this run belongs to.
    :param str config_sha256: Hash of the config snapshot this runner executes.
    :param str started_at: ISO-8601 UTC launch timestamp recorded by the server.
    :param bool allow_cancel: Cancel permission frozen at launch time.
    :raises RuntimeError: If Linux ``/proc`` start time is unavailable, so the
        handle could not be made PID-reuse safe, or the server never created a
        pending handle for ``run_id``.
    """
    store = RunStore(state_dir)
    pid = os.getpid()
    pgid = os.getpgrp() if hasattr(os, "getpgrp") else pid
    pid_starttime = read_proc_starttime(pid)
    if pid_starttime is None:
        raise RuntimeError(
            "cannot persist a PID-reuse-safe MCP runner handle because Linux "
            "/proc start time is unavailable"
        )
    pending = store.get(run_id)
    if pending is None:
        raise RuntimeError(f"cannot persist runner identity for unknown run {run_id!r}")
    store.update(
        RunHandle(
            run_id=run_id,
            experiment_id=experiment_id,
            config_sha256=config_sha256,
            pid=pid,
            pgid=pgid,
            pid_starttime=pid_starttime,
            started_at=started_at,
            launch_state="spawned",
            allow_cancel=allow_cancel,
            visible_params_at_launch=pending.visible_params_at_launch,
            # Binds pid/pid_starttime to this boot: after a reboot the pair can
            # name an unrelated process, and a reader that knows the boot
            # differs can rule the runner dead without signalling anything.
            boot_id=read_boot_id(),
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run one config to completion and record its terminal cause in status.json.

    :param list[str] | None argv: Optional argument vector; defaults to ``sys.argv`` when omitted.
    :return int: Process exit code, zero on successful sweep completion.
    :raises RuntimeError: The per-run config snapshot could not be read or did
        not match its recorded hash.
    :raises PhaseSweepShutdown: The run was cancelled; re-raised after the
        cancellation cause is recorded in status.json.
    :raises ProcessCleanupUncertainError: Child-process cleanup could not be
        confirmed; re-raised after status.json records the uncertainty.
    :raises BaseException: Whatever the handle write or engine run raised,
        re-raised after its terminal cause reaches status.json.
    """
    parser = argparse.ArgumentParser(prog="phasesweep mcp runner")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True, type=Path)  # snapshot, server-supplied
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--status-path", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--started-at", required=True)
    # The catalog entry's frozen working directory. The process does not start
    # there (see the module docstring), so it is an explicit argument.
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--allow-cancel", action="store_true")
    parser.add_argument("--from-phase", default=None)
    args = parser.parse_args(argv)

    # The server supplies absolute paths because this process starts in the
    # server-owned state directory rather than the experiment directory.
    project_cwd = args.cwd.expanduser()
    config_path = args.config
    status_path = args.status_path
    state_dir = args.state_dir

    # This process's stdout/stderr are the server-redirected run log. Log to
    # stderr; never print to stdout here. (The engine's own run.log under the
    # workdir is separate and durable.)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname).1s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Install shutdown handlers before any other work (run_config re-invokes
    # this; it is idempotent). A cancel can arrive while this runner is still
    # loading config or persisting its handle; without handlers the default
    # SIGTERM disposition kills the process before status.json is written and
    # the run derives "running" behind its cleanup-uncertainty marker forever.
    install_signal_handlers()

    status: dict = {
        "run_id": args.run_id,
        "returncode": 0,
        "error_class": None,
        "cleanup_confirmed": True,
        "failure": None,
    }
    result_snapshot: dict | None = None
    result_snapshot_error: str | None = None
    terminal_report: TerminalReport | None = None
    config: Experiment | None = None
    try:
        # The server also saves this handle after Popen returns. The runner's
        # self-write closes the restart-recovery window if the server dies
        # after Popen but before its own spawned-handle save reaches disk.
        _persist_spawned_handle(
            state_dir=state_dir,
            run_id=args.run_id,
            experiment_id=args.experiment_id,
            config_sha256=args.config_sha256,
            started_at=args.started_at,
            allow_cancel=args.allow_cancel,
        )
        # Only now enter the experiment's project directory: every import has
        # already resolved against the trusted interpreter path, and this
        # runner's PID/PGID are durable, so anything the project directory
        # influences from here on is attributable to a process the server can
        # find and terminate.
        os.chdir(project_cwd)
        try:
            config = load_experiment_snapshot(
                config_path,
                args.config_sha256,
                source=f"run snapshot {args.run_id}",
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc

        def capture_terminal(report: TerminalReport) -> None:
            """Capture immutable results while ``run_experiment`` still owns its lock.

            :param TerminalReport report: Engine outcome and cleanup evidence.
            """
            nonlocal result_snapshot, result_snapshot_error, terminal_report
            terminal_report = report
            status["cleanup_confirmed"] = report.cleanup_confirmed
            status["recovered_attempt_ids"] = sorted(report.recovered_attempt_ids)
            if report.primary_error is not None:
                status["failure"] = _terminal_failure_payload(
                    report.primary_error,
                    stage=report.failure_stage,
                    cleanup_confirmed=report.cleanup_confirmed,
                )
            try:
                result_snapshot = capture_result_snapshot(
                    config,
                    generation_id=report.generation_id,
                    engine_winners=report.winners,
                )
            except Exception as exc:  # noqa: BLE001 - preserve the engine's terminal cause
                result_snapshot_error = type(exc).__name__
                logging.getLogger("phasesweep.mcp.runner").exception(
                    "failed to capture terminal result snapshot under the experiment lock"
                )

        run_experiment(
            config,
            from_phase=args.from_phase,
            dry_run=False,
            terminal_callback=capture_terminal,
            generation_id=args.run_id,
        )
    except PhaseSweepShutdown as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        status["returncode"] = code
        status["error_class"] = "cancelled"
        status["cleanup_confirmed"] = (
            terminal_report.cleanup_confirmed
            if terminal_report is not None
            else exc.report.cleanup_confirmed
        )
        primary, failure_stage = _terminal_error(terminal_report, exc)
        status["failure"] = _terminal_failure_payload(
            primary,
            stage=failure_stage,
            cleanup_confirmed=status["cleanup_confirmed"],
        )
        raise
    except ProcessCleanupUncertainError as exc:
        status["returncode"] = 1
        primary, failure_stage = _terminal_error(terminal_report, exc)
        status["error_class"] = type(exc).__name__
        status["cleanup_confirmed"] = (
            terminal_report.cleanup_confirmed if terminal_report is not None else False
        )
        status["failure"] = _terminal_failure_payload(
            primary,
            stage=failure_stage,
            cleanup_confirmed=False,
        )
        raise
    except BaseException as exc:  # noqa: BLE001 - record every terminal cause, then re-raise
        status["returncode"] = 1
        primary, failure_stage = _terminal_error(terminal_report, exc)
        status["error_class"] = type(primary).__name__
        status["cleanup_confirmed"] = (
            terminal_report.cleanup_confirmed if terminal_report is not None else True
        )
        status["failure"] = _terminal_failure_payload(
            primary,
            stage=failure_stage,
            cleanup_confirmed=status["cleanup_confirmed"],
        )
        raise
    finally:
        if result_snapshot is None and terminal_report is None and config is not None:
            try:
                result_snapshot = capture_pre_generation_result_snapshot(config)
            except Exception as exc:  # noqa: BLE001 - preserve the terminal engine failure
                result_snapshot_error = type(exc).__name__
            else:
                status["generation_unavailable_reason"] = "engine_generation_not_claimed"
        _write_status(
            status_path,
            status,
            result_snapshot=result_snapshot,
            result_snapshot_error=result_snapshot_error,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
