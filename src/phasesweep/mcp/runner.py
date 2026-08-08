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
import time
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from phasesweep.config import Experiment
from phasesweep.engine import NoFeasibleTrialError, TerminalReport, run_experiment
from phasesweep.engine.errors import (
    ActiveAttemptPersistenceError,
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
    absorb_shutdown_signals,
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
    if isinstance(error, ActiveAttemptPersistenceError):
        # The unavailable workdir still fits the storage category, but this
        # pre-launch refusal is recorded as a terminal fatal trial and durable
        # phase abort at the accepted target. MCP experiments use persistent
        # storage, so restoring write access does not make the unchanged config
        # retryable: recovery must explicitly schedule a higher supported
        # target or start a new experiment. Nothing was launched, so there is
        # no process cleanup step (PR #5 re-review, P1).
        return {
            "code": "storage_unavailable",
            "stage": failure_stage,
            "retryable": False,
            "actor": "operator",
            "remediation": (
                "Ask the operator to restore write access to the experiment workdir, then "
                "increase the affected phase's n_trials above the failed run's accepted "
                "target when sampler continuation is supported, or use a new experiment "
                "name, before starting another run."
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
        # ``cause`` is diagnostic history, not a second action contract. It
        # intentionally retains the primary failure's own actor/retryability
        # (for example, a user cancellation) while the authoritative outer
        # cleanup_uncertain verdict blocks every relaunch pending recovery.
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


# Backoff between terminal status write attempts. This runs on the exit path
# of an already-finished sweep, so the whole retry budget stays well under a
# second: it exists to ride out a momentary ENOSPC/EIO/EINTR, not to wait out
# an operator repairing the filesystem.
_STATUS_WRITE_BACKOFF_SECONDS: tuple[float, ...] = (0.05, 0.25)


def _write_status_file_with_retry(
    status_path: Path,
    payload: dict,
    *,
    attempts: int = 3,
) -> None:
    """Write one terminal status record, retrying only transient persistence errors.

    Only :class:`OSError` is retried (PR #5 review / reviewer 2 pass 2, blocker
    6). A serialization error is a defect of the payload, not of the
    filesystem: retrying it cannot succeed, and the caller depends on seeing it
    immediately so it can downgrade the record to a serializable one.

    :param Path status_path: Destination ``status.json`` path for the run.
    :param dict payload: JSON-serializable terminal status payload.
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


def _write_status(
    status_path: Path,
    payload: dict,
    *,
    result_snapshot: dict | None,
    result_snapshot_error: str | None,
) -> bool:
    """Persist terminal evidence and its already-captured result snapshot.

    Nothing raises out of the persistence work itself, by design: this runs
    from ``main``'s ``finally`` with the run's own exception possibly in
    flight, so raising a persistence error would mask the run's primary exit
    (PR #5 review / reviewer 2 pass 2, blocker 6). The failure is reported
    through the return value instead, which ``main`` turns into a nonzero exit
    code on the otherwise-successful path. A shutdown signal deferred by the
    window below is still delivered at its exit, after the terminal record is
    durable; that is the run's own cancellation, not a persistence error.

    Every write is monotonic: state only ever moves toward more evidence.
    A ``complete`` transition that cannot be persisted leaves the durable
    ``pending`` record and its embedded snapshot in place rather than
    downgrading it to ``failed`` - a dead runner with a pending record is
    finalized by ``phasesweep mcp recover-run``, whereas ``failed`` is
    permanent.

    :param Path status_path: JSON file where terminal cause should be recorded.
    :param dict payload: Status payload containing run id, return code, and error class.
    :param dict | None result_snapshot: Raw snapshot captured under the experiment lock.
    :param str | None result_snapshot_error: Capture error class when no snapshot exists.
    :return bool: Whether durable terminal evidence exists, i.e. some record
        (``pending``, ``complete``, or ``failed``) reached disk. ``False`` only
        when nothing could be persisted at all.
    :raises RuntimeError: Raised and handled in-process when no snapshot was
        captured; it never reaches the caller, because a missing snapshot is
        recorded as ``result_snapshot_state="failed"`` instead of failing the
        already-durable terminal evidence.
    """
    log = logging.getLogger("phasesweep.mcp.runner")
    # A catchable shutdown may arrive after the durable pending write. Defer it
    # until the complete/failed replacement is durable so cancellation cannot
    # strand an otherwise terminal run in the intermediate state. This window
    # is also the checkpoint that services a shutdown absorbed by the terminal
    # snapshot capture (see ``capture_terminal``).
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
            # The engine outcome is already committed; only the MCP-side
            # terminal record is missing, so the run derives "running" until an
            # operator recovers it. Report that upward instead of exiting zero.
            log.exception(
                "failed to persist terminal status.json after every retry; the sweep's own "
                "outcome is unaffected but this run has no terminal MCP evidence"
            )
            return False
        except Exception as exc:  # noqa: BLE001 - terminal evidence must not mask the run's exit
            # Not a persistence failure: the captured snapshot cannot be
            # serialized at all. Record the minimal terminal evidence without
            # it rather than returning with no status file, which would leave
            # the run deriving "running" forever.
            log.exception("failed to write status.json with the captured result snapshot")
            terminal.pop("result_snapshot", None)
            terminal["result_snapshot_state"] = "failed"
            terminal["result_snapshot_error"] = result_snapshot_error or type(exc).__name__
            try:
                _write_status_file_with_retry(status_path, terminal)
            except Exception:  # noqa: BLE001 - no further persistence fallback is available
                log.exception("failed to write status.json without the result snapshot")
                return False
            return True

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
            log.exception("failed to finalize terminal result snapshot")
        else:
            terminal["result_snapshot_state"] = "complete"

        try:
            _write_status_file_with_retry(status_path, terminal)
        except OSError:
            # Deliberately no downgrade write here: the durable pending record
            # still holds the raw snapshot, and a dead runner with a pending
            # record is exactly what `recover-run` finalizes. Replacing it with
            # "failed" would trade a recoverable state for a permanent one -
            # and the same OSError would most likely defeat that write too.
            log.exception(
                "failed to persist the finalized result snapshot state after every retry; "
                "the durable pending record still holds the captured snapshot and remains "
                "recoverable with `phasesweep mcp recover-run`"
            )
            return True
        except Exception as exc:  # noqa: BLE001 - preserve a serializable failed state
            log.exception("failed to finalize result snapshot state in status.json")
            terminal.pop("result_snapshot", None)
            terminal["result_snapshot_state"] = "failed"
            terminal["result_snapshot_error"] = type(exc).__name__
            try:
                _write_status_file_with_retry(status_path, terminal)
            except Exception:  # noqa: BLE001 - no further persistence fallback is available
                log.exception("failed to persist result snapshot finalization failure")
        return True


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
    :return int: Process exit code: zero when the sweep completed and its
        terminal evidence is durable, and ``1`` when the sweep itself completed
        but no status.json record could be persisted at all - the run has no
        MCP-visible outcome, so exiting zero would claim one it cannot show.
    :raises RuntimeError: The per-run config snapshot could not be read or did
        not match its recorded hash.
    :raises PhaseSweepShutdown: The run was cancelled; re-raised after the
        cancellation cause is recorded in status.json. Also raised on an
        otherwise-successful run when a shutdown absorbed during the terminal
        snapshot capture is serviced at the terminal status write: status.json
        then records the engine's own successful outcome and this process exits
        with the POSIX signalled code.
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
    status_persisted = False
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

            The whole body is an absorbed-shutdown critical section (PR #5
            review / reviewer 2 pass 2, blocker 3). The engine calls this
            callback inside a diagnostic boundary that swallows
            :class:`BaseException` on purpose, so a shutdown raised here by the
            installed handler - :class:`PhaseSweepShutdown` is a
            :class:`SystemExit`, which the local ``except Exception`` below
            does not catch - is consumed there and vanishes: an already
            published run would end with no frozen result, an unrecoverable
            ``result_snapshot_state="failed"``, and no record of the
            cancellation at all.

            Absorbing rather than deferring is the point:
            :func:`defer_shutdown_signals` services the signal at window exit,
            which is still inside the callback, so the raise would be swallowed
            just the same. Absorption keeps it pending for the next checkpoint
            - ``_write_status``'s defer window, which the engine's publication
            transaction already names as the MCP runner's terminal status write
            - so the shutdown is honored only after terminal evidence is
            durable.

            :param TerminalReport report: Engine outcome and cleanup evidence.
            """
            nonlocal result_snapshot, result_snapshot_error, terminal_report
            with absorb_shutdown_signals() as absorbed:
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
        status_persisted = _write_status(
            status_path,
            status,
            result_snapshot=result_snapshot,
            result_snapshot_error=result_snapshot_error,
        )
    # Only reachable on the success path; every failure path re-raises above.
    # A sweep whose terminal evidence never reached disk is not a clean exit:
    # the server would see no status.json, derive "running" indefinitely, and
    # keep this run's concurrency slot until an operator recovers it.
    return 0 if status_persisted else 1


if __name__ == "__main__":
    raise SystemExit(main())
