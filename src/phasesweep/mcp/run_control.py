"""The MCP tools that change run state: launch_run and cancel_run.

RunControl owns the state every tool reads (catalog registry, run store, audit
log) and implements launch and cancel, both audited. Launch records a durable
handle before spawning the detached runner and settles ownership through a
ready/ack handshake; cancel stops the runner's process group SIGTERM -> grace
-> SIGKILL, signalling only a process whose boot id and start time match.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from phasesweep.engine.artifacts import _load_winner
from phasesweep.engine.publication import _last_successful_generation_id
from phasesweep.engine.state import Winner
from phasesweep.mcp.audit import AuditLogger
from phasesweep.mcp.errors import (
    ConcurrencyLimitError,
    ConfigChangedError,
    ExperimentBusyError,
    InvalidPhaseError,
    LaunchInProgressError,
    McpToolError,
    PermissionDeniedError,
    ResumeNotReadyError,
    RunCapacityUnknownError,
    RunLaunchUnsettledError,
    UnknownExperimentError,
    UnknownRunError,
)
from phasesweep.mcp.registry import RegisteredExperiment, Registry
from phasesweep.mcp.runs import (
    LAUNCH_ACK_BYTE,
    LAUNCH_READY_BYTE,
    PreparedRun,
    RunHandle,
    RunState,
    RunStore,
    write_status_file_if_absent,
)
from phasesweep.mcp.tool_names import TOOL_CANCEL_RUN, TOOL_LAUNCH_RUN
from phasesweep.runtime.files import ensure_private_dir, open_private_text
from phasesweep.runtime.process import fd_ready
from phasesweep.runtime.reaper import (
    identity_from_earlier_boot,
    kill_stale_group,
    read_boot_id,
    read_proc_starttime,
)
from phasesweep.runtime.time import utc_now_iso

# The tools log on the server's channel, so one logger name covers everything served.
log = logging.getLogger("phasesweep.mcp.server")

_RUNNER_READY_TIMEOUT_SECONDS = 10.0


class _SpawnBookkeepingError(Exception):
    """Carry cleanup evidence for a failure after a runner process was created."""

    def __init__(self, original_error: BaseException, *, cleanup_confirmed: bool) -> None:
        """Create an internal post-spawn ownership failure.

        :param BaseException original_error: Failure raised after ``Popen`` succeeded.
        :param bool cleanup_confirmed: Whether the spawned process group is confirmed gone.
        """
        super().__init__("runner bookkeeping failed after process creation")
        self.original_error = original_error
        self.cleanup_confirmed = cleanup_confirmed


# Environment variables that make the interpreter execute code before the
# detached runner's first statement. They are dropped from the child's
# environment (review v0.5.17 / blocker 9); PYTHONNOUSERSITE backs up the
# ``-s`` flag for the same reason.
_PRE_IMPORT_CODE_ENV_VARS = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONEXECUTABLE",
)


def _runner_env() -> dict[str, str]:
    """Build the detached runner's environment without pre-import code hooks.

    Everything else the operator exported is preserved: the runner's trials
    legitimately need ``PATH``, ``CUDA_*``, credentials, and the rest of the
    ambient environment, so this strips exactly the variables that can run
    code before ``phasesweep.mcp.runner`` gets control.

    :return dict[str, str]: Copy of this server's environment with the
        pre-import code hooks removed and ``PYTHONNOUSERSITE`` forced on.
    """
    env = {
        name: value for name, value in os.environ.items() if name not in _PRE_IMPORT_CODE_ENV_VARS
    }
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _runner_protocol_argv(
    *,
    run_id: str,
    config_snapshot_path: Path,
    config_sha256: str,
    status_path: Path,
    state_dir: Path,
    experiment_id: str,
    started_at: str,
    launch_ready_fd: int | None = None,
    launch_ack_fd: int | None = None,
    launch_lease_fd: int | None = None,
) -> list[str]:
    """Build the required detached-runner protocol arguments.

    Interpreter hardening flags and the project cwd stay at the spawn boundary:
    they govern how the child starts, while this list is the stable identity and
    persistence contract consumed by :mod:`phasesweep.mcp.runner`.

    :param str run_id: Claimed run identifier.
    :param Path config_snapshot_path: Immutable config snapshot for the run.
    :param str config_sha256: Expected digest of the config snapshot.
    :param Path status_path: Terminal-status destination.
    :param Path state_dir: MCP state directory containing the run store.
    :param str experiment_id: Catalog experiment identifier.
    :param str started_at: Claimed launch timestamp.
    :param int | None launch_ready_fd: Child pipe used to report a durable process identity.
    :param int | None launch_ack_fd: Child pipe blocking work until server acknowledgement.
    :param int | None launch_lease_fd: Inherited preparation lease closed after acknowledgement.
    :return list[str]: Runner arguments without interpreter prefix, cwd, or optional grants.
    """
    argv = [
        "--run-id",
        run_id,
        "--config",
        str(config_snapshot_path),
        "--config-sha256",
        config_sha256,
        "--status-path",
        str(status_path),
        "--state-dir",
        str(state_dir),
        "--experiment-id",
        experiment_id,
        "--started-at",
        started_at,
    ]
    if launch_ready_fd is not None:
        argv += ["--launch-ready-fd", str(launch_ready_fd)]
    if launch_ack_fd is not None:
        argv += ["--launch-ack-fd", str(launch_ack_fd)]
    if launch_lease_fd is not None:
        argv += ["--launch-lease-fd", str(launch_lease_fd)]
    return argv


class RunControl:
    """Launch and cancel, the audit trail they write, and the state every tool reads."""

    def __init__(
        self, registry: Registry, runs: RunStore, audit: AuditLogger | None = None
    ) -> None:
        """Create the SDK-free MCP implementation.

        :param Registry registry: Validated catalog registry.
        :param RunStore runs: Persistent detached-run store.
        :param AuditLogger | None audit: Optional audit sink override; defaults to state_dir/audit.jsonl.
        """
        self._registry = registry
        self._runs = runs
        self._audit = audit or AuditLogger(registry.state_dir / "audit.jsonl")

    def _audit_success(
        self,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        resolved: dict[str, Any] | None = None,
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
        result_counts: dict[str, int] | None = None,
    ) -> None:
        """Record a successful state-changing tool call.

        :param str tool: MCP tool name.
        :param dict[str, Any] | None args: Safe agent-supplied arguments.
        :param dict[str, Any] | None resolved: Safe server-resolved identifiers.
        :param dict[str, Any] | None state_before: Safe state summary before the call.
        :param dict[str, Any] | None state_after: Safe state summary after the call.
        :param dict[str, int] | None result_counts: Result counts that avoid copying full payloads.
        """
        self._audit.record(
            tool=tool,
            args=args,
            outcome="success",
            resolved=resolved,
            state_before=state_before,
            state_after=state_after,
            result_counts=result_counts,
        )

    def _audit_error(
        self,
        tool: str,
        args: dict[str, Any] | None,
        exc: Exception,
        *,
        resolved: dict[str, Any] | None = None,
        state_before: dict[str, Any] | None = None,
    ) -> None:
        """Record a failed state-changing tool call.

        :param str tool: MCP tool name.
        :param dict[str, Any] | None args: Safe agent-supplied arguments.
        :param Exception exc: Exception that will be surfaced through the safe tool wrapper.
        :param dict[str, Any] | None resolved: Safe server-resolved identifiers known before failure.
        :param dict[str, Any] | None state_before: Safe state summary before the failure.
        """
        message = str(exc) if isinstance(exc, McpToolError) else "internal error"
        self._audit.record(
            tool=tool,
            args=args,
            outcome="error",
            resolved=resolved,
            state_before=state_before,
            error_type=type(exc).__name__,
            error=message,
        )

    def launch(self, experiment_id: str, from_phase: str | None = None) -> dict[str, Any]:
        """Start the sweep as a detached background run; return its run_id.

        Refuses if launch is not permitted, if a ``from_phase`` resume is not
        ready (an earlier phase has no winner), if this experiment already has a
        live run, or if the server is at its max_concurrent_runs cap.

        :param str experiment_id: Catalog experiment id to launch.
        :param str | None from_phase: Optional phase to resume from after earlier winners exist.
        :return dict[str, Any]: Launch result containing run id, experiment id, and running state.
        :raises UnknownExperimentError: If ``experiment_id`` is not cataloged.
        :raises PermissionDeniedError: If the entry does not allow launching, or
            does not allow ``from_phase`` resumes.
        :raises InvalidPhaseError: If ``from_phase`` names no phase of this experiment.
        :raises ResumeNotReadyError: If an earlier phase has no compatible winner.
        :raises LaunchInProgressError: If another launch currently holds the launch lock.
        :raises ExperimentBusyError: If this experiment already has a live run.
        :raises ConcurrencyLimitError: If the server is at ``max_concurrent_runs``.
        :raises ConfigChangedError: If the cataloged config changed since startup.
        :raises RuntimeError: If no unused run id could be minted, or the spawned
            runner has no Linux ``/proc`` start time to make cancellation PID-reuse safe.
        """
        args = {"experiment_id": experiment_id, "from_phase": from_phase}
        resolved: dict[str, Any] = {}
        state_before: dict[str, Any] | None = None
        try:
            reg = self._registry.get(experiment_id)
            resolved["experiment_id"] = reg.id
            if not reg.allow_launch:
                raise PermissionDeniedError("launch", experiment_id)
            if from_phase is not None:
                if not reg.allow_from_phase:
                    raise PermissionDeniedError("from_phase", experiment_id)
                if from_phase not in reg.phase_names:
                    raise InvalidPhaseError(experiment_id, from_phase)
                self._require_resume_ready(reg, from_phase)
            # The cap check and the spawn must be atomic, or two near-simultaneous
            # launches both pass the cap and oversubscribe the GPU it protects. Hold
            # the launch lock across the whole decision. One scan then covers both
            # guards: the same experiment can't double-launch, and no more than
            # max_concurrent_runs sweeps run at once (default 1).
            with self._runs.launch_lock() as acquired:
                if not acquired:
                    raise LaunchInProgressError()
                handles, unreadable = self._runs.launch_inventory()
                abandoned = {
                    handle.run_id
                    for handle in handles
                    if self._runs.launch_lease_path(handle.run_id).is_file()
                    and self._runs.is_pre_spawn_orphan(handle.run_id)
                }
                abandoned.update(
                    identity.removeprefix("run:")
                    for identity in unreadable
                    if identity.startswith("run:")
                    and self._runs.launch_lease_path(identity.removeprefix("run:")).is_file()
                    and self._runs.is_pre_spawn_orphan(identity.removeprefix("run:"))
                )
                for abandoned_run_id in sorted(abandoned):
                    recovered_log = self._runs.clear_pre_spawn_orphan(abandoned_run_id)
                    if recovered_log is not None:
                        log.info(
                            "preserved abandoned launch log run=%s at %s",
                            abandoned_run_id,
                            recovered_log,
                        )
                if abandoned:
                    handles, unreadable = self._runs.launch_inventory()
                if unreadable:
                    recoverable = sorted(
                        identity.removeprefix("run:")
                        for identity in unreadable
                        if identity.startswith("run:")
                        and self._runs.is_pre_spawn_orphan(identity.removeprefix("run:"))
                    )
                    raise RunCapacityUnknownError(len(unreadable), recoverable)
                live = [handle for handle in handles if self._runs.state(handle) == "running"]
                state_before = {"live_runs": len(live)}
                busy = next((h for h in live if h.experiment_id == experiment_id), None)
                if busy is not None:
                    raise ExperimentBusyError(experiment_id, busy.run_id)
                if len(live) >= self._registry.max_concurrent_runs:
                    blocking_run_ids = [
                        handle.run_id
                        for handle in sorted(live, key=lambda item: (item.started_at, item.run_id))
                    ]
                    raise ConcurrencyLimitError(
                        len(live),
                        self._registry.max_concurrent_runs,
                        blocking_run_ids,
                    )
                config_bytes = self._current_config_bytes(reg)
                preparation: PreparedRun | None = None
                for _ in range(10):
                    run_id = self._runs.new_run_id(reg.id)
                    pending = self._pending_handle(reg, run_id)
                    try:
                        preparation = self._runs.prepare_launch(pending, config_bytes)
                    except FileExistsError:
                        continue
                    break
                else:
                    raise RuntimeError("failed to mint an unused MCP run id")
                assert preparation is not None
                resolved["run_id"] = run_id
                handle: RunHandle | None = None
                try:
                    handle = self._spawn(reg, from_phase, preparation)
                    if handle.pid_starttime is None:
                        raise RuntimeError(
                            "spawned runner has no Linux /proc start time; refused launch because "
                            "later cancellation could not distinguish PID reuse"
                        )
                    self._runs.update(handle)
                except _SpawnBookkeepingError as spawn_exc:
                    self._record_launch_failure(
                        pending,
                        cleanup_confirmed=spawn_exc.cleanup_confirmed,
                        error_class=type(spawn_exc.original_error).__name__,
                    )
                    raise spawn_exc.original_error from None
                except BaseException as launch_exc:
                    cleanup_confirmed = (
                        True
                        if handle is None
                        else self._terminate_failed_spawn(handle, launch_exc, acknowledged=True)
                    )
                    self._record_launch_failure(
                        pending,
                        cleanup_confirmed=cleanup_confirmed,
                        error_class=type(launch_exc).__name__,
                    )
                    raise
                finally:
                    self._runs.finish_launch_preparation(preparation)
                assert handle is not None
            result = {"run_id": handle.run_id, "experiment_id": experiment_id, "state": "running"}
        except Exception as exc:
            self._audit_error(
                TOOL_LAUNCH_RUN,
                args,
                exc,
                resolved=resolved,
                state_before=state_before,
            )
            raise
        self._audit_success(
            TOOL_LAUNCH_RUN,
            args,
            resolved={"experiment_id": experiment_id, "run_id": handle.run_id},
            state_before=state_before,
            state_after={
                "run_state": "running",
                "live_runs": (state_before or {}).get("live_runs", 0) + 1,
            },
            result_counts={"runs": 1},
        )
        return result

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Stop a running sweep: SIGTERM -> grace -> SIGKILL the runner's group.

        The terminal state is reported as ``cancelled`` only when the runner
        records its cancellation status. If the runner group is gone but no
        status was written, cleanup remains uncertain because trial process
        groups may still be alive.

        :param str run_id: Detached run id to cancel.
        :return dict[str, Any]: Cancellation result containing final state and optional cleanup confirmation.
        :raises UnknownRunError: If ``run_id`` names no persisted run.
        :raises PermissionDeniedError: If the run's entry did not allow cancel at launch time.
        :raises RunLaunchUnsettledError: If the launch has not yet persisted a
            process identity, so there is nothing safe to signal.
        """
        args = {"run_id": run_id}
        resolved: dict[str, Any] = {}
        state_before: dict[str, Any] | None = None
        result: dict[str, Any]
        try:
            handle = self._runs.get(run_id)
            if handle is None:
                raise UnknownRunError(run_id)
            resolved = {"experiment_id": handle.experiment_id, "run_id": run_id}
            if not self._cancel_allowed(handle):
                raise PermissionDeniedError("cancel", handle.experiment_id)
            before = self._runs.state(handle)
            recovery_required = self._runs.recovery_required(handle)
            state_before = {
                "run_state": before,
                "recovery_required": recovery_required,
            }
            if before == "running":
                with self._runs.transition_lock(handle):
                    # Launch may have durably replaced a pending handle with
                    # its runner identity, and recovery may have finalized the
                    # run after the first read. Use the latest handle for both
                    # the state decision and any subsequent signal.
                    refreshed = self._runs.get(run_id)
                    if refreshed is None:
                        raise UnknownRunError(run_id)
                    handle = refreshed
                    before = self._runs.state(handle, transition_locked=True)
                    recovery_required = self._runs.recovery_required(handle)
                    state_before = {
                        "run_state": before,
                        "recovery_required": recovery_required,
                    }
                    if before == "running" and handle.launch_state == "launching":
                        # No PID/PGID is durable yet. Signalling an empty identity
                        # could let launch continue after a false cancel response.
                        raise RunLaunchUnsettledError(run_id)
                    if before == "running":
                        # Reserve capacity before signalling; recovery cannot
                        # clear this marker between the reread and its write.
                        self._runs.mark_cleanup_uncertain(handle)
            after: RunState
            if before != "running":
                after = before
                confirmed: bool | None = None
            else:
                # Concurrent callers still signal outside the short transition
                # lock. kill_stale_group treats an already-gone group as confirmed,
                # terminal status is runner-authoritative, and marker removal
                # uses missing_ok.
                # SIGTERM -> grace -> SIGKILL on the runner's process group. A
                # runner-written status is useful only when it includes explicit
                # cleanup evidence from the engine shutdown handler. If the server
                # had to force-kill the runner first, or the handler reported
                # uncertainty, child trial PGIDs may still live, so keep the run
                # counted as live and fail closed.
                identity = self._runs.cleanup_identity(handle)
                current_boot = read_boot_id()
                same_boot = identity.boot_id is not None and identity.boot_id == current_boot
                earlier_boot = identity_from_earlier_boot(identity.boot_id, current_boot)
                # A prior boot proves cleanup without a signal. An unknown boot
                # cannot make saved PID/starttime safe to signal after reboot.
                runner_group_gone = earlier_boot or (
                    same_boot
                    and kill_stale_group(
                        identity.pid,
                        identity.pid_starttime,
                        pgid=identity.pgid,
                        grace_seconds=30.0,
                    )
                )
                with self._runs.transition_lock(handle):
                    terminal_status = self._runs.recorded_terminal_status(handle)
                    cleanup_recovered = self._runs._cleanup_recovered(handle)
                    confirmed = runner_group_gone and (
                        earlier_boot
                        or (
                            terminal_status is not None
                            and terminal_status.get("cleanup_confirmed") is True
                        )
                        or cleanup_recovered
                    )
                    if confirmed and not cleanup_recovered:
                        # Recovery owns any remaining marker after recording
                        # cleanup; it may still be repairing the frozen result.
                        self._runs.clear_cleanup_uncertain(handle)
                    after = self._runs.state(handle, transition_locked=True)
                    recovery_required = self._runs.recovery_required(handle)
            result = {
                "run_id": run_id,
                "state": after,
                "cleanup_confirmed": confirmed,
                "recovery_required": recovery_required,
            }
        except Exception as exc:
            self._audit_error(
                TOOL_CANCEL_RUN,
                args,
                exc,
                resolved=resolved,
                state_before=state_before,
            )
            raise
        state_after = {
            "run_state": after,
            "recovery_required": recovery_required,
        }
        if confirmed is not None:
            state_after["cleanup_confirmed"] = confirmed
        self._audit_success(
            TOOL_CANCEL_RUN,
            args,
            resolved=resolved,
            state_before=state_before,
            state_after=state_after,
            result_counts={"runs": 1},
        )
        return result

    def _require_resume_ready(self, reg: RegisteredExperiment, from_phase: str) -> None:
        """Verify that every earlier phase has a compatible persisted winner.

        :param RegisteredExperiment reg: Registered experiment being resumed.
        :param str from_phase: Requested phase to resume from.
        :raises ResumeNotReadyError: If an earlier phase has no persisted
            winner, or its stored winner is unreadable or incompatible with the
            current config.
        """
        names = reg.phase_names
        winners: dict[str, Winner] = {}
        for phase in reg.experiment.phases[: names.index(from_phase)]:
            inherited = {parent: winners[parent] for parent in phase.inherits}
            try:
                winners[phase.name] = _load_winner(
                    reg.experiment,
                    phase,
                    inherited,
                    published_generation_id=_last_successful_generation_id(
                        reg.experiment, raise_on_manifest_error=True
                    ),
                )
            except FileNotFoundError:
                raise ResumeNotReadyError(reg.id, from_phase, phase.name) from None
            except (
                RuntimeError,
                KeyError,
                TypeError,
                ValueError,
                AttributeError,
                OSError,
                yaml.YAMLError,
            ) as exc:
                log.info(
                    "resume preflight rejected winner for experiment=%s phase=%s: %s",
                    reg.id,
                    phase.name,
                    exc,
                )
                raise ResumeNotReadyError(
                    reg.id,
                    from_phase,
                    phase.name,
                    reason="has no compatible winner for the current config",
                ) from None

    @staticmethod
    def _current_config_bytes(reg: RegisteredExperiment) -> bytes:
        """Read a cataloged config only when it still matches startup validation.

        :param RegisteredExperiment reg: Frozen catalog entry to verify.
        :return bytes: Current config bytes when their SHA-256 matches startup.
        :raises ConfigChangedError: If the file is unreadable or changed.
        """
        try:
            data = reg.config_path.read_bytes()
        except OSError as exc:
            log.info("cannot read cataloged config for experiment=%s: %s", reg.id, exc)
            raise ConfigChangedError(reg.id) from None
        if hashlib.sha256(data).hexdigest() != reg.config_sha256:
            raise ConfigChangedError(reg.id)
        return data

    def _pending_handle(self, reg: RegisteredExperiment, run_id: str) -> RunHandle:
        """Build the pre-spawn handle persisted before ``Popen``.

        :param RegisteredExperiment reg: Catalog entry being launched.
        :param str run_id: Server-minted run id for the pending launch.
        :return RunHandle: Launching-state handle without process identity.
        """
        return RunHandle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
            pid=None,
            pgid=None,
            pid_starttime=None,
            started_at=utc_now_iso(),
            launch_state="launching",
            allow_cancel=reg.allow_cancel,
            visible_params_at_launch=(
                list(reg.visible_params)
                if isinstance(reg.visible_params, list)
                else reg.visible_params
            ),
        )

    def _record_launch_failure(
        self,
        pending: RunHandle,
        *,
        cleanup_confirmed: bool,
        error_class: str,
    ) -> None:
        """Finalize a known failed launch without masking its original error.

        :param RunHandle pending: Durable pre-spawn handle to finalize.
        :param bool cleanup_confirmed: Whether any spawned runner group is confirmed gone.
        :param str error_class: Operator-facing class of the original launch failure.
        """
        try:
            # The runner can finish before the server detects its own bookkeeping
            # failure. Its terminal result is authoritative once cleanup is done.
            if self._runs.recorded_terminal_status(pending) is not None:
                return
            write_status_file_if_absent(
                self._runs.status_path(pending.run_id),
                {
                    "run_id": pending.run_id,
                    "returncode": 1,
                    "error_class": error_class,
                    "cleanup_confirmed": cleanup_confirmed,
                    "ended_at": utc_now_iso(),
                    "result_snapshot_state": "failed",
                    "result_snapshot_error": "LaunchDidNotProduceSnapshot",
                    "failure": {
                        "code": "internal_error",
                        "stage": "preflight",
                        "retryable": False,
                        "actor": "operator",
                        "remediation": (
                            "Ask the operator to inspect the PhaseSweep server diagnostics "
                            "before retrying."
                        ),
                    },
                },
            )
        except Exception:
            # The unresolved launching handle remains a fail-closed concurrency
            # reservation when even terminal bookkeeping cannot be persisted.
            log.exception("failed to finalize launch error for run_id=%s", pending.run_id)

    def _terminate_failed_spawn(
        self,
        handle: RunHandle,
        original_error: BaseException,
        *,
        acknowledged: bool = False,
    ) -> bool:
        """Terminate a spawned runner whose durable bookkeeping failed.

        :param RunHandle handle: Spawned runner identity available in memory.
        :param BaseException original_error: Launch failure preserved for diagnostics.
        :param bool acknowledged: Whether the runner may already have launched trials.
        :return bool: Whether runner and possible trial groups are confirmed gone.
        """
        marker_written = False
        try:
            self._runs.mark_cleanup_uncertain(handle)
            marker_written = True
        except BaseException as marker_exc:
            log.error(
                "cleanup uncertain after failed runner launch bookkeeping for "
                "run_id=%s pgid=%s, but failed to persist cleanup uncertainty marker; "
                "original error: %r",
                handle.run_id,
                handle.pgid,
                original_error,
                exc_info=(type(marker_exc), marker_exc, marker_exc.__traceback__),
            )
        try:
            assert handle.pgid is not None
            cleanup_confirmed = kill_stale_group(
                handle.pid,
                handle.pid_starttime,
                pgid=handle.pgid,
            )
        except BaseException:
            log.exception(
                "failed to terminate untracked runner run_id=%s pgid=%s",
                handle.run_id,
                handle.pgid,
            )
            return False
        if cleanup_confirmed and acknowledged:
            # After acknowledgement the runner can start trials in separate
            # process groups. Its own terminal cleanup report is the only
            # evidence that those groups were also stopped.
            terminal = self._runs.recorded_terminal_status(handle)
            cleanup_confirmed = terminal is not None and terminal.get("cleanup_confirmed") is True
        if cleanup_confirmed:
            if marker_written:
                try:
                    self._runs.clear_cleanup_uncertain(handle)
                except BaseException:
                    log.exception(
                        "runner cleanup succeeded but its uncertainty marker could not be "
                        "cleared for run_id=%s; retaining the recovery reservation",
                        handle.run_id,
                    )
                    return False
        else:
            log.error(
                "cleanup uncertain after failed runner launch bookkeeping for run_id=%s pgid=%s",
                handle.run_id,
                handle.pgid,
            )
        return cleanup_confirmed

    def _spawn(
        self,
        reg: RegisteredExperiment,
        from_phase: str | None,
        preparation: PreparedRun,
    ) -> RunHandle:
        """Spawn a blocked runner and acknowledge its durable process receipt.

        :param RegisteredExperiment reg: Registered experiment to run.
        :param str | None from_phase: Optional phase to resume from.
        :param PreparedRun preparation: Durable preparation and inherited launch lease.
        :raises OSError: The log or detached runner cannot be opened before spawn.
        :raises _SpawnBookkeepingError: Post-spawn identity bookkeeping fails; the
            exception records whether cleanup of the spawned process group was confirmed.
        :return RunHandle: Runner-validated spawned handle.
        """
        pending = preparation.handle
        run_id = pending.run_id
        config_snapshot_path = preparation.config_snapshot_path
        log_path = self._runs.log_path(run_id)
        status_path = self._runs.status_path(run_id)
        ready_read, ready_write = os.pipe()
        ack_read, ack_write = os.pipe()
        cmd = [
            sys.executable,
            # -P: never prepend the cwd or script dir to sys.path. -s: no
            # per-user site dir. Both close the pre-identity window described
            # below; keep them together with the sanitized env.
            "-P",
            "-s",
            "-m",
            "phasesweep.mcp.runner",
            *_runner_protocol_argv(
                run_id=run_id,
                config_snapshot_path=config_snapshot_path,
                config_sha256=reg.config_sha256,
                status_path=status_path,
                state_dir=self._registry.state_dir,
                experiment_id=reg.id,
                started_at=pending.started_at,
                launch_ready_fd=ready_write,
                launch_ack_fd=ack_read,
                launch_lease_fd=preparation.lease_fd,
            ),
            # The runner chdirs here itself once its identity is durable; see
            # the trust-boundary note below for why Popen must not do it.
            "--cwd",
            str(reg.cwd),
        ]
        if reg.allow_cancel:
            cmd.append("--allow-cancel")
        if from_phase is not None:
            cmd += ["--from-phase", from_phase]
        # Pre-identity trust boundary (review v0.5.17 / blocker 9). Everything
        # between exec and the runner's own durable handle write runs before
        # this server can name the process it just created. Spawning with the
        # experiment's project directory as cwd put that window inside the
        # project's reach: interpreter startup would import a project-local
        # `phasesweep/` shadow package or `sitecustomize.py`, and a
        # PYTHONPATH/PYTHONHOME/PYTHONSTARTUP-injected module would run
        # earlier still. Any of it could fork+setsid out of the process group
        # recorded below, after which "cleanup confirmed" would be a claim
        # about an empty group. So the child starts in the server-owned state
        # directory, with -P/-s and a sanitized environment, and receives the
        # project directory as an explicit argument it applies only after its
        # identity is on disk.
        spawn_cwd = self._neutral_spawn_cwd()
        # Open the log here, hand the fd to the child, then close our copy. The
        # child keeps it. stdin is /dev/null so the runner never blocks on input.
        # Once Popen returns, every later operation is inside the BaseException
        # boundary: shutdown interrupts are ownership failures too, not proof
        # that no child was created.
        proc: subprocess.Popen[bytes] | None = None
        handle: RunHandle | None = None
        pid_starttime: int | None = None
        boot_id: str | None = None
        runner_acknowledged = False
        try:
            with open_private_text(log_path, "w") as log_file:
                proc = subprocess.Popen(  # noqa: S603 - argv list, no shell, server-controlled
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # own session/pgid; survives restart; signal as a group
                    cwd=str(spawn_cwd),
                    env=_runner_env(),
                    pass_fds=(ready_write, ack_read, preparation.lease_fd),
                )
                os.close(ready_write)
                ready_write = -1
                os.close(ack_read)
                ack_read = -1
                # Build the minimum usable identity before log close, /proc,
                # boot-id, or handle enrichment can fail. start_new_session=True
                # makes the child the group leader, so pgid == pid without a
                # getpgid() race if it exits quickly.
                handle = RunHandle(
                    run_id=run_id,
                    experiment_id=reg.id,
                    config_sha256=reg.config_sha256,
                    pid=proc.pid,
                    pgid=proc.pid,
                    pid_starttime=None,
                    started_at=pending.started_at,
                    launch_state="spawned",
                    allow_cancel=pending.allow_cancel,
                    visible_params_at_launch=pending.visible_params_at_launch,
                    boot_id=None,
                )
                pid_starttime = read_proc_starttime(proc.pid)
                handle = replace(handle, pid_starttime=pid_starttime)
                boot_id = read_boot_id()
                handle = replace(handle, boot_id=boot_id)
                if handle.pid_starttime is None:
                    raise RuntimeError(
                        "spawned runner has no Linux /proc start time; refused launch because "
                        "later cancellation could not distinguish PID reuse"
                    )
                if handle.boot_id is None:
                    raise RuntimeError(
                        "spawned runner has no Linux boot id; refused launch because later "
                        "cancellation could not distinguish PID reuse after reboot"
                    )
                readable = fd_ready(ready_read, timeout=_RUNNER_READY_TIMEOUT_SECONDS)
                ready = os.read(ready_read, 1) if readable else b""
                if ready != LAUNCH_READY_BYTE:
                    raise RuntimeError(
                        "detached runner did not persist its launch receipt before launch"
                    )
                persisted = self._runs.get(run_id)
                if persisted != handle:
                    raise RuntimeError("detached runner launch receipt did not match its process")
            # An asynchronous exception may arrive after the byte reaches the
            # runner but before os.write returns. From this point onward,
            # cleanup must assume separately-sessioned trials could start.
            runner_acknowledged = True
            if os.write(ack_write, LAUNCH_ACK_BYTE) != len(LAUNCH_ACK_BYTE):
                raise RuntimeError("could not acknowledge the detached runner launch")
            acknowledged_fd = ack_write
            ack_write = -1
            with contextlib.suppress(OSError):
                os.close(acknowledged_fd)
        except BaseException as exc:
            if proc is None:
                # Opening the log or Popen itself failed: no child exists.
                raise
            cleanup_handle = RunHandle(
                run_id=run_id,
                experiment_id=reg.id,
                config_sha256=reg.config_sha256,
                pid=proc.pid,
                pgid=proc.pid,
                pid_starttime=pid_starttime,
                started_at=pending.started_at,
                launch_state="spawned",
                allow_cancel=pending.allow_cancel,
                visible_params_at_launch=pending.visible_params_at_launch,
                boot_id=boot_id,
            )
            cleanup_confirmed = self._terminate_failed_spawn(
                cleanup_handle, exc, acknowledged=runner_acknowledged
            )
            raise _SpawnBookkeepingError(
                exc,
                cleanup_confirmed=cleanup_confirmed,
            ) from exc
        finally:
            for fd in (ready_read, ready_write, ack_read, ack_write):
                if fd >= 0:
                    with contextlib.suppress(OSError):
                        os.close(fd)
        assert handle is not None
        return handle

    def _neutral_spawn_cwd(self) -> Path:
        """Return the server-owned directory the detached runner is spawned in.

        The runner must not start in a directory the experiment's project can
        write to, so use the operator-owned MCP state directory that already
        holds run handles, logs, and config snapshots. It is re-validated here
        because it must exist and stay owner-only at the moment of the spawn.

        :return Path: Private state directory used as the child's initial cwd.
        """
        state_dir = self._registry.state_dir
        ensure_private_dir(state_dir)
        return state_dir

    def _cancel_allowed(self, handle: RunHandle) -> bool:
        """Return whether launch-time and current policy permit cancellation.

        A catalog edit may revoke cancellation but never grant it, so a still
        cataloged id is intersected with the permission frozen at launch.

        When the id is gone from the catalog there is no current policy to
        intersect with, and the launch-time permission stands alone. Refusing
        there would strand a live detached runner: ``_resolve_read_target``
        still reports it as ``running`` from its snapshot, and
        ``phasesweep mcp recover-run`` refuses while the runner is alive, so
        the operator would be left hunting the PGID by hand. Cancellation is
        risk-reducing and was authorized when this run started.

        :param RunHandle handle: Run handle whose cancellation permission should be checked.
        :return bool: Whether both launch-time and current permission are true,
            or the launch-time permission alone when the id is no longer cataloged.
        """
        if not handle.allow_cancel:
            # A launch-time denial is permanent for this run.
            return False
        try:
            return self._registry.get(handle.experiment_id).allow_cancel
        except UnknownExperimentError:
            return True
