"""Read-time composition of the ``failure`` payload an MCP run read returns.

A run read reports one safe failure payload chosen from the runner's recorded terminal status, its
cleanup record and what the read can observe of the runner now. The choice reads only the run store
and the run handle, never the catalog or the server, so it lives apart from the tool methods that
call it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping

from pydantic import ValidationError

from phasesweep.mcp.runner import FailurePayload
from phasesweep.mcp.runs import RunHandle, RunnerView, RunObservation, RunStore
from phasesweep.mcp.snapshots import parse_result_snapshot
from phasesweep.runtime.json import JsonObject, json_object

_CLEANUP_SETTLED_REMEDIATION = (
    "Operator recovery settled this run's process cleanup, so no recover-run step remains. The "
    "run itself still ended without completing; start a new run only if the user still wants "
    "the sweep to continue."
)


def run_failure_payload(
    runs: RunStore,
    handle: RunHandle,
    *,
    view: RunnerView | None = None,
    observation: RunObservation | None = None,
    force_snapshot_unavailable: bool = False,
) -> JsonObject | None:
    """Return a validated safe terminal failure, never the raw exception text.

    :param RunStore runs: Store holding the run's durable state.
    :param RunHandle handle: Run handle whose recorded terminal status
        should be inspected.
    :param RunnerView | None view: The read's derivation of the runner, used
        when a fresh observation must be derived here.
    :param RunObservation | None observation: State and terminal evidence already read
        together for this payload; otherwise read a fresh decision here.
    :param bool force_snapshot_unavailable: The result response has already
        refused a non-finalized or missing snapshot even if run state stays live.
    :return JsonObject | None: The run's ``failure`` payload validated
        against :class:`FailurePayload` and dumped to JSON-safe types. A
        terminal run with no usable result snapshot receives a generated
        snapshot-unavailable failure; otherwise returns ``None`` when no
        valid failure was recorded.
    """
    if view is None:
        view = runs.runner_view(handle)
    if observation is None:
        observation = runs.observe(handle, view=view)
    terminal: Mapping[str, object] = observation.terminal_status or {}
    persisted: JsonObject | None = None
    recorded_failure = terminal.get("failure")
    if recorded_failure is None and observation.terminal_status is None:
        recorded_failure = runs.cleanup_failure(handle)
    if recorded_failure is not None:
        with contextlib.suppress(ValidationError):
            persisted = json_object(
                FailurePayload.model_validate(recorded_failure).model_dump(
                    mode="json", exclude_none=True
                )
            )
    if persisted is not None and observation.cleanup_recovery_required:
        return persisted
    if observation.cleanup_recovery_required and terminal.get("returncode") == 0:
        # A succeeded run records no failure, but its MLflow upload worker's cleanup can be
        # unconfirmed, and that leaves the agent a recovery with no step to relay.
        return {
            "code": "cleanup_uncertain",
            "stage": "cleanup",
            "retryable": False,
            "actor": "operator",
            "remediation": (
                "The run succeeded and its published results are unaffected, but cleanup of "
                "its MLflow upload worker is unconfirmed. Stop monitoring and ask the operator "
                "to retry delivery and that cleanup with `phasesweep sync-tracking CONFIG "
                f"--generation {handle.run_id}`, then run `phasesweep mcp recover-run` from a "
                "host shell, review its preview and repeat with `--confirm`, before another "
                "launch."
            ),
        }
    if persisted is not None and persisted["code"] == "cleanup_uncertain":
        # The recorded step was recover-run, and cleanup recovery is no longer required.
        persisted = {**persisted, "remediation": _CLEANUP_SETTLED_REMEDIATION}
    runner_ended_with_boot = (
        observation.terminal_status is None
        and handle.launch_state == "spawned"
        and view.visibility == "gone"
    )
    runner_exited = terminal.get("error_class") == "RunnerExitedWithoutStatus" or (
        observation.terminal_status is None
        and handle.launch_state == "spawned"
        and observation.recovery_required
        and not observation.runner_heartbeat_stale
        and not runs.runner_may_be_alive(handle, view=view)
    )
    if runner_exited or runner_ended_with_boot or observation.runner_heartbeat_stale:
        recovery_required = observation.recovery_required or observation.runner_heartbeat_stale
        if runner_ended_with_boot:
            remedy = (
                "The runner ended with an earlier host boot. No process from that boot "
                "remains, so process cleanup and recover-run are not required. This "
                "run's historical results are unavailable; do not substitute mutable "
                "experiment-level results."
            )
        elif observation.runner_heartbeat_stale:
            remedy = (
                "The runner is in another PID namespace and its heartbeat is stale; "
                "it may have stopped. Stop monitoring and ask the operator to inspect "
                "this run with `phasesweep mcp recover-run` from a host shell outside "
                "the sandbox. Liveness and trial cleanup must be checked there."
            )
        else:
            remedy = (
                "The runner exited without writing a final status; its trial processes may "
                "still be alive. Stop monitoring and ask the operator to inspect this run "
                "with `phasesweep mcp recover-run` from a host shell, then repeat with "
                "`--confirm` after reviewing the process cleanup preview."
                if recovery_required
                else "The runner exited without writing a final status. Operator recovery "
                "settled process cleanup; this run's historical results remain unavailable. "
                "Do not substitute mutable experiment-level results."
            )
        return {
            "code": ("cleanup_uncertain" if recovery_required else "result_snapshot_unavailable"),
            "stage": "cleanup",
            "retryable": False,
            "actor": "operator",
            "remediation": remedy,
        }
    if force_snapshot_unavailable or (
        observation.state != "running" and parse_result_snapshot(terminal) is None
    ):
        failure: JsonObject = {
            "code": "result_snapshot_unavailable",
            "stage": "cleanup",
            "retryable": False,
            "actor": "operator",
            "remediation": (
                "Report that this run's historical results are unavailable and ask "
                "the operator to inspect the PhaseSweep run or server diagnostics; "
                "do not substitute mutable experiment-level results."
            ),
        }
        if persisted is not None:
            failure["cause"] = persisted
        return failure
    # Other terminal diagnoses and remediations survive operator recovery; the live
    # recovery_required field says whether recovery is still needed.
    return persisted
