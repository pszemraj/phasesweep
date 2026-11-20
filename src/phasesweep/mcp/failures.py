"""Read-time composition of the ``failure`` payload an MCP run read returns.

A run read reports one safe failure payload chosen from the runner's recorded terminal status, its
cleanup record and what the read can observe of the runner now. The choice reads only the run store,
the run handle and the publication verdict the read already holds, never the catalog or the server,
so it lives apart from the tool methods that call it.

A response is the agent's only instruction at the moment it must stop, so every read that reports
``recovery_required`` carries a failure naming the operator's step, and so does a read over a
publication it cannot use. Nothing here is stored.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import ValidationError

from phasesweep.mcp.runner import FailurePayload, _cleanup_uncertain_payload
from phasesweep.mcp.runs import RunHandle, RunnerView, RunObservation, RunStore
from phasesweep.mcp.snapshots import McpPublicationState, parse_result_snapshot
from phasesweep.runtime.json import JsonObject, json_object

_CLEANUP_SETTLED_REMEDIATION = (
    "Operator recovery settled this run's process cleanup, so no recover-run step remains. The "
    "run itself still ended without completing; a new run of this experiment keeps its finished "
    "trials and reruns its interrupted ones first."
)
_RECOVER_RUN_STEP = (
    "Stop monitoring and ask the operator to run `phasesweep mcp recover-run` from a host shell, "
    "review its preview and repeat with `--confirm`."
)
# A run's claim refuses either verdict, so the step precedes the next launch's preflight.
_PUBLICATION_FAILURES: dict[str, JsonObject] = {
    "failed": {
        "code": "publication_corrupt",
        "stage": "preflight",
        "retryable": False,
        "actor": "operator",
        "remediation": (
            "The experiment's last published result no longer validates. Ask the operator to "
            "inspect it with `phasesweep status CONFIG` and restore it before any launch of this "
            "experiment: a successful run would move the publication pointer past it, and "
            "nothing would report it again."
        ),
    },
    "permission_denied": {
        "code": "publication_unreadable",
        "stage": "preflight",
        "retryable": False,
        "actor": "operator",
        "remediation": (
            "This server cannot read the experiment's published result to validate it, which is "
            "not evidence of corruption. Ask the operator to read it as the publishing user or "
            "restore the server's read permission."
        ),
    },
}


def response_failure(
    runs: RunStore,
    handle: RunHandle,
    *,
    view: RunnerView | None = None,
    observation: RunObservation | None = None,
    force_snapshot_unavailable: bool = False,
    publication_integrity: McpPublicationState | None = None,
) -> JsonObject | None:
    """Return the failure a run read reports, carrying the step whenever the agent must stop.

    The run's own diagnosis is reported as :func:`run_failure_payload` composes it, except that
    a run requiring recovery always reports a failure naming ``recover-run``. A run with no
    failure of its own reports a ``failed`` or ``permission_denied`` publication instead;
    ``unknown`` already reads as an unavailable snapshot.

    :param RunStore runs: Store holding the run's durable state.
    :param RunHandle handle: Run being read.
    :param RunnerView | None view: The read's derivation of the runner, or ``None`` to derive it.
    :param RunObservation | None observation: The read's observation of the run, or ``None``
        to observe it here.
    :param bool force_snapshot_unavailable: The response already reports the terminal snapshot
        unavailable.
    :param McpPublicationState | None publication_integrity: The response's publication verdict,
        or ``None`` when it reports none.
    :return JsonObject | None: The validated failure, or ``None`` when the run has none.
    """
    if view is None:
        view = runs.runner_view(handle)
    if observation is None:
        observation = runs.observe(handle, view=view)
    diagnosis = _diagnose(
        runs,
        handle,
        view=view,
        observation=observation,
        force_snapshot_unavailable=force_snapshot_unavailable,
    )
    if observation.recovery_required and not diagnosis.names_recover_run:
        return _recovery_failure(handle, observation, diagnosis.recorded)
    if diagnosis.failure is None and publication_integrity is not None:
        publication = _PUBLICATION_FAILURES.get(publication_integrity)
        if publication is not None:
            return dict(publication)
    return diagnosis.failure


def _recovery_failure(
    handle: RunHandle, observation: RunObservation, recorded: JsonObject | None
) -> JsonObject:
    """Name the operator's recover-run step for a run whose diagnosis carries none.

    :param RunHandle handle: Run that requires recovery.
    :param RunObservation observation: The read's observation of the run.
    :param JsonObject | None recorded: The run's own recorded failure, kept as the cause.
    :return JsonObject: A ``cleanup_uncertain`` failure for an unresolved launch or unconfirmed
        cleanup, or ``result_snapshot_unavailable`` for an unfinished result snapshot.
    """
    cause: JsonObject = {}
    if recorded is not None:
        cause = {"cause": {key: value for key, value in recorded.items() if key != "cause"}}
    if handle.launch_state == "launching" and observation.terminal_status is None:
        return {
            "code": "cleanup_uncertain",
            "stage": "cleanup",
            "retryable": False,
            "actor": "operator",
            "remediation": (
                "The server saved this run's launch but never recorded whether its runner "
                f"started. {_RECOVER_RUN_STEP}"
            ),
        }
    if observation.snapshot_recovery_required and not observation.cleanup_recovery_required:
        return {
            "code": "result_snapshot_unavailable",
            "stage": "cleanup",
            "retryable": False,
            "actor": "operator",
            "remediation": (
                "The runner stopped before it finished recording this run's result snapshot. "
                f"{_RECOVER_RUN_STEP}"
            ),
            **cause,
        }
    return {**json_object(_cleanup_uncertain_payload([])), **cause}


def run_failure_payload(
    runs: RunStore,
    handle: RunHandle,
    *,
    view: RunnerView | None = None,
    observation: RunObservation | None = None,
    force_snapshot_unavailable: bool = False,
) -> JsonObject | None:
    """Return a validated safe terminal failure, never the raw exception text.

    Cancellation composes its own remedy when this diagnosis has none, so it reads this
    rather than :func:`response_failure`.

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
    return _diagnose(
        runs,
        handle,
        view=view,
        observation=observation,
        force_snapshot_unavailable=force_snapshot_unavailable,
    ).failure


@dataclass(frozen=True)
class _Diagnosis:
    """A run's diagnosed failure, its own recorded failure, and whether the first names recover-run."""

    failure: JsonObject | None
    recorded: JsonObject | None
    names_recover_run: bool


def _diagnose(
    runs: RunStore,
    handle: RunHandle,
    *,
    view: RunnerView,
    observation: RunObservation,
    force_snapshot_unavailable: bool,
) -> _Diagnosis:
    """Diagnose one run's failure from its recorded status, cleanup record and runner.

    :param RunStore runs: Store holding the run's durable state.
    :param RunHandle handle: Run being diagnosed.
    :param RunnerView view: The read's derivation of the runner.
    :param RunObservation observation: The read's observation of the run.
    :param bool force_snapshot_unavailable: The response already reports the terminal snapshot
        unavailable.
    :return _Diagnosis: The failure to report, the run's validated recorded failure, and
        whether that failure asks the operator for ``recover-run``.
    """
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
        return _Diagnosis(persisted, persisted, persisted["code"] == "cleanup_uncertain")
    if observation.cleanup_recovery_required and terminal.get("returncode") == 0:
        # A succeeded run records no failure, but its MLflow upload worker's cleanup can be
        # unconfirmed, and that leaves the agent a recovery with no step to relay.
        upload_cleanup: JsonObject = {
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
        return _Diagnosis(upload_cleanup, persisted, True)
    recorded = persisted
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
        runner_failure: JsonObject = {
            "code": ("cleanup_uncertain" if recovery_required else "result_snapshot_unavailable"),
            "stage": "cleanup",
            "retryable": False,
            "actor": "operator",
            "remediation": remedy,
        }
        return _Diagnosis(runner_failure, recorded, recovery_required)
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
        return _Diagnosis(failure, recorded, False)
    # Other terminal diagnoses and remediations survive operator recovery; the live
    # recovery_required field says whether recovery is still needed.
    return _Diagnosis(persisted, recorded, False)
