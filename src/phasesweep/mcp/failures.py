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

from phasesweep.mcp.errors import NEW_RUN_RESUMES, recover_run_step, run_log_reference
from phasesweep.mcp.runner import (
    FailureCode,
    FailurePayload,
    FailureStage,
    _cleanup_uncertain_payload,
)
from phasesweep.mcp.runs import RunHandle, RunnerView, RunObservation, RunStore
from phasesweep.mcp.snapshots import McpPublicationState, parse_result_snapshot
from phasesweep.runtime.json import JsonObject, json_object


def operator_failure(code: FailureCode, stage: FailureStage, remediation: str) -> JsonObject:
    """Build a failure only the operator can clear, which the agent must not retry.

    :param FailureCode code: Failure category.
    :param FailureStage stage: Stage the failure is reported at.
    :param str remediation: The operator's step, as the agent relays it.
    :return JsonObject: The failure, with no cause.
    """
    return {
        "code": code,
        "stage": stage,
        "retryable": False,
        "actor": "operator",
        "remediation": remediation,
    }


def _with_cause(failure: JsonObject, cause: JsonObject | None) -> JsonObject:
    """Attach a failure beneath another as its cause, dropping the cause's own cause.

    :class:`FailurePayload` holds one level of cause, so a recorded failure that already
    carries one contributes only its own fields.

    :param JsonObject failure: Failure the response reports.
    :param JsonObject | None cause: Failure to keep beneath it, or ``None`` for none.
    :return JsonObject: ``failure`` with ``cause`` attached when one is given.
    """
    if cause is None:
        return failure
    return {**failure, "cause": {key: value for key, value in cause.items() if key != "cause"}}


def _validated(failure: JsonObject | None) -> JsonObject | None:
    """Validate a composed failure against the schema every response carries it under.

    :param JsonObject | None failure: Failure to report, or ``None``.
    :return JsonObject | None: The failure dumped through :class:`FailurePayload`.
    :raises pydantic.ValidationError: If the composition breaks the response schema.
    """
    if failure is None:
        return None
    return json_object(
        FailurePayload.model_validate(failure).model_dump(mode="json", exclude_none=True)
    )


_CLEANUP_SETTLED_REMEDIATION = (
    "Operator recovery settled this run's process cleanup, so no recover-run step remains. The "
    f"run itself still ended without completing; {NEW_RUN_RESUMES}."
)


def _stop_for_recovery(run_id: str, *, outside_sandbox: bool = False) -> str:
    """Tell the agent to stop and relay the operator's recover-run step for one run.

    :param str run_id: Run the operator recovers.
    :param bool outside_sandbox: The runner is in another PID namespace.
    :return str: One sentence.
    """
    step = recover_run_step(run_id, outside_sandbox=outside_sandbox)
    return f"Stop monitoring and ask the operator to {step}."


# A run's claim refuses either verdict, so the step precedes the next launch's preflight.
_PUBLICATION_FAILURES: dict[str, JsonObject] = {
    "failed": operator_failure(
        "publication_corrupt",
        "preflight",
        "The experiment's last published result no longer validates. Ask the operator to "
        "inspect it with `phasesweep status CONFIG` and restore it; until then a new run of "
        "this experiment fails before its first trial.",
    ),
    "permission_denied": operator_failure(
        "publication_unreadable",
        "preflight",
        "This server cannot read the experiment's published result to validate it, which is "
        "not evidence of corruption. Ask the operator to read it as the publishing user or "
        "restore the server's read permission; until then a new run of this experiment fails "
        "before its first trial.",
    ),
}


def response_failure(
    runs: RunStore,
    handle: RunHandle,
    *,
    view: RunnerView,
    observation: RunObservation | None = None,
    force_snapshot_unavailable: bool = False,
    publication_integrity: McpPublicationState | None = None,
) -> JsonObject | None:
    """Return the failure a run read reports, carrying the step whenever the agent must stop.

    The run's own diagnosis is reported as :func:`run_failure_payload` composes it, except that
    a run requiring recovery always reports a failure naming ``recover-run``. A stopped run with
    no failure of its own reports a ``failed`` or ``permission_denied`` publication instead; a
    run still running passed its preflight and is not stopped by one, and ``unknown`` already
    reads as an unavailable snapshot.

    :param RunStore runs: Store holding the run's durable state.
    :param RunHandle handle: Run being read.
    :param RunnerView view: The read's derivation of the runner.
    :param RunObservation | None observation: The read's observation of the run, or ``None``
        to observe it here.
    :param bool force_snapshot_unavailable: The response already reports the terminal snapshot
        unavailable.
    :param McpPublicationState | None publication_integrity: The response's publication verdict,
        or ``None`` when it reports none.
    :return JsonObject | None: The validated failure, or ``None`` when the run has none.
    """
    if observation is None:
        observation = runs.observe(handle, view=view)
    diagnosis = _diagnose(
        runs,
        handle,
        view=view,
        observation=observation,
        force_snapshot_unavailable=force_snapshot_unavailable,
    )
    failure = diagnosis.failure
    if observation.recovery_required and not diagnosis.names_recover_run:
        failure = _recovery_failure(runs, handle, observation, diagnosis.recorded)
    elif failure is None and publication_integrity is not None and observation.state != "running":
        failure = _PUBLICATION_FAILURES.get(publication_integrity)
    return _validated(failure)


def _recovery_failure(
    runs: RunStore, handle: RunHandle, observation: RunObservation, recorded: JsonObject | None
) -> JsonObject:
    """Name the recovery step for a run whose diagnosis carries none.

    :param RunStore runs: Store that can verify a launch preparation was abandoned.
    :param RunHandle handle: Run that requires recovery.
    :param RunObservation observation: The read's observation of the run.
    :param JsonObject | None recorded: The run's own recorded failure, kept as the cause.
    :return JsonObject: A ``cleanup_uncertain`` failure for an unresolved launch or unconfirmed
        cleanup, or ``result_snapshot_unavailable`` for an unfinished result snapshot.
    """
    if handle.launch_state == "launching" and observation.terminal_status is None:
        if runs.is_pre_spawn_orphan(handle.run_id):
            return operator_failure(
                "cleanup_uncertain",
                "cleanup",
                "Stop monitoring this abandoned launch. The next permitted launch clears this "
                "preparation automatically. To remove it without launching, ask the operator "
                f"to {recover_run_step(handle.run_id)}.",
            )
        return operator_failure(
            "cleanup_uncertain",
            "cleanup",
            "The server saved this run's launch but never recorded whether its runner "
            f"started. {_stop_for_recovery(handle.run_id)}",
        )
    if observation.snapshot_recovery_required and not observation.cleanup_recovery_required:
        snapshot_unfinished = operator_failure(
            "result_snapshot_unavailable",
            "cleanup",
            "The runner stopped before it finished recording this run's result snapshot. "
            f"{_stop_for_recovery(handle.run_id)}",
        )
        return _with_cause(snapshot_unfinished, recorded)
    cleanup = _cleanup_uncertain_payload([], run_id=handle.run_id)
    return _with_cause(json_object(cleanup), recorded)


def run_failure_payload(
    runs: RunStore,
    handle: RunHandle,
    *,
    view: RunnerView,
    observation: RunObservation,
    force_snapshot_unavailable: bool = False,
) -> JsonObject | None:
    """Return a validated safe terminal failure, never the raw exception text.

    The cancel refusal for a runner this server cannot observe composes its own remedy when
    this diagnosis has none, so it reads this rather than :func:`response_failure`.

    :param RunStore runs: Store holding the run's durable state.
    :param RunHandle handle: Run handle whose recorded terminal status
        should be inspected.
    :param RunnerView view: The read's derivation of the runner.
    :param RunObservation observation: State and terminal evidence already read together
        for this payload.
    :param bool force_snapshot_unavailable: The result response has already
        refused a non-finalized or missing snapshot even if run state stays live.
    :return JsonObject | None: The run's ``failure`` payload validated
        against :class:`FailurePayload` and dumped to JSON-safe types. A
        terminal run with no usable result snapshot receives a generated
        snapshot-unavailable failure; otherwise returns ``None`` when no
        valid failure was recorded.
    """
    diagnosis = _diagnose(
        runs,
        handle,
        view=view,
        observation=observation,
        force_snapshot_unavailable=force_snapshot_unavailable,
    )
    return _validated(diagnosis.failure)


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
        # unconfirmed, and that leaves the agent a recovery with no step to relay. The step is
        # unscoped: the worker may have been sending an earlier generation's recovered outcome,
        # whose record a sync scoped to this run's generation leaves pending.
        upload_cleanup = operator_failure(
            "cleanup_uncertain",
            "cleanup",
            "The run succeeded and its published results are unaffected, but cleanup of "
            "its MLflow upload worker is unconfirmed. Stop monitoring and ask the operator "
            "to retry delivery and that cleanup with `phasesweep sync-tracking CONFIG`, using "
            f"the config this run launched with, then {recover_run_step(handle.run_id)}, before "
            "another launch.",
        )
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
                "The runner ended with an earlier host boot, so no process from that boot "
                "remains and no recover-run step is needed. This run recorded no results; "
                f"{NEW_RUN_RESUMES}."
            )
        elif observation.runner_heartbeat_stale:
            remedy = (
                "The runner is in another PID namespace and its heartbeat is stale; it may "
                "have stopped, and only a shell outside the sandbox can check it and its "
                f"trials. {_stop_for_recovery(handle.run_id, outside_sandbox=True)}"
            )
        else:
            remedy = (
                "The runner exited without writing a final status; its trial processes may "
                f"still be alive. {_stop_for_recovery(handle.run_id)}"
                if recovery_required
                else "The runner exited without writing a final status, and operator recovery "
                f"settled its process cleanup. This run recorded no results; {NEW_RUN_RESUMES}."
            )
        runner_failure = operator_failure(
            "cleanup_uncertain" if recovery_required else "result_snapshot_unavailable",
            "cleanup",
            remedy,
        )
        return _Diagnosis(runner_failure, recorded, recovery_required)
    if force_snapshot_unavailable or (
        observation.state != "running" and parse_result_snapshot(terminal) is None
    ):
        snapshot_unavailable = operator_failure(
            "result_snapshot_unavailable",
            "cleanup",
            "Report that this run's results are unavailable, not another run's results in "
            f"their place, and ask the operator to inspect {run_log_reference(handle.run_id)}.",
        )
        return _Diagnosis(_with_cause(snapshot_unavailable, persisted), recorded, False)
    # Other terminal diagnoses and remediations survive operator recovery; the live
    # recovery_required field says whether recovery is still needed.
    return _Diagnosis(persisted, recorded, False)
