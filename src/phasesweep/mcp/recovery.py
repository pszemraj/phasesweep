"""Operator recovery of MCP process cleanup and stored terminal results."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna

from phasesweep.config import Experiment
from phasesweep.engine.artifact_roots import _check_published_phase_studies
from phasesweep.engine.attempts import (
    _inspect_active_attempts,
    _preflight_active_attempts,
    _PreflightCleanupReport,
    _retire_active_attempt,
)
from phasesweep.engine.cleanup import (
    _inspect_cleanup_uncertain_trials,
    _inspect_stale_running_trials,
    _previously_recovered_attempt_locations,
    _reap_stale_trials,
    _recover_cleanup_uncertain_trials,
)
from phasesweep.engine.errors import PublishedStudyMissingError, StudyStorageUnavailableError
from phasesweep.engine.locking import _experiment_lock
from phasesweep.engine.optuna import _load_existing_phase_study
from phasesweep.engine.publication import _resolve_publication_pointer
from phasesweep.errors import PhaseSweepError
from phasesweep.mcp.config_snapshot import load_experiment_snapshot
from phasesweep.mcp.runs import (
    ProcessIdentity,
    RunHandle,
    RunStore,
    identity_from_earlier_boot,
    write_status_file,
)
from phasesweep.mcp.snapshots import (
    RunResultSnapshot,
    finalize_result_snapshot,
    mark_result_snapshot_published,
    parse_result_snapshot,
)
from phasesweep.runtime.files import private_atomic_write_text
from phasesweep.runtime.process import is_same_live_process, kill_stale_group
from phasesweep.runtime.time import utc_now_iso


class RunRecoveryError(PhaseSweepError):
    """An operator recovery request cannot be completed safely."""


@dataclass(frozen=True)
class _RecoveryNeeds:
    """Recovery decisions derived from the recorded run and its existing evidence."""

    terminal_status: dict[str, Any] | None
    stored_snapshot: RunResultSnapshot | None
    prepared_publication_generation: object
    cleanup_needed: bool
    terminal_cleanup_uncertain: bool
    ownership_storage_unavailable: bool
    snapshot_recovery_required: bool
    snapshot_unavailable: bool
    snapshot_finalize_needed: bool


@dataclass
class _CleanupEvidence:
    """Trial cleanup evidence to persist before releasing the run reservation."""

    reaped_attempt_ids: set[str]
    reaped_attempt_locations: dict[str, tuple[str, int, str]]
    registered_recovery_attempt_ids: set[str] = field(default_factory=set)
    reaped: int = 0
    registered_attempts_reconciled: int = 0
    cleanup_recovered: int = 0


def recover_run(
    state_dir: Path,
    run_id: str,
    *,
    confirm: bool,
    emit: Callable[[str], None],
) -> None:
    """Inspect or perform recovery while retaining its lock and write ordering.

    :param Path state_dir: Existing, normalized MCP state directory.
    :param str run_id: Run identity to recover.
    :param bool confirm: Perform recovery instead of reporting the proposed actions.
    :param Callable emit: Report operator messages as each recovery step completes.
    :raises RunRecoveryError: The run cannot be inspected or safely recovered.
    """
    try:
        store = RunStore.open_existing(state_dir)
    except ValueError as exc:
        raise RunRecoveryError(str(exc)) from None
    handle = store.get(run_id)
    if handle is None:
        _recover_pre_spawn_orphan(store, run_id, confirm=confirm, emit=emit)
        return
    if handle.launch_state == "launching" and store.is_pre_spawn_orphan(run_id):
        _recover_pre_spawn_orphan(store, run_id, confirm=confirm, emit=emit)
        return
    handle, terminal_status = _resolve_launch_state(store, handle)
    needs = _recovery_needs(store, handle, terminal_status)
    if (
        not needs.cleanup_needed
        and not needs.snapshot_finalize_needed
        and not needs.snapshot_unavailable
    ):
        emit("No cleanup uncertainty or terminal result repair is needed for this run.")
        return

    identity = store.cleanup_identity(handle)
    earlier_boot = identity_from_earlier_boot(identity.boot_id)
    _require_dead_runner(identity, earlier_boot=earlier_boot, cleanup_needed=needs.cleanup_needed)
    config = _load_recovery_config(store, handle)
    recovery_lock = _experiment_lock(config) if confirm else contextlib.nullcontext()
    try:
        with recovery_lock:
            _cleanup_runner(identity, needs, confirm=confirm, earlier_boot=earlier_boot)
            publication_action = _publication_recovery_action(config, needs)
            evidence = _recover_trial_evidence(store, handle, config, needs, confirm=confirm)
            if not confirm:
                emit(_preflight_message(run_id, needs, evidence, publication_action))
                return

            if terminal_status is None:
                terminal_status = {
                    "run_id": run_id,
                    "returncode": 1,
                    "error_class": "RunnerExitedWithoutStatus",
                    "cleanup_confirmed": True,
                    "ended_at": utc_now_iso(),
                    "result_snapshot_state": "failed",
                    "result_snapshot_error": "HistoricalSnapshotUnavailable",
                }
                write_status_file(store.status_path(run_id), terminal_status)
            if needs.cleanup_needed:
                _persist_cleanup_recovery(store, handle, config, evidence, emit=emit)
            _finish_result_recovery(
                store, run_id, terminal_status, needs, evidence, publication_action, emit=emit
            )
    except RunRecoveryError:
        raise
    except RuntimeError as exc:
        raise RunRecoveryError(str(exc)) from None


def _recover_pre_spawn_orphan(
    store: RunStore, run_id: str, *, confirm: bool, emit: Callable[[str], None]
) -> None:
    """Inspect or remove an abandoned launch preparation.

    Acquires the launch lock before checking the orphan state. With confirmation,
    clears only a confirmed pre-spawn orphan while holding that lock. The orphan
    may be a legacy config snapshot without a handle or a transactional launch
    whose persisted launching handle has a free inherited lease.

    :param RunStore store: Existing run store that owns the launch reservation.
    :param str run_id: Identity of the possible pre-spawn orphan.
    :param bool confirm: Remove the orphan when true; otherwise only report the action.
    :param Callable[[str], None] emit: Callback that receives the operator-facing result.
    :raises RunRecoveryError: A launch is in progress, the run is unknown, or removal fails.
    """
    with store.launch_lock() as acquired:
        if not acquired:
            raise RunRecoveryError(
                "another MCP launch is in progress; wait for it to finish and retry"
            )
        if store.is_pre_spawn_orphan(run_id):
            if not confirm:
                log_path = store.log_path(run_id)
                archive_path = log_path.with_suffix(".log.recovered")
                log_action = ""
                if log_path.exists() or log_path.is_symlink():
                    log_action = f" Would preserve runner log {log_path} as {archive_path}."
                elif archive_path.exists() or archive_path.is_symlink():
                    log_action = f" Runner log already preserved at {archive_path}."
                emit(
                    f"Recovery preflight for {run_id}: would remove the abandoned launch "
                    "preparation after verifying no runner can still start a trainer under "
                    f"this identity.{log_action} Re-run with --confirm to perform that action."
                )
                return
            try:
                recovered_log = store.clear_pre_spawn_orphan(run_id)
            except ValueError as exc:
                raise RunRecoveryError(str(exc)) from None
            log_result = (
                f" Preserved runner log at {recovered_log}." if recovered_log is not None else ""
            )
            emit(
                f"Removed abandoned launch preparation for {run_id}; no runner can still "
                f"start a trainer under this identity.{log_result}"
            )
            return
    raise RunRecoveryError(f"unknown run id: {run_id}")


def _resolve_launch_state(
    store: RunStore, handle: RunHandle
) -> tuple[RunHandle, dict[str, Any] | None]:
    """Resolve a recorded launch state, retrying one unsettled launch read.

    Reads the handle and status again only when the initial state is ``launching`` without a
    terminal status, then refuses recovery if that state remains unresolved.

    :param RunStore store: Existing run store containing the durable handle and status.
    :param RunHandle handle: Initially loaded handle for the requested run.
    :raises RunRecoveryError: A launching run still has no terminal status after the refresh.
    :return tuple[RunHandle, dict[str, Any] | None]: Resolved handle and its terminal status,
        if recorded.
    """
    terminal_status = store.recorded_terminal_status(handle)
    if handle.launch_state == "launching" and terminal_status is None:
        refreshed_handle = store.get(handle.run_id)
        if refreshed_handle is not None:
            refreshed_status = store.recorded_terminal_status(refreshed_handle)
            if refreshed_handle.launch_state != "launching" or refreshed_status is not None:
                handle = refreshed_handle
                terminal_status = refreshed_status
        if handle.launch_state == "launching" and terminal_status is None:
            raise RunRecoveryError(
                "launch outcome is unresolved: the durable handle cannot distinguish a "
                "pre-spawn failure from a child that has not persisted its process identity. "
                "The run remains reserved. Wait briefly and retry; if this persists after a "
                "server crash, inspect the host because automated recovery cannot safely "
                "declare that no runner was spawned."
            )
    return handle, terminal_status


def _recovery_needs(
    store: RunStore, handle: RunHandle, terminal_status: dict[str, Any] | None
) -> _RecoveryNeeds:
    """Derive cleanup, publication, and snapshot recovery decisions from stored evidence.

    Inspects run-store recovery state and parses the recorded status; it does not persist a
    recovery result.

    :param RunStore store: Existing run store used to inspect recovery markers.
    :param RunHandle handle: Resolved durable handle for the run.
    :param dict[str, Any] | None terminal_status: Recorded terminal status, if available.
    :raises RunRecoveryError: No immutable snapshot remains for a result that cannot be rebuilt.
    :return _RecoveryNeeds: Decisions and parsed terminal evidence used by later recovery steps.
    """
    cleanup_required = store.cleanup_recovery_required(handle)
    snapshot_recovery_required = store.snapshot_recovery_required(handle)
    terminal_cleanup_uncertain = (
        terminal_status is not None
        and terminal_status.get("cleanup_confirmed") is False
        and cleanup_required
    )
    failure = terminal_status.get("failure") if terminal_status is not None else None
    failure_cause = failure.get("cause") if isinstance(failure, dict) else None
    ownership_storage_unavailable = (
        terminal_status is not None
        and terminal_status.get("generation_unavailable_reason") == "engine_generation_not_claimed"
        and isinstance(failure, dict)
        and failure.get("code") == "cleanup_uncertain"
        and isinstance(failure_cause, dict)
        and failure_cause.get("code") == "storage_unavailable"
        and failure_cause.get("stage") == "preflight"
    )
    cleanup_already_recovered = (
        terminal_status is not None
        and terminal_status.get("cleanup_confirmed") is False
        and not cleanup_required
    )
    runner_without_status = handle.launch_state == "spawned" and terminal_status is None
    cleanup_needed = cleanup_required or runner_without_status
    stored_snapshot = (
        parse_result_snapshot(terminal_status) if terminal_status is not None else None
    )
    prepared_publication_generation = (
        terminal_status.get("result_publication_generation_id")
        if terminal_status is not None
        and terminal_status.get("result_publication_state") == "prepared"
        else None
    )
    snapshot_unavailable = (
        terminal_status is not None and stored_snapshot is None
    ) or runner_without_status
    snapshot_finalize_needed = stored_snapshot is not None and (
        terminal_cleanup_uncertain or cleanup_already_recovered or snapshot_recovery_required
    )
    if not cleanup_needed and snapshot_unavailable and not snapshot_recovery_required:
        raise RunRecoveryError(
            "this run has no immutable terminal result snapshot. Historical results cannot "
            "be rebuilt from the current shared study because later runs may have changed it."
        )
    return _RecoveryNeeds(
        terminal_status=terminal_status,
        stored_snapshot=stored_snapshot,
        prepared_publication_generation=prepared_publication_generation,
        cleanup_needed=cleanup_needed,
        terminal_cleanup_uncertain=terminal_cleanup_uncertain,
        ownership_storage_unavailable=ownership_storage_unavailable,
        snapshot_recovery_required=snapshot_recovery_required,
        snapshot_unavailable=snapshot_unavailable,
        snapshot_finalize_needed=snapshot_finalize_needed,
    )


def _require_dead_runner(
    identity: ProcessIdentity, *, earlier_boot: bool, cleanup_needed: bool
) -> None:
    """Reject a live runner or an identity that cannot be checked safely.

    A prior boot proves recorded processes are gone. Otherwise, a cleanup requiring a PID must
    include its Linux start time before the live-process check can safely distinguish PID reuse.

    :param ProcessIdentity identity: Recorded runner PID, start time, process group, and boot ID.
    :param bool earlier_boot: Whether the recorded boot predates the current host boot.
    :param bool cleanup_needed: Whether recovery would need to clean up the runner group.
    :raises RunRecoveryError: The runner is live or PID reuse cannot be excluded.
    """
    # PID and /proc start time are unique only within one boot. An earlier boot
    # proves the runner and descendants are gone without signalling a recycled PID.
    if (
        cleanup_needed
        and not earlier_boot
        and identity.pid is not None
        and identity.pid_starttime is None
    ):
        raise RunRecoveryError(
            "runner process identity has no Linux /proc start time; refusing automated "
            "recovery because PID reuse cannot be ruled out"
        )
    if not earlier_boot and is_same_live_process(identity.pid, identity.pid_starttime):
        raise RunRecoveryError("runner still appears live; use cancel_run first")


def _load_recovery_config(store: RunStore, handle: RunHandle) -> Experiment:
    """Load and verify the configuration snapshot bound to a run.

    :param RunStore store: Existing run store containing the config snapshot path.
    :param RunHandle handle: Handle whose run ID and config digest select the snapshot.
    :raises RunRecoveryError: The snapshot is missing, unreadable, invalid, or digest-mismatched.
    :return Experiment: Parsed experiment configuration from the verified stored snapshot.
    """
    snapshot = store.config_snapshot_path(handle.run_id)
    if not snapshot.is_file():
        raise RunRecoveryError(f"run config snapshot is missing: {snapshot}")
    try:
        return load_experiment_snapshot(
            snapshot, handle.config_sha256, source=f"run snapshot {handle.run_id}"
        )
    except OSError as exc:
        raise RunRecoveryError(f"cannot read run config snapshot: {snapshot}") from exc
    except ValueError as exc:
        raise RunRecoveryError(f"{exc}; refusing recovery") from None


def _cleanup_runner(
    identity: ProcessIdentity, needs: _RecoveryNeeds, *, confirm: bool, earlier_boot: bool
) -> None:
    """Re-check and, when confirmed, clean the recorded runner process group.

    For confirmed calls, the caller holds the experiment lock across this liveness re-check and
    any signals so study recovery and recovery-state writes cannot race the runner.

    :param ProcessIdentity identity: Recorded runner process identity and process-group ID.
    :param _RecoveryNeeds needs: Decisions indicating whether runner cleanup is required.
    :param bool confirm: Enable liveness checks and any required process-group cleanup.
    :param bool earlier_boot: Whether the recorded process belongs to an earlier system boot.
    :raises RunRecoveryError: A runner is still live or process-group cleanup remains uncertain.
    """
    # Keep the lock from this liveness check through every signal, study
    # mutation, and recovery-state write in recover_run.
    if confirm and not earlier_boot and is_same_live_process(identity.pid, identity.pid_starttime):
        raise RunRecoveryError("runner still appears live; use cancel_run first")
    if (
        needs.cleanup_needed
        and confirm
        and not earlier_boot
        and not kill_stale_group(
            identity.pid, identity.pid_starttime, pgid=identity.pgid, grace_seconds=30.0
        )
    ):
        raise RunRecoveryError("runner process-group cleanup is still uncertain")


def _publication_recovery_action(config: Experiment, needs: _RecoveryNeeds) -> str | None:
    """Choose how to reconcile a prepared result publication.

    :param Experiment config: Experiment whose last-success publication pointer is inspected.
    :param _RecoveryNeeds needs: Recorded prepared generation and other recovery decisions.
    :raises RunRecoveryError: The publication pointer cannot be interpreted safely.
    :return str | None: ``"commit"`` when the pointer names the prepared generation,
        ``"abort"`` when it names another generation or is absent, or ``None`` when none was
        prepared.
    """
    if isinstance(needs.prepared_publication_generation, str):
        publication = _resolve_publication_pointer(config)
        if publication.generation_id == needs.prepared_publication_generation:
            return "commit"
        if publication.state == "absent" or publication.generation_id is not None:
            return "abort"
        raise RunRecoveryError(
            "the prepared run result cannot be reconciled because the "
            "last-success pointer is unreadable or malformed. Restore that "
            "pointer before retrying recovery."
        )
    return None


def _load_recovery_studies(config: Experiment, needs: _RecoveryNeeds) -> dict[str, optuna.Study]:
    """Load phase studies needed to inspect or reconcile trial cleanup evidence.

    When ownership storage was unavailable during the run, verifies the published phase history
    before allowing recovery to treat the run as having no owned trial.

    :param Experiment config: Experiment defining phase names and storage locations.
    :param _RecoveryNeeds needs: Recovery decisions that may require published-history checks.
    :raises RunRecoveryError: Required storage or published studies cannot be read.
    :return dict[str, optuna.Study]: Loadable phase studies keyed by phase name.
    """
    loaded_studies = {}
    for phase in config.phases:
        try:
            study = _load_existing_phase_study(config, phase)
        except StudyStorageUnavailableError as exc:
            raise RunRecoveryError(
                f"{exc} Restore the original complete storage ledger and access "
                "to it, then retry phasesweep mcp recover-run."
            ) from exc
        if study is not None:
            loaded_studies[phase.name] = study
    if needs.ownership_storage_unavailable:
        try:
            _check_published_phase_studies(
                config,
                loaded_studies,
                from_phase=(
                    needs.terminal_status.get("from_phase")
                    if needs.terminal_status is not None
                    else None
                ),
            )
        except (PublishedStudyMissingError, StudyStorageUnavailableError) as exc:
            raise RunRecoveryError(
                f"{exc} Restore the original complete storage ledger "
                "and study with access to it, then retry "
                "phasesweep mcp recover-run."
            ) from exc
    return loaded_studies


def _recover_trial_evidence(
    store: RunStore,
    handle: RunHandle,
    config: Experiment,
    needs: _RecoveryNeeds,
    *,
    confirm: bool,
) -> _CleanupEvidence:
    """Collect attributable cleanup evidence, optionally reconciling attempts and trials.

    When cleanup is needed, both modes read run, registry, and study evidence. Confirmed recovery
    invokes the cleanup helpers; preflight uses their inspection counterparts and does not
    persist the collected run evidence.

    :param RunStore store: Existing run store containing prior cleanup evidence and attempt IDs.
    :param RunHandle handle: Durable handle for the run being recovered.
    :param Experiment config: Experiment used to load attempt registries and phase studies.
    :param _RecoveryNeeds needs: Decisions indicating whether trial cleanup is required.
    :param bool confirm: Reconcile registered attempts and trials when true; otherwise inspect
        the corresponding evidence for the preflight report.
    :raises RunRecoveryError: Required studies are unavailable or evidence cannot support cleanup.
    :return _CleanupEvidence: Cleanup counts, run-attributable evidence, and registered attempts
        to retire.
    """
    run_id = handle.run_id
    reaped_ids, reaped_locations = store.cleanup_recovered_attempt_evidence(handle)
    evidence = _CleanupEvidence(reaped_ids, reaped_locations)
    causal_attempt_ids = store.cleanup_uncertain_attempt_ids(handle)
    inspected_attempt_ids: set[str] = set()
    inspected_attempt_generations: dict[str, str] = {}
    inspected_attempt_locations: dict[str, tuple[str, int, str]] = {}
    inspected_studies = 0
    if needs.cleanup_needed:
        loaded_studies = _load_recovery_studies(config, needs)
        if confirm:
            active_report = _PreflightCleanupReport()
            registered_attempts = _preflight_active_attempts(
                config, active_report, retain_recovery_evidence=True
            )
            registered_evidence = active_report.recovered_attempt_generations
            evidence.registered_recovery_attempt_ids.update(active_report.recovered_attempt_ids)
        else:
            registered_attempts = _inspect_active_attempts(config)
            registered_evidence = registered_attempts
        evidence.registered_attempts_reconciled = len(registered_attempts)
        evidence.reaped_attempt_ids.update(
            attempt_id
            for attempt_id, generation_id in registered_evidence.items()
            if generation_id == run_id or attempt_id in causal_attempt_ids
        )
        for phase in config.phases:
            study = loaded_studies.get(phase.name)
            if study is None:
                continue
            inspected_studies += 1
            # The study ledger precedes the run recovery record. Read evidence
            # left by an interrupted pass before this pass can append to it.
            previously_recovered = _previously_recovered_attempt_locations(
                study, phase.name, run_id, causal_attempt_ids=causal_attempt_ids
            )
            evidence.reaped_attempt_ids.update(previously_recovered)
            evidence.reaped_attempt_locations.update(previously_recovered)
            if confirm:
                evidence.cleanup_recovered += _recover_cleanup_uncertain_trials(
                    study,
                    config,
                    phase.name,
                    recovered_attempt_ids=inspected_attempt_ids,
                    recovered_attempt_generations=inspected_attempt_generations,
                    recovered_attempt_locations=inspected_attempt_locations,
                )
                evidence.reaped += _reap_stale_trials(
                    study,
                    config,
                    phase.name,
                    recovered_attempt_ids=inspected_attempt_ids,
                    recovered_attempt_generations=inspected_attempt_generations,
                    recovered_attempt_locations=inspected_attempt_locations,
                )
            else:
                evidence.cleanup_recovered += _inspect_cleanup_uncertain_trials(
                    study,
                    phase.name,
                    recovered_attempt_ids=inspected_attempt_ids,
                    recovered_attempt_generations=inspected_attempt_generations,
                    recovered_attempt_locations=inspected_attempt_locations,
                )
                evidence.reaped += _inspect_stale_running_trials(
                    study,
                    config,
                    phase.name,
                    recovered_attempt_ids=inspected_attempt_ids,
                    recovered_attempt_generations=inspected_attempt_generations,
                    recovered_attempt_locations=inspected_attempt_locations,
                )
        for attempt_id in inspected_attempt_ids:
            if (
                inspected_attempt_generations.get(attempt_id) == run_id
                or attempt_id in causal_attempt_ids
            ):
                evidence.reaped_attempt_ids.add(attempt_id)
                location = inspected_attempt_locations.get(attempt_id)
                if location is not None:
                    evidence.reaped_attempt_locations[attempt_id] = location
    _require_cleanup_evidence(needs, evidence, inspected_studies)
    return evidence


def _require_cleanup_evidence(
    needs: _RecoveryNeeds, evidence: _CleanupEvidence, inspected_studies: int
) -> None:
    """Require evidence before clearing terminal cleanup uncertainty.

    An ownership storage failure may have occurred before a trial was allocated, so it is allowed
    to proceed without reaped attempts after the applicable registry and study inspection.

    :param _RecoveryNeeds needs: Decisions including terminal uncertainty and storage ownership.
    :param _CleanupEvidence evidence: Attributable attempts and trial cleanup results observed.
    :param int inspected_studies: Number of phase studies examined for cleanup evidence.
    :raises RunRecoveryError: Terminal uncertainty lacks attributable cleanup evidence.
    """
    # An ownership read can fail before this run allocates a trial. Successful
    # registry/study inspection resolves that failure without run-owned trials.
    if (
        needs.terminal_cleanup_uncertain
        and not evidence.reaped_attempt_ids
        and not needs.ownership_storage_unavailable
    ):
        if inspected_studies == 0:
            detail = "no existing Optuna studies could be loaded from the run snapshot storage"
        else:
            detail = (
                "no RUNNING trials were reaped, no terminal trials recorded cleanup "
                "uncertainty, and no prior recovery pass left durable trial-level "
                "evidence"
            )
        raise RunRecoveryError(
            "runner status recorded cleanup_confirmed=false, but recovery could not "
            f"confirm any trial-level cleanup evidence ({detail}). Refusing to clear "
            "cleanup uncertainty."
        )


def _preflight_message(
    run_id: str, needs: _RecoveryNeeds, evidence: _CleanupEvidence, publication_action: str | None
) -> str:
    """Format the operator message describing a confirmed recovery's planned actions.

    :param str run_id: Identity of the run named in the operator message.
    :param _RecoveryNeeds needs: Decisions determining which recovery actions are described.
    :param _CleanupEvidence evidence: Counts included for proposed attempt and trial recovery.
    :param str | None publication_action: Prepared-publication action, if one is required.
    :return str: Preflight message instructing the operator to re-run with confirmation.
    """
    actions = []
    if needs.cleanup_needed:
        actions.append(
            "attempt runner process-group cleanup, "
            f"reconcile {evidence.registered_attempts_reconciled} registered attempt(s), "
            f"reap {evidence.reaped} stale trial(s), and recover {evidence.cleanup_recovered} "
            "cleanup-uncertain terminal trial(s)"
        )
    if needs.snapshot_finalize_needed:
        if publication_action == "commit":
            actions.append(
                "bind the prepared snapshot to its committed publication and "
                "finalize it with cleanup evidence"
            )
        elif publication_action == "abort":
            actions.append(
                "record the prepared generation as unpublished and finalize its stored snapshot"
            )
        else:
            actions.append("finalize the stored terminal snapshot with cleanup evidence")
    elif needs.snapshot_unavailable:
        actions.append("record that the historical terminal snapshot is unavailable")
    return (
        f"Recovery preflight for {run_id}: would {' and '.join(actions)}. "
        "Re-run with --confirm to perform those actions."
    )


def _persist_cleanup_recovery(
    store: RunStore,
    handle: RunHandle,
    config: Experiment,
    evidence: _CleanupEvidence,
    *,
    emit: Callable[[str], None],
) -> None:
    """Persist cleanup evidence, then retire recovered attempts and clear uncertainty.

    Atomically writes the evidence record before retiring registered attempts and then clears the
    run's cleanup-uncertain marker.

    :param RunStore store: Existing run store that receives the cleanup recovery record.
    :param RunHandle handle: Durable handle whose cleanup uncertainty is being cleared.
    :param Experiment config: Experiment used to retire registered active attempts.
    :param _CleanupEvidence evidence: Attributable cleanup counts, IDs, and trial locations.
    :param Callable[[str], None] emit: Callback that receives the completed-recovery message.
    """
    run_id = handle.run_id
    payload = {
        "run_id": run_id,
        "config_sha256": handle.config_sha256,
        "recovered_at": utc_now_iso(),
        "cleanup_confirmed": True,
        "reaped_running_trials": evidence.reaped,
        "registered_attempts_reconciled": evidence.registered_attempts_reconciled,
        "reaped_attempt_ids": sorted(evidence.reaped_attempt_ids),
        "reaped_attempt_locations": {
            attempt_id: {
                "phase": phase_name,
                "trial_number": trial_number,
                "generation_id": generation_id,
            }
            for attempt_id, (phase_name, trial_number, generation_id) in sorted(
                evidence.reaped_attempt_locations.items()
            )
        },
        "cleanup_uncertain_terminal_trials": evidence.cleanup_recovered,
    }
    private_atomic_write_text(
        store.cleanup_recovery_path(run_id), json.dumps(payload, indent=2) + "\n"
    )
    for attempt_id in evidence.registered_recovery_attempt_ids:
        _retire_active_attempt(config, attempt_id)
    store.clear_cleanup_uncertain(handle)
    emit(
        f"Cleared cleanup uncertainty for {run_id}; reconciled "
        f"{evidence.registered_attempts_reconciled} registered attempt(s), reaped {evidence.reaped} "
        f"stale trial(s), and confirmed {evidence.cleanup_recovered} cleanup-uncertain "
        "trial(s)."
    )


def _finish_result_recovery(
    store: RunStore,
    run_id: str,
    terminal_status: dict[str, Any],
    needs: _RecoveryNeeds,
    evidence: _CleanupEvidence,
    publication_action: str | None,
    *,
    emit: Callable[[str], None],
) -> None:
    """Persist terminal snapshot recovery and reconcile a prepared publication.

    May write a failed state for interrupted finalization, update publication fields, and persist
    finalization of the immutable stored snapshot. It reports an unavailable historical snapshot
    without rebuilding it from mutable shared study state.

    :param RunStore store: Existing run store containing the terminal status file.
    :param str run_id: Identity of the run whose result state is being recovered.
    :param dict[str, Any] terminal_status: Mutable recorded status updated during recovery.
    :param _RecoveryNeeds needs: Decisions governing unavailable, pending, and final snapshots.
    :param _CleanupEvidence evidence: Confirmed attempt IDs and locations for finalization.
    :param str | None publication_action: ``"commit"``, ``"abort"``, or no publication action.
    :param Callable[[str], None] emit: Callback that receives result-recovery messages.
    :raises RunRecoveryError: Finalizing a required stored snapshot fails.
    """
    if needs.snapshot_recovery_required and needs.stored_snapshot is None:
        terminal_status["result_snapshot_state"] = "failed"
        terminal_status["result_snapshot_error"] = "InterruptedFinalization"
        write_status_file(store.status_path(run_id), terminal_status)
    if publication_action is not None:
        assert needs.stored_snapshot is not None
        if publication_action == "commit":
            assert isinstance(needs.prepared_publication_generation, str)
            terminal_status["result_snapshot"] = mark_result_snapshot_published(
                needs.stored_snapshot.model_dump(mode="json"),
                generation_id=needs.prepared_publication_generation,
            )
            terminal_status["result_publication_state"] = "committed"
        else:
            terminal_status["returncode"] = 1
            terminal_status["error_class"] = "PublicationNotCommitted"
            terminal_status.pop("result_publication_state", None)
            terminal_status.pop("result_publication_generation_id", None)
    if needs.snapshot_finalize_needed:
        _finalize_stored_terminal_result_snapshot(
            store,
            run_id,
            terminal_status,
            confirmed_attempt_ids=evidence.reaped_attempt_ids,
            confirmed_attempt_locations=evidence.reaped_attempt_locations,
        )
        emit(f"Finalized stored terminal result snapshot for {run_id}.")
    elif needs.snapshot_unavailable:
        emit(
            f"Historical terminal result snapshot for {run_id} is unavailable and was "
            "not rebuilt from mutable shared state."
        )


def _finalize_stored_terminal_result_snapshot(
    store: RunStore,
    run_id: str,
    terminal_status: dict,
    *,
    confirmed_attempt_ids: set[str],
    confirmed_attempt_locations: dict[str, tuple[str, int, str]],
) -> None:
    """Finalize and persist the snapshot captured under the experiment lock.

    :param RunStore store: Existing run store containing the terminal status.
    :param str run_id: Run whose stored terminal snapshot should be finalized.
    :param dict terminal_status: Validated terminal process status to enrich.
    :param set[str] confirmed_attempt_ids: Exact attempts durably reconciled to FAIL.
    :param dict confirmed_attempt_locations: Phase, trial, and generation of reconciled attempts.
    :raises RunRecoveryError: The stored snapshot is unavailable or persistence fails.
    """
    snapshot = parse_result_snapshot(terminal_status)
    if snapshot is None:
        raise RunRecoveryError(
            f"run {run_id} has no immutable snapshot to finalize; refusing to read current "
            "shared state as historical evidence"
        )
    raw_snapshot = snapshot.model_dump(mode="json")
    terminal_status.pop("result_snapshot_error", None)
    terminal_status["result_snapshot_state"] = "pending"
    try:
        write_status_file(store.status_path(run_id), terminal_status)
        terminal_status["result_snapshot"] = finalize_result_snapshot(
            raw_snapshot,
            confirmed_attempt_ids=confirmed_attempt_ids,
            confirmed_attempt_locations=confirmed_attempt_locations,
        )
        terminal_status["result_snapshot_state"] = "complete"
        write_status_file(store.status_path(run_id), terminal_status)
    except Exception as exc:  # noqa: BLE001 - report operator repair failures
        terminal_status["result_snapshot"] = raw_snapshot
        terminal_status["result_snapshot_state"] = "pending"
        terminal_status["result_snapshot_error"] = type(exc).__name__
        with contextlib.suppress(Exception):
            write_status_file(store.status_path(run_id), terminal_status)
        raise RunRecoveryError(
            f"failed to finalize terminal result snapshot for {run_id}: {type(exc).__name__}"
        ) from None
