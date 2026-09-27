"""SDK-free implementation of every MCP tool.

PhaseSweepMCP answers the read-only tools here: catalog listing and
inspection, run lookup, status, await, and results. It inherits launch and
cancel, and the state all tools share, from RunControl. The FastMCP adapter in
:mod:`phasesweep.mcp.server` wraps each method as a tool.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any, NoReturn

from pydantic import ValidationError

from phasesweep.config import Experiment
from phasesweep.engine import (
    ArtifactRootConflictError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    read_status,
    read_winners,
)
from phasesweep.engine.fingerprints import _experiment_semantic_fingerprint
from phasesweep.mcp.errors import (
    McpToolError,
    RunPersistentStateUnavailableError,
    RunResultSnapshotUnavailableError,
    RunSnapshotUnavailableError,
    UnknownExperimentError,
    UnknownRunError,
)
from phasesweep.mcp.redaction import (
    ResultSource,
    intersect_visible_params,
    status_payload,
    winners_payload,
)
from phasesweep.mcp.registry import VisibleParamsPolicy
from phasesweep.mcp.run_control import RunControl
from phasesweep.mcp.runner import FailurePayload
from phasesweep.mcp.runs import RunHandle, RunState, RunStore, load_experiment_snapshot
from phasesweep.mcp.snapshots import (
    McpPublicationState,
    RunResultSnapshot,
    capture_pre_generation_result_snapshot,
    parse_result_snapshot,
)
from phasesweep.runtime.time import parse_utc_iso

# The tools log on the server's channel, so one logger name covers everything served.
log = logging.getLogger("phasesweep.mcp.server")

DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 100
# await_run blocks server-side so one call replaces many status polls. Keep the
# default below common 30-second MCP client deadlines; callers with a longer
# tool-call budget can opt into the larger supported range.
AWAIT_DEFAULT_TIMEOUT_SECONDS = 20
AWAIT_MIN_TIMEOUT_SECONDS = 5
AWAIT_MAX_TIMEOUT_SECONDS = 600
# Recheck cadence inside one await. Kept at the minimum timeout so a
# default-length wait rechecks run state several times mid-wait instead of only
# at entry and at the deadline.
AWAIT_RECHECK_SECONDS = 5


def _cursor_offset(cursor: str | None) -> int:
    """Decode the opaque v1 cursor into a list offset.

    :param str | None cursor: Cursor supplied by the agent.
    :return int: Zero-based catalog offset.
    :raises McpToolError: If the cursor is not a non-negative integer produced
        by ``list_experiments``.
    """
    if cursor is None:
        return 0
    try:
        offset = int(cursor)
    except ValueError:
        raise McpToolError("invalid cursor; use next_cursor returned by list_experiments") from None
    if offset < 0:
        raise McpToolError("invalid cursor; use next_cursor returned by list_experiments")
    return offset


def _run_elapsed_seconds(store: RunStore, handle: RunHandle, state: str) -> int | None:
    """Compute wall seconds for a run: launch-to-now while running, total when terminal.

    Terminal runs use the runner-stamped ``ended_at`` in status.json. A
    terminal run with no readable endpoint reports ``None`` rather than a guess.

    ``RunStore`` refuses to load a handle whose ``started_at`` does not parse,
    so the start endpoint is established for every handle that reaches here.
    It is still re-checked rather than asserted: an assertion would be
    compiled out under ``python -O`` and the unparsed ``None`` would surface
    as a ``TypeError`` from the subtraction below instead of the documented
    "endpoints unknown" answer.

    :param RunStore store: Run store used to read the terminal status.
    :param RunHandle handle: Persisted run whose duration is measured.
    :param str state: Derived run state for ``handle``.
    :return int | None: Non-negative whole seconds, or ``None`` when the
        endpoints cannot be established.
    """
    started = parse_utc_iso(handle.started_at)
    if started is None:
        return None
    ended: datetime | None
    if state == "running":
        ended = datetime.now(UTC)
    else:
        status = store.recorded_terminal_status(handle)
        ended = parse_utc_iso(status.get("ended_at")) if status is not None else None
    if ended is None:
        return None
    return max(0, round((ended - started).total_seconds()))


def _await_snapshot(
    state: str,
    recovery_required: bool,
    status: dict[str, Any],
) -> tuple[Any, ...]:
    """Reduce a status read to the comparable facts await_run watches.

    :param str state: Derived run state at read time.
    :param bool recovery_required: Whether the run needs operator recovery.
    :param dict[str, Any] status: Path-free ``read_status`` payload.
    :return tuple[Any, ...]: Hashable snapshot of run state, recovery requirement,
        and per-phase winner presence and dense trial-state counts.
    """
    return (
        state,
        recovery_required,
        tuple(
            (
                phase["phase"],
                phase["winner_present"],
                tuple(sorted(phase["trials"].items())),
            )
            for phase in status["phases"]
        ),
    )


def _phase_gained_winner(baseline: tuple[Any, ...], snapshot: tuple[Any, ...]) -> bool:
    """Return whether any phase gained a winner between two await snapshots.

    :param tuple[Any, ...] baseline: Snapshot taken when the wait began.
    :param tuple[Any, ...] snapshot: Snapshot from the latest recheck.
    :return bool: ``True`` when a phase's winner artifact appeared mid-wait.
    """
    had_winner = {phase: winner for phase, winner, _counts in baseline[2]}
    return any(
        winner and not had_winner.get(phase, False) for phase, winner, _counts in snapshot[2]
    )


class PhaseSweepMCP(RunControl):
    """SDK-free implementation of every tool. Methods raise ``McpToolError``."""

    def list_experiments(
        self,
        *,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Return a path-free catalog page (ids, descriptions, phases, metric).

        :param int limit: Maximum catalog entries to return.
        :param str | None cursor: Optional pagination cursor returned by a prior call.
        :return dict[str, Any]: Catalog page safe for the agent.
        :raises McpToolError: If ``limit`` is outside ``1..MAX_LIST_LIMIT``, or
            the cursor is not one this tool returned.
        """
        if limit < 1 or limit > MAX_LIST_LIMIT:
            raise McpToolError(f"limit must be between 1 and {MAX_LIST_LIMIT}")
        offset = _cursor_offset(cursor)
        experiments = self._registry.summaries()
        total_count = len(experiments)
        page = experiments[offset : offset + limit]
        next_offset = offset + len(page)
        return {
            "experiments": page,
            "total_count": total_count,
            "next_cursor": str(next_offset) if next_offset < total_count else None,
        }

    def validate(self, experiment_id: str) -> dict[str, Any]:
        """Report an experiment's phase structure (never the command/env/storage).

        :param str experiment_id: Catalog experiment id to inspect.
        :return dict[str, Any]: Path-free validation payload for the agent.
        """
        reg = self._registry.get(experiment_id)
        self._current_config_bytes(reg)
        exp = reg.experiment
        phases = [
            {
                "name": p.name,
                "n_trials": p.n_trials,
                "sampler": p.sampler.type,
                "inherits": p.inherits,
                "search_space": sorted(p.search_space),  # keys only, not ranges
            }
            for p in exp.phases
        ]
        return {
            "experiment_id": reg.id,
            "metric": reg.metric_payload,
            "capabilities": reg.capabilities,
            "phases": phases,
        }

    def latest_run(self, experiment_id: str) -> dict[str, Any]:
        """Return the newest durable MCP run for one catalog experiment.

        :param str experiment_id: Catalog id whose latest run should be resolved.
        :return dict[str, Any]: Path-free result with one computed run or ``found=false``.
        """
        reg = self._registry.get(experiment_id)
        handle = self._runs.latest_run_for(reg.id)
        run = self._run_payload(handle) if handle is not None else None
        return {
            "experiment_id": reg.id,
            "found": run is not None,
            "run": run,
        }

    def _run_payload(self, handle: RunHandle) -> dict[str, Any]:
        """Build one agent-visible run state with explicit recovery metadata.

        :param RunHandle handle: Persisted run handle to derive.
        :return dict[str, Any]: Run id, state, launch timestamp, and recovery flag.
        """
        state = self._runs.state(handle)
        return {
            "run_id": handle.run_id,
            "state": state,
            "started_at": handle.started_at,
            "recovery_required": self._runs.recovery_required(handle),
            "failure": self._run_failure_payload(handle, state=state),
        }

    def _run_failure_payload(
        self,
        handle: RunHandle,
        *,
        state: RunState | None = None,
        force_snapshot_unavailable: bool = False,
    ) -> dict[str, Any] | None:
        """Return a validated safe terminal failure, never the raw exception text.

        :param RunHandle handle: Run handle whose recorded terminal status
            should be inspected.
        :param RunState | None state: Already-derived run state, when available.
        :param bool force_snapshot_unavailable: The result response has already
            refused a non-finalized or missing snapshot even if run state stays live.
        :return dict[str, Any] | None: The run's ``failure`` payload validated
            against :class:`FailurePayload` and dumped to JSON-safe types. A
            terminal run with no usable result snapshot receives a generated
            snapshot-unavailable failure; otherwise returns ``None`` when no
            valid failure was recorded.
        """
        # A runner lost across a reboot is terminal even without a status file.
        terminal = self._runs.recorded_terminal_status(handle) or {}
        persisted: dict[str, Any] | None = None
        try:
            if terminal.get("failure") is not None:
                persisted = FailurePayload.model_validate(terminal["failure"]).model_dump(
                    mode="json", exclude_none=True
                )
        except ValidationError:
            pass
        if force_snapshot_unavailable or (
            (state if state is not None else self._runs.state(handle)) != "running"
            and parse_result_snapshot(terminal) is None
        ):
            failure: dict[str, Any] = {
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
        # Preserve the terminal diagnosis and remediation after operator recovery;
        # the live recovery_required field says whether recovery is still needed.
        return persisted

    def status(self, run_id: str) -> dict[str, Any]:
        """Per-phase trial counts and winner presence plus the run process state.

        Reads are always scoped to one persisted ``run_id``. A live run reads
        its immutable configuration snapshot against current local storage;
        a terminal run reads its frozen result snapshot.

        :param str run_id: Detached run id returned by ``launch`` or ``latest_run``.
        :return dict: Path-free status payload for the agent.
        """
        target_id, status, run, handle, result_source = self._read_status_target(run_id=run_id)
        return self._status_result(target_id, status, run, handle, result_source)

    async def await_run(
        self,
        run_id: str,
        timeout_seconds: int = AWAIT_DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Wait until a launched run changes, then return its full status.

        A bounded sleep-and-recheck loop over the same reads ``status`` does:
        no new state, no side effects. Returns early when the run reaches a
        terminal state, operator recovery is required, or a phase gains a
        winner; otherwise returns the current status when the (clamped)
        timeout elapses. A blocking status read already in progress cannot be
        preempted safely and may finish after that deadline. Rechecks run every
        ``AWAIT_RECHECK_SECONDS`` (or as late as one estimated read duration
        before the deadline) and re-resolve the run from disk, so state written
        while this request is active is observed. After a server restart, a
        fresh call resumes from the same durable run.

        :param str run_id: Detached run id to wait on.
        :param int timeout_seconds: Requested wait, clamped to
            [``AWAIT_MIN_TIMEOUT_SECONDS``, ``AWAIT_MAX_TIMEOUT_SECONDS``].
        :return dict[str, Any]: ``status`` payload plus ``changed`` and
            ``reason`` (``recovery_required`` / ``terminal`` /
            ``phase_completed`` / ``timeout``).
        """
        timeout = min(
            AWAIT_MAX_TIMEOUT_SECONDS, max(AWAIT_MIN_TIMEOUT_SECONDS, int(timeout_seconds))
        )
        deadline = time.monotonic() + timeout
        baseline: tuple[Any, ...] | None = None
        read_seconds = 0.0
        while True:
            read_started = time.monotonic()
            target_id, status, run, handle, result_source = await asyncio.to_thread(
                self._read_status_target,
                run_id=run_id,
            )
            read_seconds = time.monotonic() - read_started
            snapshot = _await_snapshot(
                run["state"],
                run["recovery_required"],
                status,
            )
            if baseline is None:
                baseline = snapshot
            if run["recovery_required"]:
                reason = "recovery_required"
            elif run["state"] in ("succeeded", "failed", "cancelled"):
                reason = "terminal"
            elif _phase_gained_winner(baseline, snapshot):
                reason = "phase_completed"
            elif time.monotonic() >= deadline:
                reason = "timeout"
            elif time.monotonic() + read_seconds >= deadline:
                # Another read as slow as the one just finished would end past
                # the deadline. Do not start it, but wait out the caller's
                # remaining budget before returning the fresh snapshot in hand.
                # The status read itself is blocking filesystem/storage work in
                # a worker thread and cannot be preempted safely; if that read
                # already crossed the deadline, the branch above returns as
                # soon as it finishes.
                await asyncio.sleep(max(0.0, deadline - time.monotonic()))
                reason = "timeout"
            else:
                remaining_before_next_read = max(
                    0.0,
                    deadline - time.monotonic() - read_seconds,
                )
                await asyncio.sleep(min(AWAIT_RECHECK_SECONDS, remaining_before_next_read))
                continue
            break
        result = self._status_result(target_id, status, run, handle, result_source)
        result["changed"] = snapshot != baseline
        result["reason"] = reason
        return result

    def _read_status_target(
        self,
        *,
        run_id: str,
    ) -> tuple[
        str,
        dict[str, Any],
        dict[str, Any],
        RunHandle,
        ResultSource,
    ]:
        """Resolve one run-specific status read to its current or frozen payload.

        :param str run_id: Persisted run id for the read.
        :return tuple: Target id, status data, optional run state and handle,
            and result provenance.
        """
        target_id, experiment, run, handle, snapshot = self._resolve_read_target(
            run_id=run_id,
            include_run=True,
        )
        assert run is not None
        snapshot, result_source = self._result_snapshot_view(experiment, handle, snapshot)
        # Keep the captured frozen results while pairing them with the latest
        # cleanup and recovery state. Both snapshot finalization and a cleanup
        # reservation can change the run after target resolution.
        run = self._run_payload(handle)
        if snapshot is not None:
            status = self._snapshot_status_payload(
                target_id,
                snapshot,
                result_source=result_source,
            )
        else:
            assert experiment is not None
            status = self._live_status_payload(target_id, experiment, handle)
            # A runner can finish freezing its results during the live
            # storage read. Discard that view once its snapshot is durable.
            completed, completed_source = self._result_snapshot_view(experiment, handle, None)
            if completed is not None:
                status = self._snapshot_status_payload(
                    target_id, completed, result_source=completed_source
                )
                result_source = completed_source
            # The final probe can also reserve cleanup or discover an
            # orphaned pending snapshot without producing a result view.
            run = self._run_payload(handle)
        if result_source == "terminal_snapshot_unavailable":
            run["failure"] = self._run_failure_payload(
                handle,
                state=run["state"],
                force_snapshot_unavailable=True,
            )
        return target_id, status, run, handle, result_source

    def _catalog_comparison_experiment(self, experiment_id: str) -> Experiment | None:
        """Return the config a future catalog launch would execute, if available.

        :param str experiment_id: Catalog id associated with a current or frozen read.
        :return Experiment | None: Current catalog config, or ``None`` after removal.
        """
        try:
            return self._registry.get(experiment_id).experiment
        except UnknownExperimentError:
            return None

    def _live_status_payload(
        self,
        experiment_id: str,
        experiment: Experiment,
        handle: RunHandle,
    ) -> dict[str, Any]:
        """Read live status and leave catalog drift unknown after decataloging.

        ``read_status`` normally compares against its read config when no
        comparison config is supplied. A run snapshot remains the correct
        artifact locator after its catalog entry is removed, but it is not a
        current catalog config and therefore cannot support a drift verdict.

        :param str experiment_id: Catalog id associated with the read target.
        :param Experiment experiment: Live run snapshot config.
        :param RunHandle handle: Live run pinning the represented generation.
        :return dict[str, Any]: Path-free live status payload.
        """
        comparison = self._catalog_comparison_experiment(experiment_id)
        try:
            status = read_status(
                experiment,
                generation_id=handle.run_id,
                comparison_experiment=comparison,
            )
        except (
            ArtifactRootConflictError,
            StudySchemaMismatchError,
            StudyStorageUnavailableError,
        ) as exc:
            self._raise_persistent_read_error(handle, exc)
        if comparison is None:
            status["published_config_matches_current"] = None
        return status

    @staticmethod
    def _raise_persistent_read_error(
        handle: RunHandle,
        exc: ArtifactRootConflictError | StudySchemaMismatchError | StudyStorageUnavailableError,
    ) -> NoReturn:
        """Replace a path-bearing live-study read failure with a safe tool error.

        :param RunHandle handle: Run whose immutable config selected the failed state.
        :param ArtifactRootConflictError | StudySchemaMismatchError | StudyStorageUnavailableError exc:
            Explicit persistent-state failure to redact.
        :raises RunPersistentStateUnavailableError: Always, without storage details.
        """
        log.info("persistent state read failed for run=%s: %s", handle.run_id, exc)
        raise RunPersistentStateUnavailableError(handle.run_id) from None

    def _snapshot_status_payload(
        self,
        experiment_id: str,
        snapshot: RunResultSnapshot,
        *,
        result_source: ResultSource,
    ) -> dict[str, Any]:
        """Return frozen run facts with a config-only live drift comparison.

        A terminal run snapshot is the historical read authority for that run.
        Publication pointers and integrity stay as captured: consulting the
        live artifact tree here would make an intact snapshot unreadable, and
        a later generation's damage could erase this run's
        already-validated winners. Config drift is the sole live field and
        compares the frozen represented-config fingerprint with the current
        catalog.

        :param str experiment_id: Catalog id associated with the run.
        :param RunResultSnapshot snapshot: Validated frozen result snapshot.
        :param ResultSource result_source: Snapshot provenance, including the
            unavailable placeholder case.
        :return dict[str, Any]: Frozen status with a current config-drift verdict.
        """
        status = snapshot.status_payload()
        comparison = self._catalog_comparison_experiment(experiment_id)
        represented_fingerprint = snapshot.represented_config_fingerprint
        status["published_config_matches_current"] = (
            represented_fingerprint == _experiment_semantic_fingerprint(comparison)
            if comparison is not None
            and represented_fingerprint is not None
            and result_source != "terminal_snapshot_unavailable"
            else None
        )
        return status

    def _status_result(
        self,
        target_id: str,
        status: dict[str, Any],
        run: dict[str, Any],
        handle: RunHandle,
        result_source: ResultSource,
    ) -> dict[str, Any]:
        """Build the common path-free response for status and await_run.

        :param str target_id: Experiment id represented by ``status``.
        :param dict[str, Any] status: Current or frozen engine status payload.
        :param dict[str, Any] run: Derived detached-run state.
        :param RunHandle handle: Already-resolved detached-run handle.
        :param ResultSource result_source: Current shared state or frozen run snapshot.
        :return dict[str, Any]: Agent-safe status response.
        """
        elapsed_seconds = _run_elapsed_seconds(self._runs, handle, run["state"])
        return status_payload(
            target_id,
            status,
            run,
            result_source=result_source,
            elapsed_seconds=elapsed_seconds,
        )

    def _resolve_read_target(
        self,
        *,
        run_id: str,
        include_run: bool,
    ) -> tuple[
        str,
        Experiment | None,
        dict[str, Any] | None,
        RunHandle,
        RunResultSnapshot | None,
    ]:
        """Resolve a run read to its immutable config or terminal snapshot.

        :param str run_id: Persisted run id for immutable run-specific reads.
        :param bool include_run: Whether to include live run state in the returned payload.
        :return tuple: Target id, parsed experiment when a live read needs it, optional run
            payload, handle, and captured terminal result snapshot.
        :raises UnknownRunError: If ``run_id`` names no persisted run.
        """
        handle = self._runs.get(run_id)
        if handle is None:
            raise UnknownRunError(run_id)
        try:
            frozen_snapshot = self._terminal_result_snapshot(handle)
        except RunResultSnapshotUnavailableError:
            frozen_snapshot = None
        # A complete terminal result snapshot contains every historical status
        # and winner fact this read exposes. The sibling config snapshot is
        # needed only for a live read or unavailable-result placeholder.
        experiment = None if frozen_snapshot is not None else self._load_run_experiment(handle)
        run = self._run_payload(handle) if include_run else None
        return handle.experiment_id, experiment, run, handle, frozen_snapshot

    def _load_run_experiment(self, handle: RunHandle) -> Experiment:
        """Load and verify the immutable config snapshot for a persisted run.

        Run-specific monitoring must follow the exact config the detached
        runner received, even if the cataloged config has since been edited and
        the MCP server restarted. The saved handle's hash is the guardrail: a
        missing, corrupted, or non-experiment snapshot is rejected instead of
        silently reporting status for the wrong storage or workdir.

        :param RunHandle handle: Persisted run identity whose snapshot should be loaded.
        :return Experiment: Parsed experiment from the launched per-run snapshot.
        :raises RunSnapshotUnavailableError: If the snapshot is missing,
            unreadable, not an experiment, or no longer matches the hash
            recorded in the handle.
        """
        snapshot_path = self._runs.config_snapshot_path(handle.run_id)
        try:
            return load_experiment_snapshot(
                snapshot_path,
                handle.config_sha256,
                source=f"run snapshot {handle.run_id}",
            )
        except (OSError, ValueError) as exc:
            log.info("invalid config snapshot for run=%s: %s", handle.run_id, exc)
            raise RunSnapshotUnavailableError(handle.run_id) from None

    def winners(self, run_id: str) -> dict[str, Any]:
        """Return the winning hyperparameters per completed phase.

        Reads are scoped to the immutable config and terminal result snapshots
        recorded for ``run_id``. Live reads use only that run's saved config;
        neither path reconstructs results from a catalog experiment.

        The payload carries ``publication_integrity`` so an empty winner list
        is never ambiguous: ``"absent"`` means nothing has published yet,
        ``"failed"`` means a recorded publication no longer validates;
        ``"permission_denied"`` means this user cannot validate it without
        implying corruption. Both expose no winners and require operator action.

        Every label describing the winners -- metric name and goal, objective
        evidence, and the declared phase plan the completeness fields are
        measured against -- comes from the represented generation's own
        recorded semantics, never from the current catalog config (review
        v0.5.16 / blocker 4). Reading them from the current config relabeled
        historical numbers with a metric they were never optimized for,
        asserted evidence guarantees the run never had, and, after a phase
        rename, hid the published winner while reporting the new name as
        missing. ``result_context`` and ``published_config_matches_current``
        report which config supplied the labels and whether it still matches
        the one a further run would execute. Visibility is unchanged: the
        current policy still decides which historical *values* are redacted.

        :param str run_id: Detached run id whose snapshot should be read.
        :return dict[str, Any]: Path-free winners payload for the agent.
        """
        target_id, experiment, _run, handle, snapshot = self._resolve_read_target(
            run_id=run_id,
            include_run=False,
        )
        snapshot, result_source = self._result_snapshot_view(experiment, handle, snapshot)
        if snapshot is not None:
            # The frozen snapshot already carries the represented generation's
            # own metric, phase plan, and drift verdict, captured under the
            # config that run executed -- so this branch reads its labels from
            # the same place the live branch does.
            status = self._snapshot_status_payload(
                target_id,
                snapshot,
                result_source=result_source,
            )
            winner_views = (
                []
                if status["publication_integrity"] in {"failed", "permission_denied", "unknown"}
                else snapshot.winner_views()
            )
        else:
            assert experiment is not None
            # Resolve the represented generation once via read_status, then
            # reuse that exact id for read_winners: two independent pointer
            # resolutions here could otherwise mix identities from different
            # moments. The represented generation id is the queried run's
            # generation, never a later mutable pointer.
            status = self._live_status_payload(
                target_id,
                experiment,
                handle,
            )
            # Enumerate the plan that generation published under, not the one
            # the config declares now: a phase renamed since publication used
            # to drop its winner from this payload entirely while reporting
            # the new name as missing (review v0.5.16 / blocker 4).
            try:
                winner_views = read_winners(
                    experiment,
                    generation_id=status["represented_generation_id"],
                    phase_names=status["result_phase_plan"],
                )
            except (
                ArtifactRootConflictError,
                StudySchemaMismatchError,
                StudyStorageUnavailableError,
            ) as exc:
                self._raise_persistent_read_error(handle, exc)
            if status["publication_integrity"] in {"failed", "permission_denied", "unknown"}:
                winner_views = []
            # The live read may span the runner's final snapshot write.
            # Results for that run now come from its frozen publication.
            completed, completed_source = self._result_snapshot_view(experiment, handle, None)
            if completed is not None:
                status = self._snapshot_status_payload(
                    target_id, completed, result_source=completed_source
                )
                winner_views = (
                    []
                    if status["publication_integrity"] in {"failed", "permission_denied", "unknown"}
                    else completed.winner_views()
                )
                result_source = completed_source
            elif self._runs.recovery_required(handle) or not self._runs._runner_is_live(handle):
                # No frozen result is available after the runner stopped or
                # cleanup became uncertain. Do not reuse the mutable winner
                # view captured before the final probe.
                unavailable = RunResultSnapshot.model_validate(
                    capture_pre_generation_result_snapshot(experiment)
                )
                status = self._snapshot_status_payload(
                    target_id, unavailable, result_source="terminal_snapshot_unavailable"
                )
                winner_views = []
                result_source = "terminal_snapshot_unavailable"
        represented_generation_id: str | None = status["represented_generation_id"]
        publication_integrity: McpPublicationState = status["publication_integrity"]
        visible_params = self._effective_visible_params(target_id, handle)
        result = winners_payload(
            target_id,
            winner_views,
            metric=status["metric"],
            declared_phases=status["result_phase_plan"],
            result_source=result_source,
            publication_integrity=publication_integrity,
            run_id=run_id,
            represented_generation_id=represented_generation_id,
            visible_params=visible_params,
            result_context=status["result_context"],
            published_config_matches_current=status["published_config_matches_current"],
        )
        result["failure"] = self._run_failure_payload(
            handle,
            force_snapshot_unavailable=result_source == "terminal_snapshot_unavailable",
        )
        return result

    def _effective_visible_params(
        self,
        experiment_id: str,
        handle: RunHandle,
    ) -> VisibleParamsPolicy:
        """Resolve current or launch/current-intersected winner visibility.

        :param str experiment_id: Catalog id associated with the represented results.
        :param RunHandle handle: MCP run whose generation is represented.
        :return VisibleParamsPolicy: Effective sampled-parameter visibility.
        """
        try:
            current_policy = self._registry.get(experiment_id).visible_params
        except UnknownExperimentError:
            return "none"
        launch_policy = handle.visible_params_at_launch or "none"
        return intersect_visible_params(launch_policy, current_policy)

    def _result_snapshot_view(
        self,
        experiment: Experiment | None,
        handle: RunHandle,
        snapshot: RunResultSnapshot | None,
    ) -> tuple[RunResultSnapshot | None, ResultSource]:
        """Resolve one run result without falling back to mutable terminal state.

        A failed terminal snapshot is represented by the same config-only
        unavailable-data shape used for pre-generation failures. This lets
        monitoring return the terminal run and an actionable failure while
        every phase count remains explicitly untrusted.

        :param Experiment | None experiment: Exact catalog or saved run configuration when a
            current-state read or unavailable-result placeholder needs one. A complete terminal
            snapshot is self-contained and does not require it.
        :param RunHandle handle: Detached run being read.
        :param RunResultSnapshot | None snapshot: Terminal snapshot already captured during
            target resolution, reused without rereading mutable finalization state.
        :return tuple: Optional result view and its agent-visible provenance.
        """
        if snapshot is not None:
            return snapshot, "frozen_run_snapshot"
        try:
            snapshot = self._terminal_result_snapshot(handle)
        except RunResultSnapshotUnavailableError:
            assert experiment is not None
            placeholder = capture_pre_generation_result_snapshot(experiment)
            return (
                RunResultSnapshot.model_validate(placeholder),
                "terminal_snapshot_unavailable",
            )
        return (
            snapshot,
            "frozen_run_snapshot" if snapshot is not None else "current_shared_study",
        )

    def _terminal_result_snapshot(self, handle: RunHandle) -> RunResultSnapshot | None:
        """Return a live run's current view or require its frozen terminal snapshot.

        :param RunHandle handle: Resolved run whose terminal status should be inspected.
        :return RunResultSnapshot | None: Validated snapshot, or ``None`` while
            the run remains non-terminal.
        :raises RunResultSnapshotUnavailableError: A terminal run has no
            terminal status, or its immutable result snapshot is absent or invalid.
        """
        terminal_status = self._runs.recorded_terminal_status(handle)
        if terminal_status is None:
            # A reboot proves a spawned runner and every descendant are gone,
            # so it can release capacity even when a hard exit prevented the
            # runner from writing status.json. It cannot, however, make the
            # mutable shared study a historical result for that run. Re-read
            # after deriving state: the runner may have completed its status
            # write between the first read and the state check.
            if self._runs._runner_is_live(handle):
                return None
            # Reserve cleanup for a dead runner before this read fails closed.
            # A concurrent recovery may own the transition lock, but that never
            # makes mutable shared state historical evidence for this run ID.
            self._runs.state(handle)
            terminal_status = self._runs.recorded_terminal_status(handle)
            if terminal_status is None:
                raise RunResultSnapshotUnavailableError(handle.run_id)
        if terminal_status.get("result_snapshot_state") == "pending":
            if self._runs._runner_is_live(handle):
                return None
            # The runner can publish the complete snapshot and exit between
            # the status read and its liveness probe. Re-read before treating
            # the pending state as orphaned.
            latest_status = self._runs.recorded_terminal_status(handle)
            if latest_status is None or latest_status.get("result_snapshot_state") == "pending":
                raise RunResultSnapshotUnavailableError(handle.run_id, "pending")
            terminal_status = latest_status
        snapshot = parse_result_snapshot(terminal_status)
        if snapshot is not None:
            return snapshot
        if self._runs.state(handle) == "running":
            return None
        # The runner may have completed the second atomic status write between
        # the first read and the state check. Re-read before declaring a
        # terminal snapshot unavailable.
        latest_status = self._runs.recorded_terminal_status(handle)
        if latest_status is not None:
            terminal_status = latest_status
            snapshot = (
                None
                if terminal_status.get("result_snapshot_state") == "pending"
                else parse_result_snapshot(terminal_status)
            )
            if snapshot is not None:
                return snapshot
        finalization_state = terminal_status.get("result_snapshot_state")
        raise RunResultSnapshotUnavailableError(
            handle.run_id,
            finalization_state if isinstance(finalization_state, str) else None,
        )
