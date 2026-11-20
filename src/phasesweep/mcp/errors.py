"""MCP errors for agents and operators.

``McpToolError`` messages exclude paths, commands, environment values and storage URLs.
``CatalogError`` messages are operator-facing and may include paths.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Literal, TypeAlias

from phasesweep.mcp.runs import decode_unreadable_record

_MAX_CAPACITY_BLOCKING_RUN_IDS = 5
_MAX_LISTED_UNREADABLE_RECORDS = 5

CatalogReloadRefusal: TypeAlias = Literal["invalid", "state_dir_changed"]

#: What the next launch does with an experiment whose run ended early, for a run that stopped.
NEW_RUN_RESUMES = (
    "a new run of this experiment keeps its finished trials and reruns its interrupted ones first"
)


def recover_run_step(run_id: str, *, outside_sandbox: bool = False) -> str:
    """Return the operator's whole ``recover-run`` step for one run, to follow "ask the operator to".

    The command refuses to run without its state directory and run id, and the agent never
    sees the state directory's path, so the step names it as the catalog's ``state_dir``.

    :param str run_id: Run to recover, or a placeholder such as ``<run id>`` for several.
    :param bool outside_sandbox: The run's processes are in another PID namespace, which only
        a shell outside every sandbox can prove gone.
    :return str: A lower-case clause ending without a full stop.
    """
    shell = "a host shell outside any sandbox" if outside_sandbox else "a host shell"
    return (
        f"run `phasesweep mcp recover-run --state-dir STATE_DIR --run-id {run_id}` from {shell}, "
        "with the catalog's `state_dir` as STATE_DIR, read its preview and repeat it with "
        "`--confirm`"
    )


def run_log_reference(run_id: str) -> str:
    """Name a run's own log, which holds its errors, for an agent that never sees the state path.

    :param str run_id: Run whose runner log is named.
    :return str: A noun phrase locating the log under the catalog's ``state_dir``.
    """
    return f"this run's log, `logs/{run_id}.log` under the catalog's `state_dir`"


def live_runner_step(pid: int | None, pgid: int | None) -> str:
    """Return the host step for a runner that cancellation or recovery found still alive.

    ``cancel_run`` and ``recover-run`` give the operator this one step, so neither refusal
    leaves a runner that survived SIGKILL unnamed.

    :param int | None pid: The runner's recorded PID.
    :param int | None pgid: The runner's recorded process group.
    :return str: One sentence naming the PID and process group and what to clear on the host.
    """
    return (
        f"If runner PID {pid} (process group {pgid}) is still running on the host, clear the "
        "cause there; a process in uninterruptible I/O ignores SIGKILL until the I/O ends."
    )


def runner_stop_step(run_id: str, pid: int | None, pgid: int | None) -> str:
    """Return the operator's route to stop a running run that ``cancel_run`` may not stop.

    The runner leads its own process group and handles SIGTERM as cancellation, so a
    group SIGTERM ends the run ``cancelled`` with its trials cleaned up.

    :param str run_id: The run to stop.
    :param int | None pid: The runner's PID when this server can probe it, else ``None``.
    :param int | None pgid: The runner's process group when this server can probe it.
    :return str: One sentence naming the host command.
    """
    if pid is not None and pgid is not None:
        found = (
            f"run `kill -TERM -- -{pgid}` from a host shell, which signals runner PID {pid} and "
            f"its process group {pgid}"
        )
    else:
        found = (
            f"run `pgrep -af {run_id}` from a host shell, take the PID of its "
            f"`runner.py --run-id {run_id}` line and run `kill -TERM -- -<that PID>`, since the "
            "runner leads its own process group"
        )
    return (
        f"To stop this run, the operator can {found}; the runner handles SIGTERM as cancellation."
    )


class CatalogError(Exception):
    """Catalog could not be loaded or validated. Operator-facing.

    A catalog-level error (unreadable or malformed catalog, unusable ``state_dir``,
    unsupported host) is fatal at startup. An entry error fails only that entry: the
    server starts on the entries that load and refuses launches, and a running server
    that cannot reload an edited catalog keeps serving reads from the last valid one and
    refuses launches.
    """

    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        """Create an operator-facing catalog error.

        :param str message: Failure description; may reference paths.
        :param str | None suggestion: Optional concrete fix, surfaced by ``phasesweep mcp check``.
        """
        super().__init__(message)
        self.suggestion = suggestion


class StateDirChangedError(CatalogError):
    """Raised when a reloaded catalog names another state directory than the server serves."""


class McpToolError(Exception):
    """Base for agent-facing tool failures with redacted messages."""


class UnknownExperimentError(McpToolError):
    """Raised when a tool references an experiment id absent from the catalog."""

    def __init__(self, experiment_id: str) -> None:
        """Create an unknown-experiment tool error.

        :param str experiment_id: Agent-supplied catalog id that is not served.
        """
        super().__init__(self._message(experiment_id))

    @staticmethod
    def _message(experiment_id: str) -> str:
        """Say why the id is not served and what the agent does next.

        :param str experiment_id: Agent-supplied catalog id.
        :return str: The tool error text.
        """
        return (
            f"unknown experiment id {experiment_id!r}; call "
            "list_experiments and use an id from its response"
        )


class CatalogEntryInvalidError(UnknownExperimentError):
    """Raised when a tool references a catalog entry that failed to load.

    A subclass of :class:`UnknownExperimentError` so run reads that fall back to a saved
    snapshot when an id is not served treat a failed entry the same way.
    """

    @staticmethod
    def _message(experiment_id: str) -> str:
        """Say that the entry failed to load, without paths or validation detail.

        :param str experiment_id: Agent-supplied catalog id whose entry failed to load.
        :return str: The tool error text.
        """
        return (
            f"experiment {experiment_id!r} is in the catalog but its entry failed to load, so "
            "this server cannot serve it. Ask the operator to run `phasesweep mcp check` and "
            "fix it; the server rereads the catalog on the next call. Do not retry until the "
            "operator reports the fix."
        )


class UnknownRunError(McpToolError):
    """Raised when a tool references a run id with no on-disk handle."""

    def __init__(self, run_id: str) -> None:
        """Create an unknown-run tool error.

        :param str run_id: Agent-supplied run id that did not match a persisted handle.
        """
        super().__init__(
            f"unknown run id {run_id!r}; use the exact run_id returned by "
            "launch_run. If it was lost, call get_latest_run "
            "with the experiment id instead of launching a replacement."
        )


class InvalidPhaseError(McpToolError):
    """Raised when a ``from_phase`` is not a declared phase of the experiment."""

    def __init__(self, experiment_id: str, phase: str) -> None:
        """Create an invalid-phase tool error.

        :param str experiment_id: Catalog id whose phase list was checked.
        :param str phase: Agent-supplied phase name that was not declared.
        """
        super().__init__(
            f"phase {phase!r} is not a phase of experiment {experiment_id!r}; call "
            "inspect_experiment and use a phase name from its response"
        )


class PermissionDeniedError(McpToolError):
    """Raised when the catalog forbids an action (launch/cancel/from_phase)."""

    def __init__(self, action: str, experiment_id: str, *, stop_step: str | None = None) -> None:
        """Create a permission-denied tool error.

        :param str action: Forbidden action name.
        :param str experiment_id: Catalog id whose permissions denied the action.
        :param str | None stop_step: For a refused cancel of a running run, the operator's
            route to stop it from :func:`runner_stop_step`.
        """
        # Launch-time permission caps cancellation, so a later catalog grant cannot reach
        # a run launched without it.
        applies = (
            "which applies on the next call but never to a run launched without it"
            if action == "cancel"
            else "which applies on the next call"
        )
        stop = f" {stop_step}" if stop_step is not None else ""
        super().__init__(
            f"action {action!r} is not permitted for experiment {experiment_id!r}; "
            "this is deliberate catalog policy. Report it to the user - only the "
            f"operator can enable it (allow.{action}: true in the catalog), {applies}.{stop} "
            "Do not retry."
        )


class ConfigChangedError(McpToolError):
    """Raised when a cataloged config changes between the server's reread and its use."""

    def __init__(self, experiment_id: str) -> None:
        """Create a config-changed tool error.

        :param str experiment_id: Catalog id whose config bytes changed during the call.
        """
        super().__init__(
            f"the config for experiment {experiment_id!r} changed on disk while this call was "
            "reading it, so the call stopped. The server rereads an edited catalog and its "
            "configs before each call; retry this call once, and report it to the operator if "
            "it is refused again."
        )


class CatalogReloadRefusedError(McpToolError):
    """Raised by launch while the catalog or a config it lists does not fully validate.

    That is the case for the latest refused reread of an edited catalog, and from startup
    when an entry failed to load.
    """

    def __init__(self, reason: CatalogReloadRefusal, failed_ids: Sequence[str]) -> None:
        """Create a refused-reload tool error.

        Launch is refused for every experiment, so the error names the entries that fail;
        otherwise an agent reads the healthy entries' ``launch: false`` as their own fault.

        :param CatalogReloadRefusal reason: ``invalid`` when the catalog or a config it
            lists fails validation; ``state_dir_changed`` when it names another state
            directory than the one this server was started with.
        :param Sequence[str] failed_ids: Catalog ids of the entries that failed to load;
            empty when the catalog file itself does not validate.
        """
        if reason == "state_dir_changed":
            message = (
                "the catalog was edited to name a different state_dir than the one this server "
                "was started with. A running server keeps its run records in one state "
                "directory, so it refuses launches, showing `launch: false` for every "
                "experiment, until the operator restarts it; reads keep "
                "using the original state directory and the last catalog loaded with it. "
                "Report this to the operator. Do not retry."
            )
        else:
            if failed_ids:
                names = ", ".join(repr(experiment_id) for experiment_id in failed_ids)
                failing = (
                    f"catalog entry {names} failed to load"
                    if len(failed_ids) == 1
                    else f"catalog entries {names} failed to load"
                )
                them = "it" if len(failed_ids) == 1 else "them"
            else:
                failing, them = "the catalog file does not validate", "it"
            message = (
                f"{failing}, so this server refuses launches of every experiment, showing "
                "`launch: false` for each, and inspection of a changed config, until the "
                f"operator fixes {them}; "
                "`phasesweep mcp check` shows why. Other reads keep using the entries that "
                "loaded and saved run snapshots, and the server rereads the catalog on the "
                "next call. Do not retry until the operator reports the fix."
            )
        super().__init__(message)


class RunSnapshotUnavailableError(McpToolError):
    """Raised when a persisted run cannot be matched to its config snapshot."""

    def __init__(self, run_id: str) -> None:
        """Create a run-snapshot tool error.

        :param str run_id: MCP run id whose saved config snapshot is unusable.
        """
        super().__init__(
            f"saved config snapshot for run {run_id!r} is unavailable or invalid; "
            "the run cannot be monitored safely. Report this to the operator, not another "
            "run's results in its place."
        )


class RunResultSnapshotUnavailableError(McpToolError):
    """Raised when a terminal run lacks its immutable result snapshot."""

    def __init__(self, run_id: str, finalization_state: str | None = None) -> None:
        """Create a terminal-result-snapshot tool error.

        :param str run_id: MCP run id whose terminal result snapshot is unusable.
        :param str | None finalization_state: Persisted snapshot finalization state, if known.
        """
        state = f" (finalization state: {finalization_state})" if finalization_state else ""
        super().__init__(
            f"terminal result snapshot for run {run_id!r} is unavailable or invalid{state}; "
            "retry once after a short delay only if finalization is still pending. "
            "If the error persists, report it to the operator; this run's results cannot be "
            "rebuilt, so do not report another run's results in their place."
        )


class RunPersistentStateUnavailableError(McpToolError):
    """Raised when a live run's configured persistent state cannot be read safely."""

    def __init__(self, run_id: str) -> None:
        """Create a redacted persistent-state read failure.

        :param str run_id: MCP run id whose configured study state is unreadable.
        """
        super().__init__(
            f"run {run_id!r} cannot be read safely because its persistent study state is "
            "unavailable or incompatible. Report this to the operator; do not retry, "
            "launch a replacement, or report another run's results in its place."
        )


class ExperimentBusyError(McpToolError):
    """Raised when a second launch is attempted while a run is already live."""

    def __init__(
        self,
        experiment_id: str,
        run_id: str,
        *,
        recovery_required: bool = False,
        runner_observable: bool = True,
    ) -> None:
        """Create an experiment-busy tool error.

        :param str experiment_id: Catalog id that already has a live run.
        :param str run_id: Existing live run id blocking a new launch.
        :param bool recovery_required: Whether the blocker requires operator recovery,
            including an unobservable runner whose heartbeat is stale.
        :param bool runner_observable: Whether this server can verify runner liveness.
        """
        step = recover_run_step(run_id, outside_sandbox=not runner_observable)
        if recovery_required:
            advice = f"Ask the operator to {step} before retrying the launch."
            if not runner_observable:
                advice += (
                    " Its runner is in another PID namespace and has probably stopped or left "
                    "cleanup uncertain."
                )
        else:
            advice = (
                "If you just launched and lost the response, this is likely that run; "
                f"monitor it with await_run using run_id {run_id!r}."
            )
            if not runner_observable:
                advice += (
                    " This server cannot verify runner liveness across PID namespaces. "
                    f"Monitor durable progress; if it stops, ask the operator to {step}."
                )
        super().__init__(
            f"experiment {experiment_id!r} already has a running sweep "
            f"(run_id {run_id!r}). {advice}"
        )


class ConcurrencyLimitError(McpToolError):
    """Raised when launching would exceed the server's max concurrent runs."""

    def __init__(
        self,
        running: int,
        limit: int,
        blocking_run_ids: Sequence[str],
        *,
        recovery_required_run_ids: Sequence[str] = (),
        unobservable_run_ids: Sequence[str] = (),
    ) -> None:
        """Create a concurrency-limit tool error.

        :param int running: Number of currently live runs.
        :param int limit: Configured maximum number of concurrent runs.
        :param Sequence[str] blocking_run_ids: Live run ids the agent can await.
        :param Sequence[str] recovery_required_run_ids: Blockers requiring operator
            recovery, including unobservable runners whose heartbeat is stale.
        :param Sequence[str] unobservable_run_ids: Blockers with unverifiable runner liveness.
        """
        recovery_ids = set(recovery_required_run_ids)
        ordered = sorted(
            dict.fromkeys(blocking_run_ids), key=lambda run_id: run_id not in recovery_ids
        )
        shown = ordered[:_MAX_CAPACITY_BLOCKING_RUN_IDS]
        blockers = ", ".join(repr(run_id) for run_id in shown)
        omitted = max(0, running - len(shown))
        omitted_note = f" ({omitted} more active)" if omitted else ""
        unobservable_ids = set(unobservable_run_ids)
        if recovery_ids:
            needs_recovery = ", ".join(repr(run_id) for run_id in shown if run_id in recovery_ids)
            step = recover_run_step(
                "<run id>", outside_sandbox=bool(unobservable_ids & recovery_ids)
            )
            advice = (
                f"Run_ids requiring operator recovery: {needs_recovery}. Ask the operator to "
                f"{step} for each of them before retrying the launch."
            )
        else:
            advice = (
                "Call await_run with one of these run_ids, then retry this launch only "
                "after that run is terminal. Ask the user before cancelling a blocking run."
            )
        if unobservable_ids - recovery_ids:
            step = recover_run_step("<run id>", outside_sandbox=True)
            advice += (
                " This server cannot verify runner liveness for blockers in another PID "
                f"namespace. Monitor durable progress; if one stops, ask the operator to {step} "
                "for it."
            )
        super().__init__(
            f"concurrency limit reached (max_concurrent_runs={limit}); {running} other "
            f"sweep(s) are active. Blocking run_ids: {blockers}{omitted_note}. {advice}"
        )


class RunCapacityUnknownError(McpToolError):
    """Raised when persisted run records cannot prove launch capacity."""

    def __init__(
        self, unreadable: Collection[str], recoverable_run_ids: Sequence[str] = ()
    ) -> None:
        """Create a fail-closed capacity error without exposing state paths.

        :param Collection[str] unreadable: Distinct malformed or orphaned record identities,
            ``run:<id>`` or ``handle:<file name under runs/>``.
        :param Sequence[str] recoverable_run_ids: Safe run ids whose only
            evidence is a provably pre-spawn config snapshot.
        """
        recovery = ""
        if recoverable_run_ids:
            ids = ", ".join(repr(run_id) for run_id in recoverable_run_ids)
            recovery = (
                f" Pre-spawn orphan run id(s): {ids}. Ask the operator to "
                f"{recover_run_step('<run id>')} for each of them."
            )
        decoded = (decode_unreadable_record(identity) for identity in unreadable)
        # Each label is a run id, or ``runs/<file name>`` for a handle file that names none.
        listed = sorted(
            (label, run_id is None)
            for run_id, label in decoded
            if run_id is None or run_id not in recoverable_run_ids
        )
        remaining = ""
        if listed:
            shown_records = listed[:_MAX_LISTED_UNREADABLE_RECORDS]
            shown = ", ".join(f"`{label}`" for label, _is_file in shown_records)
            more = len(listed) - _MAX_LISTED_UNREADABLE_RECORDS
            more_note = f" and {more} more" if more > 0 else ""
            file_note = (
                "; a listed `runs/` file moves out as it is"
                if any(is_file for _label, is_file in shown_records)
                else ""
            )
            remaining = (
                f" Unreadable or orphaned: {shown}{more_note}. Ask the operator to run "
                "`pgrep -af <run-id>` on the server host for each run id and, where no runner "
                "is alive, move `runs/<run-id>.json` and `logs/<run-id>.*` out of the state "
                f"directory{file_note}."
            )
        super().__init__(
            f"cannot prove available launch capacity because the MCP state contains "
            f"{len(unreadable)} unreadable or orphaned run record(s).{recovery}{remaining} "
            "Do not retry or launch a replacement."
        )


class LaunchInProgressError(McpToolError):
    """Raised when another launch holds the shared launch lock; retry the request."""

    def __init__(self) -> None:
        """Create a transient launch-in-progress tool error."""
        super().__init__("another launch is in progress on this server; retry in a moment")


class RunnerExitedBeforeReceiptError(McpToolError):
    """Raised when the detached runner exits before it records its launch receipt."""

    def __init__(self, run_id: str, returncode: int) -> None:
        """Create a path-free launch error naming the runner's exit code.

        :param str run_id: Run id whose runner exited.
        :param int returncode: The runner's exit code, negative for the signal that ended it.
        """
        super().__init__(
            f"run {run_id!r} did not start: its detached runner exited with code {returncode} "
            "before it recorded its launch receipt, so no trial ran. Do not retry the launch. "
            f"Ask the operator to read {run_log_reference(run_id)}, and fix what it reports, "
            "often a PhaseSweep the server imports only through PYTHONPATH, which the runner "
            "does not inherit."
        )


class RunLaunchUnsettledError(McpToolError):
    """Raised when cancellation has no verified runner identity to target."""

    def __init__(self, run_id: str) -> None:
        """Create an unsettled-launch cancellation error.

        :param str run_id: Run id whose launch has not published process identity.
        """
        super().__init__(
            f"run {run_id!r} has not finished publishing a verified runner identity, so "
            "cancellation cannot target it safely. Retry once after a short delay. If "
            "get_run_status reports recovery_required, stop retrying and tell the "
            "user that operator recovery is required."
        )


class RunNamespaceMismatchError(McpToolError):
    """Raised when cancellation targets a runner recorded in another PID namespace."""

    def __init__(self, run_id: str) -> None:
        """Create a PID-namespace cancellation error.

        :param str run_id: Run id whose runner this server cannot signal.
        """
        step = recover_run_step(run_id, outside_sandbox=True)
        super().__init__(
            f"run {run_id!r} was launched in another PID namespace, where its recorded "
            "process ids name unrelated processes for this server, so nothing was "
            "signalled. Do not retry cancellation. If get_run_status reports recovery_required, "
            f"stop monitoring and ask the operator to {step}. Otherwise keep monitoring with "
            "await_run and tell the user this server cannot cancel it: the operator can cancel "
            "it from the environment that launched it while that environment runs, or, once "
            f"that environment has ended, {step}."
        )


class ResumeNotReadyError(McpToolError):
    """Raised when ``from_phase`` resume is requested but an earlier winner is unusable."""

    def __init__(
        self,
        experiment_id: str,
        from_phase: str,
        missing_phase: str,
        reason: str = "has no winner yet",
    ) -> None:
        """Create a resume-not-ready tool error.

        :param str experiment_id: Catalog id being resumed.
        :param str from_phase: Requested resume phase.
        :param str missing_phase: Earlier phase whose winner could not be loaded.
        :param str reason: Safe explanation of why the earlier winner is unusable.
        """
        super().__init__(
            f"cannot resume {experiment_id!r} from phase {from_phase!r}: "
            f"earlier phase {missing_phase!r} {reason}. Call get_latest_run({experiment_id!r}) "
            "first; if it finds a run, call get_run_results with its run_id. Resume only after "
            "every earlier phase the remaining phases inherit has a compatible published winner."
        )


class ResumePublicationUnusableError(McpToolError):
    """Raised when ``from_phase`` resume is requested over a publication that cannot be used."""

    def __init__(
        self,
        experiment_id: str,
        from_phase: str,
        publication_integrity: Literal["failed", "permission_denied"],
    ) -> None:
        """Create a resume-over-unusable-publication tool error.

        :param str experiment_id: Catalog id being resumed.
        :param str from_phase: Requested resume phase.
        :param Literal["failed", "permission_denied"] publication_integrity: Whether
            the recorded publication no longer validates or cannot be read by this server.
        """
        if publication_integrity == "failed":
            detail = (
                "the recorded publication no longer validates (publication_integrity: failed). "
                "Do not retry and do not launch a replacement run; stop and report that the "
                "published result needs operator inspection."
            )
        else:
            detail = (
                "this server cannot read the recorded publication to validate it "
                "(publication_integrity: permission_denied). Do not retry; stop and ask the "
                "operator to re-read it as the publishing user or restore read permission."
            )
        super().__init__(f"cannot resume {experiment_id!r} from phase {from_phase!r}: {detail}")
