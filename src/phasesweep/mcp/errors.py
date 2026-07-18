"""Error types for the MCP layer.

Two audiences, two trees:

* ``McpToolError`` and subclasses are returned to the (untrusted) agent. Their
  ``safe_message`` is built only from the experiment id and the kind of failure
  - never a path, command, env value, or storage URL.
* ``CatalogError`` is raised at startup to the operator who launched the server
  and may reference paths.
"""

from __future__ import annotations


class CatalogError(Exception):
    """Catalog could not be loaded or validated. Fatal at startup, operator-facing."""

    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        """Create an operator-facing catalog error.

        :param str message: Failure description; may reference paths.
        :param str | None suggestion: Optional concrete fix, surfaced by ``phasesweep mcp-check``.
        """
        super().__init__(message)
        self.suggestion = suggestion


class McpToolError(Exception):
    """Base for agent-facing tool failures. Carries a redacted message."""

    @property
    def safe_message(self) -> str:
        """Message safe to return to the agent (no paths/commands/secrets).

        :return str: Redacted message suitable for an MCP tool error.
        """
        return str(self)


class UnknownExperimentError(McpToolError):
    """Raised when a tool references an experiment id absent from the catalog."""

    def __init__(self, experiment_id: str) -> None:
        """Create an unknown-experiment tool error.

        :param str experiment_id: Agent-supplied catalog id that was not registered.
        """
        super().__init__(
            f"unknown experiment id {experiment_id!r}; call "
            "phasesweep_list_experiments and use an id from its response"
        )


class UnknownRunError(McpToolError):
    """Raised when a tool references a run id with no on-disk handle."""

    def __init__(self, run_id: str) -> None:
        """Create an unknown-run tool error.

        :param str run_id: Agent-supplied run id that did not match a persisted handle.
        """
        super().__init__(
            f"unknown run id {run_id!r}; use the exact run_id returned by "
            "phasesweep_launch_sweep. If it was lost, call phasesweep_get_latest_run "
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
            "phasesweep_validate_config and use a phase name from its response"
        )


class PermissionDeniedError(McpToolError):
    """Raised when the catalog forbids an action (launch/cancel/from_phase)."""

    def __init__(self, action: str, experiment_id: str) -> None:
        """Create a permission-denied tool error.

        :param str action: Forbidden action name.
        :param str experiment_id: Catalog id whose permissions denied the action.
        """
        super().__init__(
            f"action {action!r} is not permitted for experiment {experiment_id!r}; "
            "this is deliberate catalog policy. Report it to the user - only the "
            f"operator can enable it (allow.{action}: true) and restart the server. "
            "Do not retry."
        )


class ConfigChangedError(McpToolError):
    """Raised when a cataloged config no longer matches the startup snapshot."""

    def __init__(self, experiment_id: str) -> None:
        """Create a config-changed tool error.

        :param str experiment_id: Catalog id whose config hash no longer matches startup.
        """
        super().__init__(
            f"cataloged config for experiment {experiment_id!r} changed since server startup; "
            "the operator must restart the MCP server to reload and validate it. Do not retry."
        )


class RunSnapshotUnavailableError(McpToolError):
    """Raised when a persisted run cannot be matched to its config snapshot."""

    def __init__(self, run_id: str) -> None:
        """Create a run-snapshot tool error.

        :param str run_id: MCP run id whose saved config snapshot is unusable.
        """
        super().__init__(
            f"saved config snapshot for run {run_id!r} is unavailable or invalid; "
            "the run cannot be monitored safely. Report this to the operator; do not "
            "substitute experiment-level results."
        )


class RunResultSnapshotUnavailableError(McpToolError):
    """Raised when a terminal run lacks its immutable result snapshot."""

    def __init__(self, run_id: str) -> None:
        """Create a terminal-result-snapshot tool error.

        :param str run_id: MCP run id whose terminal result snapshot is unusable.
        """
        super().__init__(
            f"terminal result snapshot for run {run_id!r} is unavailable or invalid; "
            "retry once after a short delay in case finalization is still in progress. "
            "If the error persists, report it to the operator; do not substitute "
            "experiment-level results."
        )


class ToolResultTooLargeError(McpToolError):
    """Raised before an oversized result can reach the MCP client."""

    def __init__(self, tool_name: str, limit_bytes: int) -> None:
        """Create an actionable bounded-result error.

        :param str tool_name: MCP tool whose result exceeded the byte budget.
        :param int limit_bytes: Configured serialized result budget.
        """
        super().__init__(
            f"{tool_name} result exceeds the {limit_bytes}-byte response limit. "
            "For catalog listings, request a smaller limit and continue with next_cursor. "
            "For other tools, ask the operator to shorten catalog descriptions or reduce "
            "agent-visible winner parameter values."
        )


class ExperimentBusyError(McpToolError):
    """Raised when a second launch is attempted while a run is already live."""

    def __init__(self, experiment_id: str, run_id: str) -> None:
        """Create an experiment-busy tool error.

        :param str experiment_id: Catalog id that already has a live run.
        :param str run_id: Existing live run id blocking a new launch.
        """
        super().__init__(
            f"experiment {experiment_id!r} already has a running sweep "
            f"(run_id {run_id!r}); cancel it or wait for it to finish"
        )


class ConcurrencyLimitError(McpToolError):
    """Raised when launching would exceed the server's max concurrent runs."""

    def __init__(self, running: int, limit: int) -> None:
        """Create a concurrency-limit tool error.

        :param int running: Number of currently live runs.
        :param int limit: Configured maximum number of concurrent runs.
        """
        super().__init__(
            f"concurrency limit reached (max_concurrent_runs={limit}); {running} other "
            "sweep(s) are active. Wait for one to finish, or ask the user whether to "
            "cancel one. Do not retry immediately."
        )


class LaunchInProgressError(McpToolError):
    """Raised when another launch holds the launch lock. Transient; retry.

    The launch decision (count live runs against the cap, then spawn) is
    serialized so the cap can't be exceeded by a check-then-spawn race. When a
    concurrent launch holds that lock, this asks the caller to retry rather than
    silently proceeding past the cap.
    """

    def __init__(self) -> None:
        """Create a transient launch-in-progress tool error."""
        super().__init__("another launch is in progress on this server; retry in a moment")


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
            f"earlier phase {missing_phase!r} {reason}. Call phasesweep_get_winners "
            "for the experiment and resume only after every earlier phase has a winner."
        )
