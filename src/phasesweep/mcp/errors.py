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


class McpToolError(Exception):
    """Base for agent-facing tool failures. Carries a redacted message."""

    @property
    def safe_message(self) -> str:
        """Message safe to return to the agent (no paths/commands/secrets)."""
        return str(self)


class UnknownExperimentError(McpToolError):
    """Raised when a tool references an experiment id absent from the catalog."""

    def __init__(self, experiment_id: str) -> None:
        super().__init__(f"unknown experiment id {experiment_id!r}")


class UnknownRunError(McpToolError):
    """Raised when a tool references a run id with no on-disk handle."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"unknown run id {run_id!r}")


class InvalidPhaseError(McpToolError):
    """Raised when a ``from_phase`` is not a declared phase of the experiment."""

    def __init__(self, experiment_id: str, phase: str) -> None:
        super().__init__(f"phase {phase!r} is not a phase of experiment {experiment_id!r}")


class PermissionDeniedError(McpToolError):
    """Raised when the catalog forbids an action (launch/cancel/from_phase)."""

    def __init__(self, action: str, experiment_id: str) -> None:
        super().__init__(f"action {action!r} is not permitted for experiment {experiment_id!r}")


class ExperimentBusyError(McpToolError):
    """Raised when a second launch is attempted while a run is already live."""

    def __init__(self, experiment_id: str, run_id: str) -> None:
        super().__init__(
            f"experiment {experiment_id!r} already has a running sweep "
            f"(run_id {run_id!r}); cancel it or wait for it to finish"
        )


class ConcurrencyLimitError(McpToolError):
    """Raised when launching would exceed the server's max concurrent runs."""

    def __init__(self, running: int, limit: int) -> None:
        super().__init__(
            f"server is already running {running} sweep(s) (limit {limit}); "
            "wait for one to finish or cancel it"
        )


class LaunchInProgressError(McpToolError):
    """Raised when another launch holds the launch lock. Transient; retry.

    The launch decision (count live runs against the cap, then spawn) is
    serialized so the cap can't be exceeded by a check-then-spawn race. When a
    concurrent launch holds that lock, this asks the caller to retry rather than
    silently proceeding past the cap.
    """

    def __init__(self) -> None:
        super().__init__("another launch is in progress on this server; retry in a moment")


class ResumeNotReadyError(McpToolError):
    """Raised when ``from_phase`` resume is requested but an earlier winner is missing."""

    def __init__(self, experiment_id: str, from_phase: str, missing_phase: str) -> None:
        super().__init__(
            f"cannot resume {experiment_id!r} from phase {from_phase!r}: "
            f"earlier phase {missing_phase!r} has no winner yet"
        )
