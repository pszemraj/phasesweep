"""Shared operator-facing error taxonomy."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Self


class OperatorAction(StrEnum):
    """Operator remediation category for routing expected failures.

    FRESH_NAMESPACE starts a separate namespace for the affected state;
    RESTORE_LEDGER repairs storage; RESTORE_TREE repairs files or permissions;
    PROVISION_LOCK_DIR points PHASESWEEP_LOCK_DIR at an existing writable
    directory every launching context shares, or moves the work to the host.
    RUN_RECOVER_RUN retries process cleanup through the appropriate CLI or MCP
    recovery entry point. FIX_CONFIG corrects configuration or environment;
    RETRY repeats the operation after its blocker resolves; INSPECT_LOGS requires diagnosis.
    Defects do not inherit PhaseSweepError or an operator remediation.
    """

    FRESH_NAMESPACE = "fresh_namespace"
    RESTORE_LEDGER = "restore_ledger"
    RESTORE_TREE = "restore_tree"
    PROVISION_LOCK_DIR = "provision_lock_dir"
    RUN_RECOVER_RUN = "run_recover_run"
    FIX_CONFIG = "fix_config"
    RETRY = "retry"
    INSPECT_LOGS = "inspect_logs"


class PhaseSweepError(RuntimeError):
    """Expected operational failure carrying a diagnostic and remediation action.

    The CLI reports these failures without an internal-defect traceback.
    CLI and MCP boundaries render the action; ``str(error)`` and ``args`` retain
    the diagnostic for Python callers and exception translation.
    """

    # Concrete subclasses must declare their own default action.
    default_action: ClassVar[OperatorAction] = OperatorAction.INSPECT_LOGS

    def __init__(
        self,
        *args: object,
        action: OperatorAction | None = None,
        remediation: str | None = None,
    ) -> None:
        """Create an operational failure carrying the remediation it calls for.

        :param object args: Standard exception arguments; the first is the message.
        :param OperatorAction | None action: Remediation for this one raise,
            overriding the class's :attr:`default_action`.
        :param str | None remediation: Concrete operator step when the raise knows it.
        """
        super().__init__(*args)
        self.action: OperatorAction = action if action is not None else type(self).default_action
        self.remediation = remediation

    @classmethod
    def rewrap(
        cls,
        cause: BaseException,
        *args: object,
        action: OperatorAction | None = None,
        remediation: str | None = None,
    ) -> Self:
        """Construct a translated error while retaining its cause's remediation.

        An explicit action wins; otherwise a PhaseSweepError cause donates its
        action, and other causes use default_action. The caller raises the returned
        error with an explicit ``from cause``.

        :param BaseException cause: Failure being rewrapped.
        :param object args: Arguments for the new error; the first is the message.
        :param OperatorAction | None action: Remediation for this one raise.
        :param str | None remediation: Concrete operator step, inherited from the cause by default.
        :return Self: The new error, for the caller to raise ``from cause``.
        """
        if action is None and isinstance(cause, PhaseSweepError):
            action = cause.action
        if remediation is None and isinstance(cause, PhaseSweepError):
            remediation = cause.remediation
        return cls(*args, action=action, remediation=remediation)


#: Recovery guidance for both CLI and MCP entry points.
RERUN_CLEANUP_RECOVERY = (
    "run `phasesweep run` again with the same config for a CLI run, or "
    "`phasesweep mcp recover-run` for an MCP run"
)


NEW_STUDY_STEP = (
    "Set a new `experiment` name in the config to start a fresh "
    "study (with `storage: auto`, a new `workdir` also works); the old study stays readable "
    "with its original config."
)


UNPUBLISHED_WINNER_STEP = (
    "If a PhaseSweep operation is still active, wait for it to finish. If the run failed, "
    "follow the failing phase's reported remedy before rerunning the same config to "
    "resume publication."
)


# Path-free action phrases; boundary renderers supply framing and diagnostic details.
OPERATOR_STEPS: dict[OperatorAction, str] = {
    OperatorAction.FRESH_NAMESPACE: "use a fresh namespace for the state identified in the error",
    OperatorAction.RESTORE_LEDGER: "restore or repair the storage ledger and access to it",
    OperatorAction.RESTORE_TREE: (
        "restore the files and permissions identified in the error, deleting a file only if "
        "certain nothing is running"
    ),
    OperatorAction.PROVISION_LOCK_DIR: (
        "point PHASESWEEP_LOCK_DIR at an existing writable directory shared by every shell, "
        "MCP server, and sandbox on the host, or run on the host"
    ),
    OperatorAction.RUN_RECOVER_RUN: RERUN_CLEANUP_RECOVERY,
    OperatorAction.FIX_CONFIG: "correct the experiment configuration or the environment it runs in",
    OperatorAction.RETRY: (
        "Wait briefly, then retry the same operation once the reported blocker is resolved"
    ),
    OperatorAction.INSPECT_LOGS: "inspect the PhaseSweep run log and the logs its error points to",
}


def operator_step(error: PhaseSweepError) -> str:
    """Render the concrete detail of an error's routed operator action.

    :param PhaseSweepError error: Operational failure whose action must be reported.
    :return str: Its concrete instruction, or the shared action's default instruction.
    """
    return error.remediation or f"{OPERATOR_STEPS[error.action]}."


class ProcessCleanupUncertainError(PhaseSweepError):
    """Base class for failures where a subprocess group may still be alive."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RUN_RECOVER_RUN


class UnsafeProcessCleanupError(ProcessCleanupUncertainError):
    """Raised when a trial's trainer or evidence worker may still be alive.

    This must not be included in Optuna's ``catch`` tuple: uncertain cleanup
    aborts the phase and run until recovery confirms the process group is gone.
    """


class ForeignPidNamespaceError(ProcessCleanupUncertainError):
    """A recorded process belongs to another PID namespace.

    Its PID numbers cannot safely identify or signal processes here. Recovery
    from the initial host namespace can prove those recorded processes gone.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


class GpuConfigurationError(PhaseSweepError):
    """Raised when configured GPU isolation cannot be honored safely."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG
