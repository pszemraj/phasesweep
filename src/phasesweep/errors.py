"""Shared operator-facing error taxonomy."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import ClassVar, Self


class OperatorAction(StrEnum):
    """Operator remediation category for routing expected failures.

    FRESH_NAMESPACE starts a fresh study under a new ``experiment`` name;
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


_FRESH_STUDY_TAIL = (
    " a new `experiment` name in the config to start a fresh study "
    "(a new `workdir` also works unless `storage` names a ledger)"
)

#: The one way to start over, worded as a clause for a step that offers it second.
FRESH_STUDY = f"set{_FRESH_STUDY_TAIL}"

NEW_STUDY_STEP = f"Set{_FRESH_STUDY_TAIL}; the old study stays readable with its original config."


def rename_phase_step(dependents: Sequence[str] | None) -> str:
    """Say how to start one phase over when a setting of that phase changes.

    A phase's study is named after the phase, so a new name is a new study for it, and a phase
    that inherits it records the name in its own study identity and follows. The clause is
    worded for a step that offers it, so the caller adds what the rename costs.

    :param Sequence[str] | None dependents: The phases that inherit the phase, from
        :meth:`~phasesweep.config.models.Experiment.phases_inheriting`, or ``None`` where no
        experiment is at hand and the sentence can only say there may be some.
    :return str: A clause that gives the phase, and the dependents it names, a new ``name``.
    """
    if dependents is not None and not dependents:
        return "give the phase a new `name`"
    named = "" if dependents is None else f" ({', '.join(f'`{name}`' for name in dependents)})"
    return (
        f"give the phase, and each phase that inherits it{named}, a new `name`, updating "
        "every `inherits` to match"
    )


def timeout_step(phase_name: str, dependents: Sequence[str]) -> str:
    """Say how to give a phase whose trials timed out a longer per-trial timeout.

    The timeout is part of the phase's study identity, so a raised value is refused over the
    recorded trials; a new phase name is the step that loses the least work.

    :param str phase_name: Phase whose trials timed out.
    :param Sequence[str] dependents: The phases that inherit it.
    :return str: The operator's step, a sentence pair.
    """
    return (
        f"A trial timed out under phases[{phase_name}].timeout_seconds_per_trial, which is part "
        "of the phase's study identity, so raising it in place is refused as a changed study. "
        f"Raise it and {rename_phase_step(dependents)}; each renamed phase starts a new study "
        "and runs again, and every other phase keeps its results."
    )


#: The way out of an artifact root or ledger another PhaseSweep version wrote, on every
#: command and under every `storage`. A ledger the config names holds every experiment's
#: studies: a new `experiment` name leaves the unsupported study in it, and a new ledger alone
#: leaves the artifact root bound to the old one, so there both change. "Version" covers a
#: source checkout between releases, which no release can stand in for.
OTHER_FORMAT_STEP = (
    f"Set{_FRESH_STUDY_TAIL}, and a new `storage` ledger too if the config names one. "
    "The PhaseSweep version that wrote the old data still reads it."
)


def format_found(version: object) -> str:
    """Word the format version a record carries.

    :param object version: The version the record holds, or ``None`` when it holds none.
    :return str: ``uses format 14``, or ``has no format version`` for ``None``.
    """
    return "has no format version" if version is None else f"uses format {version!r}"


def format_reads(current: int) -> str:
    """Word the format version this PhaseSweep reads.

    :param int current: The format version this PhaseSweep reads.
    :return str: ``this PhaseSweep reads format 15``.
    """
    return f"this PhaseSweep reads format {current}"


def format_mismatch(version: object, current: int) -> str:
    """Word how a record's format differs from the one this PhaseSweep reads.

    :param object version: The version the record holds, or ``None`` when it holds none.
    :param int current: The format version this PhaseSweep reads.
    :return str: A clause for the record's subject to lead, such as
        ``uses format 14; this PhaseSweep reads format 15``.
    """
    return f"{format_found(version)}; {format_reads(current)}"


# Path-free action phrases; boundary renderers supply framing and diagnostic details.
OPERATOR_STEPS: dict[OperatorAction, str] = {
    OperatorAction.FRESH_NAMESPACE: FRESH_STUDY,
    OperatorAction.RESTORE_LEDGER: "restore or repair the storage ledger and access to it",
    OperatorAction.RESTORE_TREE: "restore the files and permissions identified in the error",
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
