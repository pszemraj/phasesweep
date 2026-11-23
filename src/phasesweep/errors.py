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
    RETRY repeats the operation after its blocker resolves, with no change to the command or
    config, so the CLI exits 75 for it and a process supervisor can rerun the same command;
    an error that needs an operator step first never carries it. INSPECT_LOGS requires diagnosis.
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


#: The one way to start over, worded as a clause for a step that offers it second. A new
#: `workdir` and what happens to the old study are said once, in docs/runtime.md ("Storage and
#: locks"). It names the key, not the file, since a `--set` override of it wins over the file.
FRESH_STUDY = "set a new `experiment` name to start a fresh study"

NEW_STUDY_STEP = f"{FRESH_STUDY[0].upper()}{FRESH_STUDY[1:]}."


def restore_or_new_study(recorded: str) -> str:
    """Offer restoring what a refusal found changed, or starting a fresh study.

    :param str recorded: What to restore, such as ``the recorded value(s)``.
    :return str: The step.
    """
    return f"Restore {recorded}, or {FRESH_STUDY}."


#: The fact about a rerun under `storage: null`, as a clause for each step that offers one to
#: compose with its own lead-in, so a restart is worded the same wherever it is offered.
STORAGE_NULL_KEEPS_NOTHING = "storage: null keeps no completed trials between runs"

#: The step after a run without storage stopped short of publishing, which every surface that
#: reports that state gives.
START_AGAIN_STEP = f"Rerun the same config to start the run again; {STORAGE_NULL_KEEPS_NOTHING}."

#: The same step after a failure-streak abort, which a rerun repeats until the cause is fixed,
#: as the step of a persistent study's abort leads with.
ABORT_START_AGAIN_STEP = f"Fix the cause, then {START_AGAIN_STEP[0].lower()}{START_AGAIN_STEP[1:]}"

#: The fact about a rerun under a persistent study, the counterpart of the clause above.
PERSISTENT_STUDY_KEEPS_TRIALS = "completed trials are kept"


def phase_names(names: Sequence[str]) -> str:
    """List phase names, each in backticks, the way a message that lists phases does.

    :param Sequence[str] names: Phase names, in the order to list them.
    :return str: Each name in backticks, separated by commas.
    """
    return ", ".join(f"`{name}`" for name in names)


def rename_step(names: Sequence[str]) -> str:
    """Say how to start phases over under new names, which give them new studies.

    A phase's study is named after the phase, so a new name is a new study for it, and a phase
    that inherits a renamed one records that name in its own study identity, so the caller
    passes every inheriting phase too. Only the ``inherits`` entries that name a renamed phase
    change. A new study keeps none of the old one's trials, so the clause says the renamed
    phases run again.

    :param Sequence[str] names: The phases to rename, at least one: a phase followed by the
        phases inheriting it (:meth:`~phasesweep.config.models.Experiment.phases_inheriting`),
        or the inheriting phases alone.
    :return str: A clause for a step to lead into, such as ``give `a` and `b` new names and
        update the `inherits` that name them, which runs them again from scratch``.
    """
    quoted = [f"`{name}`" for name in names]
    if len(quoted) == 1:
        return f"give {quoted[0]} a new `name`, which runs it again from scratch"
    listed = f"{', '.join(quoted[:-1])} and {quoted[-1]}"
    return (
        f"give {listed} new names and update the `inherits` that name them, which runs them "
        "again from scratch"
    )


def timeout_step(phase_name: str, dependents: Sequence[str]) -> str:
    """Say how to give a phase whose trials timed out a longer per-trial timeout.

    The timeout is part of the phase's study identity, so a raised value is refused over the
    recorded trials; a new phase name is the step that loses the least work.

    :param str phase_name: Phase whose trials timed out.
    :param Sequence[str] dependents: The phases that inherit it.
    :return str: The operator's step, one sentence.
    """
    return (
        f"For the timed-out trials, raise phases[{phase_name}].timeout_seconds_per_trial and "
        f"{rename_step([phase_name, *dependents])}."
    )


def unreadable_objective_step(*, persistent: bool) -> str:
    """Say what to fix when a trainer ran but its objective could not be read.

    The cause is the trainer's output or the config's ``metric`` settings. The settings are
    part of a persistent study's identity, so correcting them is refused over the recorded
    trials and starts a fresh study instead; a fixed trainer output retries in place, by the
    retry step that follows this sentence in a persistent study.

    :param bool persistent: Whether the study is recorded, which decides whether a corrected
        ``metric`` can continue it.
    :return str: One sentence that leads the retry step.
    """
    if not persistent:
        return (
            "A trainer ran but its objective could not be read: fix the trainer's output or the "
            "`metric` settings, then rerun."
        )
    return (
        "A trainer ran but its objective could not be read; if the `metric` settings are wrong, "
        "correct them under a new `experiment` name."
    )


#: The way out of an artifact root or ledger another PhaseSweep version wrote, on every
#: command and under every `storage`. A ledger the config names holds every experiment's
#: studies: a new `experiment` name leaves the unsupported study in it, and a new ledger alone
#: leaves the artifact root bound to the old one, so there both change.
OTHER_FORMAT_STEP = "Set a new `experiment` name, and a new ledger if `storage` names one."


#: The step for a recorded result that stopped short at a timeout, on every surface that
#: refuses to use it while ``allow_incomplete_on_timeout`` is off.
ACCEPT_INCOMPLETE_STEP = "Set allow_incomplete_on_timeout: true to use the incomplete result."


def as_sentence(text: object) -> str:
    """End recorded text as a sentence, so the text after it starts a new one.

    :param object text: A recorded cause or diagnostic, which an exception's text may leave
        unpunctuated.
    :return str: ``text`` without trailing whitespace, with a period added unless it already
        ends in sentence punctuation.
    """
    ended = str(text).rstrip()
    return ended if ended.endswith((".", "!", "?")) else f"{ended}."


def short_digest(digest: str) -> str:
    """Shorten a SHA-256 digest the way every message that prints one does.

    :param str digest: Hex digest.
    :return str: Its first 12 characters followed by ``...``.
    """
    return f"{digest[:12]}..."


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
    OperatorAction.RETRY: "Retry once the reported blocker clears",
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
