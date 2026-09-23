"""Shared operator-facing error taxonomy."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Self


class OperatorAction(StrEnum):
    """One remediation step an operator should take for a failure.

    The message stays authoritative about *what* went wrong; this says *what to
    do next*, so callers can route a failure without parsing prose.

    ``USE_PRIOR_RELEASE``
        Operate the existing state under the preserved pre-cutover release.
    ``FRESH_NAMESPACE``
        Start again under a new experiment name, artifact root, or local
        storage instead of reusing the bound one.
    ``RESTORE_LEDGER``
        Restore or repair the persistent study ledger before running again.
    ``RESTORE_TREE``
        Restore or repair the on-disk experiment tree - artifacts, evidence, or
        its permissions - before running again.
    ``RUN_RECOVER_RUN``
        Run the operator recovery command before anything else touches the run.
    ``FIX_CONFIG``
        Correct the experiment configuration or the request, then run again.
    ``RETRY``
        Wait for the current holder to finish and repeat the same request.
    ``INSPECT_LOGS``
        Read the message and the surrounding logs; no single mechanical remedy
        applies.

    There is deliberately no ``REPORT_BUG`` member. A PhaseSweep defect must not
    inherit a remediation, and defects do not subclass :class:`PhaseSweepError`.
    """

    USE_PRIOR_RELEASE = "use_prior_release"
    FRESH_NAMESPACE = "fresh_namespace"
    RESTORE_LEDGER = "restore_ledger"
    RESTORE_TREE = "restore_tree"
    RUN_RECOVER_RUN = "run_recover_run"
    FIX_CONFIG = "fix_config"
    RETRY = "retry"
    INSPECT_LOGS = "inspect_logs"


class PhaseSweepError(RuntimeError):
    """Base for expected operational failures an operator is meant to act on.

    Every subclass carries a complete diagnostic, so the CLI reports it as one
    line without an internal-bug traceback. Exceptions that indicate a
    PhaseSweep defect must not inherit from this class.

    Each instance also carries its remediation as :attr:`actions`, a routing
    attribute only: it never changes the message text an operator reads. It
    holds one :class:`OperatorAction` unless the operator must do several
    things, every one of them, in the order listed. A message that offers
    alternatives ("restore the setting, or start fresh") routes the one that
    keeps the operator's existing work, and retrying what failed is implied,
    never listed. A raise
    that lists several steps must appear in ``tests/test_error_routing.py``'s
    allowlist, so a second step is always a reviewed decision.
    """

    # The base declares a fallback solely because the base class is itself
    # raised directly at a few sites. It is not an inheritable default: every
    # concrete subclass declares its own, and tests/test_error_routing.py fails
    # when one silently inherits this one. A class default is one step by type;
    # only a single raise may require several.
    default_action: ClassVar[OperatorAction] = OperatorAction.INSPECT_LOGS

    def __init__(
        self,
        *args: object,
        action: OperatorAction | tuple[OperatorAction, ...] | None = None,
    ) -> None:
        """Create an operational failure carrying the remediation it calls for.

        :param object args: Standard exception arguments; the first is the message.
        :param OperatorAction | tuple[OperatorAction, ...] | None action:
            Remediation for this one raise, overriding the class's
            :attr:`default_action`: one step, or every required step in order.
        """
        super().__init__(*args)
        if action is None:
            action = type(self).default_action
        self.actions: tuple[OperatorAction, ...] = (
            (action,) if isinstance(action, OperatorAction) else action
        )

    @classmethod
    def rewrap(
        cls,
        cause: BaseException,
        *args: object,
        action: OperatorAction | tuple[OperatorAction, ...] | None = None,
    ) -> Self:
        """Build this error from ``cause``, inheriting the remediation ``cause`` carried.

        Returns the new instance rather than raising it, so the caller still
        writes ``raise X.rewrap(exc, msg) from exc`` and the explicit ``from``
        clause that preserves ``__cause__`` stays visible at the raise site.

        An explicit ``action`` wins. Otherwise a :class:`PhaseSweepError` cause
        donates every step it carried, so a remediation survives translation
        between layers, and any other cause falls back to :attr:`default_action`.

        :param BaseException cause: Failure being rewrapped.
        :param object args: Arguments for the new error; the first is the message.
        :param OperatorAction | tuple[OperatorAction, ...] | None action:
            Remediation for this one raise.
        :return Self: The new error, for the caller to raise ``from cause``.
        """
        if action is None and isinstance(cause, PhaseSweepError):
            action = cause.actions
        return cls(*args, action=action)


class ProcessCleanupUncertainError(PhaseSweepError):
    """Base class for failures where a subprocess group may still be alive."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RUN_RECOVER_RUN


class UnsafeProcessCleanupError(ProcessCleanupUncertainError):
    """Raised when a trial's trainer or evidence worker may still be alive.

    This must not be included in Optuna's ``catch`` tuple: uncertain cleanup
    aborts the phase and run until recovery confirms the process group is gone.
    """


class LockBusyError(PhaseSweepError):
    """Raised when a required same-host lock is already held."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RETRY


class GpuConfigurationError(PhaseSweepError):
    """Raised when configured GPU isolation cannot be honored safely."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG
