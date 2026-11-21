"""Shutdown-signal handling and the live-child registry it drains.

SIGTERM, SIGINT and SIGHUP terminate registered process groups and raise
``PhaseSweepShutdown`` with cleanup evidence; a SIGHUP ignored when ownership is first
taken, as under ``nohup``, stays ignored. Scopes control handler ownership and defer or
absorb signals during critical sections.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from types import FrameType

from phasesweep.runtime.reaper import _KILL_GRACE_SECONDS, _terminate_process_groups
from phasesweep.runtime.terminal import REPORT_LOGGER, ReportText, plural

# Process supervision logs on one channel, whichever module does the work.
log = logging.getLogger("phasesweep.runtime.process")
# The one line a person reads when a signal arrives; everything else stays in run.log.
console = logging.getLogger(REPORT_LOGGER)

#: What the interrupt line calls a registered trainer; every other label names a worker.
TRAINER_LABEL = "trial"


@dataclass(frozen=True)
class _RegisteredGroup:
    """A live process group the shutdown handler stops, and what the interrupt line calls it."""

    guardian: subprocess.Popen[bytes]
    label: str


_lock = threading.Lock()
_active_children: dict[int, _RegisteredGroup] = {}  # pgid -> its guardian and label

# Serialize spawning and registration against the handler's child snapshot.
# The handler waits until each in-flight launch registers its PGID or fails.
_launch_lock = threading.Lock()
_shutdown_handler_lock = threading.Lock()
# The package must import anywhere so that
# require_supported_runtime(), not an AttributeError, reports a host without SIGHUP.
# Code that installs, masks or drains these goes through _owned_shutdown_signals().
_SHUTDOWN_SIGNALS: tuple[int, ...] = tuple(
    signal.Signals[name] for name in ("SIGTERM", "SIGINT", "SIGHUP") if hasattr(signal, name)
)
_SIGHUP = signal.Signals.__members__.get("SIGHUP")
# Fixed by the first ownership in this process.
_owned_signals: tuple[int, ...] | None = None

# What signal.getsignal returns and signal.signal accepts.
_SignalHandler = Callable[[int, FrameType | None], object] | int | signal.Handlers | None


def _owned_shutdown_signals(*, taking_ownership: bool = False) -> tuple[int, ...]:
    """Return the shutdown signals phasesweep owns in this process.

    The first ownership fixes the set for the life of the process. A SIGHUP ignored at
    that moment is left out: ``nohup`` starts a program that way so it outlives a closed
    terminal, and a program leaves an inherited ignored SIGHUP alone. PhaseSweep then never
    installs, unblocks, masks, drains, reinstalls or restores it. SIGTERM and SIGINT are
    always owned. Before the first ownership, the set follows the current disposition.

    :param bool taking_ownership: The caller is about to install shutdown handlers, which
        fixes the set unless an earlier ownership already did.
    :return tuple[int, ...]: The owned signals, in ``_SHUTDOWN_SIGNALS`` order.
    """
    global _owned_signals
    if _owned_signals is not None:
        return _owned_signals
    owned = tuple(
        sig
        for sig in _SHUTDOWN_SIGNALS
        if sig != _SIGHUP or signal.getsignal(sig) is not signal.SIG_IGN
    )
    if taking_ownership:
        _owned_signals = owned
    return owned


@dataclass(frozen=True)
class ShutdownCleanupReport:
    """Cleanup evidence captured when the orchestrator handles a shutdown signal."""

    signum: int
    cleanup_confirmed: bool
    child_pgids: tuple[int, ...]


class SignalOwnershipUnavailableError(RuntimeError):
    """Raised when shutdown-signal ownership is needed but cannot be taken here."""


class PhaseSweepShutdown(SystemExit):
    """SystemExit carrying cleanup evidence and post-publication state."""

    def __init__(self, signum: int, report: ShutdownCleanupReport) -> None:
        """Create a POSIX-style signaled exit with structured cleanup evidence.

        :param int signum: Shutdown signal that triggered the exit; the exit code is
            ``128 + signum`` per the POSIX signaled-exit convention.
        :param ShutdownCleanupReport report: Cleanup evidence captured by the shutdown handler
            for the child process groups it terminated.
        """
        super().__init__(128 + signum)
        self.signum = signum
        self.report = report
        # Set by the engine's post-publication shutdown checkpoint.
        self.published_result_committed = False


def shutdown_raised_over(exc: BaseException) -> PhaseSweepShutdown | None:
    """Return the shutdown ``exc`` is, or was raised while handling or from.

    The handler raises in the main thread at whatever bytecode is running, so a library's
    cleanup code can raise over it; the run is still stopping.

    :param BaseException exc: An exception leaving the code a shutdown may interrupt.
    :return PhaseSweepShutdown | None: The first shutdown in its cause and context chains,
        or ``None`` when they hold none.
    """
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, PhaseSweepShutdown):
            return current
        if id(current) not in seen:
            seen.add(id(current))
            pending.extend(
                link for link in (current.__cause__, current.__context__) if link is not None
            )
    return None


# Kernel masks are per-thread; CPython dispatches all Python handlers on the
# main thread, including signals delivered elsewhere. Defer there to avoid
# re-entering launch/registry locks; the outermost exit services the signal.
# These variables are mutated only on the main thread.
_deferred_shutdown_signum: int | None = None
_main_thread_defer_depth = 0

# Set before the child snapshot lock: workers record interruption and refuse
# launches, or register a child the snapshot includes. Reset at the next run.
_servicing_signum: int | None = None

# Wake worker waits after ``_servicing_signum`` is set.
_shutdown_wakers: list[Callable[[], None]] = []


def _signal_name(signum: int) -> str:
    """Name a shutdown signal as the interrupt line and the recorded cause both show it.

    :param int signum: Signal number.
    :return str: Its name, such as ``SIGINT``, or ``signal N`` for a number with none.
    """
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def interruption_cause(signum: int) -> str:
    """Return the cause text recorded for a trial the orchestrator's shutdown signal stopped.

    :param int signum: The shutdown signal being serviced.
    :return str: The trial outcome cause for that signal, such as ``orchestrator interrupted by SIGINT``.
    """
    return f"orchestrator interrupted by {_signal_name(signum)}"


def launch_refusal_text(signum: int) -> str:
    """Return the message of a trainer launch that a shutdown being serviced refused.

    :param int signum: The shutdown signal being serviced.
    :return str: The refusal message, ending in the interruption cause for that signal.
    """
    return f"trainer launch refused: {interruption_cause(signum)}"


class ShutdownLaunchRefusedError(RuntimeError):
    """Raised by a launch that a shutdown being serviced refused before any process started."""

    def __init__(self, signum: int) -> None:
        """Name the signal that refused the launch.

        :param int signum: The shutdown signal being serviced.
        """
        super().__init__(launch_refusal_text(signum))
        self.signum = signum


def servicing_shutdown_signal() -> int | None:
    """Return the shutdown signal the handler is servicing.

    :return int | None: The signal number, or ``None`` when no shutdown is under way.
    """
    return _servicing_signum


def register_shutdown_waker(waker: Callable[[], None]) -> None:
    """Have the shutdown handler call ``waker`` once a shutdown is being serviced.

    A waker runs in the main thread inside the signal handler, so it must only
    notify waiting threads: no blocking, no I/O.

    :param Callable[[], None] waker: Callable that wakes threads blocked in a wait.
    """
    _shutdown_wakers.append(waker)


def _interrupt_text(signum: int, running: Sequence[str]) -> str:
    """Say in plain words that a shutdown signal arrived and what it stops.

    :param int signum: Shutdown signal number.
    :param Sequence[str] running: The label of each process group the signal will stop, in
        registration order; :data:`TRAINER_LABEL` marks a trainer.
    :return str: For example ``Interrupted (SIGINT): stopping 2 trials and waiting for them to exit.``
    """
    name = _signal_name(signum)
    counts = Counter(running)
    trials = counts.pop(TRAINER_LABEL, 0)
    others = [plural(count, label) for label, count in counts.items()]
    if not trials:
        stopping = f"; stopping {' and '.join(others)}" if others else ""
        # "trial" keeps the commonest worker-only line, one MLflow request, within 80 columns
        # behind the report's time prefix.
        return f"Interrupted ({name}): no trial is running{stopping}."
    # A trainer that ignores SIGTERM keeps the run silent until its grace period ends, so the
    # line says the run is waiting.
    pronoun = "it" if len(running) == 1 else "them"
    stopping = " and ".join([plural(trials, TRAINER_LABEL), *others])
    return f"Interrupted ({name}): stopping {stopping} and waiting for {pronoun} to exit."


def _shutdown_handler(signum: int, _frame: FrameType | None) -> None:
    """Terminate all tracked process groups, then exit.

    One guardian snapshot under the launch lock requests cancellation; each guardian
    records its intent and owns trainer termination while this handler confirms cleanup.
    Never call ``proc.wait()`` here: its interrupted waitpid lock is non-reentrant.
    A main-thread critical section records the signal for its exit checkpoint;
    a concurrent handler invocation returns without interrupting cleanup.

    :param int signum: Shutdown signal number.
    :param FrameType | None _frame: Unused frame required by the signal handler protocol.
    :raises PhaseSweepShutdown: Exit code ``128 + signum``, unless deferred or cleanup is
        already running.
    """
    global _deferred_shutdown_signum, _servicing_signum

    # A second signal must not re-enter registry locks or replace the first
    # handler's cleanup evidence.
    if not _shutdown_handler_lock.acquire(blocking=False):
        return

    try:
        if _main_thread_defer_depth > 0:
            # Avoid reacquiring locks held by the interrupted critical section.
            _deferred_shutdown_signum = signum
            return
        # Any recorded-but-unserviced signal is superseded by this invocation.
        _deferred_shutdown_signum = None
        _servicing_signum = signum
        for waker in _shutdown_wakers:
            try:
                waker()
            except Exception:
                log.exception("Failed to wake a waiting thread after %s", _signal_name(signum))

        with _launch_lock, _lock:
            registered = dict(_active_children)
        guardians = {pgid: group.guardian for pgid, group in registered.items()}
        pgids = tuple(registered)

        console.info(
            ReportText(_interrupt_text(signum, [group.label for group in registered.values()]))
        )

        confirmed_by_pgid: dict[int, bool] = {}
        try:
            confirmed_by_pgid = _terminate_process_groups(
                pgids,
                grace_seconds=_KILL_GRACE_SECONDS,
                guardians=guardians,
            )
        except Exception:
            log.exception(
                "Failed while cleaning child process groups after %s", _signal_name(signum)
            )

        cleanup_confirmed = all(confirmed_by_pgid.get(pgid, False) for pgid in pgids)
        report = ShutdownCleanupReport(
            signum=signum,
            cleanup_confirmed=cleanup_confirmed,
            child_pgids=pgids,
        )
        if cleanup_confirmed:
            # The expected outcome of a shutdown, so the console says nothing more about it.
            log.info(
                "Received %s; confirmed cleanup for %s",
                _signal_name(signum),
                plural(len(pgids), "child group"),
            )
        else:
            uncertain = [pgid for pgid in pgids if not confirmed_by_pgid.get(pgid, False)]
            log.error(
                "Received %s; cleanup is uncertain for %s: %s",
                _signal_name(signum),
                plural(len(uncertain), "child group"),
                uncertain,
            )

        raise PhaseSweepShutdown(signum, report)
    finally:
        _shutdown_handler_lock.release()


def _unblock_shutdown_signals() -> None:
    """Ensure the owned shutdown signals can reach the main-thread handler."""
    if threading.current_thread() is not threading.main_thread():
        return
    signal.pthread_sigmask(signal.SIG_UNBLOCK, _owned_shutdown_signals())


# Main-thread ownership state.
_process_lifetime_owner = False
_scope_depth = 0


def install_signal_handlers() -> None:
    """Take process-lifetime ownership of the shutdown signals.

    Those are SIGTERM, SIGINT and SIGHUP, less a SIGHUP ignored when ownership is first
    taken (:func:`_owned_shutdown_signals`). Main-thread only. Reinstall missing handlers
    before unblocking shutdown signals on each call. An open temporary scope hands its
    installation over and skips restoration; later scopes leave process-lifetime ownership
    intact.

    :raises SignalOwnershipUnavailableError: Called outside the main thread.
    """
    global _process_lifetime_owner
    if threading.current_thread() is not threading.main_thread():
        raise SignalOwnershipUnavailableError(
            "install_signal_handlers() must be called from the main thread: "
            "signal.signal and process-lifetime shutdown-signal ownership are "
            "main-thread-only."
        )
    owned = _owned_shutdown_signals(taking_ownership=True)
    if not all(signal.getsignal(sig) is _shutdown_handler for sig in owned):
        for sig in owned:
            signal.signal(sig, _shutdown_handler)
    _process_lifetime_owner = True
    _unblock_shutdown_signals()


def _reassert_process_lifetime_handlers() -> None:
    """Repair replaced handlers when process-lifetime ownership is active.

    Only the main thread can repair them; worker calls leave repair to its next scope.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    stale = [
        sig for sig in _owned_shutdown_signals() if signal.getsignal(sig) is not _shutdown_handler
    ]
    if not stale:
        return
    log.warning(
        "Shutdown handler(s) for signal(s) %s were replaced after phasesweep "
        "took process-lifetime ownership; reinstalling them for child cleanup.",
        stale,
    )
    install_signal_handlers()


def _restore_host_signal_state(
    prior_handlers: dict[int, _SignalHandler],
    prior_mask: Collection[int],
) -> Exception | None:
    """Restore the embedding process's shutdown handlers and signal mask.

    Block signals, restore handlers, then restore the mask. Continue past individual
    handler failures so the remaining handlers are still restored.

    :param dict[int, _SignalHandler] prior_handlers: Handlers of the owned signals,
        captured before the scope installed its own, keyed by signal number. A ``None``
        value means the signal had a C-level default that Python cannot reinstall.
    :param Collection[int] prior_mask: Thread signal mask captured before the
        scope unblocked shutdown signals.
    :return Exception | None: The first restoration failure, or ``None`` when
        every handler was restored.
    """
    signal.pthread_sigmask(signal.SIG_BLOCK, _owned_shutdown_signals())
    restoration_error: Exception | None = None
    for sig, handler in prior_handlers.items():
        if handler is None:
            log.debug(
                "Signal %d had no Python-level handler before this scope "
                "(a C-level default phasesweep cannot reinstall); leaving "
                "phasesweep's shutdown handler installed for it.",
                sig,
            )
            continue
        try:
            signal.signal(sig, handler)
        except Exception as exc:  # continue past one bad restore
            log.exception("Failed to restore the prior handler for signal %d after scope exit", sig)
            if restoration_error is None:
                restoration_error = exc
    signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
    return restoration_error


@contextlib.contextmanager
def signal_handler_scope() -> Iterator[None]:
    """Temporarily own shutdown signals for one outermost main-thread run.

    Nested calls share ownership. Process-lifetime ownership skips installation and
    restoration, including when claimed inside the scope. Worker calls require that
    ownership. Restoration reports its first error only when the body did not fail.

    :raises SignalOwnershipUnavailableError: Called from a non-main thread when no
        :func:`install_signal_handlers` call owns the shutdown signals for the process
        lifetime; a scope open on the main thread does not count.
    :return Iterator[None]: Context for the run whose children require shutdown cleanup.
    """
    global _scope_depth, _servicing_signum

    if _scope_depth == 0:
        # A new outermost run starts with no shutdown under way.
        _servicing_signum = None
    if _process_lifetime_owner:
        _reassert_process_lifetime_handlers()
        yield
        return

    if threading.current_thread() is not threading.main_thread():
        raise SignalOwnershipUnavailableError(
            "Shutdown-signal handlers are not installed on this process, and "
            "this thread is not the main thread, so they cannot be installed "
            "here (signal.signal only works on the main thread). Call "
            "install_signal_handlers() from the main thread at process "
            "start, or run this from the main thread."
        )

    if _scope_depth > 0:
        _scope_depth += 1
        try:
            yield
        finally:
            _scope_depth -= 1
        return

    owned = _owned_shutdown_signals(taking_ownership=True)
    prior_handlers: dict[int, _SignalHandler] = {sig: signal.getsignal(sig) for sig in owned}
    # Read the mask without changing it.
    prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    _scope_depth = 1
    try:
        for sig in owned:
            signal.signal(sig, _shutdown_handler)
        _unblock_shutdown_signals()
        yield
    finally:
        _scope_depth = 0
        restoration_error: Exception | None = None
        if not _process_lifetime_owner:
            restoration_error = _restore_host_signal_state(prior_handlers, prior_mask)
        if restoration_error is not None:
            if sys.exc_info()[1] is None:
                raise restoration_error
            log.error(
                "Signal handler restoration failed while another exception was already "
                "propagating out of this scope; not overriding it: %s",
                restoration_error,
            )


@contextlib.contextmanager
def defer_shutdown_signals() -> Iterator[None]:
    """Defer shutdown handling until the outermost critical section exits.

    Kernel masking blocks this thread's signals; Python-level main-thread deferral
    also covers signals delivered to other threads. Worker launchers instead make
    the handler wait on the launch lock.

    :return Iterator[None]: Context for the critical section.
    """
    global _main_thread_defer_depth
    is_main = threading.current_thread() is threading.main_thread()
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _owned_shutdown_signals())
    if is_main:
        _main_thread_defer_depth += 1
    try:
        yield
    finally:
        if is_main:
            _main_thread_defer_depth -= 1
        # Mask restoration delivers queued signals after deferral ends.
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        if is_main and _main_thread_defer_depth == 0 and _deferred_shutdown_signum is not None:
            service_pending_shutdown()


def service_pending_shutdown() -> None:
    """Deliver a deferred shutdown signal before starting new work.

    Does nothing when no signal is pending.

    :raises PhaseSweepShutdown: Via ``_shutdown_handler``, carrying the cleanup evidence for
        the recorded signal, unless a main-thread critical section is still open (the signal
        is recorded again) or another handler is already running (the call returns).
    """
    global _deferred_shutdown_signum
    signum = _deferred_shutdown_signum
    _deferred_shutdown_signum = None
    if signum is not None:
        _shutdown_handler(signum, None)


@dataclass
class AbsorbedShutdown:
    """Outcome of one :func:`absorb_shutdown_signals` window."""

    signum: int | None = None


@contextlib.contextmanager
def absorb_shutdown_signals() -> Iterator[AbsorbedShutdown]:
    """Complete a one-way commit before servicing shutdown.

    Signals arriving inside the window are reported on the yielded object and
    retained for a later shutdown checkpoint. Signals arriving before entry raise
    normally. Kernel-queued signals are drained while blocked where timed waits
    exist, or delivered into main-thread deferral before restoring the original mask.

    :return Iterator[AbsorbedShutdown]: Window outcome whose ``signum`` is the absorbed
        shutdown signal, or ``None`` when no shutdown arrived during the window; it is set on
        window exit.
    """
    global _main_thread_defer_depth
    global _deferred_shutdown_signum
    absorbed = AbsorbedShutdown()
    is_main = threading.current_thread() is threading.main_thread()
    # One set for the whole window: a SIGHUP left ignored is neither blocked nor drained.
    owned = _owned_shutdown_signals()
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, owned)
    if is_main:
        _main_thread_defer_depth += 1
    try:
        yield absorbed
    finally:
        drained: int | None = None
        if hasattr(signal, "sigtimedwait"):
            # A timed wait cannot block if another thread receives the signal first.
            while True:
                info = signal.sigtimedwait(owned, 0)
                if info is None:
                    break
                drained = info.si_signo
        elif is_main:
            # Deliver even an outer scope's masked signals into Python deferral,
            # which stays active until after the original mask is restored.
            signal.pthread_sigmask(signal.SIG_UNBLOCK, owned)
        else:
            # Workers cannot dispatch Python handlers. Observe without consuming
            # so restoration still delivers shutdown to the owning main thread.
            pending = signal.sigpending()
            drained = next((sig for sig in owned if sig in pending), None)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        if is_main:
            _main_thread_defer_depth -= 1
            recorded = _deferred_shutdown_signum
            absorbed.signum = recorded if recorded is not None else drained
            if recorded is None and drained is not None:
                # Keep the drained signal pending for the next checkpoint.
                _deferred_shutdown_signum = drained
        else:
            absorbed.signum = drained


def _register(proc: subprocess.Popen[bytes], *, label: str, pgid: int) -> int:
    """Add a freshly-launched subprocess to the global child registry.

    :param subprocess.Popen[bytes] proc: Newly launched guardian.
    :param str label: What the interrupt line calls this group, counted with others of its
        label: :data:`TRAINER_LABEL` for a trainer, a noun such as ``MLflow request`` for a worker.
    :param int pgid: The child process group ``proc`` guards, which is not ``proc``'s own.
    :return int: Registered process-group ID for removal and signal targeting.
    """
    with _lock:
        _active_children[pgid] = _RegisteredGroup(proc, label)
    return pgid


def _unregister(pgid: int) -> None:
    """Remove a finished process group from the global child registry.

    Defer shutdown while holding ``_lock`` to prevent handler re-entry deadlock.

    :param int pgid: The process-group ID returned by :func:`_register`. A pgid not currently
        registered is silently ignored.
    """
    with defer_shutdown_signals(), _lock:
        _active_children.pop(pgid, None)
