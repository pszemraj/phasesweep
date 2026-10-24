"""Shutdown-signal handling and the live-child registry it drains.

SIGTERM, SIGINT and SIGHUP terminate registered process groups and raise
``PhaseSweepShutdown`` with cleanup evidence. Scopes control handler ownership and
defer or absorb signals during critical sections.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from types import FrameType
from typing import Any

from phasesweep.runtime.reaper import _KILL_GRACE_SECONDS, _terminate_process_groups

# Process supervision logs on one channel, whichever module does the work.
log = logging.getLogger("phasesweep.runtime.process")

_lock = threading.Lock()
_active_children: dict[int, subprocess.Popen[bytes]] = {}  # pgid -> Popen

# Serialize spawning and registration against the handler's child snapshot.
# The handler waits until each in-flight launch registers its PGID or fails.
_launch_lock = threading.Lock()
_shutdown_handler_lock = threading.Lock()
# The package must import anywhere so that
# require_supported_runtime(), not an AttributeError, reports a host without SIGHUP.
_SHUTDOWN_SIGNALS: tuple[int, ...] = tuple(
    signal.Signals[name] for name in ("SIGTERM", "SIGINT", "SIGHUP") if hasattr(signal, name)
)


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


class ShutdownLaunchRefusedError(RuntimeError):
    """Raised by a launch that a shutdown being serviced refused before any process started."""

    def __init__(self, signum: int) -> None:
        """Name the signal that refused the launch.

        :param int signum: The shutdown signal being serviced.
        """
        super().__init__(f"trainer launch refused: orchestrator interrupted by signal {signum}")
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
                log.exception("Failed to wake a waiting thread after signal %d", signum)

        with _launch_lock, _lock:
            guardians = {pgid: proc.pid for pgid, proc in _active_children.items()}
            pgids = tuple(guardians)

        log.warning("Received signal %d — killing %d active child group(s)", signum, len(pgids))

        confirmed_by_pgid: dict[int, bool] = {}
        try:
            confirmed_by_pgid = _terminate_process_groups(
                pgids,
                grace_seconds=_KILL_GRACE_SECONDS,
                guardians=guardians,
            )
        except Exception:
            log.exception("Failed while cleaning child process groups after signal %d", signum)

        cleanup_confirmed = all(confirmed_by_pgid.get(pgid, False) for pgid in pgids)
        report = ShutdownCleanupReport(
            signum=signum,
            cleanup_confirmed=cleanup_confirmed,
            child_pgids=pgids,
        )
        if cleanup_confirmed:
            log.warning(
                "Received signal %d; confirmed cleanup for %d child group(s)",
                signum,
                len(pgids),
            )
        else:
            uncertain = [pgid for pgid in pgids if not confirmed_by_pgid.get(pgid, False)]
            log.error(
                "Received signal %d; cleanup is uncertain for child group(s): %s",
                signum,
                uncertain,
            )

        raise PhaseSweepShutdown(signum, report)
    finally:
        _shutdown_handler_lock.release()


def _unblock_shutdown_signals() -> None:
    """Ensure shutdown signals can reach the main-thread handler."""
    if threading.current_thread() is not threading.main_thread():
        return
    signal.pthread_sigmask(signal.SIG_UNBLOCK, _SHUTDOWN_SIGNALS)


# Main-thread ownership state.
_process_lifetime_owner = False
_scope_depth = 0


def install_signal_handlers() -> None:
    """Take process-lifetime ownership of SIGTERM, SIGINT and SIGHUP.

    Main-thread only. Reinstall missing handlers and unblock shutdown signals on
    each call. An open temporary scope hands its installation over and skips
    restoration; later scopes leave process-lifetime ownership intact.

    :raises SignalOwnershipUnavailableError: Called outside the main thread.
    """
    global _process_lifetime_owner
    if threading.current_thread() is not threading.main_thread():
        raise SignalOwnershipUnavailableError(
            "install_signal_handlers() must be called from the main thread: "
            "signal.signal and process-lifetime shutdown-signal ownership are "
            "main-thread-only."
        )
    _unblock_shutdown_signals()
    if all(signal.getsignal(sig) is _shutdown_handler for sig in _SHUTDOWN_SIGNALS):
        _process_lifetime_owner = True
        return
    for sig in _SHUTDOWN_SIGNALS:
        signal.signal(sig, _shutdown_handler)
    _process_lifetime_owner = True


def _reassert_process_lifetime_handlers() -> None:
    """Repair replaced handlers when process-lifetime ownership is active.

    Only the main thread can repair them; worker calls leave repair to its next scope.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    stale = [sig for sig in _SHUTDOWN_SIGNALS if signal.getsignal(sig) is not _shutdown_handler]
    if not stale:
        return
    log.warning(
        "Shutdown handler(s) for signal(s) %s were replaced after phasesweep "
        "took process-lifetime ownership; reinstalling them for child cleanup.",
        stale,
    )
    for sig in stale:
        signal.signal(sig, _shutdown_handler)
    _unblock_shutdown_signals()


def _restore_host_signal_state(
    prior_handlers: dict[int, Any],
    prior_mask: Collection[int],
) -> Exception | None:
    """Restore the embedding process's shutdown handlers and signal mask.

    Block signals, restore handlers, then restore the mask. Continue past individual
    handler failures so the remaining handlers are still restored.

    :param dict[int, Any] prior_handlers: Handlers captured before the scope
        installed its own, keyed by signal number. A ``None`` value means the
        signal had a C-level default that Python cannot reinstall.
    :param Collection[int] prior_mask: Thread signal mask captured before the
        scope unblocked shutdown signals.
    :return Exception | None: The first restoration failure, or ``None`` when
        every handler was restored.
    """
    signal.pthread_sigmask(signal.SIG_BLOCK, _SHUTDOWN_SIGNALS)
    restoration_error: Exception | None = None
    for sig, handler in prior_handlers.items():
        if handler is None:
            log.warning(
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

    prior_handlers: dict[int, Any] = {sig: signal.getsignal(sig) for sig in _SHUTDOWN_SIGNALS}
    # Read the mask without changing it.
    prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    _scope_depth = 1
    try:
        for sig in _SHUTDOWN_SIGNALS:
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
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _SHUTDOWN_SIGNALS)
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
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _SHUTDOWN_SIGNALS)
    if is_main:
        _main_thread_defer_depth += 1
    try:
        yield absorbed
    finally:
        drained: int | None = None
        if hasattr(signal, "sigtimedwait"):
            # A timed wait cannot block if another thread receives the signal first.
            while True:
                info = signal.sigtimedwait(_SHUTDOWN_SIGNALS, 0)
                if info is None:
                    break
                drained = info.si_signo
        elif is_main:
            # Deliver even an outer scope's masked signals into Python deferral,
            # which stays active until after the original mask is restored.
            signal.pthread_sigmask(signal.SIG_UNBLOCK, _SHUTDOWN_SIGNALS)
        else:
            # Workers cannot dispatch Python handlers. Observe without consuming
            # so restoration still delivers shutdown to the owning main thread.
            pending = signal.sigpending()
            drained = next((sig for sig in _SHUTDOWN_SIGNALS if sig in pending), None)
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


def _register(proc: subprocess.Popen[bytes], *, pgid: int | None = None) -> int:
    """Add a freshly-launched subprocess to the global child registry.

    :param subprocess.Popen[bytes] proc: Newly launched subprocess.
    :param int | None pgid: Explicit child process group guarded by ``proc``. When omitted,
        register ``proc``'s own process group.
    :return int: Registered process-group ID for removal and signal targeting.
    """
    resolved_pgid = os.getpgid(proc.pid) if pgid is None else pgid
    with _lock:
        _active_children[resolved_pgid] = proc
    return resolved_pgid


def _unregister(pgid: int) -> None:
    """Remove a finished process group from the global child registry.

    Defer shutdown while holding ``_lock`` to prevent handler re-entry deadlock.

    :param int pgid: The process-group ID returned by :func:`_register`. A pgid not currently
        registered is silently ignored.
    """
    with defer_shutdown_signals(), _lock:
        _active_children.pop(pgid, None)
