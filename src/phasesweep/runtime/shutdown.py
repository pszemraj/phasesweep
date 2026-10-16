"""Shutdown-signal handling and the live-child registry it drains.

A SIGTERM, SIGINT, or SIGHUP runs ``_shutdown_handler``, which terminates every
registered process group and raises ``PhaseSweepShutdown`` carrying the
cleanup evidence. The rest of this module decides who owns those handlers
(``install_signal_handlers``, ``signal_handler_scope``) and what happens to a
signal that arrives inside a critical section (``defer_shutdown_signals``,
``absorb_shutdown_signals``).
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
_SHUTDOWN_SIGNALS: tuple[int, ...] = tuple(
    sig
    for sig in (
        signal.SIGTERM,
        signal.SIGINT,
        getattr(signal, "SIGHUP", None),
    )
    if sig is not None
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

        Args:
            signum: Shutdown signal that triggered the exit; the exit code is
                ``128 + signum`` per the POSIX signaled-exit convention.
            report: Cleanup evidence captured by the shutdown handler for the
                child process groups it terminated.

        """
        super().__init__(128 + signum)
        self.signum = signum
        self.report = report
        # The engine sets this only when its post-publication checkpoint
        # services a shutdown that was absorbed while committing results.
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

# Callables the handler runs once it has set ``_servicing_signum``, so waits
# that would otherwise block a worker thread (a GPU lease queue) wake and see it.
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

    One PGID snapshot under the launch lock drives both termination stages.
    Never call ``proc.wait()`` here: its interrupted waitpid lock is non-reentrant.
    A main-thread critical section records the signal for its exit checkpoint;
    a concurrent handler invocation returns without interrupting cleanup.

    Args:
        signum: The signal number that fired (``SIGTERM``, ``SIGINT``, or
            ``SIGHUP`` where available).
        _frame: The interrupted stack frame at signal-delivery time; unused
            but required by the ``signal.signal`` handler protocol.

    Raises:
        SystemExit: With exit code ``128 + signum`` (POSIX ``signaled-exit``
            convention) — always, except when deferred mid-critical-section
            or ignored during an active handler invocation as described above.

    """
    global _deferred_shutdown_signum, _servicing_signum  # noqa: PLW0603

    # A second signal must not re-enter registry locks or replace the first
    # handler's cleanup evidence.
    if not _shutdown_handler_lock.acquire(blocking=False):
        return

    try:
        if _main_thread_defer_depth > 0:
            # The main thread is inside a launch/unregister critical section and
            # may already hold the locks below. Record and return; the outermost
            # window exit services the shutdown.
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
            pgids = tuple(_active_children)

        log.warning("Received signal %d — killing %d active child group(s)", signum, len(pgids))

        confirmed_by_pgid: dict[int, bool] = {}
        try:
            confirmed_by_pgid = _terminate_process_groups(
                pgids,
                grace_seconds=_KILL_GRACE_SECONDS,
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
    if not hasattr(signal, "pthread_sigmask"):
        return
    if threading.current_thread() is not threading.main_thread():
        return
    signal.pthread_sigmask(signal.SIG_UNBLOCK, _SHUTDOWN_SIGNALS)


# Track temporary and process-lifetime ownership explicitly rather than
# inferring nesting from handler identity. Only the main thread mutates these.
_process_lifetime_owner = False
_scope_depth = 0


def install_signal_handlers() -> None:
    """Take process-lifetime ownership of SIGTERM, SIGINT and available SIGHUP.

    Main-thread only. Reinstall missing handlers and unblock shutdown signals on
    each call. An open temporary scope hands its installation over and skips
    restoration; later scopes leave process-lifetime ownership intact.

    Raises:
        SignalOwnershipUnavailableError: Called from a thread other than the
            main thread. ``signal.signal`` is main-thread-only, and so is the
            ownership handover above; entry points must install from the main
            thread at process start.

    """
    global _process_lifetime_owner  # noqa: PLW0603
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
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _shutdown_handler)
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
    prior_mask: Collection[int] | None,
) -> Exception | None:
    """Restore the embedding process's shutdown handlers and signal mask.

    Block signals, restore handlers, then restore the mask. Continue past individual
    handler failures so the remaining handlers are still restored.

    :param dict[int, Any] prior_handlers: Handlers captured before the scope
        installed its own, keyed by signal number. A ``None`` value means the
        signal had a C-level default that Python cannot reinstall.
    :param Collection[int] | None prior_mask: Thread signal mask captured
        before the scope unblocked shutdown signals, or ``None`` on platforms
        without ``signal.pthread_sigmask``.
    :return Exception | None: The first restoration failure, or ``None`` when
        every handler was restored.
    """
    if hasattr(signal, "pthread_sigmask"):
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
        except Exception as exc:  # noqa: BLE001 - continue past one bad restore
            log.exception("Failed to restore the prior handler for signal %d after scope exit", sig)
            if restoration_error is None:
                restoration_error = exc
    if prior_mask is not None:
        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
    return restoration_error


@contextlib.contextmanager
def signal_handler_scope() -> Iterator[None]:
    """Temporarily own shutdown signals for one outermost main-thread run.

    Nested calls share ownership. Process-lifetime ownership skips installation and
    restoration, including when claimed inside the scope. Worker calls require that
    ownership. Restoration reports its first error only when the body did not fail.

    Raises:
        SignalOwnershipUnavailableError: Called from a non-main thread while
            no enclosing scope or explicit :func:`install_signal_handlers`
            call already owns the shutdown signals.

    Yields:
        ``None``. Use as ``with signal_handler_scope(): ...`` around the
        outermost run whose child processes must be cleaned up on shutdown.

    """
    global _scope_depth, _servicing_signum  # noqa: PLW0603

    if _scope_depth == 0:
        # A new outermost run starts with no shutdown under way.
        _servicing_signum = None
    if _process_lifetime_owner:
        # Process-lifetime ownership skips restoration. Repair externally replaced
        # handlers on the main thread before running.
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
        # True lexical nesting on the main thread: an enclosing scope call
        # already installed the handlers and mask. Just track depth so only
        # the outermost exit restores anything.
        _scope_depth += 1
        try:
            yield
        finally:
            _scope_depth -= 1
        return

    prior_handlers: dict[int, Any] = {sig: signal.getsignal(sig) for sig in _SHUTDOWN_SIGNALS}
    prior_mask = None
    if hasattr(signal, "pthread_sigmask"):
        # An empty-set SIG_BLOCK call changes nothing; it is a pure read that
        # returns the mask already in effect, for restoration later.
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
    the handler wait on the launch lock. Python deferral remains active on platforms
    without ``pthread_sigmask``.

    Yields:
        ``None``. Use as ``with defer_shutdown_signals(): ...``.

    """
    global _main_thread_defer_depth  # noqa: PLW0603
    is_main = threading.current_thread() is threading.main_thread()
    old_mask = None
    if hasattr(signal, "pthread_sigmask"):
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _SHUTDOWN_SIGNALS)
    if is_main:
        _main_thread_defer_depth += 1
    try:
        yield
    finally:
        if is_main:
            _main_thread_defer_depth -= 1
        if old_mask is not None:
            # Restoring the mask delivers any kernel-queued signal right here;
            # its handler runs normally (depth is already back to zero) and
            # clears any deferred marker before raising.
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        if is_main and _main_thread_defer_depth == 0 and _deferred_shutdown_signum is not None:
            _service_deferred_shutdown()


def _service_deferred_shutdown() -> None:
    """Run the shutdown handler for a signal recorded mid-critical-section.

    Raises:
        PhaseSweepShutdown: Via ``_shutdown_handler``, carrying the cleanup
            evidence for the recorded signal.

    """
    global _deferred_shutdown_signum  # noqa: PLW0603
    signum = _deferred_shutdown_signum
    _deferred_shutdown_signum = None
    if signum is not None:
        _shutdown_handler(signum, None)


def service_pending_shutdown() -> None:
    """Deliver a shutdown signal absorbed by an earlier commit window.

    Call before starting new work. Does nothing when no signal is pending.

    Raises:
        PhaseSweepShutdown: Via ``_shutdown_handler``, when an absorbed
            shutdown signal is still pending.

    """
    _service_deferred_shutdown()


@dataclass
class AbsorbedShutdown:
    """Outcome of one :func:`absorb_shutdown_signals` window."""

    signum: int | None = None


@contextlib.contextmanager
def absorb_shutdown_signals() -> Iterator[AbsorbedShutdown]:
    """Complete a one-way commit before servicing shutdown.

    Signals arriving inside the window are reported on the yielded object and
    retained for a later shutdown checkpoint. Signals arriving before entry raise
    normally. Kernel-queued signals are drained while blocked using ``sigtimedwait``;
    without it, delivery can occur at window exit, with the terminal record remaining
    authoritative.

    Yields:
        AbsorbedShutdown: ``signum`` is the absorbed shutdown signal, or
        ``None`` when no shutdown arrived during the window.

    """
    global _main_thread_defer_depth  # noqa: PLW0603
    global _deferred_shutdown_signum  # noqa: PLW0603
    absorbed = AbsorbedShutdown()
    is_main = threading.current_thread() is threading.main_thread()
    old_mask = None
    if hasattr(signal, "pthread_sigmask"):
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _SHUTDOWN_SIGNALS)
    if is_main:
        _main_thread_defer_depth += 1
    try:
        yield absorbed
    finally:
        drained: int | None = None
        if old_mask is not None and hasattr(signal, "sigtimedwait"):
            # Consume any kernel-queued shutdown signal for this thread while
            # it is still blocked, so restoring the mask cannot deliver it as
            # an asynchronous exception after the window closes.
            while True:
                info = signal.sigtimedwait(_SHUTDOWN_SIGNALS, 0)
                if info is None:
                    break
                drained = info.si_signo
        if old_mask is not None:
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

    Args:
        proc: The ``Popen`` object returned by a just-completed ``Popen()`` call.
        pgid: Explicit child process group guarded by ``proc``. When omitted,
            register ``proc``'s own process group.

    Returns:
        The process-group ID (``pgid``) the subprocess was registered under.
        Callers store this for later ``_unregister`` and signal targeting.

    """
    resolved_pgid = os.getpgid(proc.pid) if pgid is None else pgid
    with _lock:
        _active_children[resolved_pgid] = proc
    return resolved_pgid


def _unregister(pgid: int) -> None:
    """Remove a finished process group from the global child registry.

    Defers shutdown signals while holding ``_lock`` so the signal handler
    (which also acquires ``_lock``) cannot interrupt and deadlock against
    the same thread.

    Args:
        pgid: The process-group ID returned by :func:`_register`. A pgid not
            currently registered is silently ignored.

    """
    with defer_shutdown_signals(), _lock:
        _active_children.pop(pgid, None)
