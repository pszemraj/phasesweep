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
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from types import FrameType
from typing import Any

from phasesweep.runtime.reaper import _KILL_GRACE_SECONDS, _terminate_process_groups

# Process supervision logs on one channel, whichever module does the work.
log = logging.getLogger("phasesweep.runtime.process")

_lock = threading.Lock()
_active_children: dict[int, subprocess.Popen[bytes]] = {}  # pgid -> Popen

# The launch lock guards the Popen() -> _register() critical section so the
# shutdown handler cannot snapshot _active_children while a child has been
# spawned but not yet registered (review v0.5.7 / blocker 3). The handler
# acquires the same lock before snapshotting, which forces it to wait until
# every in-flight launch has either registered its PGID or failed. Cost: if
# _spawn_blocked_supervisor's except-block runs _kill_group while this lock is
# still held (run_supervised's caller holds it across the whole spawn), that
# one failing launch can delay the shutdown handler's snapshot — and thus
# global shutdown response — by up to ~27s worst case: 10s awaiting supervisor
# readiness plus ~17s in _kill_group.
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


# Python-level shutdown deferral. Kernel signal masks are per-thread, but
# CPython runs Python signal handlers only in the main thread — and it does so
# whenever ANY thread's C-level handler tripped the pending flag, regardless of
# the main thread's kernel mask. Library thread pools (e.g. BLAS workers pulled
# in via numpy/optuna) keep shutdown signals unblocked, so a signal sent while
# the main thread is inside a ``defer_shutdown_signals()`` window can still
# execute ``_shutdown_handler`` in the main thread mid-critical-section, where
# re-acquiring ``_launch_lock``/``_lock`` self-deadlocks. These state variables
# extend the deferral to the Python level: while the main thread is inside a
# window, the handler records the signal and returns; the outermost window exit
# services it. Both are touched only from the main thread (window bookkeeping
# by construction, the handler by CPython's main-thread guarantee), so no lock
# is needed.
_deferred_shutdown_signum: int | None = None
_main_thread_defer_depth = 0


def _shutdown_handler(signum: int, _frame: FrameType | None) -> None:
    """Kill all tracked child process groups, then exit.

    CRITICAL: we must NOT call proc.wait() here because the main thread may
    already be inside proc.wait() on the same Popen object, and Python's
    internal _waitpid_lock is non-reentrant. Calling wait() from the signal
    handler would deadlock.

    Instead: send SIGTERM to all groups, brief sleep, SIGKILL for stragglers,
    then raise SystemExit. The original wait() call unblocks when the child dies.

    We snapshot the PGID set once under the lock and use that snapshot for both
    the SIGTERM and SIGKILL phases (review v0.5.5 / blocker 1). Without this,
    a worker thread can unregister a PGID after the root process exits on
    SIGTERM but before we escalate to SIGKILL — leaving descendants alive.

    The handler also acquires ``_launch_lock`` before snapshotting (review
    v0.5.7 / blocker 3) so a child that was just ``Popen()``-ed but not yet
    ``_register()``-ed cannot escape the snapshot. When the launcher is a
    worker thread, this handler (main thread) blocks until the launch site
    releases the lock, then sees the new PGID in ``_active_children``. When
    the launcher IS the main thread, blocking on the lock would self-deadlock:
    kernel masking cannot prevent that (a signal delivered to any unblocked
    library thread still runs this handler in the main thread), so if a
    main-thread ``defer_shutdown_signals()`` window is open the handler
    records the signal and returns; the window exit re-invokes it.

    A second signal delivered while this handler is already running returns
    immediately. The first invocation remains responsible for the complete
    PGID snapshot and its cleanup evidence instead of re-entering the
    non-reentrant registry locks or interrupting the kill loop.

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
    global _deferred_shutdown_signum  # noqa: PLW0603

    # Python signal handlers can interrupt an earlier invocation of this
    # handler. Let the first signal finish the authoritative cleanup pass;
    # re-entering could deadlock on the non-reentrant registry locks or replace
    # the first signal's cleanup evidence partway through the kill loop.
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


# Explicit shutdown-signal ownership tokens (review v0.5.15 / blocker 2B),
# replacing inference from ``signal.getsignal(sig) is _shutdown_handler``.
# That inference could not distinguish "this call's own enclosing scope" from
# "an unrelated concurrent top-level run in another thread" — both look like
# "the shutdown handler is already installed" — so the first of two
# concurrent top-level runs to finish would tear down handlers the second
# run still depends on (false nesting). Both globals are mutated only from
# the main thread (``install_signal_handlers`` and ``signal_handler_scope``'s
# install/restore branch both run there; the main thread is where
# ``signal.signal`` is even callable), so no lock is needed — CPython's GIL
# already makes a single attribute assignment atomic, and there is never a
# writer to race against a writer.
_process_lifetime_owner = False
_scope_depth = 0


def install_signal_handlers() -> None:
    """Install shutdown handlers that clean up child process groups.

    Handles SIGTERM/SIGINT and SIGHUP where the platform exposes it. Safe to
    call multiple times; only installs once. Idempotence is checked against
    the OS ground truth (every shutdown signal already dispatching to
    ``_shutdown_handler``) rather than a separate flag, so an external reset
    of a handler is never masked by stale bookkeeping. The main thread's
    shutdown signals are unblocked on each call so an inherited signal mask
    cannot prevent the handlers from running.

    Main-thread only, enforced BEFORE any ownership state is inspected or
    mutated (review v0.5.16 / blocker 5). The old off-main-thread behavior
    silently fell through to the idempotent check; during an open main-thread
    :func:`signal_handler_scope` every handler already points at
    ``_shutdown_handler``, so a worker thread could take the fast path and
    convert the scope's *temporary* installation into *process-lifetime*
    ownership — the scope's exit then skipped restoring the host's handlers,
    permanently stealing them from a worker thread that could never legally
    call ``signal.signal`` itself.

    On success — including the idempotent already-installed path — this
    takes process-lifetime ownership of the shutdown signals: every later
    :func:`signal_handler_scope` call, on any thread, becomes a no-op for the
    rest of the process (review v0.5.15 / blocker 2B).

    Calling this while a :func:`signal_handler_scope` is open is a valid
    handover, not a conflict: the scope's exit sees the ownership claim and
    leaves the handlers and mask in place instead of restoring the host's.
    Without that handover the idempotent path above would take ownership on
    the strength of the *scope's* installation, the scope would then tear
    that installation down on exit, and every later scope would no-op with
    no handler installed at all — silently disabling child-group cleanup for
    the rest of the process.

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
    """Reinstall any process-lifetime shutdown handler an outside party replaced.

    Runs only when :data:`_process_lifetime_owner` is already ``True``: the
    entry point committed this whole process to phasesweep's shutdown
    cleanup, so a handler some other library re-bound afterward would leave
    child process groups (and their GPUs) leaking on SIGTERM while the
    boolean claim still says otherwise. Off the main thread this can only
    observe, not repair (``signal.signal`` is main-thread-only), so it
    returns silently and the next main-thread scope entry repairs it.
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
    """Give the embedding process its shutdown handlers and signal mask back.

    Ordering matters (review v0.5.15 / blocker 2A): block first, THEN restore
    handlers, THEN restore the mask. Restoring the mask before the handlers
    would deliver a pending shutdown signal while ``_shutdown_handler`` is
    still installed for it, raising :class:`PhaseSweepShutdown` out of this
    cleanup path and leaving the remaining handlers unrestored.

    Restoration continues past an individual ``signal.signal`` failure rather
    than aborting the loop, so one bad signal cannot strand the others.

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
    """Own shutdown signals for the duration of one outermost run, then restore the host.

    A library must not leave its shutdown handlers and unblocked signal mask
    installed after it returns control — that permanently steals the
    embedding process's own SIGTERM/SIGINT/SIGHUP handling (review v0.5.14 /
    blocker 6). This context manager scopes ownership to one call tree,
    tracked with explicit tokens rather than handler-identity inference
    (review v0.5.15 / blocker 2B). Ownership contract:

      1. A process-lifetime :func:`install_signal_handlers` call (CLI/MCP
         entry points) owns the shutdown signals for the rest of the
         process; every scope call afterward, on any thread, is a no-op.
         An open scope that sees that call happen inside its body hands its
         installation over and skips restoration on exit, so the ownership
         the entry point just claimed is the ownership that survives.
      2. Absent that, the MAIN THREAD may take temporary ownership for one
         call tree: the outermost call installs and restores; lexically
         nested calls on that same thread share the ownership and touch
         nothing.
      3. A call from a thread that is neither the main thread nor covered by
         (1) raises :class:`SignalOwnershipUnavailableError` instead of
         silently running unprotected, because ``signal.signal`` only works
         on the main thread.

    Only the outermost main-thread call restores anything, and only when no
    :func:`install_signal_handlers` call took process-lifetime ownership
    while the body ran; it delegates to
    :func:`_restore_host_signal_state` for the ordering that makes
    restoration safe. The first restoration error is raised only when the
    scope body itself did not already raise (see the ``finally`` block).

    Raises:
        SignalOwnershipUnavailableError: Called from a non-main thread while
            no enclosing scope or explicit :func:`install_signal_handlers`
            call already owns the shutdown signals.

    Yields:
        ``None``. Use as ``with signal_handler_scope(): ...`` around the
        outermost run whose child processes must be cleaned up on shutdown.

    """
    global _scope_depth  # noqa: PLW0603

    if _process_lifetime_owner:
        # An entry point (CLI/MCP) already owns shutdown signals for the
        # whole process. Nothing to install or restore, on any thread — but
        # the ownership claim is a Python-side boolean, and another library
        # may have replaced an OS handler since the install. Re-assert the
        # OS ground truth (main thread only) so a stale claim cannot make
        # this run silently execute without child-group cleanup (review
        # v0.5.16 / blocker 5 follow-up).
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
    """Defer shutdown handling while the calling thread is in a critical section.

    Used to keep the ``Popen() -> _register()`` window atomic from the
    perspective of the signal handler (review v0.5.7 / blocker 3). CPython
    runs Python signal handlers in the main thread, so when the launcher is
    the main thread, taking ``_launch_lock`` from the handler would deadlock
    against the launcher's own lock acquisition. Two layers close that:

    1. Kernel mask: the calling thread blocks shutdown signals, so a signal
       aimed at it queues until the critical section ends.
    2. Python-level deferral (main thread only): the kernel mask is per-thread
       and cannot stop a signal delivered to some other unblocked thread (e.g.
       a BLAS pool worker) from tripping CPython's pending flag — the Python
       handler then runs in the main thread mid-window anyway. While a
       main-thread window is open, ``_shutdown_handler`` records the signal
       and returns; the outermost window exit re-invokes it after the kernel
       mask is restored.

    For worker-thread launchers (``n_jobs > 1``) the handler still runs on the
    main thread, which is NOT inside the window, so it simply blocks on
    ``_launch_lock`` until the worker finishes registration — the designed
    behavior, with no self-deadlock possible.

    Kernel masking is skipped on platforms without ``signal.pthread_sigmask``
    (Windows); the Python-level deferral still applies.

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
    """Deliver a shutdown signal absorbed by an earlier one-way commit window.

    Explicit checkpoint counterpart to :func:`absorb_shutdown_signals`: a
    caller that must not start new work after an absorbed shutdown calls this
    at its decision point. A no-op when nothing is pending.

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
    """Make a one-way commit window win any race against a shutdown signal.

    :func:`defer_shutdown_signals` delays a shutdown only to *service* it at
    window exit — correct for launch bookkeeping, but wrong for a publication
    transaction: once the last-success pointer has committed, a shutdown
    delivered at window exit would propagate as a failure for work that is
    already durably successful (review v0.5.16 / blocker 1). This window
    instead *absorbs* the signal: the transaction runs to completion, the
    window exit reports the absorbed signal on the yielded
    :class:`AbsorbedShutdown` instead of raising, and the signal stays
    recorded in the module's deferred-shutdown marker so a later checkpoint
    (:func:`service_pending_shutdown`, or any :func:`defer_shutdown_signals`
    exit, e.g. the next trial launch) still honors it before new work starts.

    Deterministic race outcome: a shutdown that arrives before this window
    opens raises normally and no publication happens; one that arrives inside
    the window loses to the commit and is delivered afterward.

    Signals kernel-queued to this thread during the window are drained
    synchronously via ``signal.sigtimedwait`` while still blocked, so they can
    never fire as an exception at window exit. On platforms without
    ``sigtimedwait``, a queued thread-directed signal is delivered after the
    window closes (razor-thin race); the on-disk outcome is still protected by
    the write-once terminal record. Signals routed through another unblocked
    thread run the Python handler mid-window, which records them via the
    Python-level deferral exactly like :func:`defer_shutdown_signals`.

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
    the same thread (review v0.5.9 / blocker 2).

    Args:
        pgid: The process-group ID returned by :func:`_register`. A pgid not
            currently registered is silently ignored.

    """
    with defer_shutdown_signals(), _lock:
        _active_children.pop(pgid, None)
