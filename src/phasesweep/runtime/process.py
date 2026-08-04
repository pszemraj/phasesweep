"""Child process lifecycle management.

Owns the dangerous parts: process groups, signal forwarding, PID tracking,
and graceful + forceful termination. Every child subprocess created by
phasesweep goes through this module.

Design:
  - Children run in their own process group (start_new_session=True) so we
    can kill the whole tree with os.killpg, not just the shell.
  - A global registry of live children lets us clean up on orchestrator death.
  - PID files in each trial_dir let operators identify orphans manually.

Launch barrier (review v0.5.15 / blocker 1): every trial launches through a
two-phase supervisor. Before the parent acknowledges, the supervisor is a
stdlib-only helper script (``phasesweep.runtime.supervisor``) run directly
(never ``-m``) under ``python -I -S`` with a minimal sanitized environment —
just ``PATH``. It cannot import the phasesweep package, a poisoned/shadowed
package from the trainer's ``PYTHONPATH``, or a trainer-composed
``sitecustomize``, and it starts in ~30ms instead of paying phasesweep's
package-import cost. Only after this process durably persists
``process_identity.json`` does it send the trainer's shell command and full
environment to the supervisor as a framed JSON payload; the supervisor then
``execve``s ``/bin/sh -c cmd`` under that environment, replacing itself as
the same PID/process group already registered here.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import select
import signal
import subprocess
import sys
import threading
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import IO, Any

from phasesweep.runtime import supervisor as _supervisor
from phasesweep.runtime.files import atomic_write_text
from phasesweep.runtime.json import strict_json_loads

log = logging.getLogger("phasesweep.runtime.process")

_KILL_GRACE_SECONDS = 10.0
_DIRECT_CHILD_REAP_TIMEOUT_SECONDS = 5.0
# With the -I -S stdlib-only launch (review v0.5.15 / blocker 1), supervisor
# startup no longer pays phasesweep's package-import cost (~0.55s pre-fix) —
# real-world readiness lands in ~30ms. 10s stays generous headroom for a
# loaded host, not a tight bound.
_SUPERVISOR_READY_TIMEOUT_SECONDS = 10.0
PROCESS_IDENTITY_FILE = "process_identity.json"
PROCESS_IDENTITY_SCHEMA_VERSION = 1


def _supervisor_script_path() -> str:
    """Return the absolute on-disk path to the stdlib-only supervisor script.

    Importing :mod:`phasesweep.runtime.supervisor` here (in the PARENT) is
    safe — the parent already has the full ``phasesweep`` package loaded.
    Only the supervisor *subprocess* must stay import-free before the
    ready/ack barrier (review v0.5.15 / blocker 1); this import never runs in
    that subprocess, which launches the file directly via its path instead.

    :return str: Absolute path to ``supervisor.py`` on disk.
    :raises RuntimeError: If the imported module has no ``__file__`` (e.g. a
        namespace package or frozen build), making a subprocess launch by
        path impossible.
    """
    module_file = _supervisor.__file__
    if module_file is None:
        raise RuntimeError("phasesweep.runtime.supervisor has no __file__; cannot launch it.")
    return str(Path(module_file).resolve())


_SUPERVISOR_SCRIPT_PATH = _supervisor_script_path()

# ---------------------------------------------------------------------------
# Global child registry + signal handler
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_active_children: dict[int, subprocess.Popen] = {}  # pgid -> Popen

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
    """SystemExit carrying child process-group cleanup evidence."""

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
    caller that must not start new work after an absorbed shutdown (e.g. the
    suite loop before its next study) calls this at its decision point. A
    no-op when nothing is pending.

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


def _register(proc: subprocess.Popen) -> int:
    """Add a freshly-launched subprocess to the global child registry.

    Args:
        proc: The ``Popen`` object returned by a just-completed ``Popen()`` call.

    Returns:
        The process-group ID (``pgid``) the subprocess was registered under.
        Callers store this for later ``_unregister`` and signal targeting.

    """
    pgid = os.getpgid(proc.pid)
    with _lock:
        _active_children[pgid] = proc
    return pgid


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


def _kill_group(pgid: int, proc: subprocess.Popen) -> bool:
    """Terminate the trial process group and return whether cleanup is confirmed.

    Returns ``True`` when the group is confirmed gone, ``False`` when cleanup
    is uncertain (survived SIGKILL, permission denied, etc.). Callers must
    propagate uncertainty so the orchestrator can refuse to schedule more work
    onto a potentially-leaked GPU (review v0.5.9 / blocker 3).

    Pre-v0.5.15 this only reaped the direct child (``proc``) when a single
    nonblocking ``poll()`` already observed it exited — a race where the
    child died right after ``poll()`` returned ``None`` left it an unreaped
    zombie despite ``cleanup_confirmed`` being reported ``True`` (review
    v0.5.15 / blocker 4). Once :func:`_terminate_process_group` confirms every
    group member is gone, this now unconditionally does one bounded blocking
    ``wait`` on the direct child so it is provably reaped (or cleanup is
    reported unconfirmed) before returning. This never calls ``proc.wait()``
    from ``_shutdown_handler`` itself — that path documents why it must not
    block (see :func:`_shutdown_handler`); this function only runs on normal
    (non-signal-handler) cleanup paths.

    Worst case this blocks for roughly ``_KILL_GRACE_SECONDS`` (10s) SIGTERM
    grace + ~2s SIGKILL confirm inside :func:`_terminate_process_group`, plus
    ``_DIRECT_CHILD_REAP_TIMEOUT_SECONDS`` (5s) for the direct-child reap
    above — up to ~17s total. Before one caller,
    ``_spawn_blocked_supervisor``'s except-block, reaches this cleanup it can
    spend up to ``_SUPERVISOR_READY_TIMEOUT_SECONDS`` (10s) awaiting
    readiness. Because ``run_supervised`` still holds ``_launch_lock`` with
    shutdown signals deferred throughout, that combined path can delay the
    global shutdown handler by up to ~27s (see the ``_launch_lock`` comment
    block).

    Args:
        pgid: Process-group ID of the trial subprocess.
        proc: The root subprocess's :class:`subprocess.Popen` handle. Reaped
            via a bounded ``wait`` once the group is confirmed terminated.

    Returns:
        ``True`` if every process in the group is gone after the SIGTERM →
        SIGKILL escalation AND the direct child was reaped within
        ``_DIRECT_CHILD_REAP_TIMEOUT_SECONDS``; ``False`` if at least one
        group member survived, cleanup status was inconclusive, or the direct
        child failed to reap in time.

    """
    cleanup_confirmed = _terminate_process_group(pgid, grace_seconds=_KILL_GRACE_SECONDS)
    try:
        proc.wait(timeout=_DIRECT_CHILD_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        log.error(
            "Direct child PID %d (pgid %d) did not reap within %.1fs after group "
            "termination; treating cleanup as unconfirmed",
            proc.pid,
            pgid,
            _DIRECT_CHILD_REAP_TIMEOUT_SECONDS,
        )
        cleanup_confirmed = False
    except ChildProcessError:
        pass  # Already reaped elsewhere (e.g. a concurrent wait()).
    return cleanup_confirmed


def _abort_launch(proc: subprocess.Popen, pgid: int | None) -> bool:
    """Kill and unregister a subprocess whose launch failed before hand-off.

    Shared by both launch-failure paths — the readiness-wait failure in
    :func:`_spawn_blocked_supervisor` and the identity/payload-delivery
    failure in :func:`run_supervised` — which otherwise ran this identical
    three-step sequence independently. Resolves the target process group
    (the registered ``pgid``, or the bare PID when registration never
    happened), kills it, and unregisters it from the global registry if it
    was registered.

    Args:
        proc: The subprocess whose launch is being aborted.
        pgid: The process-group ID it was registered under, or ``None`` if
            registration never completed.

    Returns:
        Whatever :func:`_kill_group` returns for the resolved target group —
        ``True`` only if the group is confirmed terminated.

    """
    target_pgid = pgid if pgid is not None else proc.pid
    cleanup_confirmed = _kill_group(target_pgid, proc)
    if pgid is not None:
        _unregister(pgid)
    return cleanup_confirmed


# ---------------------------------------------------------------------------
# Supervised subprocess execution
# ---------------------------------------------------------------------------


@dataclass
class ProcessResult:
    """Result of a supervised subprocess execution."""

    return_code: int
    timed_out: bool
    pid: int
    duration_seconds: float
    failure_reason: str | None = None
    cleanup_confirmed: bool = True


class _LaunchDeadlineExpired(Exception):
    """The launch deadline expired before the trainer payload was delivered.

    Raised inside the supervised-launch critical section (review v0.5.16 /
    blocker 6) so :func:`run_supervised` can classify the outcome as a
    timeout — the trainer never started — instead of a generic launch
    failure. Carries the cleanup confirmation and pid when the raising site
    already aborted the blocked supervisor itself.
    """

    def __init__(
        self,
        message: str,
        *,
        cleanup_confirmed: bool = True,
        pid: int = -1,
    ) -> None:
        """Record the expiry context for :func:`run_supervised`'s timeout result.

        :param str message: Human-readable description of which launch stage expired.
        :param bool cleanup_confirmed: Whether the aborted supervisor's group
            is confirmed gone (set by the abort site when it already ran).
        :param int pid: PID of the aborted supervisor, or ``-1`` when unknown.
        """
        super().__init__(message)
        self.cleanup_confirmed = cleanup_confirmed
        self.pid = pid


@dataclass(frozen=True)
class StaleProcessIdentity:
    """Durable identity of one launched trial process group."""

    schema_version: int
    attempt_id: str
    pid: int
    pgid: int
    proc_starttime: int | None
    boot_id: str | None


def read_boot_id() -> str | None:
    """Return the current Linux boot identity, or ``None`` when unavailable."""
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def _trial_process_identity(
    *,
    attempt_id: str,
    pid: int,
    pgid: int,
) -> StaleProcessIdentity:
    """Build the identity persisted before a blocked supervisor may exec the trainer.

    Args:
        attempt_id: Immutable attempt identity already persisted in Optuna;
            binds this identity record to exactly one attempt so a later
            reader cannot mistake it for a different trial's process.
        pid: PID of the just-launched supervisor process.
        pgid: Process-group ID the supervisor was registered under.

    Returns:
        A :class:`StaleProcessIdentity` combining the given fields with the
        current schema version, this process's ``/proc`` start time (``None``
        off-Linux or if unreadable), and the current boot id (``None`` when
        unavailable).

    """
    return StaleProcessIdentity(
        schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
        attempt_id=attempt_id,
        pid=pid,
        pgid=pgid,
        proc_starttime=read_proc_starttime(pid),
        boot_id=read_boot_id(),
    )


def _write_process_identity(path: Path, identity: StaleProcessIdentity) -> None:
    """Atomically persist one complete process identity record.

    Args:
        path: Destination ``process_identity.json`` path under the trial dir.
        identity: Complete identity record to serialize as sorted-key JSON.

    """
    atomic_write_text(
        path,
        json.dumps(
            {
                "schema_version": identity.schema_version,
                "attempt_id": identity.attempt_id,
                "pid": identity.pid,
                "pgid": identity.pgid,
                "proc_starttime": identity.proc_starttime,
                "boot_id": identity.boot_id,
            },
            sort_keys=True,
        )
        + "\n",
    )


ATTEMPT_LIFECYCLE_FILE = "attempt_lifecycle.json"
ATTEMPT_LIFECYCLE_SCHEMA_VERSION = 1
_ATTEMPT_LIFECYCLE_STATES = frozenset({"allocated", "exited"})


@dataclass(frozen=True)
class AttemptLifecycle:
    """Durable coarse lifecycle state for one trial attempt.

    Closes the two recovery windows the transient identity file could not
    represent (review v0.5.17 / blocker 2): ``allocated`` says a durable
    Optuna ``RUNNING`` trial exists but no process was ever created (the
    worker may still be queued for a GPU), and ``exited`` says the supervised
    process group is confirmed gone even though evidence extraction and the
    Optuna terminal commit may not have happened yet. Recovery can then fail
    such trials safely instead of treating both states as unverifiable
    cleanup uncertainty.
    """

    schema_version: int
    attempt_id: str
    state: str
    return_code: int | None
    cleanup_confirmed: bool | None


def write_attempt_lifecycle(
    trial_dir: Path,
    *,
    attempt_id: str,
    state: str,
    return_code: int | None = None,
    cleanup_confirmed: bool | None = None,
) -> None:
    """Atomically persist the attempt's coarse lifecycle state.

    Args:
        trial_dir: Per-trial directory (attempt-scoped, so states from
            different attempts can never collide).
        attempt_id: Immutable attempt identity binding the record.
        state: One of ``"allocated"`` or ``"exited"``.
        return_code: Root return code; only meaningful for ``"exited"``.
        cleanup_confirmed: Whether the whole process group was confirmed
            gone; only meaningful for ``"exited"``.

    Raises:
        ValueError: ``state`` is not a known lifecycle state.
        OSError: The record could not be written.

    """
    if state not in _ATTEMPT_LIFECYCLE_STATES:
        raise ValueError(f"Unknown attempt lifecycle state {state!r}.")
    atomic_write_text(
        trial_dir / ATTEMPT_LIFECYCLE_FILE,
        json.dumps(
            {
                "schema_version": ATTEMPT_LIFECYCLE_SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "state": state,
                "return_code": return_code,
                "cleanup_confirmed": cleanup_confirmed,
            },
            sort_keys=True,
        )
        + "\n",
    )


def read_attempt_lifecycle(
    trial_dir: Path,
    *,
    expected_attempt_id: str,
) -> AttemptLifecycle | None:
    """Load and validate the attempt lifecycle record, if one exists.

    Args:
        trial_dir: Trial directory that may contain ``attempt_lifecycle.json``.
        expected_attempt_id: Attempt identity stored on the Optuna trial.

    Returns:
        The validated record, or ``None`` when no record exists (legacy
        attempts written before this schema).

    Raises:
        ValueError: The record is malformed, uses an unknown schema or state,
            or belongs to another attempt. Callers must treat this as
            cleanup uncertainty, not as absence.

    """
    path = trial_dir / ATTEMPT_LIFECYCLE_FILE
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"Attempt lifecycle record at {path} is unreadable.") from exc
    try:
        payload = strict_json_loads(raw)
    except ValueError as exc:
        raise ValueError(f"Malformed attempt lifecycle record at {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Attempt lifecycle record at {path} must be a JSON object.")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != ATTEMPT_LIFECYCLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported attempt lifecycle schema at {path}.")
    if payload.get("attempt_id") != expected_attempt_id:
        raise ValueError(f"Attempt lifecycle record at {path} belongs to another attempt.")
    state = payload.get("state")
    if state not in _ATTEMPT_LIFECYCLE_STATES:
        raise ValueError(f"Unknown attempt lifecycle state at {path}: {state!r}.")
    return_code = payload.get("return_code")
    if return_code is not None and (isinstance(return_code, bool) or type(return_code) is not int):
        raise ValueError(f"Attempt lifecycle field 'return_code' is invalid at {path}.")
    cleanup_confirmed = payload.get("cleanup_confirmed")
    if cleanup_confirmed is not None and not isinstance(cleanup_confirmed, bool):
        raise ValueError(f"Attempt lifecycle field 'cleanup_confirmed' is invalid at {path}.")
    return AttemptLifecycle(
        schema_version=schema_version,
        attempt_id=expected_attempt_id,
        state=state,
        return_code=return_code,
        cleanup_confirmed=cleanup_confirmed,
    )


def _record_attempt_exited(
    trial_dir: Path,
    *,
    attempt_id: str,
    return_code: int,
) -> None:
    """Best-effort durable transition to the ``exited`` lifecycle state.

    Called only after the supervised group is confirmed gone. A write failure
    must not fail the trial: the retained identity file still lets recovery
    verify the (now dead) process the slow way.

    Args:
        trial_dir: Per-trial directory holding the lifecycle record.
        attempt_id: Immutable attempt identity binding the record.
        return_code: Root process return code observed by the supervisor wait.

    """
    try:
        write_attempt_lifecycle(
            trial_dir,
            attempt_id=attempt_id,
            state="exited",
            return_code=return_code,
            cleanup_confirmed=True,
        )
    except OSError:
        log.warning(
            "Could not persist the 'exited' lifecycle transition for attempt %s; "
            "recovery will fall back to verifying the retained process identity.",
            attempt_id,
        )


def _sanitized_supervisor_env() -> dict[str, str]:
    """Build the minimal environment for the pre-ACK supervisor interpreter.

    Only ``PATH`` is passed through. Everything else — the full trainer
    environment, ``PYTHONPATH``/``PYTHONHOME``, and any ``CUDA_*`` var
    composed for this trial — is intentionally withheld: the interpreter
    that runs ``supervisor.py`` has not yet passed the ready/ack barrier, so
    it must not see anything a trainer-composed environment could use to
    execute code before phasesweep's process identity is durable (review
    v0.5.15 / blocker 1). The trainer environment is delivered separately,
    over the ack pipe, only after identity persistence succeeds.

    :return dict[str, str]: ``{"PATH": ...}`` using the parent's ``PATH``, or
        a conservative fallback when the parent has none.
    """
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


def _encode_launch_payload(cmd: str, env: dict[str, str], cwd: str | None = None) -> bytes:
    """Frame the trainer command, environment, and cwd for the supervisor's ack pipe.

    Decoded on the other end by
    :func:`phasesweep.runtime.supervisor._read_launch_payload` (via
    ``_read_exact``); keep both sides in sync if the wire format changes.

    :param str cmd: Shell command string the supervisor execs with ``/bin/sh``.
    :param dict[str, str] env: Full trainer process environment.
    :param str | None cwd: Optional working directory the supervisor changes
        into before exec'ing the trainer (review v0.5.17 / blocker 4).
    :return bytes: An ASCII decimal length header (byte length of the UTF-8
        JSON body) immediately followed by that many body bytes.
    """
    payload: dict[str, Any] = {"cmd": cmd, "env": env}
    if cwd is not None:
        payload["cwd"] = cwd
    body = json.dumps(payload).encode("utf-8")
    # supervisor.py's _HEADER_LEN is the single source of truth for the frame
    # header width; derive the format width from it rather than hardcoding
    # the digit count here.
    return f"{len(body):0{_supervisor._HEADER_LEN}d}".encode("ascii") + body


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte of ``data`` to ``fd``, looping past partial pipe writes.

    :param int fd: Open file descriptor to write to.
    :param bytes data: Bytes to write in full.
    :raises OSError: If the underlying ``os.write`` call fails.
    """
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _spawn_blocked_supervisor(
    *,
    stdout: IO[str],
    stderr: IO[str],
    deadline: float | None = None,
) -> tuple[subprocess.Popen, int, int]:
    """Spawn a supervisor that cannot exec the trainer until its parent delivers a payload.

    Launches the stdlib-only ``phasesweep.runtime.supervisor`` script
    directly (never ``-m phasesweep.runtime.supervisor``) under
    ``python -I -S`` with a minimal sanitized environment — see
    :func:`_sanitized_supervisor_env`. Passes it a readiness pipe and an
    acknowledgement pipe. Blocks (via ``select``) until the supervisor
    signals readiness or ``_SUPERVISOR_READY_TIMEOUT_SECONDS`` elapses —
    capped by the remaining launch ``deadline`` when one is given (review
    v0.5.16 / blocker 6), so a slow supervisor startup can never outlive the
    trial's own wallclock budget. Then registers the new process group. On
    any failure — timeout, an unexpected readiness byte, or an exception
    from ``Popen`` itself — any spawned process group is killed and
    unregistered before the exception propagates.

    Args:
        stdout: Already-open file handle that receives the subprocess stdout.
        stderr: Already-open file handle that receives the subprocess stderr.
        deadline: Optional ``time.monotonic()`` launch deadline. When it
            expires during the readiness wait, the spawn is aborted and
            :class:`_LaunchDeadlineExpired` is raised so the caller reports a
            timeout instead of a generic launch failure.

    Returns:
        A ``(proc, pgid, ack_write)`` tuple: the supervisor's ``Popen`` handle,
        its registered process-group id, and the write end of the
        acknowledgement pipe. The caller owns ``ack_write`` and must send the
        framed launch payload (see :func:`_encode_launch_payload`) and close
        it once the trial's process identity is durably persisted.

    Raises:
        _LaunchDeadlineExpired: The launch deadline expired before the
            supervisor became ready; the spawned group is already aborted and
            the exception carries the cleanup confirmation.
        RuntimeError: If the supervisor does not signal readiness within
            ``_SUPERVISOR_READY_TIMEOUT_SECONDS``, or signals something other
            than ``b"R"``.

    """
    import time

    ready_read, ready_write = os.pipe()
    ack_read, ack_write = os.pipe()
    proc: subprocess.Popen | None = None
    pgid: int | None = None
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                _SUPERVISOR_SCRIPT_PATH,
                str(ready_write),
                str(ack_read),
            ],
            env=_sanitized_supervisor_env(),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            pass_fds=(ready_write, ack_read),
        )
        closed_fd = ready_write
        ready_write = -1
        os.close(closed_fd)
        closed_fd = ack_read
        ack_read = -1
        os.close(closed_fd)

        pgid = _register(proc)
        ready_timeout = _SUPERVISOR_READY_TIMEOUT_SECONDS
        if deadline is not None:
            ready_timeout = min(ready_timeout, max(0.0, deadline - time.monotonic()))
        readable, _, _ = select.select(
            [ready_read],
            [],
            [],
            ready_timeout,
        )
        if not readable and deadline is not None and time.monotonic() >= deadline:
            raise _LaunchDeadlineExpired(
                "trial launch deadline expired while waiting for the supervisor",
                pid=proc.pid,
            )
        # Written by phasesweep.runtime.supervisor.main's os.write(ready_fd, b"R").
        if not readable or os.read(ready_read, 1) != b"R":
            raise RuntimeError("trial supervisor did not become ready before launch")
        closed_fd = ready_read
        ready_read = -1
        os.close(closed_fd)
        return proc, pgid, ack_write
    except Exception as exc:
        if ack_write >= 0:
            closed_fd = ack_write
            ack_write = -1
            os.close(closed_fd)
        if proc is not None:
            confirmed = _abort_launch(proc, pgid)
            if isinstance(exc, _LaunchDeadlineExpired):
                exc.cleanup_confirmed = confirmed
        raise
    finally:
        for fd in (ready_read, ready_write, ack_read):
            if fd >= 0:
                os.close(fd)


def run_supervised(
    cmd: str,
    *,
    env: dict[str, str],
    stdout: IO[str],
    stderr: IO[str],
    timeout: float | None,
    trial_dir: Path,
    attempt_id: str,
    cwd: str | None = None,
) -> ProcessResult:
    """Launch a shell command in its own process group with full lifecycle management.

    Launches a small stdlib-only supervisor first (see module docstring for
    the pre-ACK/post-ACK launch boundary; review v0.5.15 / blocker 1). The
    supervisor waits on an inherited acknowledgement pipe while the parent
    atomically persists ``process_identity.json``. Only after the identity is
    durable does the parent send the trainer command and full trainer
    environment to the supervisor as a framed JSON payload, which the
    supervisor then execs the trainer command under. If the parent dies
    before delivering that payload, pipe EOF makes the supervisor exit
    without starting training.

    The identity record is always retained. On a fully clean exit — root
    returned 0 and no descendant processes were left alive — the attempt
    lifecycle is durably advanced to ``exited``. If the root exits cleanly
    but leaves GPU-holding descendants running, we treat that as a lifecycle
    failure and kill the group (review v0.5.5 / blocker 1).

    On timeout: SIGTERM -> grace -> SIGKILL on the entire group.

    ``timeout`` is the total launch-plus-execution budget, converted to an
    absolute ``time.monotonic()`` deadline at entry (review v0.5.16 /
    blocker 6). Every stage consumes it: the supervisor readiness wait is
    capped to the remaining budget, the deadline is re-checked after
    identity persistence and *before* the trainer payload crosses the ack
    pipe — an expired deadline aborts the blocked supervisor and returns a
    timeout without ever starting the trainer — and the final ``proc.wait``
    uses the recomputed remainder, never the original duration. Cleanup
    grace after a timeout is explicitly post-deadline time.

    Once the supervisor is spawned, launch failures — an expired deadline, a
    failed identity write, a failed payload delivery — never propagate: the
    group is terminated and the failure is reported through the returned
    :class:`ProcessResult`.

    Args:
        cmd: Shell command string the acknowledged supervisor execs with ``/bin/sh``.
        env: Full process environment for the subprocess.
        stdout: Already-open file handle that receives the subprocess stdout.
        stderr: Already-open file handle that receives the subprocess stderr.
        timeout: Total wall-clock budget in seconds measured from this call's
            entry (launch overhead included), or ``None`` for no timeout.
        trial_dir: Per-trial directory where ``process_identity.json`` is written.
        attempt_id: Immutable attempt identity already persisted in Optuna.
        cwd: Optional working directory the supervisor changes into before
            exec'ing the trainer, delivered over the ack pipe with the rest
            of the launch payload (review v0.5.17 / blocker 4). ``None``
            keeps the invocation cwd.

    Returns:
        :class:`ProcessResult` capturing return code, wall-clock duration,
        timeout flag, ``failure_reason`` (set on timeout or descendant
        survival), and ``cleanup_confirmed`` (``False`` when SIGKILL did not
        confirm the group is gone).

    Raises:
        RuntimeError: The supervisor never signalled readiness, or signalled an
            unexpected byte, so no process group was ever created.
        OSError: The supervisor process or its pipes could not be created.

    """
    import time

    started = time.monotonic()
    deadline = None if timeout is None else started + timeout

    # The launch + register + identity write must be atomic from the
    # signal handler's perspective. Signal deferral MUST come first so that
    # SIGTERM/SIGINT cannot land between ``_launch_lock`` acquisition and
    # signal masking (review v0.5.9 / blocker 2). Reversing the order
    # (``_launch_lock`` first, then ``defer_shutdown_signals()``) creates
    # two deadlock windows:
    #
    # 1. Signal lands after lock acquired but before mask set: handler runs
    #    in same thread and blocks on the lock it already holds.
    # 2. On exit, mask is restored before lock released: pending signal is
    #    delivered while the thread still owns the lock -> same deadlock.
    #
    # Correct ordering: block signals -> take lock -> work -> release lock
    # -> unblock signals. Any pending signal is delivered after the lock is
    # released, so the handler can safely acquire it.
    proc: subprocess.Popen | None = None
    pgid: int | None = None
    ack_write: int | None = None
    identity_path = trial_dir / PROCESS_IDENTITY_FILE

    try:
        with defer_shutdown_signals(), _launch_lock:
            proc, pgid, ack_write = _spawn_blocked_supervisor(
                stdout=stdout,
                stderr=stderr,
                deadline=deadline,
            )
            identity = _trial_process_identity(
                attempt_id=attempt_id,
                pid=proc.pid,
                pgid=pgid,
            )
            _write_process_identity(identity_path, identity)
            if deadline is not None and time.monotonic() >= deadline:
                # The budget expired during launch bookkeeping. The payload
                # has NOT been delivered, so the blocked supervisor can be
                # aborted without any trainer work ever starting (review
                # v0.5.16 / blocker 6).
                raise _LaunchDeadlineExpired(
                    "trial launch deadline expired before the trainer payload was delivered",
                    pid=proc.pid,
                )
            # Only now does the trainer command and full trainer environment
            # cross into the supervisor — after identity is durable, over the
            # ack pipe as a framed JSON payload (review v0.5.15 / blocker 1).
            _write_all(ack_write, _encode_launch_payload(cmd, env, cwd))
            os.close(ack_write)
            ack_write = None
    except _LaunchDeadlineExpired as exc:
        if ack_write is not None:
            os.close(ack_write)
        cleanup_confirmed = exc.cleanup_confirmed
        pid = exc.pid
        return_code = -9
        if proc is not None:
            # Raised at the pre-payload recheck: the supervisor is still
            # blocked on its ack pipe; kill and reap it here.
            cleanup_confirmed = _abort_launch(proc, pgid)
            pid = proc.pid
            if proc.returncode is not None:
                return_code = proc.returncode
        if cleanup_confirmed:
            _record_attempt_exited(trial_dir, attempt_id=attempt_id, return_code=return_code)
        log.warning(
            "Trial launch deadline expired before the trainer started (pid %d): %s",
            pid,
            exc,
        )
        return ProcessResult(
            return_code=return_code,
            timed_out=True,
            pid=pid,
            duration_seconds=time.monotonic() - started,
            failure_reason=f"timeout after {timeout}s before trainer launch",
            cleanup_confirmed=cleanup_confirmed,
        )
    except Exception as exc:
        if ack_write is not None:
            os.close(ack_write)
        if proc is None:
            raise
        target_pgid = pgid if pgid is not None else proc.pid
        # Covers both a failed identity write and a failed payload delivery;
        # the substring "failed to persist process identity" is kept stable
        # because it is asserted on by existing tests.
        launch_failure_reason = (
            f"failed to persist process identity or deliver launch payload: {exc}"
        )
        log.exception(
            "Trial PID %d (pgid %d) launched but identity persistence or launch payload "
            "delivery failed; terminating group",
            proc.pid,
            target_pgid,
        )
        cleanup_confirmed = _abort_launch(proc, pgid)
        if cleanup_confirmed:
            _record_attempt_exited(
                trial_dir,
                attempt_id=attempt_id,
                return_code=proc.returncode if proc.returncode is not None else -9,
            )
        duration = time.monotonic() - started
        return ProcessResult(
            return_code=proc.returncode if proc.returncode is not None else -9,
            timed_out=False,
            pid=proc.pid,
            duration_seconds=duration,
            failure_reason=launch_failure_reason,
            cleanup_confirmed=cleanup_confirmed,
        )

    assert proc is not None
    assert pgid is not None

    timed_out = False
    failure_reason: str | None = None
    cleanup_confirmed = True

    try:
        try:
            # Recompute the remainder: the original duration would silently
            # extend the budget by however long launch bookkeeping took
            # (review v0.5.16 / blocker 6).
            proc.wait(timeout=None if deadline is None else max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            failure_reason = f"timeout after {timeout}s"
            log.warning("Trial PID %d (pgid %d) timed out — terminating group", proc.pid, pgid)
            cleanup_confirmed = _kill_group(pgid, proc)
        else:
            # Root process exited normally. That is not sufficient — the trial
            # is only clean once the entire process group is gone. A common
            # pathological case: `python launcher.py &` exits immediately while
            # the training worker stays alive holding GPU memory.
            if _process_group_alive(pgid):
                failure_reason = (
                    f"root process exited with code {proc.returncode}, "
                    f"but process group {pgid} still had live descendants"
                )
                log.warning(
                    "Trial PID %d exited with code %s but process group %d "
                    "still has live descendants — terminating group",
                    proc.pid,
                    proc.returncode,
                    pgid,
                )
                cleanup_confirmed = _kill_group(pgid, proc)

    finally:
        _unregister(pgid)

    # The identity record is deliberately RETAINED on clean exit (review
    # v0.5.17 / blocker 2 gap B): the orchestrator can still die between
    # here and the Optuna terminal commit (evidence extraction, gates), and
    # recovery must be able to distinguish "safely exited" from "identity
    # missing". The durable 'exited' transition records that the whole group
    # is confirmed gone; it is only written outside the exception paths
    # above, so an interrupted wait can never claim a confirmed exit.
    if cleanup_confirmed:
        _record_attempt_exited(
            trial_dir,
            attempt_id=attempt_id,
            return_code=proc.returncode if proc.returncode is not None else -9,
        )

    duration = time.monotonic() - started
    return ProcessResult(
        return_code=proc.returncode if proc.returncode is not None else -9,
        timed_out=timed_out,
        pid=proc.pid,
        duration_seconds=duration,
        failure_reason=failure_reason,
        cleanup_confirmed=cleanup_confirmed,
    )


# ---------------------------------------------------------------------------
# Stale process utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProcStat:
    """Parsed fields from one Linux ``/proc/<pid>/stat`` record."""

    state: str
    pgrp: int
    starttime: int


def _read_proc_stat(proc_entry: Path) -> _ProcStat | None:
    """Parse the proc stat fields phasesweep uses for liveness checks.

    :param Path proc_entry: ``/proc/<pid>`` directory to inspect.
    :return _ProcStat | None: Parsed state, process group, and starttime, or ``None`` when
        unreadable.
    """
    try:
        data = (proc_entry / "stat").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    rparen = data.rfind(b")")
    if rparen < 0:
        return None
    rest = data[rparen + 1 :].strip().split()
    if len(rest) < 20:
        return None
    try:
        return _ProcStat(state=rest[0].decode("ascii"), pgrp=int(rest[2]), starttime=int(rest[19]))
    except (UnicodeDecodeError, ValueError):
        return None


def read_proc_starttime(pid: int) -> int | None:
    """Read the start time of a process from /proc/<pid>/stat.

    On Linux, (pid, starttime) uniquely identifies a process across its
    lifetime. This is the only reliable way to avoid PID-reuse hazards
    when killing stale processes from a prior orchestrator run.

    Args:
        pid: The process ID to inspect.

    Returns:
        The starttime in clock ticks (``/proc/<pid>/stat`` field 22), or
        ``None`` on non-Linux systems and when the proc entry is unreadable.

    """
    stat = _read_proc_stat(Path("/proc") / str(pid))
    return None if stat is None else stat.starttime


def is_pid_alive(pid: int) -> bool:
    """Check if a PID exists (best-effort; race-free check is impossible).

    Args:
        pid: The process ID to probe via ``kill(pid, 0)``.

    Returns:
        ``True`` if the PID exists (or exists but is owned by another user);
        ``False`` if the kernel reports ``ProcessLookupError``.

    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user


def is_same_live_process(pid: int | None, saved_starttime: int | None) -> bool:
    """Return whether a PID identifies the same live, non-zombie process.

    :param int | None pid: Process identifier, if one was recorded.
    :param int | None saved_starttime: Recorded Linux process start time.
    :return bool: Whether the identity still names a live non-zombie process.
    """
    if pid is None or not is_pid_alive(pid):
        return False
    stat = _read_proc_stat(Path("/proc") / str(pid))
    if saved_starttime is not None and (stat is None or stat.starttime != saved_starttime):
        return False
    return stat is None or stat.state != "Z"


def reap_child(pid: int) -> bool:
    """Best-effort non-blocking reap of one exited child.

    A long-lived parent (the MCP server) that spawns detached runners and never
    waits on them accumulates a zombie per runner as each one exits. Call this
    for a known runner pid to reap it if it has already exited; it is a no-op if
    the process is still running, was never our child, or has already been reaped.
    This runs on status-read paths, so it must stay strictly non-blocking. A
    single ``waitpid(WNOHANG)`` reduces zombie buildup; it does not guarantee
    that a child exiting immediately after this call is reaped before the next
    status scan.

    Args:
        pid: PID of a runner this process spawned.

    Returns:
        ``True`` when this call reaped ``pid``; ``False`` otherwise.

    """
    try:
        waited, _status = os.waitpid(pid, os.WNOHANG)
    except OSError:
        return False
    return waited == pid


def read_stale_process_identity(
    trial_dir: Path,
    *,
    expected_attempt_id: str,
) -> StaleProcessIdentity:
    """Load and validate the atomic identity for a possibly-stale trial.

    Args:
        trial_dir: Trial directory containing ``process_identity.json``.
        expected_attempt_id: Attempt identity stored on the Optuna trial.

    Returns:
        A complete, attempt-bound :class:`StaleProcessIdentity`.

    Raises:
        OSError: The identity record is missing or unreadable.
        ValueError: The identity record is malformed or belongs to another attempt.

    """
    path = trial_dir / PROCESS_IDENTITY_FILE
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"Malformed trial process identity at {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Trial process identity at {path} must be a JSON object.")
    required_fields = {
        "schema_version",
        "attempt_id",
        "pid",
        "pgid",
        "proc_starttime",
        "boot_id",
    }
    if not required_fields.issubset(payload):
        raise ValueError(f"Trial process identity at {path} is partial.")
    schema_version = payload["schema_version"]
    if type(schema_version) is not int or schema_version != PROCESS_IDENTITY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported trial process identity schema at {path}.")
    attempt_id = payload.get("attempt_id")
    if attempt_id != expected_attempt_id:
        raise ValueError(f"Trial process identity at {path} belongs to another attempt.")

    def positive_int(field: str) -> int:
        """Validate and return one required positive-int identity field.

        Args:
            field: Name of the top-level identity field to validate, used
                only to build the error message.

        Returns:
            The field's value, guaranteed to be a non-bool ``int`` greater
            than zero.

        Raises:
            ValueError: If the field is missing, not an ``int`` (or is a
                ``bool``), or not strictly positive.

        """
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Trial process identity field {field!r} is invalid at {path}.")
        return value

    proc_starttime = payload.get("proc_starttime")
    if proc_starttime is not None and (
        isinstance(proc_starttime, bool)
        or not isinstance(proc_starttime, int)
        or proc_starttime <= 0
    ):
        raise ValueError(f"Trial process identity field 'proc_starttime' is invalid at {path}.")
    boot_id = payload.get("boot_id")
    if boot_id is not None and (not isinstance(boot_id, str) or not boot_id):
        raise ValueError(f"Trial process identity field 'boot_id' is invalid at {path}.")
    return StaleProcessIdentity(
        schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
        attempt_id=attempt_id,
        pid=positive_int("pid"),
        pgid=positive_int("pgid"),
        proc_starttime=proc_starttime,
        boot_id=boot_id,
    )


def cleanup_stale_trial_process(
    identity: StaleProcessIdentity,
    *,
    grace_seconds: float = _KILL_GRACE_SECONDS,
) -> bool:
    """Safely clean a stale trial group using boot- and process-bound identity.

    Refuses to act when ``identity`` lacks a recorded boot id or start time, or
    the current boot id cannot be read, since PID-reuse safety cannot be
    verified in that case. When ``identity.boot_id`` differs from the current
    boot id, the host has rebooted since launch, so no process from that boot
    can still be alive and cleanup is trivially complete. Otherwise delegates
    to :func:`kill_stale_group` with the identity's ``pid``/``pgid``/starttime.

    Args:
        identity: Durable process identity read from ``process_identity.json``.
        grace_seconds: Seconds to wait after SIGTERM before escalating to
            SIGKILL; forwarded to :func:`kill_stale_group`.

    Returns:
        ``True`` when it is safe to mark the trial ``FAIL`` (nothing to clean,
        the host rebooted, or cleanup was confirmed). ``False`` when identity
        cannot be verified or :func:`kill_stale_group` reports cleanup is
        uncertain; callers must not advance state in that case.

    """
    current_boot_id = read_boot_id()
    if identity.boot_id is None or identity.proc_starttime is None or current_boot_id is None:
        log.warning(
            "Refusing automatic cleanup for attempt %s because robust process-birth identity "
            "is unavailable on this platform.",
            identity.attempt_id,
        )
        return False
    if identity.boot_id != current_boot_id:
        log.warning(
            "Attempt %s belongs to an earlier host boot; no process from that boot remains.",
            identity.attempt_id,
        )
        return True
    return kill_stale_group(
        identity.pid,
        identity.proc_starttime,
        pgid=identity.pgid,
        grace_seconds=grace_seconds,
    )


def _terminate_process_group(pgid: int, *, grace_seconds: float) -> bool:
    """Send SIGTERM, wait, then SIGKILL — and confirm the group is actually gone.

    Returns ``True`` only when the process group is confirmed gone. Returns
    ``False`` when delivery fails (permission denied, OS error) or the group
    is still alive after SIGKILL. Pre-v0.5.8 this function returned ``True``
    even when the group survived SIGKILL — callers then marked the trial
    ``FAIL`` and proceeded, potentially launching new trials onto a GPU still
    held by the leaked process (review v0.5.7 / blocker 2).

    ``ProcessLookupError`` from ``killpg`` means the group is already gone, so
    those branches return ``True``.

    Args:
        pgid: Target process-group ID.
        grace_seconds: Seconds to wait after SIGTERM before escalating to SIGKILL.

    Returns:
        ``True`` if the group is confirmed dead (already gone, or died within
        the SIGTERM grace, or died within 2s after SIGKILL). ``False`` if
        signal delivery failed for non-``ProcessLookupError`` reasons, or the
        group is still alive 2s after SIGKILL.

    """
    return _terminate_process_groups((pgid,), grace_seconds=grace_seconds)[pgid]


def _terminate_process_groups(pgids: tuple[int, ...], *, grace_seconds: float) -> dict[int, bool]:
    """Escalate SIGTERM → SIGKILL across process groups with shared wait windows.

    Per-group semantics match :func:`_terminate_process_group`; the SIGTERM
    grace and the post-SIGKILL confirmation window are shared across all
    groups, so total worst-case latency stays roughly ``grace_seconds + 2``
    seconds instead of scaling with the number of live groups. The shutdown
    handler kills every active trial group of an ``n_jobs > 1`` phase through
    this path — trainers that ignore SIGTERM fail correlated, not
    independently, so a serial escalation would multiply the documented
    worst case by the trial parallelism.

    Args:
        pgids: Target process-group IDs.
        grace_seconds: Seconds to wait after SIGTERM before escalating to
            SIGKILL, shared across all groups.

    Returns:
        Confirmation verdict per requested pgid — ``True`` only when that
        group is confirmed dead.

    """
    import time

    confirmed: dict[int, bool] = {}
    members: dict[int, set[int]] = {}
    pending: list[int] = []
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            confirmed[pgid] = True
            continue
        except (PermissionError, OSError) as exc:
            log.error("Failed to send SIGTERM to process group %d: %s", pgid, exc)
            confirmed[pgid] = False
            continue
        members[pgid] = set(_group_member_pids(pgid))
        pending.append(pgid)

    deadline = time.monotonic() + grace_seconds
    while pending and time.monotonic() < deadline:
        still_alive = []
        for pgid in pending:
            if _process_group_alive_with_members(pgid, members[pgid]):
                still_alive.append(pgid)
            else:
                confirmed[pgid] = True
        pending = still_alive
        if pending:
            time.sleep(0.1)

    survivors: list[int] = []
    for pgid in pending:
        log.warning("Stale process group %d survived SIGTERM — sending SIGKILL", pgid)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            confirmed[pgid] = True
            continue
        except (PermissionError, OSError) as exc:
            log.error("Failed to send SIGKILL to process group %d: %s", pgid, exc)
            confirmed[pgid] = False
            continue
        survivors.append(pgid)

    # Wait briefly for the kernel to actually reap descendants so the reaper's
    # "marked FAIL" state means cleanup completed, not requested.
    kill_deadline = time.monotonic() + 2.0
    while survivors and time.monotonic() < kill_deadline:
        still_alive = []
        for pgid in survivors:
            if _process_group_alive_with_members(pgid, members[pgid]):
                still_alive.append(pgid)
            else:
                confirmed[pgid] = True
        survivors = still_alive
        if survivors:
            time.sleep(0.05)

    for pgid in survivors:
        log.error("Process group %d still appears alive after SIGKILL", pgid)
        confirmed[pgid] = False
    return confirmed


def _process_group_alive(pgid: int) -> bool:
    """Check whether any non-zombie process in the group ``pgid`` is alive.

    ``os.killpg(pgid, 0)`` returns success for zombie processes too, because
    they still occupy the PID table. For our purposes a zombie is dead — it
    holds no GPU memory, no file descriptors, no shared resources. Skipping
    zombies stops the cleanup escalation from looping after SIGKILL when the
    parent hasn't reaped its child yet (review v0.5.7 / blocker 2 follow-up).

    On non-Linux, ``/proc/<pid>/stat`` doesn't exist; we fall back to the
    previous behavior (``killpg(0)`` semantics).

    Args:
        pgid: Process-group ID to probe.

    Returns:
        ``True`` if at least one non-zombie member of the group exists.
        ``False`` if the group is gone, or every remaining member is a zombie
        (state ``Z``/``X``).

    """
    return _process_group_alive_with_members(pgid, None)


def _process_group_exists(pgid: int) -> bool:
    """Return whether the process group has any PID-table entry.

    :param int pgid: Process-group ID to probe.
    :return bool: ``True`` when the group exists or exists but is not inspectable.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Group exists but is owned by a different user. Treat as alive.
        return True
    return True


def _stored_pgid_is_reused_group_leader(pgid: int, saved_starttime: int) -> bool | None:
    """Return whether ``pgid`` identifies a new group leader, or is unverifiable.

    A PGID number can later be reused as an ordinary PID in another process
    group; that does not make ``killpg(pgid, ...)`` unsafe because the process
    is not a member of the target group. The unsafe case is narrower: a live
    ``/proc/<pgid>`` entry belongs to process group ``pgid`` but has a different
    starttime, meaning the saved group ID is now led by an unrelated process.

    :param int pgid: Stored process-group ID whose leader identity is checked.
    :param int saved_starttime: Starttime recorded for the original group leader.
    :return bool | None: ``True`` for a reused leader, ``False`` when no live
        leader exists or its identity is compatible, and ``None`` when a live
        leader's proc identity is unreadable.
    """
    stat = _read_proc_stat(Path("/proc") / str(pgid))
    if stat is None:
        return None if is_pid_alive(pgid) else False
    return stat.pgrp == pgid and stat.starttime != saved_starttime


def _process_group_alive_with_members(pgid: int, member_pids: set[int] | None) -> bool:
    """Check group liveness, optionally using and refreshing a cached member set.

    :param int pgid: Process-group ID to probe.
    :param set[int] | None member_pids: Cached group members, or ``None`` to scan once.
    :return bool: ``True`` when a non-zombie member of the group is still alive.
    """
    if not _process_group_exists(pgid):
        return False
    proc_root = Path("/proc")
    if not proc_root.exists():
        return True
    if member_pids is None:
        return _member_pids_alive(pgid, _group_member_pids(pgid))
    if _member_pids_alive(pgid, member_pids):
        return True
    refreshed = set(_group_member_pids(pgid))
    member_pids.clear()
    member_pids.update(refreshed)
    return _member_pids_alive(pgid, member_pids)


def _group_member_pids(pgid: int) -> list[int]:
    """Return current ``/proc`` PIDs that belong to process group ``pgid``.

    :param int pgid: Process-group ID to find under ``/proc``.
    :return list[int]: PIDs currently reporting membership in ``pgid``.
    """
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []
    member_pids: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        stat = _read_proc_stat(entry)
        if stat is None:
            continue
        if stat.pgrp == pgid:
            member_pids.append(int(entry.name))
    return member_pids


def _member_pids_alive(pgid: int, member_pids: set[int] | list[int]) -> bool:
    """Return whether any known member PID is still live and in ``pgid``.

    :param int pgid: Process-group ID each PID must still belong to.
    :param set[int] | list[int] member_pids: Candidate member PIDs to inspect.
    :return bool: ``True`` when any candidate is a live, non-zombie member.
    """
    for pid in member_pids:
        stat = _read_proc_stat(Path("/proc") / str(pid))
        if stat is None:
            continue
        if stat.pgrp != pgid:
            continue
        if stat.state in {"Z", "X"}:
            continue
        return True
    return False


def kill_stale_group(
    pid: int | None,
    saved_starttime: int | None,
    *,
    pgid: int | None = None,
    grace_seconds: float = _KILL_GRACE_SECONDS,
) -> bool:
    """Terminate a stale trial process group, escalating SIGTERM -> SIGKILL.

    Returns ``True`` only when it is safe to mark the trial ``FAIL``: either
    nothing was alive to clean up, or cleanup ran and the process group is
    confirmed gone. Returns ``False`` when cleanup is uncertain — PID/PGID
    identity was reused unsafely, permission was denied, or the group survived
    SIGKILL. Callers must not advance state when this returns ``False``
    (review v0.5.7 / blocker 2): a leaked training process can still hold a
    GPU, scribble over W&B runs, or starve the host scheduler.

    Recovery order (review v0.5.3 / blocker 2):

    1. ``pid`` is alive AND saved starttime matches: derive PGID from the live
       PID and kill the group. Starttime check guards against PID reuse.
    2. ``pid`` is alive but starttime mismatches: this is PID reuse by an
       unrelated process. If the stored PGID proves the old group is gone,
       cleanup is complete. If the PGID still exists, use it only when the
       reused PID is not the leader of that group; otherwise fail closed to
       avoid killing an unrelated process group.
    3. ``pid`` is dead but its verified same-boot identity includes ``pgid``:
       use the group only when its leader identity has not been reused. The
       root shell may have exited while descendants still hold GPU memory.
    4. No PID and no PGID: nothing to clean up, return ``True``.

    Args:
        pid: Root PID from a durable process identity, or ``None`` if absent.
        saved_starttime: Saved Linux process start time, or ``None`` when identity
            cannot be verified. A live PID/PGID is never signalled in that case.
        pgid: Process-group ID from the same durable identity.
        grace_seconds: Seconds to wait after SIGTERM before escalating to
            SIGKILL; passed through to :func:`_terminate_process_group`.

    Returns:
        ``True`` when it is safe to mark the trial ``FAIL`` (cleanup
        confirmed or nothing to clean). ``False`` when cleanup is uncertain
        and callers must NOT advance state.

    """
    target_pgid: int | None = None

    if saved_starttime is None:
        pid_alive = pid is not None and is_pid_alive(pid)
        pgid_alive = pgid is not None and _process_group_exists(pgid)
        if pid_alive or pgid_alive:
            log.warning(
                "Refusing to signal stale pid=%s pgid=%s without a saved process start time.",
                pid,
                pgid,
            )
            return False
        return True

    if pid is not None:
        pid_alive = is_pid_alive(pid)
        same_process = False
        if pid_alive:
            current_starttime = read_proc_starttime(pid)
            if current_starttime is None:
                log.warning(
                    "PID %d is alive but its /proc start time is unreadable; "
                    "refusing cleanup because process identity is unknown.",
                    pid,
                )
                return False
            same_process = current_starttime == saved_starttime
        if same_process:
            try:
                target_pgid = os.getpgid(pid)
            except ProcessLookupError:
                target_pgid = None
            except (PermissionError, OSError) as exc:
                log.error("Failed reading PGID for stale PID %d: %s", pid, exc)
                return False
        elif pid_alive:
            # PID reuse detected. A persisted PGID can still prove cleanup is
            # complete (group gone) or target original descendants (group alive
            # but not led by the reused PID). Refuse only when the stored PGID
            # itself appears to be a reused group leader.
            if pgid is not None and not _process_group_exists(pgid):
                log.warning(
                    "PID %d is alive but starttime does not match saved value; "
                    "PID was reused, but stored process group %d no longer "
                    "exists.",
                    pid,
                    pgid,
                )
                return True
            if pgid is None:
                log.warning(
                    "PID %d is alive but starttime does not match saved value; "
                    "PID was reused and no stored PGID is available. Cleanup "
                    "status is uncertain.",
                    pid,
                )
                return False
            pgid_reused = _stored_pgid_is_reused_group_leader(pgid, saved_starttime)
            if pgid_reused is None:
                log.warning(
                    "Stored PGID %d has a live but unreadable leader; refusing "
                    "PGID fallback because process identity is unknown.",
                    pgid,
                )
                return False
            if pgid_reused:
                log.warning(
                    "PID %d is alive but starttime does not match saved value; "
                    "stored PGID %d is led by a different process. Refusing "
                    "PGID fallback to avoid killing an unrelated process group.",
                    pid,
                    pgid,
                )
                return False

    if target_pgid is None and pgid is not None and saved_starttime is not None:
        if not _process_group_exists(pgid):
            return True
        pgid_reused = _stored_pgid_is_reused_group_leader(pgid, saved_starttime)
        if pgid_reused is None:
            log.warning(
                "Stored PGID %d has a live but unreadable leader; refusing PGID "
                "fallback because process identity is unknown.",
                pgid,
            )
            return False
        if pgid_reused:
            log.warning(
                "Stored PGID %d is now led by a different process. Refusing "
                "PGID fallback to avoid killing an unrelated process group "
                "(saved starttime %s).",
                pgid,
                saved_starttime,
            )
            return False

    if target_pgid is None:
        if pgid is None:
            # No identity at all → nothing alive to clean up.
            return True
        log.warning(
            "Root PID is gone, reused, or unrecoverable; using stored PGID %d for "
            "best-effort cleanup of stale trial process group.",
            pgid,
        )
        target_pgid = pgid

    if not _process_group_alive(target_pgid):
        return True

    log.warning("Terminating stale training process group pgid=%d (pid=%s)", target_pgid, pid)
    return _terminate_process_group(target_pgid, grace_seconds=grace_seconds)
