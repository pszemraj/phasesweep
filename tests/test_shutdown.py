"""``phasesweep.runtime.shutdown``: signal-handler scope, deferral, absorption, restoration and ownership tokens."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from phasesweep import run_experiment
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.runtime import shutdown as runtime_shutdown
from phasesweep.runtime.shutdown import (
    PhaseSweepShutdown,
    SignalOwnershipUnavailableError,
    _shutdown_handler,
    defer_shutdown_signals,
    install_signal_handlers,
    signal_handler_scope,
)
from tests.conftest import make_experiment, write_constant_trainer, write_trainer


@pytest.mark.integration
def test_signal_handler_scope_restores_host_signal_state_on_success_and_failure(
    tmp_path: Path,
) -> None:
    """run_experiment restores the host's prior signal handlers and mask on every exit path.

    A library that leaves its own SIGTERM/SIGINT/SIGHUP handlers and unblocked
    mask installed after returning steals the embedding process's own
    shutdown handling permanently (review v0.5.14 / blocker 6). This must be
    undone whether the run succeeds or raises.
    """

    def host_handler(_signum: int, _frame: object) -> None:
        raise AssertionError("host handler should never fire during this test")

    def assert_host_state_active() -> None:
        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is host_handler
        current_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        assert set(runtime_shutdown._SHUTDOWN_SIGNALS) <= current_mask

    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_shutdown._SHUTDOWN_SIGNALS}
    prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    try:
        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            signal.signal(sig, host_handler)
        signal.pthread_sigmask(signal.SIG_BLOCK, set(runtime_shutdown._SHUTDOWN_SIGNALS))

        trainer = write_constant_trainer(tmp_path)
        experiment = make_experiment(workdir=tmp_path / "runs", trainer=trainer, n_trials=1)
        run_experiment(experiment)
        assert_host_state_active()

        failing_trainer = write_trainer(tmp_path / "failing.py", "raise SystemExit(1)")
        failing_experiment = make_experiment(
            experiment="fails",
            workdir=tmp_path / "runs",
            trainer=failing_trainer,
            n_trials=1,
            max_consecutive_failures=1,
        )
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(failing_experiment)
        assert_host_state_active()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


@pytest.mark.integration
@pytest.mark.signals_own_pid
def test_signal_handler_scope_delivers_pending_signal_to_host_handler_after_restore() -> None:
    """A signal pending at scope-exit must reach the restored HOST handler, not
    ``_shutdown_handler`` mid-restoration (review v0.5.15 / blocker 2A).

    Pre-fix, the mask was restored before the host handlers, so a pending
    SIGTERM fired while ``_shutdown_handler`` was still installed for it,
    raising ``PhaseSweepShutdown`` out of the cleanup path and leaving some
    host handlers unrestored. The fixed order is: block, then restore
    handlers, then restore the mask.
    """
    if not hasattr(signal, "pthread_sigmask"):
        pytest.skip("pthread_sigmask not available")

    received: list[int] = []

    def host_handler(signum: int, _frame: object) -> None:
        received.append(signum)

    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_shutdown._SHUTDOWN_SIGNALS}
    prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    try:
        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            signal.signal(sig, host_handler)

        with signal_handler_scope():
            # Block SIGTERM ourselves so the kill below queues instead of
            # firing immediately (the scope's own entry unblocks it).
            signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGTERM,))
            os.kill(os.getpid(), signal.SIGTERM)
            assert received == [], "signal fired before scope exit"

        # Scope exit order: block -> restore handlers -> restore mask. The pending
        # SIGTERM is delivered on the final unblock, by which point the HOST handler
        # (not phasesweep's) is installed. CPython only invokes the Python-level handler
        # at the next eval-breaker check, not necessarily synchronously with the
        # unblocking call, so poll briefly instead of asserting immediately.
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.001)
        assert received == [signal.SIGTERM]
        assert signal.getsignal(signal.SIGTERM) is host_handler
        assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == prior_mask
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_signal_handler_scope_continues_restoring_after_one_signal_signal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failed ``signal.signal`` restore must not abort the rest of the loop.

    Every other prior handler must still be restored, and the first
    restoration error surfaces once the scope body itself did not already
    raise (review v0.5.15 / blocker 2A).
    """
    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_shutdown._SHUTDOWN_SIGNALS}
    try:

        def make_sentinel(tag: int):
            def handler(signum: int, _frame: object) -> None:
                return None

            handler.__name__ = f"sentinel_{tag}"
            return handler

        sentinel_handlers = {sig: make_sentinel(sig) for sig in runtime_shutdown._SHUTDOWN_SIGNALS}
        for sig, handler in sentinel_handlers.items():
            signal.signal(sig, handler)

        first_signal = runtime_shutdown._SHUTDOWN_SIGNALS[0]
        real_signal = signal.signal
        failed_once: set[int] = set()

        def flaky_signal(signalnum: int, handler: object) -> object:
            if signalnum == first_signal and signalnum not in failed_once:
                failed_once.add(signalnum)
                raise OSError("simulated restore failure")
            return real_signal(signalnum, handler)

        with (
            pytest.raises(OSError, match="simulated restore failure"),
            monkeypatch.context() as scoped,
            signal_handler_scope(),
        ):
            # Patch only after the scope's entry-time installation used the real
            # signal.signal; the patch ends with this block, before the restore below.
            scoped.setattr(signal, "signal", flaky_signal)

        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            if sig == first_signal:
                # Restoration failed for this one; phasesweep's handler is
                # still installed until manual cleanup below.
                assert signal.getsignal(sig) is runtime_shutdown._shutdown_handler
                continue
            assert signal.getsignal(sig) is sentinel_handlers[sig]
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def _worker_errors(operation: Callable[[], None]) -> list[BaseException]:
    """Run one operation on a worker and return captured failures."""
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            operation()
        except BaseException as exc:  # noqa: BLE001 - returned for the main thread to assert on
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    return errors


def _enter_signal_handler_scope() -> None:
    """Enter and leave one signal-handler scope."""
    with signal_handler_scope():
        pass


def test_signal_handler_scope_raises_off_main_thread_without_prior_install() -> None:
    """Off the main thread, with nothing already owning shutdown signals, the scope refuses.

    ``signal.signal`` only works on the main thread, so a scope entered from a
    worker thread with no enclosing install cannot safely take ownership; it
    must raise a typed error instead of silently running unprotected.
    """
    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_shutdown._SHUTDOWN_SIGNALS}
    try:
        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            if signal.getsignal(sig) is runtime_shutdown._shutdown_handler:
                signal.signal(sig, signal.SIG_DFL)

        errors = _worker_errors(_enter_signal_handler_scope)

        assert len(errors) == 1
        assert isinstance(errors[0], SignalOwnershipUnavailableError)
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_signal_handler_scope_is_noop_once_process_lifetime_install_owns_signals() -> None:
    """A process-lifetime ``install_signal_handlers()`` call is never undone by a nested scope.

    Entry points (CLI, MCP server) install shutdown handlers once for the
    whole process. A later ``signal_handler_scope()`` — even from a worker
    thread, where taking ownership from scratch would be impossible — must
    see that ownership is already established and do nothing, on entry or
    exit.
    """
    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_shutdown._SHUTDOWN_SIGNALS}
    try:
        install_signal_handlers()
        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is runtime_shutdown._shutdown_handler

        errors = _worker_errors(_enter_signal_handler_scope)

        assert errors == []
        # Entry-point ownership persists: the nested scope did not tear it down.
        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is runtime_shutdown._shutdown_handler
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_install_signal_handlers_inside_open_scope_survives_that_scope_exit() -> None:
    """An ``install_signal_handlers()`` call inside an open scope must outlive the scope.

    ``install_signal_handlers()`` recognizes an already-installed handler set
    as its own idempotent path, so calling it while a
    ``signal_handler_scope()`` is open took process-lifetime ownership on the
    strength of the *scope's* installation. The scope then restored the host's
    handlers on exit while ownership stayed claimed, so every later scope
    no-opped with nothing installed and child process groups leaked on
    shutdown. The scope now hands its installation over instead of restoring.
    """

    def host_handler(_signum: int, _frame: object) -> None:
        raise AssertionError("host handler should never fire during this test")

    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_shutdown._SHUTDOWN_SIGNALS}
    prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    try:
        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            signal.signal(sig, host_handler)
        signal.pthread_sigmask(signal.SIG_BLOCK, set(runtime_shutdown._SHUTDOWN_SIGNALS))

        with signal_handler_scope():
            install_signal_handlers()

        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is runtime_shutdown._shutdown_handler
        assert not set(runtime_shutdown._SHUTDOWN_SIGNALS) & signal.pthread_sigmask(
            signal.SIG_BLOCK, set()
        )

        # The ownership claim is now truthful, so a later no-op scope is safe.
        with signal_handler_scope():
            pass
        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is runtime_shutdown._shutdown_handler
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_worker_thread_install_cannot_steal_scope_ownership() -> None:
    """A worker-thread ``install_signal_handlers()`` raises and takes nothing.

    During an open main-thread ``signal_handler_scope()`` every shutdown
    handler already points at ``_shutdown_handler``, so a worker thread used
    to take the idempotent fast path, flip ``_process_lifetime_owner``, and
    silently convert the scope's temporary installation into permanent
    process ownership — the scope exit then skipped restoring the host's
    handlers (review v0.5.16 / blocker 5). The install must now reject
    off-main-thread callers before touching any ownership state.
    """

    def host_handler(_signum: int, _frame: object) -> None:
        raise AssertionError("host handler should never fire during this test")

    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_shutdown._SHUTDOWN_SIGNALS}
    try:
        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            signal.signal(sig, host_handler)

        with signal_handler_scope():
            errors = _worker_errors(install_signal_handlers)
            assert not runtime_shutdown._process_lifetime_owner

        assert len(errors) == 1
        assert isinstance(errors[0], SignalOwnershipUnavailableError)
        assert not runtime_shutdown._process_lifetime_owner
        # The scope's exit restored the host's handlers because no legitimate
        # process-lifetime handover happened.
        for sig in runtime_shutdown._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is host_handler
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


@pytest.mark.integration
@pytest.mark.signals_own_pid
def test_absorb_shutdown_signals_reports_signal_and_defers_it_to_next_checkpoint() -> None:
    """A shutdown inside an absorb window is reported, not raised — then honored later.

    The publication transaction uses this to win its race against a shutdown signal
    deterministically (review v0.5.16 / blocker 1): the window exit reports the absorbed signal on
    the yielded object instead of raising, and the next ``defer_shutdown_signals()`` exit (e.g. the
    next trial launch) still delivers the shutdown before new work starts.
    """
    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_shutdown._SHUTDOWN_SIGNALS}
    try:
        install_signal_handlers()

        with runtime_shutdown.absorb_shutdown_signals() as absorbed:
            os.kill(os.getpid(), signal.SIGTERM)
            # Give an unblocked sibling thread's delivery path (if any) a
            # chance to run the Python-level handler; either delivery route
            # must end up recorded, never raised, inside the window.
            time.sleep(0.05)

        assert absorbed.signum == signal.SIGTERM

        # The absorbed signal is still pending: the next deferral checkpoint
        # delivers it before any new work could start.
        with (
            pytest.raises(runtime_shutdown.PhaseSweepShutdown) as exc_info,
            runtime_shutdown.defer_shutdown_signals(),
        ):
            pass
        assert exc_info.value.signum == signal.SIGTERM
    finally:
        runtime_shutdown._deferred_shutdown_signum = None
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_service_pending_shutdown_is_noop_without_absorbed_signal() -> None:
    """The explicit checkpoint does nothing when no shutdown was absorbed."""
    assert runtime_shutdown.service_pending_shutdown() is None


def test_stale_process_lifetime_claim_is_reasserted_on_scope_entry() -> None:
    """A scope entered under a stale ownership claim reinstalls the OS handlers.

    ``_process_lifetime_owner`` is a Python-side boolean; another library can
    re-bind a shutdown signal after the entry point installed. A later
    main-thread scope must notice the divergence and reinstall phasesweep's
    handler so the run does not silently execute without child-group cleanup.
    """
    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_shutdown._SHUTDOWN_SIGNALS}
    try:
        install_signal_handlers()
        interloper_sig = runtime_shutdown._SHUTDOWN_SIGNALS[0]
        signal.signal(interloper_sig, signal.SIG_IGN)

        with signal_handler_scope():
            assert signal.getsignal(interloper_sig) is runtime_shutdown._shutdown_handler
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def _install_signal_probe(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """Patch ``signal_handler_scope`` so tests can observe whether it was entered.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to replace the scope.
    :return dict[str, bool]: Mutable probe; ``"called"`` flips to ``True`` on entry.
    """
    installed = {"called": False}

    @contextlib.contextmanager
    def fake_scope():
        installed["called"] = True
        yield

    monkeypatch.setattr("phasesweep.engine.run.signal_handler_scope", fake_scope)
    return installed


@pytest.mark.integration
def test_public_run_experiment_enters_signal_handler_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Library callers using ``run_experiment`` directly get the same cleanup
    contract as CLI callers."""
    installed = _install_signal_probe(monkeypatch)

    # Minimal trial_command that emits the metric captured by the log extractor.
    script = "print('x=1')"
    exp = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f'{sys.executable} -c "{script}" trial_dir={{trial_dir}} {{overrides}}',
        override_format="argparse",
    )
    run_experiment(exp)
    assert installed["called"] is True


@pytest.mark.integration
def test_dry_run_does_not_enter_signal_handler_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry-run launches no children, so it must not perturb the signal mask."""
    installed = _install_signal_probe(monkeypatch)

    exp = make_experiment(workdir=tmp_path / "runs")
    run_experiment(exp, dry_run=True)
    assert installed["called"] is False


@pytest.mark.integration
def test_defer_shutdown_signals_blocks_and_restores() -> None:
    """The context manager must add SIGTERM/SIGINT to the thread mask on entry
    and restore the original mask on exit."""
    if not hasattr(signal, "pthread_sigmask"):
        pytest.skip("pthread_sigmask not available")

    before = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    try:
        with defer_shutdown_signals():
            inside = signal.pthread_sigmask(signal.SIG_BLOCK, set())
            assert signal.SIGTERM in inside
            assert signal.SIGINT in inside
        after = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        assert after == before
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, before)


@pytest.mark.integration
def test_install_signal_handlers_unblocks_inherited_shutdown_mask() -> None:
    """Startup should recover if the orchestrator inherited blocked SIGTERM."""
    if not hasattr(signal, "pthread_sigmask"):
        pytest.skip("pthread_sigmask not available")

    code = r"""
import os, signal
from phasesweep.runtime.shutdown import install_signal_handlers, defer_shutdown_signals
signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGTERM,))
install_signal_handlers()
with defer_shutdown_signals():
    print("queued", flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
print("post-context", flush=True)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=5.0,
        check=False,
    )
    assert "queued" in proc.stdout
    assert proc.returncode == 128 + signal.SIGTERM


@pytest.mark.integration
def test_pending_sigterm_inside_signal_deferred_sections_does_not_deadlock() -> None:
    """Queued SIGTERM must not deadlock while launch or registry locks are held."""
    cases = [
        (
            "launch_lock",
            "queued",
            r"""
import os, signal
from phasesweep.runtime.shutdown import install_signal_handlers, defer_shutdown_signals, _launch_lock
install_signal_handlers()
with defer_shutdown_signals(), _launch_lock:
    print("queued", flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
print("post-context", flush=True)
""",
        ),
        (
            "registry_lock",
            "locked",
            r"""
import os, signal
from phasesweep.runtime.shutdown import install_signal_handlers, defer_shutdown_signals, _lock
install_signal_handlers()
with defer_shutdown_signals():
    with _lock:
        print("locked", flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
print("post-context", flush=True)
""",
        ),
    ]

    for case, marker, code in cases:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            timeout=5.0,
            check=False,
        )
        assert marker in proc.stdout, case
        assert proc.returncode == 128 + signal.SIGTERM, case


@pytest.mark.integration
def test_sigterm_via_worker_thread_mid_launch_window_defers_instead_of_deadlocking() -> None:
    """A signal tripped by a non-main thread mid-window must defer, not deadlock.

    Kernel masking in ``defer_shutdown_signals`` only covers the main thread.
    Library pools (e.g. BLAS workers pulled in via numpy/optuna) keep SIGTERM
    unblocked, so a process-directed SIGTERM sent during the masked launch
    window is delivered to one of them — and CPython then runs the Python
    handler in the main thread anyway, mid-critical-section. Pre-fix the
    handler re-acquired ``_launch_lock`` held by that same thread and hung
    until the MCP server's 30s grace SIGKILLed the runner with no status.json
    written (the flaky-cancel e2e failures). The handler must record the signal
    and let the window exit service it.
    """
    code = r"""
import os, signal, threading, time
import phasesweep.runtime.shutdown as shutdown_mod
from phasesweep.runtime.shutdown import install_signal_handlers, defer_shutdown_signals, _launch_lock

install_signal_handlers()

# Stand-in for a BLAS pool worker: SIGTERM stays unblocked here, so the kernel
# delivers the process-directed signal to this thread while the main thread is
# masked inside the launch window.
ready = threading.Event()
def helper():
    ready.set()
    threading.Event().wait(30)
threading.Thread(target=helper, daemon=True).start()
assert ready.wait(5)

with defer_shutdown_signals(), _launch_lock:
    os.kill(os.getpid(), signal.SIGTERM)
    deadline = time.time() + 5
    while shutdown_mod._deferred_shutdown_signum is None and time.time() < deadline:
        time.sleep(0.005)
    print("recorded-mid-window" if shutdown_mod._deferred_shutdown_signum is not None
          else "never-recorded", flush=True)
print("post-context", flush=True)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=15.0,
        check=False,
    )
    assert "recorded-mid-window" in proc.stdout, proc.stdout + proc.stderr
    assert "post-context" not in proc.stdout  # window exit must raise the shutdown
    assert proc.returncode == 128 + signal.SIGTERM


@pytest.mark.integration
def test_deferred_shutdown_services_at_outermost_window_exit() -> None:
    """A shutdown recorded mid-window fires only when the outermost window exits."""
    inner_exited = False
    with pytest.raises(PhaseSweepShutdown) as excinfo, defer_shutdown_signals():
        with defer_shutdown_signals():
            # Emulates CPython invoking the handler in the main thread
            # after a worker-thread delivery: it must record and return.
            _shutdown_handler(signal.SIGTERM, None)
        inner_exited = True
    assert inner_exited, "inner window exit must not service the deferred shutdown"
    assert excinfo.value.code == 128 + signal.SIGTERM
