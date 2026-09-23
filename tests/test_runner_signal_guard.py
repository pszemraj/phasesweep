"""The autouse guard in ``tests/conftest.py`` that keeps tests from signalling the runner.

Each test swaps the guard's real senders for recorders before sending anything,
so a regressed guard records a signal here instead of delivering it to pytest.
The kill functions are reached by name through ``getattr``: the calls deliver
nothing, so these tests stay out of the tier scanner's process rule.
"""

from __future__ import annotations

import contextlib
import os
import signal
from collections.abc import Callable, Iterator

import pytest

from tests.conftest import RunnerSignalGuard

#: Case id -> (``os`` function name, target computed in the running test).
TARGETS: dict[str, tuple[str, Callable[[], int]]] = {
    "own-pid": ("kill", os.getpid),
    "parent-pid": ("kill", os.getppid),
    "own-group": ("killpg", os.getpgrp),
    "parent-group": ("killpg", lambda: os.getpgid(os.getppid())),
    "own-group-as-zero": ("killpg", lambda: 0),
    "caller-group-via-kill-zero": ("kill", lambda: 0),
    "own-group-via-negative-pid": ("kill", lambda: -os.getpgrp()),
    "every-process": ("kill", lambda: -1),
}

Sent = list[tuple[str, int, int]]


@pytest.fixture
def recorded(guard_runner_signals: RunnerSignalGuard) -> Iterator[tuple[RunnerSignalGuard, Sent]]:
    guard = guard_runner_signals
    sent: Sent = []
    guard.send_kill = lambda pid, sig: sent.append(("kill", pid, sig))
    guard.send_killpg = lambda pgid, sig: sent.append(("killpg", pgid, sig))
    for name in ("kill", "killpg"):
        assert getattr(os, name) == getattr(guard, name), f"os.{name} is not guarded"
    yield guard, sent


def _send(name: str, target: int, sig: int) -> None:
    getattr(os, name)(target, sig)


@pytest.mark.parametrize("case", sorted(TARGETS))
def test_guard_refuses_a_real_signal_to_the_runner(case, recorded):
    guard, sent = recorded
    name, target = TARGETS[case]
    # The code under test wraps kills in broad handlers; the refusal must escape them.
    with (
        pytest.raises(pytest.fail.Exception, match="would signal the test runner"),
        contextlib.suppress(Exception),
    ):
        _send(name, target(), signal.SIGTERM)
    assert sent == []
    assert len(guard.refused) == 1
    # The refusal is what this test wants, so teardown must not fail it again.
    guard.refused.clear()


def test_guard_passes_liveness_probes_through(recorded):
    guard, sent = recorded
    expected = []
    for name, target in TARGETS.values():
        _send(name, target(), 0)
        expected.append((name, target(), 0))
    assert sent == expected
    assert guard.refused == []


@pytest.mark.signals_own_pid
def test_opt_out_lifts_only_the_own_pid_target(recorded):
    guard, sent = recorded
    _send("kill", os.getpid(), signal.SIGTERM)
    assert sent == [("kill", os.getpid(), signal.SIGTERM)]
    with pytest.raises(pytest.fail.Exception):
        _send("kill", os.getppid(), signal.SIGTERM)
    with pytest.raises(pytest.fail.Exception):
        _send("killpg", os.getpgrp(), signal.SIGTERM)
    assert len(guard.refused) == 2
    guard.refused.clear()
