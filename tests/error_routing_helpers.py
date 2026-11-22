"""Shared setup and assertions for the operator-action routing contract tests.

``tests/test_error_routing.py`` and ``tests/test_error_routing_mcp.py`` drive real raise sites
through the same case record and check them with the same assertions, so both import them from here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

import phasesweep.engine.ledger as engine_ledger
from phasesweep.cli import _operational_error_text
from phasesweep.config.models import Experiment
from phasesweep.engine.attempts import _register_active_attempt
from phasesweep.engine.paths import _attempts_dir
from phasesweep.errors import OperatorAction, PhaseSweepError, ProcessCleanupUncertainError
from phasesweep.mcp import runner as mcp_runner
from tests.ledger_fixtures import Materialized, materialize

Trigger = Callable[[Path, pytest.MonkeyPatch], object]


@dataclass(frozen=True)
class RoutingCase:
    """One real raise site and the routing an operator must receive from it."""

    id: str
    #: Drives the real entry point until the raise under test.
    trigger: Trigger
    raised: type[PhaseSweepError]
    action: OperatorAction
    #: Stable substring of the operator text the site raises today.
    message: str
    #: Exact ``__cause__`` type, pinned only where the runner routes by it.
    cause: type[BaseException] | None = None
    marks: tuple[pytest.MarkDecorator, ...] = ()


def raiser(error: BaseException) -> Callable[..., object]:
    """Return a stand-in that raises ``error`` however it is called."""

    def fail(*_args: object, **_kwargs: object) -> object:
        raise error

    return fail


def garbage(path: Path) -> None:
    """Overwrite a ledger file with bytes that fail to parse as journal records."""
    path.write_bytes(b"this is not a journal record\n" * 64)


def truncated(path: Path) -> None:
    """Cut the journal's last record short, as an interrupted append leaves it."""
    path.write_bytes(path.read_bytes()[:-8])


def materialize_damaged(
    tmp_path: Path, fixture: str, mode: str, damage: Callable[[Path], None]
) -> Materialized:
    """Copy a current golden ledger and apply ``damage`` to its ledger file."""
    materialized = materialize(fixture, tmp_path, mode=mode)
    damage(materialized.ledger_path)
    return materialized


@contextmanager
def read_only_once_locked(journal: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Acquire the journal inode, then deny creation of its repair backup.

    Restore the directory mode on exit so the tree can be cleaned up.
    """
    real_acquire = engine_ledger._JournalLock.acquire
    mode = journal.parent.stat().st_mode

    def acquire_then_deny(lock: engine_ledger._JournalLock) -> bool:
        result = real_acquire(lock)
        journal.parent.chmod(0o555)
        return result

    monkeypatch.setattr(engine_ledger._JournalLock, "acquire", acquire_then_deny)
    try:
        yield
    finally:
        journal.parent.chmod(mode)


def registered_entry(experiment: Experiment, trial_dir: Path) -> Path:
    """Register one attempt of phase ``p`` and return its registry entry file."""
    _register_active_attempt(
        experiment,
        attempt_id="attempt",
        phase_name="p",
        study_name="t::p",
        trial_number=0,
        trial_dir=trial_dir,
        generation_id="generation",
    )
    return _attempts_dir(experiment) / "attempt.json"


def assert_routes_declared_action(
    case: RoutingCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Drive a case's raise site and check the action, text and runner payload it yields.

    :param RoutingCase case: The raise site and the routing it must produce.
    :param Path tmp_path: The test's scratch directory.
    :param pytest.MonkeyPatch monkeypatch: The test's monkeypatch fixture.
    :param pytest.LogCaptureFixture caplog: The test's log capture, which must not hold the remedy.
    """
    with pytest.raises(case.raised) as excinfo:
        case.trigger(tmp_path, monkeypatch)

    error = excinfo.value
    assert type(error) is case.raised
    assert error.action is case.action
    if case.cause is not None:
        assert type(error.__cause__) is case.cause
    assert case.message in _operational_error_text(error)
    if error.remediation is not None:
        rendered = _operational_error_text(error)
        assert rendered.split("\nfix: ", 1)[1] == error.remediation
        assert "correct the experiment configuration" not in rendered
        assert rendered.count(error.remediation) == 1
        assert error.remediation not in str(error)
        assert error.remediation not in caplog.text
        failure = mcp_runner._terminal_failure_payload(
            error, run_id="r0", stage="execution", cleanup_confirmed=True
        )
        if isinstance(error, ProcessCleanupUncertainError):
            # A cleanup-uncertain failure routes its repair and then recover-run, and points at
            # the run log, which holds the step the error carries.
            assert failure["code"] == "cleanup_uncertain"
        else:
            expected = (
                error.remediation
                if error.action is OperatorAction.RETRY
                else f"Ask the operator: {error.remediation} {mcp_runner._run_log_details('r0')}"
            )
            assert failure["remediation"] == expected
        if case.id.startswith("live_abort_with_"):
            # The run log records the failure itself and then its step, once.
            logged = (tmp_path / "runs" / "t" / "run.log").read_text(encoding="utf-8")
            assert logged.count(error.remediation) == 1
            assert logged.index(str(error)) < logged.index(f"fix: {error.remediation}")
    if case.action is OperatorAction.RETRY:
        rendered = _operational_error_text(error)
        if error.remediation is None:
            assert "fix: Retry once the reported blocker clears." in rendered
        else:
            assert error.remediation in rendered
        assert "start a new run" not in rendered


_STEP_MARKERS: Mapping[OperatorAction, str] = MappingProxyType(
    {
        OperatorAction.FRESH_NAMESPACE: "set a new `experiment` name to start",
        OperatorAction.RESTORE_LEDGER: "storage ledger",
        OperatorAction.RESTORE_TREE: "restore the files and permissions identified in the error",
        OperatorAction.PROVISION_LOCK_DIR: "point PHASESWEEP_LOCK_DIR at an existing writable",
        OperatorAction.RUN_RECOVER_RUN: "recover-run",
        OperatorAction.FIX_CONFIG: "correct the experiment configuration",
        OperatorAction.INSPECT_LOGS: "inspect `logs/",
    }
)


def assert_payload_follows(failure: Mapping[str, object], routed: list[OperatorAction]) -> None:
    """Assert a runner failure payload says exactly the ``routed`` steps, in order.

    :param Mapping[str, object] failure: The runner's terminal failure payload.
    :param list[OperatorAction] routed: The actions the payload must name, in order.
    """
    remediation = failure["remediation"]
    assert isinstance(remediation, str), remediation
    steps = [step for step in routed if step is not OperatorAction.RETRY]
    if failure["code"] == "cleanup_uncertain":
        # The server launches nothing more for such a run until recover-run
        # confirms it, so recover-run is how it is retried.
        last = OperatorAction.RUN_RECOVER_RUN
        steps = [*(step for step in steps if step is not last), last]
    if steps:
        assert (failure["retryable"], failure["actor"]) == (False, "operator")
    else:
        assert (failure["retryable"], failure["actor"]) == (True, "agent")
    positions = [remediation.find(_STEP_MARKERS[step]) for step in steps]
    assert -1 not in positions, remediation
    assert positions == sorted(positions), remediation
    unrouted = [marker for action, marker in _STEP_MARKERS.items() if action not in steps]
    assert not [marker for marker in unrouted if marker in remediation], remediation
    # A step whose specifics are in the message points at the run log holding it.
    self_contained = {OperatorAction.RUN_RECOVER_RUN, OperatorAction.INSPECT_LOGS}
    points_at_log = " Details: `logs/" in remediation
    assert points_at_log is not self_contained.issuperset(steps), remediation


def assert_runner_payload_follows_routing(errors: Iterable[PhaseSweepError]) -> None:
    """Assert each error's preflight runner payload says the step its own action routes.

    :param Iterable[PhaseSweepError] errors: Errors whose routed action the payload must follow.
    """
    for error in errors:
        failure = mcp_runner._terminal_failure_payload(
            error, run_id="r0", stage="preflight", cleanup_confirmed=True
        )
        assert_payload_follows(failure, [error.action])
