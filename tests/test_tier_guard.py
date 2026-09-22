"""The integration-tier scanner that ``conftest.pytest_collection_modifyitems`` enforces.

Each source below is synthetic text handed straight to ``scan_module``; nothing
here imports or executes it. Keeping the probes as module-level constants also
keeps this module out of its own scanner's results: the rule (b) string probe
would otherwise match the literal embedded in a test body.
"""

from __future__ import annotations

from pathlib import Path

from tests.tiers import scan_module

DIRECT_POPEN_SOURCE = """
import subprocess
import sys


def test_spawns_a_child():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    assert proc.wait() == 0
"""

HELPER_POPEN_SOURCE = """
import subprocess
import sys


def _spawn_worker():
    return subprocess.Popen([sys.executable, "-c", "pass"])


def test_reaches_popen_through_a_helper():
    assert _spawn_worker().wait() == 0
"""

TRAINER_STRING_SLEEP_SOURCE = """
def test_trainer_script_waits(tmp_path):
    trainer = tmp_path / "trainer.py"
    trainer.write_text("import time; time.sleep(5)\\n")
    assert trainer.exists()
"""

SESSION_LEADER_SOURCE = """
import subprocess
import sys


def test_becomes_a_session_leader():
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    assert proc.wait() == 0
"""

START_NEW_SESSION_ONLY_SOURCE = """
def test_forwards_a_session_request(spawn):
    assert spawn(start_new_session=True) is not None
"""

CLEAN_SOURCE = """
from phasesweep.engine.state import Winner


def _build_winner():
    return Winner(phase="p", params={"x": 1})


def test_winner_round_trips():
    assert _build_winner().phase == "p"
"""

MONOTONIC_CLOCK_SOURCE = """
import time


def _elapsed(start):
    return time.monotonic() - start


def test_elapsed_uses_a_monotonic_clock(monkeypatch):
    start = time.perf_counter()
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    assert _elapsed(start) >= 0.0
"""


def test_tier_scanner_flags_process_and_sleep_primitives():
    assert "subprocess.Popen" in scan_module(DIRECT_POPEN_SOURCE)["test_spawns_a_child"]

    helper_reason = scan_module(HELPER_POPEN_SOURCE)["test_reaches_popen_through_a_helper"]
    assert "subprocess.Popen" in helper_reason
    assert "_spawn_worker" in helper_reason, "the helper chain must name where the spawn lives"

    assert "string literal" in scan_module(TRAINER_STRING_SLEEP_SOURCE)["test_trainer_script_waits"]

    assert scan_module(SESSION_LEADER_SOURCE)["test_becomes_a_session_leader"]
    assert (
        "start_new_session"
        in scan_module(START_NEW_SESSION_ONLY_SOURCE)["test_forwards_a_session_request"]
    )

    assert scan_module(CLEAN_SOURCE) == {}, "pure unit tests must stay in the fast tier"

    # The guard module itself must survive its own rules, or every probe added
    # here would have to be marked as an integration test.
    assert scan_module(Path(__file__).read_text(encoding="utf-8")) == {}


def test_tier_scanner_ignores_monotonic_clock_reads():
    # Measuring elapsed time is not spending it: monotonic/perf_counter reads
    # are exactly how the fast tier fakes clocks instead of waiting on one.
    assert scan_module(MONOTONIC_CLOCK_SOURCE) == {}
