"""The integration-tier scanner and slow-test report that ``tests/conftest.py`` applies.

Each source below is synthetic text handed straight to ``scan_module``; nothing
here imports or executes it. Keeping the probes as module-level constants also
keeps this module out of its own scanner's results: the rule (b) string probe
would otherwise match the literal embedded in a test body.

Every rule has a probe that only that rule flags, so a rule that stops firing
fails a named case here instead of silently letting a slow test into the fast
tier.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.tiers import (
    PROCESS_DRIVER_CALLS,
    SLOW_CALL_SECONDS,
    WALLCLOCK_HELPER_CALLS,
    excludes_integration,
    scan_helpers,
    scan_module,
    slow_unmarked,
)

#: Rule id -> (probe source defining ``test_probe``, fragment its reason must contain).
RULE_PROBES: dict[str, tuple[str, str]] = {
    "subprocess-attribute": (
        "import subprocess\n"
        "import sys\n\n"
        "def test_probe():\n"
        "    assert subprocess.Popen([sys.executable, '-c', 'pass']).wait() == 0\n",
        "references subprocess.Popen",
    ),
    "module-alias": (
        "import subprocess as sp\n\ndef test_probe():\n    sp.run(['true'])\n",
        "references subprocess.run",
    ),
    "from-import-alias": (
        "from subprocess import check_output as co\n\ndef test_probe():\n    co(['true'])\n",
        "references subprocess.check_output",
    ),
    "os-killpg": (
        "import os\nimport signal\n\ndef test_probe():\n    os.killpg(1, signal.SIGTERM)\n",
        "references os.killpg",
    ),
    "os-system": (
        "import os\n\ndef test_probe():\n    os.system('true')\n",
        "references os.system",
    ),
    "os-popen": (
        "import os\n\ndef test_probe():\n    os.popen('true').close()\n",
        "references os.popen",
    ),
    "os-fork": (
        "from os import fork\n\ndef test_probe():\n    fork()\n",
        "references os.fork",
    ),
    "os-spawn-family": (
        "import os\n\ndef test_probe():\n    os.spawnlp(os.P_WAIT, 'true', 'true')\n",
        "references os.spawnlp",
    ),
    "os-posix-spawn": (
        "import os\n\ndef test_probe():\n    os.posix_spawnp('true', ['true'], {})\n",
        "references os.posix_spawnp",
    ),
    "multiprocessing-process": (
        "import multiprocessing\n\ndef test_probe():\n"
        "    multiprocessing.Process(target=print).start()\n",
        "references multiprocessing.Process",
    ),
    "multiprocessing-pool": (
        "from multiprocessing.pool import Pool\n\ndef test_probe():\n    Pool(1).close()\n",
        "references multiprocessing.pool.Pool",
    ),
    "process-pool-executor": (
        "from concurrent import futures\n\ndef test_probe():\n"
        "    futures.ProcessPoolExecutor().shutdown()\n",
        "references concurrent.futures.ProcessPoolExecutor",
    ),
    "asyncio-subprocess-exec": (
        "import asyncio\n\nasync def test_probe():\n"
        "    await asyncio.create_subprocess_exec('true')\n",
        "references asyncio.create_subprocess_exec",
    ),
    "asyncio-subprocess-shell": (
        "import asyncio\n\nasync def test_probe():\n"
        "    await asyncio.create_subprocess_shell('true')\n",
        "references asyncio.create_subprocess_shell",
    ),
    "pty-spawn": (
        "import pty\n\ndef test_probe():\n    pty.spawn('true')\n",
        "references pty.spawn",
    ),
    "start-new-session": (
        "def test_probe(spawn):\n    assert spawn(start_new_session=True) is not None\n",
        "passes start_new_session=",
    ),
    "runner-main": (
        "from phasesweep.mcp.runner import main as runner_main\n\n"
        "def test_probe(tmp_path):\n    runner_main([str(tmp_path)])\n",
        "calls runner_main()",
    ),
    "sleep-call": (
        "import time\n\ndef test_probe():\n    time.sleep(1)\n",
        "calls sleep()",
    ),
    "sleep-in-string": (
        "def test_probe(tmp_path):\n"
        "    (tmp_path / 'trainer.py').write_text('import time; time.sleep(5)\\n')\n",
        "string literal",
    ),
    # Nothing else holds the event, so the wait always runs its full timeout.
    "wait-on-fresh-event": (
        "import threading\n\ndef test_probe():\n    threading.Event().wait(0.5)\n",
        "waits out a .wait() timeout",
    ),
    "wait-asserted-to-time-out": (
        "def test_probe(ready):\n    assert not ready.wait(timeout=0.5)\n",
        "waits out a .wait() timeout",
    ),
    "wait-compared-to-false": (
        "def test_probe(ready):\n    assert ready.wait(0.5) is False\n",
        "waits out a .wait() timeout",
    ),
    "threading-timer": (
        "from threading import Timer\n\ndef test_probe():\n    Timer(0.1, print).start()\n",
        "references threading.Timer",
    ),
    "select-select": (
        "import select\n\ndef test_probe():\n    select.select([], [], [], 0.1)\n",
        "references select.select",
    ),
    "signal-pause": (
        "import signal\n\ndef test_probe():\n    signal.pause()\n",
        "references signal.pause",
    ),
    "wallclock-helper": (
        "from tests.mcp_helpers import wait_for_mcp_running_trial\n\n"
        "def test_probe(app):\n    wait_for_mcp_running_trial(app, 'r', timeout=1.0)\n",
        "calls wait_for_mcp_running_trial()",
    ),
    # A registered name also counts when a test requests it as a fixture.
    "shared-fixture": (
        "def test_probe(runner_main):\n    assert runner_main\n",
        "requests fixture runner_main",
    ),
    "helper": (
        "import subprocess\n\n"
        "def _spawn_worker():\n    return subprocess.Popen(['true'])\n\n"
        "def test_probe():\n    assert _spawn_worker().wait() == 0\n",
        "(via _spawn_worker)",
    ),
    # The fixture is requested but never named in the body, so only the
    # signature leads the scanner to it.
    "fixture-parameter": (
        "import subprocess\nimport pytest\n\n"
        "@pytest.fixture\n"
        "def worker():\n"
        "    proc = subprocess.Popen(['true'])\n"
        "    yield\n"
        "    proc.wait()\n\n"
        "def test_probe(worker):\n    assert True\n",
        "(via worker)",
    ),
    "test-class-method": (
        "import subprocess\n\n"
        "class TestProbe:\n"
        "    def test_probe(self):\n"
        "        subprocess.run(['true'])\n",
        "references subprocess.run",
    ),
}

#: Case id -> probe source whose ``test_probe`` must stay in the fast tier.
FAST_TIER_PROBES: dict[str, str] = {
    "pure-unit-test": (
        "from phasesweep.engine.state import Winner\n\n"
        "def _build_winner():\n    return Winner(phase='p', params={'x': 1})\n\n"
        "def test_probe():\n    assert _build_winner().phase == 'p'\n"
    ),
    # Measuring elapsed time is not spending it: monotonic/perf_counter reads
    # are exactly how the fast tier fakes clocks instead of waiting on one.
    "monotonic-clock": (
        "import time\n\n"
        "def _elapsed(start):\n    return time.monotonic() - start\n\n"
        "def test_probe(monkeypatch):\n"
        "    start = time.perf_counter()\n"
        "    monkeypatch.setattr(time, 'monotonic', lambda: 10.0)\n"
        "    assert _elapsed(start) >= 0.0\n"
    ),
    # Replacing a spawner by name never starts a process.
    "patched-spawner": (
        "import subprocess\n\n"
        "def test_probe(monkeypatch):\n"
        "    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: None)\n"
    ),
    "thread-pools": (
        "from concurrent.futures import ThreadPoolExecutor\n"
        "from multiprocessing.pool import ThreadPool\n\n"
        "def test_probe():\n"
        "    ThreadPoolExecutor().shutdown()\n"
        "    ThreadPool(1).close()\n"
    ),
    # A handshake returns once the peer sets the event; the timeout only bounds a hang.
    "handshake-wait": (
        "import threading\n\n"
        "def test_probe():\n"
        "    ready = threading.Event()\n"
        "    threading.Thread(target=ready.set).start()\n"
        "    assert ready.wait(timeout=5)\n"
        "    if not ready.wait(5):\n"
        "        raise AssertionError('peer never signalled')\n"
    ),
    # A quick engine run spawns its trainer inside the package, out of the
    # scanner's sight by design: these runs are the fast tier's engine coverage.
    "quick-engine-run": (
        "from phasesweep import run_experiment\n"
        "from tests.conftest import make_experiment\n\n"
        "def test_probe(tmp_path):\n"
        "    run_experiment(make_experiment(workdir=tmp_path, trial_command='echo {overrides}'))\n"
    ),
}


@pytest.mark.parametrize("rule", sorted(RULE_PROBES))
def test_each_tier_rule_flags_its_probe(rule):
    source, fragment = RULE_PROBES[rule]
    reason = scan_module(source).get("test_probe")
    assert reason is not None, f"rule {rule!r} no longer flags its probe"
    assert fragment in reason


@pytest.mark.parametrize("case", sorted(FAST_TIER_PROBES))
def test_tier_scanner_leaves_fast_tests_alone(case):
    assert scan_module(FAST_TIER_PROBES[case]) == {}


def test_guard_module_survives_its_own_rules():
    # Otherwise every probe added here would have to be marked as an
    # integration test.
    assert scan_module(Path(__file__).read_text(encoding="utf-8")) == {}


#: A shared helper module with one spawner, one waiter, and one plain builder.
HELPER_MODULE_PROBE = (
    "import subprocess\nimport time\n\n"
    "def spawn_worker():\n    return subprocess.Popen(['true'])\n\n"
    "def poll_until_ready():\n    time.sleep(0.1)\n\n"
    "def build_config():\n    return {}\n"
)


def test_helper_scan_finds_shared_spawners_and_waiters():
    assert set(scan_helpers(HELPER_MODULE_PROBE)) == {"spawn_worker", "poll_until_ready"}


def test_shared_helpers_that_spawn_or_wait_are_registered():
    # A test that imports a helper is out of the scanner's same-module reach, so
    # only registration by name classifies it. tiers.py is skipped because its
    # rule descriptions quote the very primitives they detect.
    registered = PROCESS_DRIVER_CALLS | WALLCLOCK_HELPER_CALLS
    unregistered = {
        f"{path.name}::{name}": reason
        for path in sorted(Path(__file__).parent.glob("*.py"))
        if not path.name.startswith("test_") and path.name != "tiers.py"
        for name, reason in scan_helpers(path.read_text(encoding="utf-8")).items()
        if name not in registered
    }
    assert unregistered == {}, "register these in tests/tiers.py"


def _report(nodeid, *, duration, when="call", markers=()):
    keywords = {nodeid.rpartition("::")[2]: 1, **dict.fromkeys(markers, 1)}
    return pytest.TestReport(
        nodeid=nodeid,
        location=("t.py", 0, nodeid),
        keywords=keywords,
        outcome="passed",
        longrepr=None,
        when=when,
        duration=duration,
    )


def test_slow_report_lists_unmarked_call_phases_at_or_over_the_threshold():
    reports = [
        _report("t.py::test_fast", duration=SLOW_CALL_SECONDS - 0.01),
        _report("t.py::test_edge", duration=SLOW_CALL_SECONDS),
        _report("t.py::test_slow", duration=SLOW_CALL_SECONDS + 1.5),
        _report("t.py::test_marked", duration=9.0, markers=("integration",)),
        _report("t.py::test_hardware", duration=9.0, markers=("hardware",)),
        _report("t.py::test_slow_setup", duration=9.0, when="setup"),
    ]

    assert slow_unmarked(reports) == [
        ("t.py::test_slow", SLOW_CALL_SECONDS + 1.5),
        ("t.py::test_edge", SLOW_CALL_SECONDS),
    ]


@pytest.mark.parametrize(
    ("markexpr", "excluded"),
    [
        ("not hardware and not integration", True),
        ("not (integration or hardware)", True),
        ("unit", True),
        ("not hardware", False),
        ("integration and not hardware", False),
        ("", False),
    ],
)
def test_slow_report_runs_only_when_the_integration_tier_is_left_out(markexpr, excluded):
    assert excludes_integration(markexpr) is excluded
