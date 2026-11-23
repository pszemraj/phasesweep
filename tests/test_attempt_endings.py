"""How an attempt ended, as its lifecycle record says, and the verdict each ending earns."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from phasesweep.runtime.process import AttemptEnd, read_attempt_lifecycle, run_supervised


@pytest.mark.parametrize(
    ("command", "env_extra", "cwd", "timeout", "cap", "ending", "return_code"),
    [
        pytest.param("exit 3", {}, None, None, False, "trainer", 3, id="nonzero-exit"),
        pytest.param("exit 75", {}, None, None, False, "trainer", 75, id="trainer-exits-75"),
        pytest.param("kill -9 $$", {}, None, None, False, "trainer", -9, id="own-sigkill"),
        pytest.param("true", {}, "/nonexistent", None, False, "not_started", None, id="bad-cwd"),
        # One environment string past MAX_ARG_STRLEN makes ``execve`` fail with E2BIG.
        pytest.param(
            "true", {"BIG": "x" * 200_000}, None, None, False, "not_started", None, id="execve"
        ),
        pytest.param("exec sleep 30", {}, None, 0.5, False, "timed_out", None, id="timeout"),
        pytest.param("exec sleep 30", {}, None, 0.5, True, "cancelled", None, id="deadline-cap"),
        pytest.param(
            "kill -9 $PPID; exec sleep 30", {}, None, None, False, "guardian_lost", None, id="lost"
        ),
    ],
)
def test_each_ending_leaves_its_own_record(
    tmp_path: Path,
    command: str,
    env_extra: dict[str, str],
    cwd: str | None,
    timeout: float | None,
    cap: bool,
    ending: AttemptEnd,
    return_code: int | None,
) -> None:
    """The record names how the attempt ended, and only a trainer's own ending carries a code."""
    with (tmp_path / "stdout.log").open("w") as out, (tmp_path / "stderr.log").open("w") as err:
        run_supervised(
            command,
            env={**os.environ, **env_extra},
            stdout=out,
            stderr=err,
            timeout=None if cap else timeout,
            wallclock_deadline=time.monotonic() + timeout if cap and timeout else None,
            trial_dir=tmp_path,
            attempt_id="attempt",
            cwd=cwd,
        )
    lifecycle = read_attempt_lifecycle(tmp_path, expected_attempt_id="attempt")
    assert lifecycle is not None
    assert (lifecycle.state, lifecycle.cleanup_confirmed) == ("exited", True)
    assert (lifecycle.ended_by, lifecycle.return_code) == (ending, return_code)
