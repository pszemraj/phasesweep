"""Attempt lifecycle reader refusal contracts and the recovery authority each state grants."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from phasesweep.engine import attempts
from phasesweep.engine.attempts import _attempt_process_resolution, _AttemptRecord, _resolve_attempt
from phasesweep.errors import OperatorAction
from phasesweep.runtime.json import JsonObject
from phasesweep.runtime.process import (
    ATTEMPT_LIFECYCLE_FILE,
    ATTEMPT_LIFECYCLE_SCHEMA_VERSION,
    AttemptLifecycle,
    read_attempt_lifecycle,
    write_attempt_lifecycle,
)
from phasesweep.runtime.reaper import StaleProcessIdentity


@pytest.mark.parametrize(
    ("state", "cleanup_confirmed", "identity_exists", "resolution"),
    [
        (None, None, False, "identity"),
        (None, None, True, "identity"),
        ("allocated", None, False, "allocated"),
        ("allocated", None, True, "identity"),
        ("launching", None, False, "payload_withheld"),
        ("launching", None, True, "identity"),
        ("terminating", False, False, "identity"),
        ("terminating", False, True, "identity"),
        ("exited", False, False, "identity"),
        ("exited", True, False, "exited"),
        ("exited", True, True, "exited"),
    ],
)
def test_attempt_process_resolution_table(
    state: str | None, cleanup_confirmed: bool | None, identity_exists: bool, resolution: str
) -> None:
    """Only proof that no process can remain skips the identity read; any identity file keeps it."""
    lifecycle = (
        None
        if state is None
        else AttemptLifecycle(
            state=state,
            return_code=None,
            cleanup_confirmed=cleanup_confirmed,
            duration_seconds=None,
        )
    )
    assert _attempt_process_resolution(lifecycle, identity_exists=identity_exists) == resolution


def test_read_attempt_lifecycle_rejects_unreadable_record(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    (trial_dir / ATTEMPT_LIFECYCLE_FILE).mkdir(parents=True)
    with pytest.raises(ValueError, match="is unreadable"):
        read_attempt_lifecycle(trial_dir, expected_attempt_id="expected")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ("{", "Malformed attempt lifecycle record"),
        ("[]", "must be a JSON object"),
        (
            {"schema_version": ATTEMPT_LIFECYCLE_SCHEMA_VERSION + 1},
            "Unsupported attempt lifecycle schema",
        ),
        ({"attempt_id": "other"}, "belongs to another attempt"),
        ({"state": "unknown"}, "Unknown attempt lifecycle state"),
        ({"state": []}, "Unknown attempt lifecycle state"),
        ({"state": {}}, "Unknown attempt lifecycle state"),
        ({"state": "exited", "return_code": True}, "field 'return_code' is invalid"),
        (
            {"state": "exited", "return_code": 0, "cleanup_confirmed": 1},
            "field 'cleanup_confirmed' is invalid",
        ),
        ({"duration_seconds": True}, "field 'duration_seconds' is invalid"),
        ({"duration_seconds": "1"}, "field 'duration_seconds' is invalid"),
        ({"duration_seconds": -1}, "field 'duration_seconds' is invalid"),
        ({"duration_seconds": 10**400}, "field 'duration_seconds' is invalid"),
    ],
)
def test_read_attempt_lifecycle_rejects_every_malformed_shape(
    tmp_path: Path, fields: JsonObject | str, message: str
) -> None:
    """Malformed state, identity and elapsed-time fields use the reader's ValueError contract."""
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    if isinstance(fields, str):
        payload = fields
    else:
        record: JsonObject = {
            "schema_version": ATTEMPT_LIFECYCLE_SCHEMA_VERSION,
            "attempt_id": "expected",
            "state": "allocated",
        }
        record.update(fields)
        payload = json.dumps(record)
    (trial_dir / ATTEMPT_LIFECYCLE_FILE).write_text(payload)
    with pytest.raises(ValueError, match=message):
        read_attempt_lifecycle(trial_dir, expected_attempt_id="expected")


@pytest.mark.parametrize(
    ("state", "return_code", "cleanup_confirmed", "text"),
    [
        pytest.param(
            "launching",
            None,
            None,
            "lifecycle=launching with no process identity: no worker received its payload; "
            "settles without signalling.",
            id="payload-withheld",
        ),
        pytest.param(
            "exited",
            -15,
            True,
            "lifecycle=exited rc=-15 confirms cleanup without signalling.",
            id="exited-with-cleanup",
        ),
    ],
)
def test_an_attempt_settled_without_signalling_is_routine_in_a_run_and_collected_in_a_preview(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    state: str,
    return_code: int | None,
    cleanup_confirmed: bool | None,
    text: str,
) -> None:
    """A run logs the settlement at INFO, which the console leaves out; a preview collects it instead."""
    write_attempt_lifecycle(
        tmp_path,
        attempt_id="a1",
        state=state,
        return_code=return_code,
        cleanup_confirmed=cleanup_confirmed,
    )
    record = _AttemptRecord("a1", tmp_path, "generation", "p", 3)
    expected = f"Attempt a1 phase='p' trial=3: {text}"

    collected: dict[_AttemptRecord, str] = {}
    with caplog.at_level(logging.INFO, logger="phasesweep"):
        for messages in (collected, None):
            settled = _resolve_attempt(
                record,
                context="RUNNING trial 3",
                evidence_action=OperatorAction.RESTORE_TREE,
                lifecycle_settles=True,
                confirm=True,
                inspection_messages=messages,
            )
            assert settled is False

    assert collected == {record: expected}
    assert [(entry.levelno, entry.getMessage()) for entry in caplog.records] == [
        (logging.INFO, expected)
    ]


@pytest.mark.parametrize(
    ("signalled", "level", "text"),
    [
        pytest.param(
            True,
            logging.WARNING,
            "terminated training process group; confirmed process cleanup (pid=41 pgid=41).",
            id="signalled",
        ),
        pytest.param(
            False,
            logging.INFO,
            "confirmed process cleanup without signalling (pid=41 pgid=41).",
            id="already-gone",
        ),
    ],
)
def test_a_stale_group_warns_only_when_cleanup_signalled_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    signalled: bool,
    level: int,
    text: str,
) -> None:
    """Terminating a live group warns; confirming one that was already gone is routine."""
    identity = StaleProcessIdentity(
        schema_version=1,
        attempt_id="a1",
        pid=41,
        pgid=41,
        proc_starttime=None,
        boot_id=None,
        pid_namespace="pid:[1]@0",
    )

    def cleanup(_identity: StaleProcessIdentity, *, before_terminate: Callable[[], None]) -> bool:
        if signalled:
            before_terminate()
        return True

    monkeypatch.setattr(attempts, "read_stale_process_identity", lambda *_a, **_k: identity)
    monkeypatch.setattr(attempts, "cleanup_stale_trial_process", cleanup)
    monkeypatch.setattr(attempts, "_record_attempt_exited", lambda *_a, **_k: None)
    record = _AttemptRecord("a1", tmp_path, "generation", "p", 3)

    with caplog.at_level(logging.INFO, logger="phasesweep"):
        assert (
            _resolve_attempt(
                record,
                context="RUNNING trial 3",
                evidence_action=OperatorAction.RESTORE_TREE,
                lifecycle_settles=False,
                confirm=True,
            )
            is signalled
        )

    assert [(entry.levelno, entry.getMessage()) for entry in caplog.records] == [
        (level, f"Attempt a1 phase='p' trial=3: {text}")
    ]
