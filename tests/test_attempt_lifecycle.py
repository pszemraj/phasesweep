"""Attempt lifecycle reader refusal contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phasesweep.runtime.json import JsonObject
from phasesweep.runtime.process import (
    ATTEMPT_LIFECYCLE_FILE,
    ATTEMPT_LIFECYCLE_SCHEMA_VERSION,
    read_attempt_lifecycle,
)


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
