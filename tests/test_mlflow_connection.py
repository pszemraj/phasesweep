"""MLflow failures identify the configured target and missing evidence."""

from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from phasesweep.runtime.json import JsonObject
from phasesweep.tracking import mlflow as worker


@pytest.mark.parametrize("explicit_cause", [False, True], ids=["context", "cause"])
def test_mlflow_refused_connection_is_identified_as_transport_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_cause: bool
) -> None:
    """A wrapped transport failure names the refused connection without private SDK text."""
    from mlflow.exceptions import MlflowException
    from requests.exceptions import ConnectionError as RequestsConnectionError

    def dispatch(_request: JsonObject) -> JsonObject:
        try:
            raise RequestsConnectionError("private connection detail") from ConnectionRefusedError(
                errno.ECONNREFUSED, "private endpoint"
            )
        except RequestsConnectionError as exc:
            if explicit_cause:
                raise MlflowException("private MLflow transport wrapper") from exc
            raise MlflowException("private MLflow transport wrapper") from None

    monkeypatch.setattr(worker, "_dispatch", dispatch)
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(json.dumps({"action": "resolve"}))
    assert worker.main([str(request), str(response)]) == 0
    message = json.loads(response.read_text())["message"]
    assert "connection" in message.lower()
    assert "refused" in message.lower()
    assert "private" not in message
