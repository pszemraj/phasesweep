"""Managed MLflow launcher keeps a script's final evaluation in its supplied run."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from phasesweep.tracking.mlflow_launch import main


@pytest.mark.parametrize("ending", ["complete", "exit_zero", "exit_none", "error", "exit_failure"])
def test_launcher_keeps_managed_run_active_for_complete_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    """The supplied run spans training and evaluation and receives script failures."""
    events: list[str] = []

    @contextmanager
    def start_run(*, run_id: str) -> Iterator[None]:
        assert run_id == "managed-run"
        events.append("start")
        try:
            yield
        except (RuntimeError, SystemExit):
            events.append("failed")
            raise
        else:
            events.append("finished")

    monkeypatch.setitem(
        sys.modules, "mlflow", SimpleNamespace(start_run=start_run, record=events.append)
    )
    monkeypatch.setenv("MLFLOW_RUN_ID", "managed-run")
    monkeypatch.setattr(sys, "argv", sys.argv.copy())
    monkeypatch.setattr(sys, "path", sys.path.copy())
    sibling = tmp_path / "launcher_sibling.py"
    sibling.write_text("VALUE = 'evaluation'\n", encoding="utf-8")
    script = tmp_path / "trainer.py"
    script.write_text(
        "import os, sys, mlflow\n"
        "from launcher_sibling import VALUE\n"
        "assert 'MLFLOW_RUN_ID' not in os.environ\n"
        "assert sys.argv[1:] == ['--learning_rate', '0.1']\n"
        "mlflow.record('training')\n"
        "mlflow.record(VALUE)\n"
        + {
            "complete": "",
            "exit_zero": "raise SystemExit(0)\n",
            "exit_none": "raise SystemExit()\n",
            "error": "raise RuntimeError('trainer failed')\n",
            "exit_failure": "raise SystemExit(2)\n",
        }[ending],
        encoding="utf-8",
    )
    if ending == "error":
        with pytest.raises(RuntimeError, match="trainer failed"):
            main([str(script), "--learning_rate", "0.1"])
    elif ending == "exit_failure":
        with pytest.raises(SystemExit) as raised:
            main([str(script), "--learning_rate", "0.1"])
        assert raised.value.code == 2
    else:
        assert main([str(script), "--learning_rate", "0.1"]) == 0
    outcome = "failed" if ending in {"error", "exit_failure"} else "finished"
    assert events == ["start", "training", "evaluation", outcome]
    sys.modules.pop("launcher_sibling", None)


def test_launcher_requires_managed_run_before_importing_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A launcher without managed identity reports a usage error without starting a run."""
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(SystemExit) as raised:
        main([str(tmp_path / "trainer.py")])
    assert raised.value.code == 2
    assert "requires mlflow.run_mode: managed" in capsys.readouterr().err
