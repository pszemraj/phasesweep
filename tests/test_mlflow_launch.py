"""Managed MLflow launcher keeps a script's final evaluation in its supplied run."""

from __future__ import annotations

import os
import subprocess
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
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    packages = tmp_path / "packages"
    packages.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:], str(packages)])
    (tmp_path / "launcher_evaluate.py").write_text(
        "raise RuntimeError('cwd shadow imported')\n", encoding="utf-8"
    )
    (packages / "launcher_evaluate.py").write_text("VALUE = 'package'\n", encoding="utf-8")
    sibling = script_dir / "launcher_sibling.py"
    sibling.write_text("VALUE = 'evaluation'\n", encoding="utf-8")
    script = script_dir / "trainer.py"
    script.write_text(
        "import os, sys, mlflow\n"
        "from launcher_sibling import VALUE\n"
        "import launcher_evaluate\n"
        "assert launcher_evaluate.VALUE == 'package'\n"
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
    sys.modules.pop("launcher_evaluate", None)


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


def test_launcher_missing_sdk_names_install_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An absent optional MLflow SDK gives the operator its install command."""
    monkeypatch.setenv("MLFLOW_RUN_ID", "managed-run")
    monkeypatch.setitem(sys.modules, "mlflow", None)
    script = tmp_path / "trainer.py"
    script.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        main([str(script)])
    assert raised.value.code == 2
    assert 'pip install "phasesweep[mlflow]"' in capsys.readouterr().err


@pytest.mark.parametrize("safe_path", ["normal", "flag", "environment"])
def test_launcher_keeps_pythonpath_and_excludes_cwd_shadow(tmp_path: Path, safe_path: str) -> None:
    """Normal and safe-path launches preserve package paths and script sibling imports."""
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    packages = tmp_path / "packages"
    packages.mkdir()
    (tmp_path / "launcher_evaluate.py").write_text(
        "raise RuntimeError('cwd shadow imported')\n", encoding="utf-8"
    )
    (packages / "launcher_evaluate.py").write_text("VALUE = 'package'\n", encoding="utf-8")
    (packages / "mlflow.py").write_text(
        "from contextlib import nullcontext\n"
        "def start_run(*, run_id: str) -> nullcontext[None]:\n"
        "    assert run_id == 'managed-run'\n"
        "    return nullcontext()\n",
        encoding="utf-8",
    )
    (script_dir / "launcher_sibling.py").write_text("VALUE = 'sibling'\n", encoding="utf-8")
    script = script_dir / "trainer.py"
    script.write_text(
        "import sys\n"
        "import launcher_evaluate, launcher_sibling\n"
        "assert launcher_evaluate.VALUE == 'package'\n"
        "assert launcher_sibling.VALUE == 'sibling'\n"
        "assert sys.argv[1:] == ['--learning_rate', '0.1']\n"
        "print('trainer completed')\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "MLFLOW_RUN_ID": "managed-run",
        "PYTHONPATH": str(packages),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.pop("PYTHONSAFEPATH", None)
    if safe_path == "environment":
        env["PYTHONSAFEPATH"] = "1"
    flags = ["-P"] if safe_path == "flag" else []
    result = subprocess.run(
        [
            sys.executable,
            *flags,
            "-m",
            "phasesweep.tracking.mlflow_launch",
            str(script),
            "--learning_rate",
            "0.1",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "trainer completed\n"


def test_launcher_missing_script_is_usage_error_before_opening_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing script names the path without importing MLflow or opening its run."""
    monkeypatch.setenv("MLFLOW_RUN_ID", "managed-run")
    monkeypatch.setitem(sys.modules, "mlflow", None)

    with pytest.raises(SystemExit) as raised:
        main([str(tmp_path / "missing.py")])

    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "trainer script does not exist or is not a file:" in error
    assert str(tmp_path / "missing.py") in error
    assert "Traceback" not in error
    assert os.environ["MLFLOW_RUN_ID"] == "managed-run"


def test_launcher_help_names_module_invocation_and_trial_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Offline help supplies the installed invocation and a complete config command."""
    with pytest.raises(SystemExit) as raised:
        main(["--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage: python -m phasesweep.tracking.mlflow_launch" in help_text
    assert "trial_command:" in help_text
    assert "my_project.train" in help_text
    assert "{overrides}" in help_text


@pytest.mark.parametrize("target", ["launcher_trainer.train", "launcher_trainer"])
def test_launcher_module_preserves_relative_imports_and_avoids_sibling_shadow(
    tmp_path: Path, target: str
) -> None:
    """Module and package entry points keep package imports and trainer arguments intact."""
    packages = tmp_path / "packages"
    packages.mkdir()
    package = packages / "launcher_trainer"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "score.py").write_text("VALUE = 'relative'\n", encoding="utf-8")
    (packages / "score.py").write_text("VALUE = 'top-level'\n", encoding="utf-8")
    (packages / "mlflow.py").write_text(
        "from contextlib import contextmanager\n"
        "active = False\n"
        "@contextmanager\n"
        "def start_run(*, run_id):\n"
        "    global active\n"
        "    assert run_id == 'managed-run'\n"
        "    active = True\n"
        "    print('managed start')\n"
        "    yield\n"
        "    print('managed finish')\n",
        encoding="utf-8",
    )
    trainer = (
        "import os, sys, mlflow, score\n"
        "from .score import VALUE\n"
        "assert VALUE == 'relative'\n"
        "assert score.VALUE == 'top-level'\n"
        "assert mlflow.active\n"
        "assert 'MLFLOW_RUN_ID' not in os.environ\n"
        "assert sys.argv[1:] == ['--learning_rate', '0.1']\n"
        "print('training and final evaluation')\n"
    )
    (package / "train.py").write_text(trainer, encoding="utf-8")
    (package / "__main__.py").write_text(trainer, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "phasesweep.tracking.mlflow_launch",
            "-m",
            target,
            "--learning_rate",
            "0.1",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "MLFLOW_RUN_ID": "managed-run",
            "PYTHONPATH": str(packages),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "managed start",
        "training and final evaluation",
        "managed finish",
    ]
