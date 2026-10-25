"""Managed MLflow command wrapping preserves trainer execution and run outcomes."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from phasesweep.cli import cli
from phasesweep.tracking import mlflow_launch
from phasesweep.tracking.mlflow_launch import run_managed


@pytest.fixture
def managed_run(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record whether the supplied run spans execution and observes trainer failures."""
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
    return events


@pytest.mark.parametrize("kind", ["script", "console"])
@pytest.mark.parametrize("ending", ["complete", "exit_zero", "exit_none", "error", "exit_failure"])
def test_launcher_runs_trainer_and_final_evaluation_in_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    managed_run: list[str],
    kind: str,
    ending: str,
) -> None:
    """The active managed run encloses training and evaluation in the caller's process."""
    original_argv, original_path = sys.argv, sys.path[:]
    trainer = (
        "import os, sys, mlflow\n"
        f"assert os.getpid() == {os.getpid()}\n"
        "assert 'MLFLOW_RUN_ID' not in os.environ\n"
        "assert sys.argv[1:] == ['--help', '--learning_rate=0.1', '-xy']\n"
        "mlflow.record('training')\n"
        "mlflow.record('evaluation')\n"
        + {
            "complete": "",
            "exit_zero": "raise SystemExit(0)\n",
            "exit_none": "raise SystemExit()\n",
            "error": "raise RuntimeError('trainer failed')\n",
            "exit_failure": "raise SystemExit(2)\n",
        }[ending]
    )
    if kind == "script":
        script = tmp_path / "trainer.py"
        script.write_text(trainer, encoding="utf-8")
        target = str(script)
    else:

        def callback() -> None:
            exec(trainer, {})

        monkeypatch.setitem(sys.modules, "launcher_trainer", SimpleNamespace(main=callback))
        monkeypatch.setattr(
            mlflow_launch,
            "entry_points",
            lambda **_filters: [
                EntryPoint(
                    name="launcher-train", value="launcher_trainer:main", group="console_scripts"
                )
            ],
        )
        target = "launcher-train"
    arguments = [target, "--help", "--learning_rate=0.1", "-xy"]
    if ending == "error":
        with pytest.raises(RuntimeError, match="trainer failed"):
            run_managed(arguments)
    elif ending == "exit_failure":
        with pytest.raises(SystemExit) as raised:
            run_managed(arguments)
        assert raised.value.code == 2
    else:
        assert run_managed(arguments) == 0
    outcome = "failed" if ending in {"error", "exit_failure"} else "finished"
    assert managed_run == ["start", "training", "evaluation", outcome]
    assert sys.argv is original_argv
    assert sys.path == original_path
    assert os.environ["MLFLOW_RUN_ID"] == "managed-run"


def test_console_return_status_marks_the_run_failed(
    monkeypatch: pytest.MonkeyPatch, managed_run: list[str]
) -> None:
    """A nonzero console-entry return has the same effect as a nonzero script exit."""
    monkeypatch.setitem(sys.modules, "launcher_trainer", SimpleNamespace(main=lambda: 3))
    monkeypatch.setattr(
        mlflow_launch,
        "entry_points",
        lambda **_filters: [
            EntryPoint(
                name="launcher-train", value="launcher_trainer:main", group="console_scripts"
            )
        ],
    )
    with pytest.raises(SystemExit) as raised:
        run_managed(["launcher-train"])
    assert raised.value.code == 3
    assert managed_run == ["start", "failed"]


def test_launcher_help_and_trainer_help_have_separate_owners(
    tmp_path: Path, managed_run: list[str]
) -> None:
    """Wrapper help describes accepted forms; help after COMMAND reaches the trainer."""
    wrapper = CliRunner().invoke(cli, ["mlflow-run", "--help"])
    assert wrapper.exit_code == 0
    assert "mlflow-run [OPTIONS] COMMAND..." in wrapper.output
    assert "trial_command:" in wrapper.output
    assert "phasesweep mlflow-run dcl train" in wrapper.output
    assert "in-process" in wrapper.output and "Child processes" in wrapper.output
    assert managed_run == []
    script = tmp_path / "trainer.py"
    script.write_text(
        "import sys\nassert sys.argv[1:] == ['--help']\nprint('trainer help')\n",
        encoding="utf-8",
    )
    trainer = CliRunner().invoke(cli, ["mlflow-run", str(script), "--help"])
    assert trainer.exit_code == 0, trainer.output
    assert trainer.output == "trainer help\n"
    assert managed_run == ["start", "finished"]


@pytest.mark.parametrize("target", ["missing-trainer", "trainer.module", "python", "missing.py"])
def test_launcher_invalid_target_is_usage_error_before_importing_sdk(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """Unresolvable commands list the supported forms without opening a run or traceback."""
    monkeypatch.setenv("MLFLOW_RUN_ID", "managed-run")
    monkeypatch.setitem(sys.modules, "mlflow", None)
    result = CliRunner().invoke(cli, ["mlflow-run", target, "--help"])
    assert result.exit_code == 2
    assert "installed console script" in result.output and "Python script path" in result.output
    assert "shell commands and module names are not supported" in result.output
    assert "Traceback" not in result.output
    assert os.environ["MLFLOW_RUN_ID"] == "managed-run"


def test_launcher_requires_managed_run_before_importing_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolved trainer without managed identity fails before importing MLflow."""
    script = tmp_path / "trainer.py"
    script.write_text("", encoding="utf-8")
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    result = CliRunner().invoke(cli, ["mlflow-run", str(script)])
    assert result.exit_code == 2
    assert "requires mlflow.run_mode: managed" in result.output


def test_launcher_missing_sdk_names_install_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent optional SDK gives its install command before executing the trainer."""
    script = tmp_path / "trainer.py"
    script.write_text("", encoding="utf-8")
    monkeypatch.setenv("MLFLOW_RUN_ID", "managed-run")
    monkeypatch.setitem(sys.modules, "mlflow", None)
    result = CliRunner().invoke(cli, ["mlflow-run", str(script)])
    assert result.exit_code == 2
    assert 'pip install "phasesweep[mlflow]"' in result.output


@pytest.mark.parametrize("kind", ["script", "console"])
@pytest.mark.parametrize("safe_path", [False, True])
def test_installed_command_avoids_cwd_shadow_and_preserves_arguments(
    tmp_path: Path, kind: str, safe_path: bool
) -> None:
    """Console imports exclude the working directory; script siblings keep normal semantics."""
    packages = tmp_path / "packages"
    packages.mkdir()
    (tmp_path / "launcher_evaluate.py").write_text(
        "raise RuntimeError('cwd shadow imported')\n", encoding="utf-8"
    )
    (packages / "launcher_evaluate.py").write_text("VALUE = 'package'\n", encoding="utf-8")
    (packages / "mlflow.py").write_text(
        "from contextlib import contextmanager\n"
        "active = False\n"
        "@contextmanager\n"
        "def start_run(*, run_id):\n"
        "    global active\n"
        "    assert run_id == 'managed-run'\n"
        "    active = True\n"
        "    yield\n",
        encoding="utf-8",
    )
    body = (
        "import os, sys, mlflow, launcher_evaluate\n"
        "assert launcher_evaluate.VALUE == 'package'\n"
        "assert mlflow.active and 'MLFLOW_RUN_ID' not in os.environ\n"
        "assert sys.argv[1:] == ['train', '--help', '--learning_rate=0.1']\n"
        "print('trainer completed')\n"
    )
    if kind == "script":
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "sibling.py").write_text("VALUE = 'sibling'\n", encoding="utf-8")
        script = scripts / "trainer.py"
        script.write_text(
            "from sibling import VALUE\nassert VALUE == 'sibling'\n" + body,
            encoding="utf-8",
        )
        target = str(script)
    else:
        distribution = packages / "launcher_trainer-0.0.dist-info"
        distribution.mkdir()
        (distribution / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: launcher-trainer\nVersion: 0.0\n", encoding="utf-8"
        )
        (distribution / "entry_points.txt").write_text(
            "[console_scripts]\nlauncher-train = launcher_trainer:main\n", encoding="utf-8"
        )
        (packages / "launcher_trainer.py").write_text(
            "def main():\n" + "".join("    " + line + "\n" for line in body.splitlines()),
            encoding="utf-8",
        )
        target = "launcher-train"
    env = {
        **os.environ,
        "MLFLOW_RUN_ID": "managed-run",
        "PYTHONPATH": str(packages),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.pop("PYTHONSAFEPATH", None)
    if safe_path:
        env["PYTHONSAFEPATH"] = "1"
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("phasesweep")),
            "mlflow-run",
            target,
            "train",
            "--help",
            "--learning_rate=0.1",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "trainer completed\n"
