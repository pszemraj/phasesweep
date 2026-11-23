"""Command help text, the ``validate`` phase count, the quiet exit on a closed output pipe and the exit on a full output device."""

from __future__ import annotations

import contextlib
import errno
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from phasesweep.cli import cli, main
from phasesweep.config.models import Phase
from phasesweep.config.search import IntParam
from tests.conftest import console_script, make_experiment, write_experiment_config


def test_top_level_help_shows_the_whole_sync_tracking_summary() -> None:
    """The command list carries sync-tracking's own short help, not a truncated long help."""
    output = CliRunner().invoke(cli, ["--help"], terminal_width=80).output

    line = next(line for line in output.splitlines() if line.lstrip().startswith("sync-tracking"))

    assert line.split(None, 1)[1] == "Reconcile and retry MLflow uploads."
    assert "..." not in line
    assert all(len(line) <= 80 for line in output.splitlines())


def test_show_winners_names_its_argument_config_like_the_other_commands() -> None:
    """The positional reads ``CONFIG``, as in ``status`` and ``validate``."""
    runner = CliRunner()

    winners = runner.invoke(cli, ["show-winners", "--help"], terminal_width=120).output

    assert "show-winners [OPTIONS] CONFIG" in winners
    assert "CONFIG_YAML" not in winners
    assert "including unpublished results from the latest run" in winners
    for command in ("status", "validate"):
        assert f"{command} [OPTIONS] CONFIG" in runner.invoke(cli, [command, "--help"]).output


def test_status_help_describes_the_trial_table() -> None:
    """``--trials`` says what the table holds and that trials.csv is the machine-readable form."""
    help_text = " ".join(CliRunner().invoke(cli, ["status", "--help"]).output.split())

    assert "a column per swept parameter" in help_text
    assert "the winner marked" in help_text
    assert "trials.csv" in help_text


@pytest.mark.parametrize(("names", "count"), [(["p"], "1 phase"), (["a", "b"], "2 phases")])
def test_validate_counts_its_phases_in_the_singular_for_one(
    tmp_path: Path, names: list[str], count: str
) -> None:
    """The ``OK`` line reads ``1 phase`` for a single phase and ``N phases`` otherwise."""
    phases = [
        Phase(
            name=name,
            n_trials=2,
            gpu_policy="none",
            search_space={"x": IntParam(type="int", low=0, high=9)},
        )
        for name in names
    ]
    experiment = make_experiment(workdir=tmp_path / "runs", phases=phases)
    config = write_experiment_config(tmp_path / "sweep.yaml", experiment)

    result = CliRunner().invoke(cli, ["validate", str(config)])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0] == f"OK: {experiment.experiment} ({count})"


def test_a_closed_output_pipe_ends_quietly_with_the_commands_own_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``phasesweep ... | head`` closes stdout early: no operational error, no traceback, status 0."""
    read_end, write_end = os.pipe()
    os.close(read_end)
    closed = os.fdopen(write_end, "w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", closed)
    monkeypatch.setattr(sys, "argv", ["phasesweep", "explain", "--notes"])

    try:
        with pytest.raises(SystemExit) as raised:
            main()
    finally:
        with contextlib.suppress(BrokenPipeError):
            closed.close()

    assert raised.value.code == 0
    assert capsys.readouterr().err == ""


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full")
def test_a_full_output_device_fails_the_command_once_with_status_1(tmp_path: Path) -> None:
    """Output to a full device ends the command with the OS error and status 1, and nothing at exit.

    The interpreter's last flush of the text the failed write left behind reports nothing more,
    so the status stays the command's own instead of 120.
    """
    with Path("/dev/full").open("w") as full:
        result = subprocess.run(
            [str(console_script("phasesweep")), "explain", "--notes"],
            stdout=full,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp_path,
            check=False,
        )

    assert (result.returncode, result.stderr) == (
        1,
        f"phasesweep: operational error: {OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))}\n",
    )
