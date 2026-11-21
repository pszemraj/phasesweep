"""Command help text, the ``validate`` phase count and the quiet exit on a closed output pipe."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from phasesweep.cli import cli, main
from phasesweep.config.models import Phase
from phasesweep.config.search import IntParam
from tests.conftest import make_experiment, write_experiment_config


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
    assert "winner named in the first column" in help_text
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


def test_a_closed_output_pipe_ends_quietly_with_status_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``phasesweep ... | head`` closes stdout early: no operational error, no traceback, status 1."""
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

    assert raised.value.code == 1
    assert capsys.readouterr().err == ""
