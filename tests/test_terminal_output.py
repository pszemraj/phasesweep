"""What the CLI writes to a terminal, laid out at fixed widths, and what it writes to a pipe."""

from __future__ import annotations

from pathlib import Path

import pytest

from phasesweep.config.models import Experiment
from phasesweep.engine.selection import PhaseTrialsFailedError
from tests.conftest import invoke_cli_boundary, terminal_stderr

_DIAGNOSIS = (
    "No completed trials in phase. The phase's trials.csv records why each failed or "
    "infeasible trial finished that way; trainer output is in its trial_* directory."
)
_STEP = (
    "Inspect runs/exp/phase/trials.csv and the trainer logs in its trial_* directories. Fix the "
    "failure, then raise phases[phase].n_trials to 10 in the config and rerun to retry the "
    "failed attempts."
)

_AT_80 = [
    "phasesweep: No completed trials in phase. The phase's trials.csv records why",
    "            each failed or infeasible trial finished that way; trainer output is",
    "            in its trial_* directory.",
    "fix: Inspect runs/exp/phase/trials.csv and the trainer logs in its trial_*",
    "     directories. Fix the failure, then raise phases[phase].n_trials to 10 in",
    "     the config and rerun to retry the failed attempts.",
]
_AT_60 = [
    "phasesweep: No completed trials in phase. The phase's",
    "            trials.csv records why each failed or infeasible",
    "            trial finished that way; trainer output is in",
    "            its trial_* directory.",
    "fix: Inspect runs/exp/phase/trials.csv and the trainer logs",
    "     in its trial_* directories. Fix the failure, then raise",
    "     phases[phase].n_trials to 10 in the config and rerun to",
    "     retry the failed attempts.",
]


def _refuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    """Run ``phasesweep run`` against a config load that refuses with the long diagnosis and step."""

    def refuse(_path: Path) -> Experiment:
        raise PhaseTrialsFailedError(_DIAGNOSIS, remediation=_STEP)

    monkeypatch.setattr("phasesweep.cli.load_experiment", refuse)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    return invoke_cli_boundary(["run", str(config_path)], monkeypatch)


@pytest.mark.parametrize(("columns", "expected"), [(80, _AT_80), (60, _AT_60)])
def test_an_operational_error_wraps_under_its_labels_on_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, columns: int, expected: list[str]
) -> None:
    """The diagnosis and the fix step each break at spaces, their text hanging under the label."""
    with terminal_stderr(columns) as stderr:
        status = _refuse(tmp_path, monkeypatch)

    assert status == 1
    assert stderr.text.split("\n") == [*expected, ""]
    assert all(len(line) <= columns for line in expected)


@pytest.mark.parametrize("columns", ["40", "200"])
def test_a_redirected_operational_error_is_one_line_per_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    columns: str,
) -> None:
    """Scripts and agents match these sentences, so a pipe gets each whole whatever ``COLUMNS`` says."""
    monkeypatch.setenv("COLUMNS", columns)

    assert _refuse(tmp_path, monkeypatch) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"phasesweep: {_DIAGNOSIS}\nfix: {_STEP}\n"
