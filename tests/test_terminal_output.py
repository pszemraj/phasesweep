"""What the CLI prints: terminal layouts at fixed widths, what a pipe gets, and closing lines."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from phasesweep.config.models import Experiment
from phasesweep.config.search import CategoricalParam, Sampler
from phasesweep.engine.selection import PhaseTrialsFailedError
from tests.conftest import (
    invoke_cli_boundary,
    make_experiment,
    terminal_stream,
    write_constant_trainer,
    write_experiment_config,
)

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


_PAD = " " * 10
_STARTER_DRY_RUN_AT_80 = [
    "HH:MM:SS  DRY RUN metric='eval_loss' goal=minimize extractor={'type':",
    f"{_PAD}'json_envelope', 'path': 'result.json', 'objective_name': 'eval_loss',",
    f"{_PAD}'split': 'validation', 'policy': 'synthetic', 'checkpoint':",
    f"{_PAD}'toy_formula', 'expected_step': 100, 'evaluation_axis': 'step'}}",
    "HH:MM:SS  DRY RUN evaluation_point check: supported (axis 'step' from the",
    f"{_PAD}envelope; any other axis from evaluation.progress)",
    "HH:MM:SS  DRY RUN environment: inherit_env=none (inherited when set, and",
    f"{_PAD}digested: PATH, HOME, LANG, LC_ALL, USER, LOGNAME, TZ); env keys:",
    f"{_PAD}none; passthrough_env: TMPDIR (built in; inherited when set, excluded",
    f"{_PAD}from the digest); a persistent study refuses to resume when any",
    f"{_PAD}digested value changes",
    "HH:MM:SS  DRY RUN trainer cwd:",
    f"{_PAD}TMP",
    f"{_PAD}(execution.cwd; part of the study identity)",
    "HH:MM:SS  DRY RUN artifact root:",
    f"{_PAD}TMP/runs/phasesweep_starter",
    "HH:MM:SS  DRY RUN phase 'depth': sampler=grid (resumable, reproducible)",
    "HH:MM:SS  DRY RUN grid order: configurations are shuffled by sampler.seed, not",
    f"{_PAD}listed choice order.",
    "HH:MM:SS  DRY RUN phase 'learning_rate': sampler=grid (resumable, reproducible)",
    "HH:MM:SS  DRY RUN grid order: configurations are shuffled by sampler.seed, not",
    f"{_PAD}listed choice order.",
    "HH:MM:SS  phase=depth study=phasesweep_starter::depth finished=0 remaining=2",
    f"{_PAD}reruns=0 n_jobs=1",
    "HH:MM:SS  GPU isolation disabled by gpu_policy='none'.",
    "HH:MM:SS  DRY RUN phase=depth would launch 2 trials",
    "HH:MM:SS  DRY RUN example command:",
    f"{_PAD}  phasesweep-fake-train --config "
    "TMP/runs/phasesweep_starter/depth/trial_dryrun/trainer_config.yaml",
    "HH:MM:SS  phase=learning_rate study=phasesweep_starter::learning_rate finished=0",
    f"{_PAD}remaining=2 reruns=0 n_jobs=1",
    "HH:MM:SS  GPU isolation disabled by gpu_policy='none'.",
    "HH:MM:SS  DRY RUN phase=learning_rate would launch 2 trials",
    "HH:MM:SS  DRY RUN example command (placeholders until 'depth' publishes a",
    f"{_PAD}winner: model.n_layers):",
    f"{_PAD}  phasesweep-fake-train --config "
    "TMP/runs/phasesweep_starter/learning_rate/trial_dryrun/trainer_config.yaml",
    "HH:MM:SS  DRY RUN total: would launch 4 trials across 2 phases",
    "HH:MM:SS  DRY RUN complete. No trials launched, no summary written.",
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
    with terminal_stream("stderr", columns) as stderr:
        status = _refuse(tmp_path, monkeypatch)

    assert status == 1
    assert stderr.text.split("\n") == [*expected, ""]
    assert all(len(line) <= columns for line in expected)


def test_a_fix_steps_command_stays_whole_and_flush_left_on_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command a fix step sets off on its own line is pasted, so a terminal narrower than it prints it whole, with no indent."""
    command = (
        "phasesweep sync-tracking /srv/experiments/sweeps/learning-rate.yaml "
        "--set mlflow.timeout_seconds=30 --generation 0123abcd"
    )

    def refuse(_path: Path) -> Experiment:
        raise PhaseTrialsFailedError(
            "MLflow delivery remains pending.",
            remediation=f"Once the server answers, run this command again:\n{command}",
        )

    monkeypatch.setattr("phasesweep.cli.load_experiment", refuse)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    with terminal_stream("stderr", 60) as stderr:
        status = invoke_cli_boundary(["run", str(config_path)], monkeypatch)

    assert status == 1
    assert len(command) > 60
    assert stderr.text.split("\n") == [
        "phasesweep: MLflow delivery remains pending.",
        "fix: Once the server answers, run this command again:",
        command,
        "",
    ]


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


def test_a_dry_run_wraps_its_lines_to_the_terminal_and_keeps_its_commands_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each flush-left line breaks at spaces under the text after the time, and a path stays whole.

    The indented example commands are meant to be copied, so they overrun the width on one line.
    """
    config = tmp_path / "starter.yaml"
    assert invoke_cli_boundary(["init", "-o", str(config)], monkeypatch) == 0

    with terminal_stream("stderr", 80) as stderr:
        assert invoke_cli_boundary(["run", str(config), "--dry-run"], monkeypatch) == 0

    root = str(tmp_path.resolve())
    lines = [
        re.sub(r"^\d\d:\d\d:\d\d", "HH:MM:SS", line).replace(root, "TMP")
        for line in stderr.text.splitlines()
    ]
    assert lines == _STARTER_DRY_RUN_AT_80


@pytest.mark.integration
def test_a_run_with_nothing_to_run_still_names_the_command_that_reads_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The replay's closing sentence is followed by the same ``Next:`` line a launching run prints."""
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_constant_trainer(tmp_path),
        n_trials=2,
        gpu_policy="none",
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[1, 2])},
    )
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 0
    next_line = f"Next: phasesweep status {config} --trials"
    assert next_line in capsys.readouterr().err.splitlines()

    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 0

    lines = capsys.readouterr().err.splitlines()
    closing = next(index for index, line in enumerate(lines) if line.startswith("Finished in"))
    assert "nothing to run" in lines[closing]
    after = lines[closing + 1 :]
    assert after[after.index("") + 1] == next_line
    assert after[after.index("") + 2].startswith("log: ")


def test_validate_says_it_checked_the_config_alone_and_names_the_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A passing validation ends by naming the command that also compares with a recorded study."""
    config = tmp_path / "my sweep.yaml"
    assert invoke_cli_boundary(["init", "-o", str(config)], monkeypatch) == 0
    capsys.readouterr()

    assert (
        invoke_cli_boundary(
            ["validate", str(config), "--set", "phases[depth].gpu_policy=none"], monkeypatch
        )
        == 0
    )

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "OK: phasesweep_starter (2 phases)"
    assert lines[-1] == (
        "Checked the config alone; phasesweep run "
        f"'{config}' --set 'phases[depth].gpu_policy=none' --dry-run "
        "also compares it with a recorded study."
    )
    assert sum(line.startswith("Checked the config alone") for line in lines) == 1
