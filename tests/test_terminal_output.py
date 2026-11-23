"""What the CLI prints: terminal layouts at fixed widths, what a pipe gets, closing lines, and the status a closed terminal leaves."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from phasesweep.cli import cli as cli_main
from phasesweep.config.models import Experiment
from phasesweep.config.search import CategoricalParam, Sampler
from phasesweep.engine.publication import _resolve_publication_pointer
from phasesweep.engine.selection import PhaseTrialsFailedError
from tests.conftest import (
    console_script,
    invoke_cli_boundary,
    make_experiment,
    terminal_stream,
    write_constant_trainer,
    write_experiment_config,
    write_trainer,
)
from tests.experiments import phase_payload, write_experiment

_DIAGNOSIS = (
    "Phase 'phase' aborted: 2 consecutive failed/infeasible trials reached "
    "max_consecutive_failures=2. Inspect runs/exp/phase for stderr logs."
)
_STEP = (
    "Fix the cause, then raise phases[phase].n_trials from 2 to 4 and rerun to retry its 2 "
    "failed cells."
)

_AT_80 = [
    "phasesweep: Phase 'phase' aborted: 2 consecutive failed/infeasible trials",
    "            reached max_consecutive_failures=2. Inspect runs/exp/phase for",
    "            stderr logs.",
    "fix: Fix the cause, then raise phases[phase].n_trials from 2 to 4 and rerun to",
    "     retry its 2 failed cells.",
]
_AT_60 = [
    "phasesweep: Phase 'phase' aborted: 2 consecutive",
    "            failed/infeasible trials reached",
    "            max_consecutive_failures=2. Inspect",
    "            runs/exp/phase for stderr logs.",
    "fix: Fix the cause, then raise phases[phase].n_trials from 2",
    "     to 4 and rerun to retry its 2 failed cells.",
]


_PAD = " " * 10
_STARTER_DRY_RUN_AT_80 = [
    "HH:MM:SS  DRY RUN metric='eval_loss' goal=minimize extractor=json_envelope",
    f"{_PAD}path=result.json objective_name=eval_loss split=validation",
    f"{_PAD}policy=synthetic checkpoint=toy_formula expected_step=100",
    f"{_PAD}evaluation_axis=step",
    "HH:MM:SS  DRY RUN evaluation_point check: supported (axis 'step' from the",
    f"{_PAD}envelope; any other axis from evaluation.progress)",
    "HH:MM:SS  DRY RUN environment: inherit_env=none (digested: PATH, HOME, LANG,",
    f"{_PAD}LC_ALL, USER, LOGNAME, TZ); env keys: none; passthrough_env: TMPDIR",
    f"{_PAD}(built in, not digested)",
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


def test_validate_wraps_its_contract_lines_under_their_prefix_without_dict_syntax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``validate`` prints the extractor as ``key=value`` words and wraps each line, hanging two columns in."""
    config = tmp_path / "starter.yaml"
    assert invoke_cli_boundary(["init", "-o", str(config)], monkeypatch) == 0

    with terminal_stream("stdout", 60) as stdout:
        assert invoke_cli_boundary(["validate", str(config)], monkeypatch) == 0

    lines = stdout.text.splitlines()
    assert lines[1:5] == [
        "  metric='eval_loss' goal=minimize extractor=json_envelope",
        "    path=result.json objective_name=eval_loss",
        "    split=validation policy=synthetic checkpoint=toy_formula",
        "    expected_step=100 evaluation_axis=step",
    ]
    assert "{" not in "\n".join(lines[1:5])
    # The experiment's contract lines up to the trainer cwd, whose path stays whole.
    assert all(len(line) <= 60 for line in lines[1:11])


def test_validate_prints_a_string_extractor_field_as_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A field with whitespace is in double quotes with its backslashes as written, so it pastes back into a config."""
    pattern = r"throughput \{tok/s\}: (?P<value>\d+\.\d+) \[avg over last 100 steps\]"
    quoted = r'say "hi" (?P<value>\d+)'
    config = write_experiment(
        tmp_path / "exp.yaml",
        observe=[
            {
                "name": "tps",
                "extractor": {"type": "log_regex", "file": "stdout.log", "pattern": pattern},
            },
            {
                "name": "greeting",
                "extractor": {"type": "log_regex", "file": "stdout.log", "pattern": quoted},
            },
        ],
    )
    monkeypatch.setenv("COLUMNS", "500")

    result = CliRunner().invoke(cli_main, ["validate", str(config)])

    assert result.exit_code == 0, result.output
    assert f'pattern="{pattern}"' in result.output
    assert 'pattern="say \\"hi\\" (?P<value>\\d+)"' in result.output


_LONG_PHASES = [
    phase_payload(gpu_policy="none", name="first_phase_with_a_long_name"),
    phase_payload(
        gpu_policy="none",
        name="second_phase_with_a_long_name",
        inherits=["first_phase_with_a_long_name"],
        search_space={"y": {"type": "int", "low": 0, "high": 3}},
    ),
    phase_payload(
        gpu_policy="none",
        name="third_phase_with_long_name_and_many_inherits",
        inherits=["first_phase_with_a_long_name", "second_phase_with_a_long_name"],
        search_space={"z": {"type": "int", "low": 0, "high": 3}},
    ),
]
_PHASE_HEADERS_AT_80 = [
    "  - first_phase_with_a_long_name: n_trials=1 sampler=random",
    "  - second_phase_with_a_long_name: n_trials=1 sampler=random",
    "    inherits=first_phase_with_a_long_name",
    "  - third_phase_with_long_name_and_many_inherits: n_trials=1 sampler=random",
    "    inherits=first_phase_with_a_long_name,second_phase_with_a_long_name",
]
_PHASE_HEADERS_AT_60 = [
    "  - first_phase_with_a_long_name: n_trials=1 sampler=random",
    "  - second_phase_with_a_long_name: n_trials=1 sampler=random",
    "    inherits=first_phase_with_a_long_name",
    "  - third_phase_with_long_name_and_many_inherits: n_trials=1",
    "    sampler=random",
    "    inherits=first_phase_with_a_long_name,second_phase_with_a_long_name",
]


@pytest.mark.parametrize(
    ("columns", "expected"), [(80, _PHASE_HEADERS_AT_80), (60, _PHASE_HEADERS_AT_60)]
)
def test_validate_wraps_a_phase_header_hanging_under_its_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, columns: int, expected: list[str]
) -> None:
    """A phase header that overruns the terminal breaks at spaces and hangs under the text after its dash, a long name or inherits list staying whole."""
    config = write_experiment(tmp_path / "exp.yaml", phases=_LONG_PHASES)

    with terminal_stream("stdout", columns) as stdout:
        assert invoke_cli_boundary(["validate", str(config)], monkeypatch) == 0

    lines = stdout.text.splitlines()
    first = next(index for index, line in enumerate(lines) if line.startswith("  - "))
    headers = [line for line in lines[first:] if re.match(r"(  - |    \S)", line)]
    assert headers == expected


def test_validate_names_inherited_phases_without_list_syntax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A phase header lists the phases it inherits as bare names, comma-separated."""
    config = write_experiment(
        tmp_path / "exp.yaml",
        phases=[
            {"name": "a", "n_trials": 1, "fixed_overrides": {"x": 1}},
            {"name": "b", "n_trials": 1, "fixed_overrides": {"y": 2}},
            {"name": "c", "n_trials": 1, "inherits": ["a"]},
            {"name": "d", "n_trials": 1, "inherits": ["a", "b"]},
        ],
    )
    monkeypatch.setenv("COLUMNS", "500")

    result = CliRunner().invoke(cli_main, ["validate", str(config)])

    assert result.exit_code == 0, result.output
    assert "- c: n_trials=1 (fixed configuration) inherits=a\n" in result.output
    assert "- d: n_trials=1 (fixed configuration) inherits=a,b\n" in result.output
    assert "[" not in result.output


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


@pytest.mark.integration
def test_a_run_started_with_stderr_closed_publishes_without_writing_to_stdout(
    tmp_path: Path,
) -> None:
    """Closing stderr at launch preserves publication and keeps stdout empty."""
    experiment = make_experiment(
        persistent=tmp_path, trainer=write_constant_trainer(tmp_path), n_trials=1, gpu_policy="none"
    )
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    # Close the descriptor before exec so the CLI starts with sys.stderr unset.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sys; os.close(2); os.execv(sys.argv[1], sys.argv[1:])",
            str(console_script("phasesweep")),
            "run",
            str(config),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert _resolve_publication_pointer(experiment).state == "ok"


# Makes the pseudo-terminal its controlling terminal and standard streams, sets SIGHUP's
# disposition, then execs the command, as a shell on that terminal starts what is typed into it.
_ON_TERMINAL = (
    "import os, signal, sys\n"
    "os.login_tty(int(sys.argv[1]))\n"
    "signal.signal(signal.SIGHUP, signal.Handlers[sys.argv[2]])\n"
    "os.execv(sys.argv[3], sys.argv[3:])\n"
)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("sighup", "status"),
    [("SIG_DFL", 128 + signal.SIGHUP), ("SIG_IGN", 0)],
    ids=["hangup-stops-the-run", "hangup-ignored"],
)
def test_a_terminal_closed_mid_trial_leaves_the_runs_own_exit_status(
    tmp_path: Path, sighup: str, status: int
) -> None:
    """Every write after a dropped SSH session fails, yet the run its hangup stops exits 129, and one that ignores the hangup completes and exits 0."""
    reached, release = tmp_path / "reached", tmp_path / "release"
    trainer = write_trainer(
        tmp_path / "gated_train.py",
        f"""
        import time
        from pathlib import Path
        Path({str(reached)!r}).touch()
        while not Path({str(release)!r}).exists():
            time.sleep(0.05)
        print("x=0.5")
        """,
    )
    experiment = make_experiment(
        workdir=tmp_path / "runs", trainer=trainer, n_trials=1, gpu_policy="none"
    )
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    master, slave = os.openpty()
    try:
        run = subprocess.Popen(
            [sys.executable, "-c", _ON_TERMINAL, str(slave), sighup]
            + [str(console_script("phasesweep")), "run", str(config)],
            pass_fds=(slave,),
        )
    finally:
        os.close(slave)
    try:
        deadline = time.monotonic() + 60
        while not reached.exists():
            assert run.poll() is None, f"run exited ({run.returncode}) before its trial started"
            assert time.monotonic() < deadline, "the trial never started"
            time.sleep(0.05)
        os.close(master)
        master = -1
        if sighup == "SIG_IGN":
            # The run outlives the hangup, so its trial finishes once released.
            release.touch()
        assert run.wait(timeout=60) == status
    finally:
        if master != -1:
            os.close(master)
        release.touch()
        if run.poll() is None:
            run.kill()
            run.wait(timeout=10)


_VALIDATE_CLOSING_AT_80 = [
    "Checked the config alone. A dry run also compares it with the recorded study and",
    "its trials' environment, and checks the trial program on PATH:",
]
_VALIDATE_CLOSING_AT_60 = [
    "Checked the config alone. A dry run also compares it with",
    "the recorded study and its trials' environment, and checks",
    "the trial program on PATH:",
]


@pytest.mark.parametrize(
    ("columns", "sentence"), [(80, _VALIDATE_CLOSING_AT_80), (60, _VALIDATE_CLOSING_AT_60)]
)
def test_validate_says_what_it_left_to_the_dry_run_and_gives_its_command_whole(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    columns: int,
    sentence: list[str],
) -> None:
    """A passing validation ends with one sentence naming the checks only a dry run makes, then the command, flush left and unbroken."""
    config = tmp_path / "my sweep.yaml"
    assert invoke_cli_boundary(["init", "-o", str(config)], monkeypatch) == 0
    capsys.readouterr()

    with terminal_stream("stdout", columns) as stdout:
        status = invoke_cli_boundary(
            ["validate", str(config), "--set", "phases[depth].gpu_policy=none"], monkeypatch
        )

    assert status == 0
    lines = stdout.text.splitlines()
    assert lines[0] == "OK: phasesweep_starter (2 phases)"
    assert lines[-(len(sentence) + 1) :] == [
        *sentence,
        f"phasesweep run '{config}' --set 'phases[depth].gpu_policy=none' --dry-run",
    ]
    assert all(len(line) <= columns for line in sentence)
    assert sum(line.startswith("Checked the config alone") for line in lines) == 1


def test_a_from_phase_naming_no_phase_is_a_usage_error_that_lists_the_phases_in_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal is click's usage error with exit status 2, naming the real phases without Python list syntax."""
    config = tmp_path / "experiment.yaml"
    assert invoke_cli_boundary(["init", "-o", str(config)], monkeypatch) == 0
    capsys.readouterr()

    status = invoke_cli_boundary(["run", str(config), "--from-phase", "nope"], monkeypatch)

    assert status == 2
    assert capsys.readouterr().err.splitlines() == [
        "Usage: phasesweep run [OPTIONS] CONFIG",
        "Try 'phasesweep run --help' for help.",
        "",
        "Error: Invalid value for --from-phase: 'nope' is not a phase of this config; "
        "its phases are `depth`, `learning_rate`.",
    ]
