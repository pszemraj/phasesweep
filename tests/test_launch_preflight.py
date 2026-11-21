"""A launch refuses trials that could not start, before it allocates any.

The trial command runs through ``/bin/sh`` on the PATH the run starts with, in the trainer's
working directory. A program that PATH lacks, or a missing ``execution.cwd``, would fail every
trial before its trainer ran and record each as a failure, so the launch preflight refuses
them up front. The program check judges a plain first word when shell control flow permits.
``validate`` checks the directory and a program written with a ``/``, which do not depend on PATH.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import optuna
import pytest
from click.testing import CliRunner

from phasesweep import run_experiment
from phasesweep.cli import _operational_error_text
from phasesweep.cli import cli as cli_main
from phasesweep.config.models import ExecutionContext, Experiment, Phase
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.errors import LaunchEnvironmentError, StudyFingerprintMismatchError
from phasesweep.engine.guards import _raise_collected_errors
from phasesweep.engine.paths import _artifact_root_binding_path
from phasesweep.engine.study_policy import (
    _launch_preflight,
    _require_launch_target,
    _trial_program_refusal,
)
from phasesweep.engine.trial import _environment_identity, _missing_trainer_cwd
from phasesweep.errors import OperatorAction, PhaseSweepError
from tests.conftest import (
    make_experiment,
    phase_trials,
    study_with_stamped_environment,
    write_constant_trainer,
    write_experiment_config,
)

_MISSING = "phasesweep-test-no-such-trainer"


def _executable(path: Path) -> Path:
    """Write an executable shell script at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _with_target(experiment: Experiment, n_trials: int) -> Experiment:
    """Return ``experiment`` with every phase's trial target set to ``n_trials``."""
    phases = [phase.model_copy(update={"n_trials": n_trials}) for phase in experiment.phases]
    return experiment.model_copy(update={"phases": phases})


@pytest.mark.parametrize("program", ["trainer", "trai\\\nner"])
def test_program_on_the_path_is_found(tmp_path: Path, program: str) -> None:
    """A plain first word that an executable on PATH matches passes the program check."""
    _executable(tmp_path / "bin" / "trainer")

    assert (
        _trial_program_refusal(f"{program} --lr 1", path=str(tmp_path / "bin"), cwd=tmp_path)
        is None
    )


def test_program_missing_from_the_path_is_refused_with_the_path_it_searched(
    tmp_path: Path,
) -> None:
    """The refusal names the PATH it searched and the way to put the program on it."""
    refusal = _trial_program_refusal(
        f"{_MISSING} --lr 1", path="/nowhere:/also/nowhere", cwd=tmp_path
    )

    assert refusal is not None
    message, remediation = refusal
    assert f"program {_MISSING!r} is not on the trainer's PATH (/nowhere:/also/nowhere)" in message
    assert remediation == (
        "Put the program on the PATH used to launch PhaseSweep (activate the environment containing "
        "the program for CLI runs, or set PATH in the MCP server entry), or name it by absolute path "
        "in trial_command."
    )


def test_program_missing_from_a_path_the_config_sets_is_sent_to_the_config(tmp_path: Path) -> None:
    """When the config's ``env`` sets ``PATH``, the fix line names that, not the shell or client."""
    refusal = _trial_program_refusal(
        f"{_MISSING} --lr 1", path="/nowhere", cwd=tmp_path, path_from_config=True
    )

    assert refusal is not None
    assert refusal[1] == (
        "Put the program on the PATH the config's `env` sets, or name it by absolute path in "
        "trial_command."
    )


def test_relative_path_entry_resolves_against_the_trainer_directory(tmp_path: Path) -> None:
    """A relative PATH entry is searched from the trainer's directory even when it contains a colon."""
    cwd = tmp_path / "trainer:project"
    _executable(cwd / "tools" / "trainer")

    assert _trial_program_refusal("trainer", path="tools", cwd=cwd) is None


@pytest.mark.parametrize("program", ["./run.sh", "scripts/run.sh"])
def test_relative_program_resolves_against_the_trainer_directory(
    tmp_path: Path, program: str
) -> None:
    """A program named by a relative path is found from the trainer's directory and nowhere else."""
    _executable(tmp_path / program)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    assert _trial_program_refusal(f"{program} --x 1", path="/nowhere", cwd=tmp_path) is None
    refusal = _trial_program_refusal(f"{program} --x 1", path="/nowhere", cwd=elsewhere)
    assert refusal is not None
    assert f"resolves to {elsewhere / program}, which is not an executable file" in refusal[0]


def test_absolute_program_must_be_an_executable_file(tmp_path: Path) -> None:
    """An absolute program that lacks the execute bit or is a directory is refused."""
    script = tmp_path / "train.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o644)

    assert _trial_program_refusal(f"{script} --x 1", path=None, cwd=tmp_path) is not None
    assert _trial_program_refusal(str(tmp_path), path=None, cwd=tmp_path) is not None
    assert _trial_program_refusal(f"{_executable(script)} --x 1", path=None, cwd=tmp_path) is None


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(f"if true; then {_MISSING}; fi", id="reserved-word"),
        pytest.param(f"exec {_MISSING} --x 1", id="builtin"),
        pytest.param("chdir .", id="dash-chdir-builtin"),
        pytest.param(f"CUDA_VISIBLE_DEVICES=0 {_MISSING}", id="assignment"),
        pytest.param(f"$PYTHON {_MISSING}.py", id="parameter-expansion"),
        pytest.param(f"~/bin/{_MISSING}", id="tilde"),
        pytest.param("{python} train.py --config {config_path}", id="placeholder"),
        pytest.param(f"{_MISSING}* --x 1", id="glob"),
        pytest.param(f"'{_MISSING}' --x 1", id="quoted"),
        pytest.param("", id="empty"),
        pytest.param(f"{_MISSING} --x 1 || true", id="or-fallback"),
        pytest.param(f"{_MISSING} ${{PATH:=/tools}}", id="argument-can-set-path"),
        pytest.param(f"{_MISSING} --x 1 ; true", id="sequence"),
        pytest.param(f"{_MISSING} --x 1 | tee out.log", id="pipe"),
        pytest.param(f"{_MISSING} --x 1 &", id="background"),
        pytest.param(rf"{_MISSING} \&& true", id="escaped-ampersand-before-background"),
        pytest.param(rf"{_MISSING} \>&1", id="escaped-redirection-before-background"),
        pytest.param(f"{_MISSING} &>out.log", id="non-posix-redirect-or-background"),
        pytest.param(f"{_MISSING} --x 1\ntrue", id="newline-between-commands"),
        pytest.param(
            f'{_MISSING} "$(echo "\'")" || echo "$(echo "\'")"',
            id="fallback-after-nested-quotes-in-command-substitution",
        ),
        pytest.param(f"{_MISSING} # \\\ntrue", id="backslash-in-comment"),
        pytest.param(f"{_MISSING} \\\\\ntrue", id="escaped-backslash-before-newline"),
        pytest.param(f"{_MISSING} --x 1 # don't\ntrue # it's", id="comment-quotes-span-lines"),
    ],
)
def test_program_check_skips_what_it_cannot_judge(tmp_path: Path, command: str) -> None:
    """None of these is refused, even on a PATH where no program can be found."""
    assert _trial_program_refusal(command, path=str(tmp_path), cwd=tmp_path) is None


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(f"{_MISSING} --x 1", id="simple"),
        pytest.param(f"{_MISSING} --note 'a b'", id="quoted-argument"),
        pytest.param(f"{_MISSING} --lr $LR", id="parameter-argument"),
        pytest.param(f"{_MISSING} >out.log", id="redirection"),
        pytest.param(f"{_MISSING} >out.log 2>&1", id="file-descriptor-duplication"),
        pytest.param(f"{_MISSING} \\\n  --x 1", id="continued-command"),
        pytest.param(f"{_MISSING} --x 1 && true", id="and-chain"),
        pytest.param(f"{_MISSING} --x 1\n", id="trailing-newline"),
        pytest.param(f"\n  {_MISSING} --x 1  ", id="surrounding-whitespace"),
    ],
)
def test_program_check_refuses_a_missing_program_in_a_simple_command(
    tmp_path: Path, command: str
) -> None:
    """A simple command whose program is missing fails before allocating a trial."""
    refusal = _trial_program_refusal(command, path=str(tmp_path), cwd=tmp_path)

    assert refusal is not None
    assert f"program {_MISSING!r} is not on the trainer's PATH" in refusal[0]


def test_program_check_skips_a_trainer_that_gets_no_path(tmp_path: Path) -> None:
    """Without PATH the shell searches its own default, which the check does not reproduce."""
    assert _trial_program_refusal(_MISSING, path=None, cwd=tmp_path) is None


def test_cohort_refusal_comes_before_the_program_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed PATH reads as a cohort change naming both values, not as a missing program."""
    experiment = make_experiment(trial_command=f"{_MISSING} {{overrides}}")
    monkeypatch.setenv("PATH", "/opt/old/bin")
    study = study_with_stamped_environment(_environment_identity(experiment, "p"))
    monkeypatch.setenv("PATH", "/opt/new/bin")

    with pytest.raises(
        StudyFingerprintMismatchError, match='PATH "/opt/old/bin" -> "/opt/new/bin"'
    ):
        _launch_preflight(experiment, "p", study)


def test_fresh_tree_refuses_a_missing_program_before_creating_anything(tmp_path: Path) -> None:
    """A launch whose program PATH lacks is refused before it binds the tree or opens a ledger."""
    experiment = make_experiment(
        persistent=tmp_path,
        trial_command=f"{_MISSING} --note 'a b' \\\n  {{overrides}} >out.log 2>&1",
    )

    with pytest.raises(PhaseSweepError, match=f"program {_MISSING!r} is not on") as caught:
        run_experiment(experiment)

    assert type(caught.value) is LaunchEnvironmentError
    assert caught.value.action is OperatorAction.FIX_CONFIG
    assert "Starting new trials requires this program" in str(caught.value)
    assert not _artifact_root_binding_path(experiment).exists()
    assert not (tmp_path / "studies.journal").exists()


def test_fresh_tree_refuses_a_missing_trainer_directory_before_creating_anything(
    tmp_path: Path,
) -> None:
    """A launch whose trainer directory is missing is refused with the fix, before it binds the tree."""
    experiment = make_experiment(
        persistent=tmp_path, execution=ExecutionContext(cwd=str(tmp_path / "gone"))
    )

    with pytest.raises(PhaseSweepError, match="which is not a directory on this host") as caught:
        run_experiment(experiment)

    assert caught.value.action is OperatorAction.FIX_CONFIG
    assert (
        caught.value.remediation
        == "Create that directory, or set execution.cwd to an existing one."
    )
    assert not _artifact_root_binding_path(experiment).exists()


@pytest.mark.integration
def test_resume_refuses_a_vanished_trainer_directory_without_recording_a_trial(
    tmp_path: Path,
) -> None:
    """The refusal precedes allocation, so the study holds exactly the trials it had."""
    trainer_dir = tmp_path / "trainer"
    trainer_dir.mkdir()
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_constant_trainer(tmp_path),
        execution=ExecutionContext(cwd=str(trainer_dir)),
        n_trials=1,
    )
    run_experiment(experiment)
    before = [(trial.number, trial.state) for trial in phase_trials(experiment)]
    trainer_dir.rmdir()

    with pytest.raises(PhaseSweepError, match="which is not a directory on this host"):
        run_experiment(_with_target(experiment, 3))
    with pytest.raises(PhaseSweepError, match="which is not a directory on this host"):
        run_experiment(_with_target(experiment, 3), dry_run=True)

    assert [(trial.number, trial.state) for trial in phase_trials(experiment)] == before
    assert before == [(0, optuna.trial.TrialState.COMPLETE)]


@pytest.mark.integration
def test_resume_reports_an_experiment_wide_refusal_once_with_its_fix(tmp_path: Path) -> None:
    """Two phases that both need trials share one refusal, which keeps its type and fix line."""
    program = tmp_path / "bin" / "trainer"
    program.parent.mkdir()
    program.write_text(
        f'#!/bin/sh\nexec {sys.executable} {write_constant_trainer(tmp_path)} "$@"\n'
    )
    program.chmod(0o755)
    phases = [
        Phase(
            name=name,
            n_trials=1,
            gpu_policy="none",
            sampler=Sampler(type="random", seed=0),
            search_space={f"{name}_x": IntParam(type="int", low=0, high=10)},
        )
        for name in ("a", "b")
    ]
    experiment = make_experiment(
        persistent=tmp_path,
        phases=phases,
        trial_command="trainer --out {trial_dir}/r.json {overrides}",
        env={"PATH": f"{program.parent}:/usr/bin:/bin"},
    )
    run_experiment(experiment)
    program.unlink()

    for dry_run in (False, True):
        with pytest.raises(PhaseSweepError, match="program 'trainer' is not on") as caught:
            run_experiment(_with_target(experiment, 3), dry_run=dry_run)

        error = caught.value
        assert type(error) is LaunchEnvironmentError
        assert str(error).count("is not on the trainer's PATH") == 1
        assert "multiple unsafe studies" not in str(error)
        assert error.action is OperatorAction.FIX_CONFIG
        assert error.remediation is not None
        assert _operational_error_text(error).endswith(f"\nfix: {error.remediation}")
    assert len(phase_trials(experiment, "a")) == len(phase_trials(experiment, "b")) == 1


def test_studies_refused_for_different_launch_environment_reasons_share_the_type() -> None:
    """Differing launch-environment refusals across studies aggregate under their own type."""
    with pytest.raises(LaunchEnvironmentError, match="multiple unsafe studies") as caught:
        _raise_collected_errors(
            [
                LaunchEnvironmentError("W&B logging is offline."),
                LaunchEnvironmentError("No program."),
            ]
        )

    assert caught.value.action is OperatorAction.FIX_CONFIG


@pytest.mark.integration
def test_dry_run_refuses_a_top_up_under_another_environment_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dry run applies the cohort check a launch applies, so it previews no refused trial."""
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_constant_trainer(tmp_path),
        execution=ExecutionContext(inherit_env=["PHASESWEEP_DATASET_REV"]),
        n_trials=1,
    )
    monkeypatch.setenv("PHASESWEEP_DATASET_REV", "revision-a")
    run_experiment(experiment)
    monkeypatch.setenv("PHASESWEEP_DATASET_REV", "revision-b")

    # The met target replays without allocating, so it needs no cohort.
    run_experiment(experiment, dry_run=True)
    for dry_run in (True, False):
        with pytest.raises(
            StudyFingerprintMismatchError,
            match='PHASESWEEP_DATASET_REV "revision-a" -> "revision-b". No trial was allocated.',
        ):
            run_experiment(_with_target(experiment, 2), dry_run=dry_run)

    assert len(phase_trials(experiment)) == 1


def test_dry_run_refuses_a_missing_program(tmp_path: Path) -> None:
    """The dry run refuses the program a launch would refuse, so it previews no doomed trial."""
    experiment = make_experiment(persistent=tmp_path, trial_command=f"{_MISSING} {{overrides}}")

    with pytest.raises(PhaseSweepError, match=f"program {_MISSING!r} is not on"):
        run_experiment(experiment, dry_run=True)


def test_validate_checks_the_trainer_directory_but_not_the_program(tmp_path: Path) -> None:
    """Every shell gives the same directory answer; the launch's PATH need not be this shell's."""
    config = tmp_path / "experiment.yaml"
    write_experiment_config(
        config,
        make_experiment(workdir=tmp_path / "runs", trial_command=f"{_MISSING} {{overrides}}"),
    )
    result = CliRunner().invoke(cli_main, ["validate", str(config)])
    assert result.exit_code == 0, result.output

    write_experiment_config(
        config,
        make_experiment(
            workdir=tmp_path / "runs", execution=ExecutionContext(cwd=str(tmp_path / "gone"))
        ),
    )
    result = CliRunner().invoke(cli_main, ["validate", str(config)])
    assert result.exit_code != 0
    assert isinstance(result.exception, PhaseSweepError)
    assert "which is not a directory on this host" in str(result.exception)


def test_launch_judges_the_program_on_the_trainer_path_not_the_shell_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pinned env.PATH is the PATH the trainer's shell searches."""
    _executable(tmp_path / "bin" / "trainer")
    pinned = make_experiment(
        trial_command="trainer {overrides}", env={"PATH": str(tmp_path / "bin")}
    )
    monkeypatch.setenv("PATH", os.defpath)
    assert shutil.which("trainer") is None

    assert _launch_preflight(pinned, "p", None).values["PATH"] == str(tmp_path / "bin")
    with pytest.raises(PhaseSweepError, match="program 'trainer' is not on"):
        _launch_preflight(make_experiment(trial_command="trainer {overrides}"), "p", None)


@pytest.mark.parametrize("program", ["/opt/nonexistent/venv/bin/python", "./train.sh"])
def test_validate_refuses_a_program_written_with_a_slash_that_is_not_executable(
    tmp_path: Path, program: str
) -> None:
    """No PATH changes the answer for a program with a ``/``, so ``validate`` refuses it as a launch does."""
    config = tmp_path / "experiment.yaml"
    write_experiment_config(
        config,
        make_experiment(
            workdir=tmp_path / "runs",
            trial_command=f"{program} train.py {{overrides}}",
            execution=ExecutionContext(cwd=str(tmp_path)),
        ),
    )

    result = CliRunner().invoke(cli_main, ["validate", str(config)])

    assert result.exit_code != 0
    assert isinstance(result.exception, PhaseSweepError)
    assert repr(program) in str(result.exception)
    assert "which is not an executable file" in str(result.exception)
    assert "Starting new trials requires this program" in str(result.exception)

    _executable(tmp_path / "train.sh")
    result = CliRunner().invoke(cli_main, ["validate", str(config)])
    assert (result.exit_code == 0) == (program == "./train.sh"), result.output


def test_the_any_shell_check_leaves_a_program_found_through_path_to_the_launch(
    tmp_path: Path,
) -> None:
    """Asked without a PATH, the check never judges a PATH program; the launch preflight does."""
    experiment = make_experiment(
        trial_command=f"{_MISSING} {{overrides}}",
        execution=ExecutionContext(cwd=str(tmp_path)),
        env={"PATH": "/nowhere"},
    )

    _require_launch_target(experiment)
    with pytest.raises(PhaseSweepError, match=f"program {_MISSING!r} is not on"):
        _launch_preflight(experiment, "p", None)


def test_a_path_that_is_already_absolute_is_printed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The directory and program refusals say where a path resolves only when it differs from the text."""
    gone = tmp_path.resolve() / "gone"
    experiment = make_experiment(execution=ExecutionContext(cwd=str(gone)))
    assert _missing_trainer_cwd(experiment) == (
        f"execution.cwd is {str(gone)!r}, which is not a directory on this host."
    )

    monkeypatch.setenv("HOME", str(tmp_path.resolve()))
    tilde = make_experiment(execution=ExecutionContext(cwd="~/gone"))
    assert _missing_trainer_cwd(tilde) == (
        f"execution.cwd '~/gone' resolves to {gone}, which is not a directory on this host."
    )

    absolute = _trial_program_refusal("/opt/nonexistent/bin/python x.py", path=None, cwd=tmp_path)
    assert absolute is not None
    assert absolute[0] == (
        "The trial command's program is '/opt/nonexistent/bin/python', "
        "which is not an executable file."
    )
