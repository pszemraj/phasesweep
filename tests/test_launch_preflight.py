"""A launch refuses trials that could not start, before it allocates any.

The trial command runs through ``/bin/sh`` on the PATH the run starts with, in the trainer's
working directory. A program that PATH lacks, or a missing ``execution.cwd``, would fail every
trial before its trainer ran and record each as a failure, so the launch preflight refuses
them up front. The program check judges only a plain first word and never refuses a command
it cannot judge. ``validate`` runs only the directory check, whose answer is the same from
any shell.
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
from phasesweep.engine.state import TRAINER_ENV_ATTR, TRAINER_ENV_DIGEST_ATTR
from phasesweep.engine.study_policy import _launch_preflight, _trial_program_refusal
from phasesweep.engine.trial import _environment_identity
from phasesweep.errors import OperatorAction, PhaseSweepError
from tests.conftest import (
    make_experiment,
    phase_trials,
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


def _completed_study(experiment: Experiment) -> optuna.Study:
    """Return a study holding one trial stamped with this process's trainer environment."""
    identity = _environment_identity(experiment, "p")
    study = optuna.create_study(study_name="t::p")
    study.add_trial(
        optuna.trial.create_trial(
            value=1.0,
            user_attrs={
                TRAINER_ENV_DIGEST_ATTR: identity.digest,
                TRAINER_ENV_ATTR: dict(identity.tracked),
            },
        )
    )
    return study


def test_program_on_the_path_is_found(tmp_path: Path) -> None:
    _executable(tmp_path / "bin" / "trainer")

    assert (
        _trial_program_refusal("trainer --lr 1", path=str(tmp_path / "bin"), cwd=tmp_path) is None
    )


def test_program_missing_from_the_path_is_refused_with_the_path_it_searched(
    tmp_path: Path,
) -> None:
    refusal = _trial_program_refusal(
        f"{_MISSING} --lr 1", path="/nowhere:/also/nowhere", cwd=tmp_path
    )

    assert refusal is not None
    message, remediation = refusal
    assert f"program {_MISSING!r} is not on the trainer's PATH (/nowhere:/also/nowhere)" in message
    assert remediation == (
        "Put the program on the PATH the run starts with, or name it by absolute path in "
        "trial_command."
    )


def test_relative_path_entry_resolves_against_the_trainer_directory(tmp_path: Path) -> None:
    _executable(tmp_path / "tools" / "trainer")

    assert _trial_program_refusal("trainer", path="tools", cwd=tmp_path) is None


@pytest.mark.parametrize("program", ["./run.sh", "scripts/run.sh"])
def test_relative_program_resolves_against_the_trainer_directory(
    tmp_path: Path, program: str
) -> None:
    _executable(tmp_path / program)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    assert _trial_program_refusal(f"{program} --x 1", path="/nowhere", cwd=tmp_path) is None
    refusal = _trial_program_refusal(f"{program} --x 1", path="/nowhere", cwd=elsewhere)
    assert refusal is not None
    assert f"resolves to {elsewhere / program}, which is not an executable file" in refusal[0]


def test_absolute_program_must_be_an_executable_file(tmp_path: Path) -> None:
    script = tmp_path / "train.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o644)

    assert _trial_program_refusal(f"{script} --x 1", path=None, cwd=tmp_path) is not None
    assert _trial_program_refusal(str(tmp_path), path=None, cwd=tmp_path) is not None
    assert _trial_program_refusal(f"{_executable(script)} --x 1", path=None, cwd=tmp_path) is None


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(f"{_MISSING} 'unbalanced", id="unsplittable"),
        pytest.param(f"if true; then {_MISSING}; fi", id="reserved-word"),
        pytest.param(f"exec {_MISSING} --x 1", id="builtin"),
        pytest.param("printf 'x=1\\n'", id="printf-builtin"),
        pytest.param(f". ./env.sh && {_MISSING}", id="dot-builtin"),
        pytest.param(f"CUDA_VISIBLE_DEVICES=0 {_MISSING}", id="assignment"),
        pytest.param(f"$PYTHON {_MISSING}.py", id="parameter-expansion"),
        pytest.param(f"~/bin/{_MISSING}", id="tilde"),
        pytest.param("{python} train.py --config {config_path}", id="placeholder"),
        pytest.param(f"{_MISSING}* --x 1", id="glob"),
        pytest.param(f"'{_MISSING}' --x 1", id="quoted"),
        pytest.param(f'{_MISSING}"" --x 1', id="partly-quoted"),
        pytest.param(f"{_MISSING}>out.log", id="redirection"),
        pytest.param("", id="empty"),
    ],
)
def test_program_check_skips_what_it_cannot_judge(tmp_path: Path, command: str) -> None:
    """None of these is refused, even on a PATH where no program can be found."""
    assert _trial_program_refusal(command, path=str(tmp_path), cwd=tmp_path) is None


def test_program_check_skips_a_trainer_that_gets_no_path(tmp_path: Path) -> None:
    """Without PATH the shell searches its own default, which the check does not reproduce."""
    assert _trial_program_refusal(_MISSING, path=None, cwd=tmp_path) is None


def test_cohort_refusal_comes_before_the_program_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed PATH reads as a cohort change naming both values, not as a missing program."""
    experiment = make_experiment(trial_command=f"{_MISSING} {{overrides}}")
    monkeypatch.setenv("PATH", "/opt/old/bin")
    study = _completed_study(experiment)
    monkeypatch.setenv("PATH", "/opt/new/bin")

    with pytest.raises(
        StudyFingerprintMismatchError, match='PATH "/opt/old/bin" -> "/opt/new/bin"'
    ):
        _launch_preflight(experiment, "p", study)


def test_fresh_tree_refuses_a_missing_program_before_creating_anything(tmp_path: Path) -> None:
    experiment = make_experiment(persistent=tmp_path, trial_command=f"{_MISSING} {{overrides}}")

    with pytest.raises(PhaseSweepError, match=f"program {_MISSING!r} is not on") as caught:
        run_experiment(experiment)

    assert type(caught.value) is LaunchEnvironmentError
    assert caught.value.action is OperatorAction.FIX_CONFIG
    assert "none was allocated" in str(caught.value)
    assert not _artifact_root_binding_path(experiment).exists()
    assert not (tmp_path / "studies.journal").exists()


def test_fresh_tree_refuses_a_missing_trainer_directory_before_creating_anything(
    tmp_path: Path,
) -> None:
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
