"""The trainer's git revision recorded in each generation's reproducibility record."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import optuna
import pytest
import yaml
from click.testing import CliRunner

from phasesweep import run_experiment
from phasesweep.cli import cli
from phasesweep.config.models import ExecutionContext, Experiment
from phasesweep.engine.paths import (
    _experiment_dir,
    _generation_dir,
    _generation_reproducibility_path,
)
from phasesweep.engine.provenance import _code_revision
from phasesweep.engine.selection import _ranked_code_revisions
from phasesweep.engine.state import GENERATION_ID_ATTR, REPRODUCIBILITY_SCHEMA_VERSION
from tests.conftest import (
    invoke_cli_boundary,
    make_experiment,
    write_experiment_config,
    write_param_echo_trainer,
    write_trial_zero_trainer,
)
from tests.ledger_fixtures import published_generation_id, winner_payload


@pytest.fixture
def hermetic_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep git in scratch repositories independent of the developer's and the caller's git state.

    A hook-run pytest inherits ``GIT_DIR`` and ``GIT_INDEX_FILE`` from the commit that
    launched it, which would point scratch-repository commands at the real repository.
    The ceiling keeps discovery from climbing out of ``tmp_path`` into an enclosing
    work tree, so a plain scratch directory is never inside one.
    """
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))


def _git(repo: Path, *args: str) -> str:
    """Run git in a scratch repository with a fixed identity and no signing.

    :param Path repo: Directory to run in.
    :param str args: Git subcommand and arguments.
    :raises subprocess.CalledProcessError: git exits nonzero.
    :return str: Stripped standard output.
    """
    command = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    completed = subprocess.run(
        [*command, *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _committed_repo(repo: Path) -> str:
    """Create a git work tree with one tracked file committed.

    :param Path repo: Directory to create.
    :return str: The commit id of HEAD.
    """
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("v1\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "init")
    return _git(repo, "rev-parse", "HEAD")


def _experiment_running_in(tmp_path: Path, trainer_cwd: Path) -> Experiment:
    """Build a persistent one-trial experiment whose trainer runs in ``trainer_cwd``.

    :param Path tmp_path: Per-test directory holding the ledger, workdir and trainer.
    :param Path trainer_cwd: Directory the trainer runs in (``execution.cwd``).
    :return Experiment: The experiment, not yet run.
    """
    return make_experiment(
        persistent=tmp_path,
        trainer=write_param_echo_trainer(tmp_path),
        n_trials=1,
        fixed_overrides={"batch_size": 8},
        execution=ExecutionContext(cwd=str(trainer_cwd)),
    )


def _published_record(experiment: Experiment) -> dict[str, object]:
    """Read the published generation's ``reproducibility.json``.

    :param Experiment experiment: Experiment that has published a generation.
    :return dict[str, object]: The parsed record.
    """
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    path = _generation_reproducibility_path(_experiment_dir(experiment), generation_id)
    return json.loads(path.read_text())


def test_code_revision_reports_commit_and_whether_tracked_files_changed(
    tmp_path: Path, hermetic_git: None
) -> None:
    """A clean tree is not dirty; a modified tracked file is; an untracked file is ignored."""
    repo = tmp_path / "repo"
    head = _committed_repo(repo)
    assert len(head) == 40

    assert _code_revision(repo) == {"commit": head, "dirty": False}

    (repo / "untracked.txt").write_text("new\n")
    assert _code_revision(repo) == {"commit": head, "dirty": False}

    (repo / "tracked.txt").write_text("v2\n")
    assert _code_revision(repo) == {"commit": head, "dirty": True}

    # A subdirectory of the work tree resolves to the same repository.
    (repo / "sub").mkdir()
    assert _code_revision(repo / "sub") == {"commit": head, "dirty": True}


def test_code_revision_ignores_a_repository_inherited_from_a_git_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hermetic_git: None
) -> None:
    """Hook repository variables redirect neither generation nor fixture provenance."""
    from tests.fixtures.make_ledger_fixtures import _git_describe

    trainer_head = _committed_repo(tmp_path / "trainer")
    expected_description = _git(tmp_path / "trainer", "describe", "--tags", "--always", "--dirty")
    other = tmp_path / "other"
    _committed_repo(other)
    (other / "tracked.txt").write_text("changed\n")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    monkeypatch.setenv("GIT_INDEX_FILE", str(other / ".git" / "index"))

    assert _code_revision(tmp_path / "trainer") == {"commit": trainer_head, "dirty": False}
    assert _git_describe(tmp_path / "trainer") == expected_description


def test_code_revision_is_none_when_git_cannot_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hermetic_git: None
) -> None:
    """Outside a work tree, before the first commit, without git, or past the timeout: ``None``."""
    plain = tmp_path / "plain"
    plain.mkdir()
    assert _code_revision(plain) is None
    assert _code_revision(tmp_path / "missing") is None

    unborn = tmp_path / "unborn"
    unborn.mkdir()
    _git(unborn, "init", "-q")
    assert _code_revision(unborn) is None

    repo = tmp_path / "repo"
    _committed_repo(repo)
    with monkeypatch.context() as patch:
        patch.setenv("PATH", str(tmp_path / "no-git-here"))
        assert _code_revision(repo) is None

    def time_out(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="git", timeout=10)

    monkeypatch.setattr(subprocess, "run", time_out)
    assert _code_revision(repo) is None


@pytest.mark.integration
def test_generation_records_the_trainer_cwd_git_revision_without_comparing_it(
    tmp_path: Path, hermetic_git: None
) -> None:
    """Each generation records its own trainer-cwd revision, and a changed revision never refuses a resume."""
    repo = tmp_path / "repo"
    first_head = _committed_repo(repo)
    experiment = _experiment_running_in(tmp_path, repo)

    run_experiment(experiment)
    first = _published_record(experiment)
    assert first["code_revision"] == {"commit": first_head, "dirty": False}
    assert first["schema_version"] == REPRODUCIBILITY_SCHEMA_VERSION

    # Tracked changes mark the next generation dirty without refusing the resume.
    (repo / "tracked.txt").write_text("v2\n")
    phase = experiment.phases[0].model_copy(update={"n_trials": 2})
    experiment = experiment.model_copy(update={"phases": [phase]})
    run_experiment(experiment)
    second = _published_record(experiment)
    assert second["generation_id"] != first["generation_id"]
    assert second["code_revision"] == {"commit": first_head, "dirty": True}
    assert second["config_fingerprint"] == first["config_fingerprint"]

    # A new commit moves the recorded revision; the study fingerprint stays put.
    _git(repo, "commit", "-q", "-am", "docs only")
    second_head = _git(repo, "rev-parse", "HEAD")
    phase = experiment.phases[0].model_copy(update={"n_trials": 3})
    experiment = experiment.model_copy(update={"phases": [phase]})
    run_experiment(experiment)
    third = _published_record(experiment)
    assert third["code_revision"] == {"commit": second_head, "dirty": False}
    assert third["config_fingerprint"] == first["config_fingerprint"]


@pytest.mark.integration
def test_generation_records_no_revision_for_a_trainer_cwd_outside_git(
    tmp_path: Path, hermetic_git: None
) -> None:
    """The record keeps the key, as ``null``, when the trainer cwd is not in a git work tree."""
    plain = tmp_path / "plain"
    plain.mkdir()
    experiment = _experiment_running_in(tmp_path, plain)

    run_experiment(experiment)
    record = _published_record(experiment)
    assert "code_revision" in record
    assert record["code_revision"] is None


@pytest.mark.integration
@pytest.mark.parametrize("state", ["clean", "dirty", "outside-git"])
def test_show_winners_names_the_code_revision_of_the_publishing_generation(
    tmp_path: Path, hermetic_git: None, state: str
) -> None:
    """The publication revision is the publishing generation's, not the checkout's today."""
    cwd = tmp_path / "repo"
    if state == "outside-git":
        cwd.mkdir()
        head = None
    else:
        head = _committed_repo(cwd)
    if state == "dirty":
        (cwd / "tracked.txt").write_text("v2\n")
    experiment = _experiment_running_in(tmp_path, cwd)
    run_experiment(experiment)
    config_path = tmp_path / "experiment.yaml"
    write_experiment_config(config_path, experiment)

    # A later commit must not change what an earlier generation reports.
    if head is not None:
        (cwd / "tracked.txt").write_text("v3\n")
        _git(cwd, "commit", "-q", "-am", "later")

    result = CliRunner().invoke(cli, ["show-winners", str(config_path)])
    assert result.exit_code == 0, result.output
    expected = {
        "clean": f"# publication code revision: {head}\n",
        "dirty": f"# publication code revision: {head} (dirty)\n",
        "outside-git": (
            "# publication code revision: unknown (not a git work tree or git unavailable)\n"
        ),
    }[state]
    assert expected in result.output
    assert result.output.index(expected) < result.output.index("=== p ===")
    assert "winner trial from generation" not in result.output


def _carried_winner_study(tmp_path: Path) -> tuple[Path, Experiment, str, str, str, str]:
    """Run two generations whose second publishes a winner carried from the first.

    :param Path tmp_path: Per-test directory holding the ledger, workdir, trainer and repository.
    :return tuple[Path, Experiment, str, str, str, str]: The config path, the experiment, the
        first and second generation ids, and the git commit each generation ran under.
    """
    repo = tmp_path / "repo"
    first_head = _committed_repo(repo)
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_trial_zero_trainer(tmp_path, otherwise='print("x=9.0")'),
        n_trials=1,
        execution=ExecutionContext(cwd=str(repo)),
    )
    run_experiment(experiment)
    first_generation = published_generation_id(experiment)

    (repo / "tracked.txt").write_text("v2\n")
    _git(repo, "commit", "-q", "-am", "change the trainer")
    second_head = _git(repo, "rev-parse", "HEAD")
    phase = experiment.phases[0].model_copy(update={"n_trials": 2})
    experiment = experiment.model_copy(update={"phases": [phase]})
    run_experiment(experiment)
    second_generation = published_generation_id(experiment)
    assert first_generation is not None and second_generation is not None
    assert second_generation != first_generation
    config_path = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    return config_path, experiment, first_generation, second_generation, first_head, second_head


def test_a_ranked_generation_without_a_readable_record_reads_as_unreadable(
    tmp_path: Path,
) -> None:
    """Selection never raises over a missing or damaged generation record; it reports ``unreadable``."""
    experiment = make_experiment(persistent=tmp_path, trainer=tmp_path / "trainer.py")
    study = optuna.create_study()
    for number in range(2):
        study.add_trial(
            optuna.trial.create_trial(
                value=0.5,
                params={"x": number},
                distributions={"x": optuna.distributions.IntDistribution(0, 9)},
                user_attrs={GENERATION_ID_ATTR: f"missing-{number}"},
            )
        )
    trials = study.get_trials(deepcopy=False)

    assert _ranked_code_revisions(experiment, trials) == {
        "missing-0": "unreadable",
        "missing-1": "unreadable",
    }
    assert _ranked_code_revisions(experiment, trials[:1]) == {}


@pytest.mark.integration
def test_a_top_up_under_another_commit_is_disclosed_on_the_winner(
    tmp_path: Path,
    hermetic_git: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Trials ranked across two commits flag ``code_revision`` in the winner, the run summary and ``show-winners``."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("COLUMNS", "500")
    repo = tmp_path / "repo"
    first_head = _committed_repo(repo)
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_trial_zero_trainer(tmp_path, otherwise='print("x=9.0")'),
        n_trials=1,
        gpu_policy="none",
        execution=ExecutionContext(cwd=str(repo)),
    )
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 0
    (repo / "tracked.txt").write_text("v2\n")
    _git(repo, "commit", "-q", "-am", "change the trainer")
    second_head = _git(repo, "rev-parse", "HEAD")
    capsys.readouterr()

    top_up = ["--set", "phases[p].n_trials=2"]
    assert invoke_cli_boundary(["run", str(config), *top_up], monkeypatch) == 0

    err = capsys.readouterr().err
    assert (
        "code_revision: ranked trials span 2 generations whose trainer code may differ: "
        f"{first_head[:12]} (trial 0), {second_head[:12]} (trial 1)"
    ) in err
    assert "flagged: code_revision\n" in err
    experiment = experiment.model_copy(
        update={"phases": [experiment.phases[0].model_copy(update={"n_trials": 2})]}
    )
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    check = winner_payload(experiment, generation_id)["confound"]["checks"]["code_revision"]
    assert check["verdict"] == "heterogeneous"
    assert [(entry["commit"], entry["trials"]) for entry in check["detail"]["generations"]] == [
        (first_head, [0]),
        (second_head, [1]),
    ]
    shown = CliRunner().invoke(cli, ["show-winners", str(config), *top_up])
    assert shown.exit_code == 0, shown.output
    summary = yaml.safe_load(shown.output.split("=== p ===", 1)[1])["confound"]["code_revision"]
    assert summary["verdict"] == "heterogeneous"
    assert (summary["generations"], summary["commits"], summary["dirty"]) == (2, 2, 0)
    assert "rename" not in summary["remedy"] and "retrain" not in summary["remedy"]


@pytest.mark.integration
def test_show_winners_names_a_carried_winners_own_code_revision(
    tmp_path: Path, hermetic_git: None
) -> None:
    """A winner carried from an earlier generation shows that generation's revision beside the publication's."""
    config_path, _, first_generation, second_generation, first_head, second_head = (
        _carried_winner_study(tmp_path)
    )

    result = CliRunner().invoke(cli, ["show-winners", str(config_path)])
    assert result.exit_code == 0, result.output
    assert f"# publication code revision: {second_head}\n" in result.output
    assert (
        f"# winner generation: {second_generation} (published); "
        f"winner trial from generation {first_generation}, code revision {first_head}\n"
    ) in result.output


@pytest.mark.integration
def test_show_winners_carries_on_when_the_source_generation_record_is_gone(
    tmp_path: Path, hermetic_git: None
) -> None:
    """A carried winner whose earlier generation was removed reads ``unreadable`` on its line, and the listing continues."""
    config_path, experiment, first_generation, second_generation, _, second_head = (
        _carried_winner_study(tmp_path)
    )
    shutil.rmtree(_generation_dir(_experiment_dir(experiment), first_generation))

    result = CliRunner().invoke(cli, ["show-winners", str(config_path)])
    assert result.exit_code == 0, result.output
    assert f"# publication code revision: {second_head}\n" in result.output
    assert (
        f"# winner generation: {second_generation} (published); "
        f"winner trial from generation {first_generation}, code revision unreadable "
        "(its reproducibility.json cannot be read)\n"
    ) in result.output
    assert "trial_number:" in result.output.split("=== p ===", 1)[1]
