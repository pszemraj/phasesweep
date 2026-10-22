"""The trainer's git revision recorded in each generation's reproducibility record."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from phasesweep import run_experiment
from phasesweep.config.models import ExecutionContext, Experiment
from phasesweep.engine.paths import _generation_reproducibility_path
from phasesweep.engine.provenance import _code_revision
from phasesweep.engine.state import REPRODUCIBILITY_SCHEMA_VERSION
from tests.conftest import make_experiment, write_param_echo_trainer
from tests.ledger_fixtures import published_generation_id


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
    path = _generation_reproducibility_path(experiment, generation_id)
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
    """A hook's exported ``GIT_DIR`` and ``GIT_INDEX_FILE`` do not redirect the read."""
    trainer_head = _committed_repo(tmp_path / "trainer")
    other = tmp_path / "other"
    _committed_repo(other)
    (other / "tracked.txt").write_text("changed\n")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(other / ".git" / "index"))

    assert _code_revision(tmp_path / "trainer") == {"commit": trainer_head, "dirty": False}


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
