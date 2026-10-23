"""On-disk layout: <workdir>/<experiment>/<phase>/ namespacing, summary.yaml placement, experiment-name validation, and the permission modes of atomically written artifacts."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from phasesweep.engine.paths import (
    _experiment_dir,
    _generation_summary_path,
    _generation_winner_path,
    _phase_dir,
)
from phasesweep.engine.read import experiment_status
from phasesweep.engine.run import run_experiment
from phasesweep.runtime.files import (
    atomic_create_text,
    atomic_text_writer,
    atomic_write_text,
    ensure_artifact_dir,
    private_atomic_write_text,
)
from tests.conftest import make_experiment, temporary_umask, write_constant_trainer
from tests.ledger_fixtures import published_generation_id


def test_experiment_artifact_paths_share_namespaced_layout(tmp_path: Path) -> None:
    """Experiment, phase, and summary artifacts share one namespaced root."""
    exp = make_experiment(workdir=str(tmp_path / "runs"))
    root = (tmp_path / "runs" / exp.experiment).resolve()

    assert _experiment_dir(exp) == root
    assert _phase_dir(exp, "p") == root / "p"
    assert (
        _generation_summary_path(exp, "generation") == root / "generations/generation/summary.yaml"
    )
    assert (
        _generation_winner_path(exp, "generation", "p")
        == root / "generations/generation/phases/p/winner.yaml"
    )


def test_two_experiments_sharing_workdir_have_disjoint_output_trees(
    tmp_path: Path,
) -> None:
    """Two configs with the same ``workdir`` but different experiment names
    must not share any output paths, or one run would silently overwrite the
    other's ``trial_*/``, ``winner.yaml``, and ``summary.yaml``.
    """
    exp_a = make_experiment(workdir=str(tmp_path / "runs"))
    exp_b = make_experiment(workdir=str(tmp_path / "runs"))
    exp_b = exp_b.model_copy(update={"experiment": "other"})

    a_dir = _experiment_dir(exp_a)
    b_dir = _experiment_dir(exp_b)
    assert a_dir != b_dir
    # Neither path is a prefix of the other.
    assert not str(a_dir).startswith(str(b_dir) + "/")
    assert not str(b_dir).startswith(str(a_dir) + "/")


@pytest.mark.parametrize(
    ("workdir_name", "existing_workdir"), [("runs", False), ("runs", True), (".", True)]
)
@pytest.mark.integration
def test_run_experiment_writes_summary_at_namespaced_path(
    tmp_path: Path, workdir_name: str, existing_workdir: bool
) -> None:
    """A run publishes generation results without experiment-root or phase-root copies."""
    # Keep the test fixture's host lock directory outside this source repository.
    tmp_path = tmp_path / "repository"
    tmp_path.mkdir()
    trainer = write_constant_trainer(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "experiment.yaml"
    source.write_text("# operator-authored configuration\n")
    exp = make_experiment(workdir=str(tmp_path / workdir_name), trainer=trainer)
    if existing_workdir:
        Path(exp.workdir).mkdir(exist_ok=True)
    run_experiment(exp)
    generation_id = published_generation_id(exp)
    assert generation_id is not None
    assert _generation_summary_path(exp, generation_id).is_file()
    assert _generation_winner_path(exp, generation_id, "p").is_file()
    assert not (_experiment_dir(exp) / "summary.yaml").exists()
    assert not (_phase_dir(exp, "p") / "winner.yaml").exists()
    # Nothing is written directly under the bare workdir.
    assert not (Path(exp.workdir).resolve() / "summary.yaml").exists()
    assert (_experiment_dir(exp) / ".gitignore").read_text() == "*\n"
    assert (_experiment_dir(exp) / ".ignore").read_text() == "!**\n"
    assert not (Path(exp.workdir) / ".gitignore").exists()
    visible = subprocess.run(
        [
            "git",
            "-c",
            "core.excludesFile=/dev/null",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert set(visible) == {f"?? {trainer.name}", f"?? {source.name}"}


@pytest.mark.parametrize("existing_ignore", [None, "keep-me\n"])
def test_existing_artifact_dir_adds_only_missing_ignore(
    tmp_path: Path, existing_ignore: str | None
) -> None:
    artifact_dir = tmp_path / "runs" / "experiment"
    artifact_dir.mkdir(parents=True)
    ignore = artifact_dir / ".gitignore"
    if existing_ignore is not None:
        ignore.write_text(existing_ignore)
    ensure_artifact_dir(artifact_dir)
    assert ignore.read_text() == (existing_ignore if existing_ignore is not None else "*\n")
    assert (artifact_dir / ".ignore").read_text() == "!**\n"


def test_new_artifact_dir_does_not_modify_repository_ignore(tmp_path: Path) -> None:
    ignore = tmp_path / ".gitignore"
    ignore.write_text("existing-rule\n")
    ensure_artifact_dir(tmp_path / "nested" / "runs" / "experiment")
    assert ignore.read_text() == "existing-rule\n"
    assert (tmp_path / "nested" / "runs" / "experiment" / ".gitignore").read_text() == "*\n"


def test_inspection_does_not_create_workdir(tmp_path: Path) -> None:
    exp = make_experiment(workdir=str(tmp_path / "runs"))
    run_experiment(exp, dry_run=True)
    experiment_status(exp)
    assert not Path(exp.workdir).exists()


@pytest.mark.parametrize(
    "bad_name",
    [
        "../../etc/evil",  # path separators escape the workdir
        "my experiment",  # whitespace breaks lock-file paths
    ],
)
def test_experiment_name_rejected(bad_name: str) -> None:
    """Experiment name is used in lock-file paths; unsafe characters break that."""
    with pytest.raises(ValueError, match="Experiment name"):
        make_experiment(experiment=bad_name)


def test_experiment_name_accepts_valid() -> None:
    """Alphanumeric, underscore, and hyphen are safe."""
    make_experiment(experiment="tiny_lm-16mb")


@pytest.mark.parametrize(("mask", "expected_mode"), [(0o022, 0o644), (0o077, 0o600)])
def test_atomic_write_text_applies_process_umask_to_new_artifacts(
    tmp_path: Path, mask: int, expected_mode: int
) -> None:
    """Fresh shared artifacts use the kernel-masked ordinary file mode."""
    path = tmp_path / "winner.yaml"

    with temporary_umask(mask):
        atomic_write_text(path, "value: 1\n")

    assert path.read_text(encoding="utf-8") == "value: 1\n"
    assert stat.S_IMODE(path.stat().st_mode) == expected_mode


def test_atomic_write_text_preserves_existing_artifact_mode(tmp_path: Path) -> None:
    """Rewriting an existing artifact keeps whatever mode the operator left on
    it, even under a umask that would have created it more restrictively.
    """
    path = tmp_path / "summary.yaml"
    path.write_text("old\n", encoding="utf-8")
    path.chmod(0o640)

    with temporary_umask(0o077):
        atomic_write_text(path, "new\n")

    assert path.read_text(encoding="utf-8") == "new\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.parametrize("mask", [0o000, 0o022, 0o777])
def test_private_atomic_write_text_stays_owner_only_under_any_umask(
    tmp_path: Path, mask: int
) -> None:
    """Private artifacts remain exactly ``0600`` regardless of process umask."""
    path = tmp_path / "state.json"

    with temporary_umask(mask):
        private_atomic_write_text(path, "{}\n")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_private_file_in_shared_tree_creates_umask_governed_parents(tmp_path: Path) -> None:
    path = tmp_path / "artifact-root" / "generation" / "config.snapshot.yaml"

    with temporary_umask(0o022):
        private_atomic_write_text(path, "secret\n", require_private_dir=False)

    assert stat.S_IMODE((tmp_path / "artifact-root").stat().st_mode) == 0o755
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o755
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_atomic_writer_does_not_inherit_mode_through_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("target\n")
    target.chmod(0o600)
    destination = tmp_path / "summary.yaml"
    destination.symlink_to(target)

    with temporary_umask(0o022):
        atomic_write_text(destination, "replacement\n")

    assert destination.is_file() and not destination.is_symlink()
    assert destination.read_text() == "replacement\n"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644
    assert target.read_text() == "target\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_atomic_text_writer_leaves_no_temporary_files(tmp_path: Path) -> None:
    """Staging files are cleaned up on both the commit path and the failure path."""
    artifacts = tmp_path / "artifacts"
    committed = artifacts / "trials.csv"
    with atomic_text_writer(committed) as handle:
        handle.write("a,b\n")
    assert sorted(p.name for p in artifacts.iterdir()) == ["trials.csv"]

    failed = artifacts / "aborted.csv"
    with (
        pytest.raises(RuntimeError, match="writer exploded"),
        atomic_text_writer(failed) as handle,
    ):
        handle.write("partial")
        raise RuntimeError("writer exploded")

    assert not failed.exists()
    assert sorted(p.name for p in artifacts.iterdir()) == ["trials.csv"]


def test_atomic_create_validation_receives_closed_fsynced_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exclusive-create validation runs after durable staging closes its descriptor."""
    target = tmp_path / "artifacts" / "config.yaml"
    fsynced: list[int] = []
    real_fsync = os.fsync

    def remember_fsync(fd: int) -> None:
        real_fsync(fd)
        fsynced.append(fd)

    def validate(staged: Path) -> None:
        assert staged.read_text() == "complete\n"
        assert len(fsynced) == 1
        with pytest.raises(OSError):
            os.fstat(fsynced[0])
        assert not target.exists()

    monkeypatch.setattr(os, "fsync", remember_fsync)
    assert atomic_create_text(target, "complete\n", validate=validate)
    assert list(target.parent.iterdir()) == [target]


def test_atomic_create_preserves_exhausted_staging_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ten staging collisions preserve the exclusive creator's diagnostic."""
    from phasesweep.runtime import files as runtime_files

    target = tmp_path / "config.yaml"
    (tmp_path / ".config.yaml.collision.tmp").write_text("reserved")
    monkeypatch.setattr(runtime_files.secrets, "token_hex", lambda _size: "collision")

    with pytest.raises(FileExistsError, match=f"cannot create a staging file beside {target}"):
        atomic_create_text(target, "new")
    assert not target.exists()
    assert (tmp_path / ".config.yaml.collision.tmp").read_text() == "reserved"


@pytest.mark.parametrize("generation_id", ["../escape", "/tmp/escape", ""])
def test_run_experiment_rejects_unsafe_generation_id_before_writing_state(
    tmp_path: Path,
    generation_id: str,
) -> None:
    experiment = make_experiment(workdir=tmp_path / "runs")

    with pytest.raises(ValueError, match="generation name"):
        run_experiment(experiment, generation_id=generation_id)

    assert not _experiment_dir(experiment).exists()
