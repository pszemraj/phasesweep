"""On-disk layout: <workdir>/<experiment>/<phase>/ namespacing, summary.yaml placement, experiment-name validation, and the permission modes of atomically written artifacts."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from phasesweep.config import (
    Experiment,
    IntParam,
    LogRegexExtractor,
    Metric,
    Phase,
    Suite,
)
from phasesweep.engine import run_experiment, run_suite
from phasesweep.engine.paths import (
    _experiment_dir,
    _phase_dir,
    _summary_path,
)
from phasesweep.engine.run import experiment_status
from phasesweep.runtime.files import (
    atomic_text_writer,
    atomic_write_text,
    ensure_workdir,
    private_atomic_write_text,
)
from tests.conftest import make_experiment, temporary_umask, write_constant_trainer


def test_experiment_artifact_paths_share_namespaced_layout(tmp_path: Path) -> None:
    """Experiment, phase, and summary artifacts share one namespaced root."""
    exp = make_experiment(workdir=str(tmp_path / "runs"))
    root = (tmp_path / "runs" / exp.experiment).resolve()

    assert _experiment_dir(exp) == root
    assert _phase_dir(exp, "p") == root / "p"
    assert _summary_path(exp) == root / "summary.yaml"


def test_two_experiments_sharing_workdir_have_disjoint_output_trees(
    tmp_path: Path,
) -> None:
    """Two configs with the same ``workdir`` but different experiment names
    must not share any output paths — pre-v0.5.7 they did, which let one run
    silently overwrite the other's ``trial_*/``, ``winner.yaml``, and
    ``summary.yaml``.
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


@pytest.mark.parametrize("existing_workdir", [False, True])
def test_run_experiment_writes_summary_at_namespaced_path(
    tmp_path: Path, existing_workdir: bool
) -> None:
    """End-to-end: a real run must write ``summary.yaml`` under the
    ``<workdir>/<experiment>/`` tree, not directly under ``<workdir>``.
    """
    trainer = write_constant_trainer(tmp_path)
    exp = make_experiment(
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
    )
    if existing_workdir:
        Path(exp.workdir).mkdir()
    run_experiment(exp)
    assert (_summary_path(exp)).is_file()
    # The pre-v0.5.7 location must NOT be created.
    assert not (Path(exp.workdir).resolve() / "summary.yaml").exists()
    assert (Path(exp.workdir) / ".gitignore").read_text() == "*\n"


@pytest.mark.parametrize("existing_ignore", [None, "keep-me\n"])
def test_existing_workdir_adds_only_missing_ignore(
    tmp_path: Path, existing_ignore: str | None
) -> None:
    workdir = tmp_path / "runs"
    workdir.mkdir()
    ignore = workdir / ".gitignore"
    if existing_ignore is not None:
        ignore.write_text(existing_ignore)
    ensure_workdir(workdir)
    assert ignore.read_text() == (existing_ignore if existing_ignore is not None else "*\n")


def test_new_workdir_does_not_modify_repository_ignore(tmp_path: Path) -> None:
    ignore = tmp_path / ".gitignore"
    ignore.write_text("existing-rule\n")
    ensure_workdir(tmp_path / "nested" / "runs")
    assert ignore.read_text() == "existing-rule\n"
    assert (tmp_path / "nested" / "runs" / ".gitignore").read_text() == "*\n"


def test_inspection_does_not_create_workdir(tmp_path: Path) -> None:
    exp = make_experiment(workdir=str(tmp_path / "runs"))
    run_experiment(exp, dry_run=True)
    experiment_status(exp)
    assert not Path(exp.workdir).exists()


def test_suite_creates_self_ignoring_workdirs(tmp_path: Path) -> None:
    exp = make_experiment(
        workdir=str(tmp_path / "component"), trial_command="echo x=0.5 {overrides}", n_trials=1
    )
    payload = exp.model_dump(mode="json")
    payload.pop("experiment")
    phases = payload.pop("phases")
    suite = Suite.model_validate(
        {
            "suite": "suite",
            "defaults": {**payload, "workdir": str(tmp_path / "runs")},
            "studies": [{"name": "study", "phases": phases, "workdir": exp.workdir}],
        }
    )
    run_suite(suite, dry_run=True)
    assert not (tmp_path / "runs").exists()
    run_suite(suite)
    for workdir in (tmp_path / "runs", Path(exp.workdir)):
        assert (workdir / ".gitignore").read_text() == "*\n"


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
        Experiment(
            experiment=bad_name,
            trial_command="echo {overrides}",
            override_format="argparse",
            metric=Metric(
                extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
            ),
            phases=[
                Phase(  # type: ignore[arg-type]
                    name="p",
                    n_trials=1,
                    search_space={"x": IntParam(type="int", low=0, high=1)},
                )
            ],
        )


def test_experiment_name_accepts_valid() -> None:
    """Alphanumeric, underscore, and hyphen are safe."""
    Experiment(
        experiment="tiny_lm-16mb",
        trial_command="echo {overrides}",
        override_format="argparse",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
        phases=[
            Phase(  # type: ignore[arg-type]
                name="p",
                n_trials=1,
                search_space={"x": IntParam(type="int", low=0, high=1)},
            )
        ],
    )


@pytest.mark.parametrize(("mask", "expected_mode"), [(0o022, 0o644), (0o077, 0o600)])
def test_atomic_write_text_applies_process_umask_to_new_artifacts(
    tmp_path: Path, mask: int, expected_mode: int
) -> None:
    """Durable artifacts (``winner.yaml``, ``summary.yaml``, ``trials.csv``, ...)
    must land with the ordinary umask-governed mode the ``workdir`` trust
    boundary in ``docs/runtime.md`` promises, not the owner-only mode a
    ``NamedTemporaryFile`` staging file would carry across ``os.replace``.
    Parametrizing the umask proves the mode is masked by the kernel rather
    than hardcoded.
    """
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


@pytest.mark.parametrize("mask", [0o000, 0o022])
def test_private_atomic_write_text_stays_owner_only_under_any_umask(
    tmp_path: Path, mask: int
) -> None:
    """The private writer serves the lock namespace and MCP ``state_dir``, which
    are deliberately hardened. It must stay ``0600`` regardless of umask — the
    artifact-tree fix above must never be generalized onto it.
    """
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


@pytest.mark.parametrize("generation_id", ["../escape", "/tmp/escape", ""])
def test_run_experiment_rejects_unsafe_generation_id_before_writing_state(
    tmp_path: Path,
    generation_id: str,
) -> None:
    experiment = make_experiment(workdir=tmp_path / "runs")

    with pytest.raises(ValueError, match="generation name"):
        run_experiment(experiment, generation_id=generation_id)

    assert not _experiment_dir(experiment).exists()
