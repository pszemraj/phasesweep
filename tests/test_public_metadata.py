"""Public packaging metadata and engine export drift checks (no build artifacts).

Also guards the other public surface PhaseSweep ships: the shareable metadata
inside a published generation namespace. Every file there except the
owner-only config snapshot is meant to be read and passed around, so none of
them may carry a configured secret.
"""

from __future__ import annotations

import fnmatch
import importlib
import importlib.resources
import re
import stat
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

import phasesweep
from phasesweep.engine.paths import _generation_dir
from phasesweep.engine.run import run_experiment
from tests.conftest import make_experiment, write_constant_trainer
from tests.ledger_fixtures import published_generation_id

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"

CONSOLE_SCRIPTS = {
    "phasesweep": "phasesweep.cli:main",
    "phasesweep-mcp": "phasesweep.mcp.server:main",
}


def _pyproject() -> dict:
    if not PYPROJECT_PATH.is_file():
        pytest.skip(f"source pyproject.toml unavailable at {PYPROJECT_PATH}")
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def test_console_scripts_declare_importable_targets() -> None:
    """Both entry points stay declared and keep pointing at real callables."""
    assert _pyproject()["project"]["scripts"] == CONSOLE_SCRIPTS

    for target in CONSOLE_SCRIPTS.values():
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute)), f"{target} is not callable"


def test_package_data_covers_shipped_resources() -> None:
    """Every non-Python file the package reads at run time is declared as package data."""
    declared = _pyproject()["tool"]["setuptools"]["package-data"]
    templates = importlib.resources.files("phasesweep").joinpath("templates")
    shipped = {
        "phasesweep": [
            "config_reference.yaml",
            *(f"templates/{entry.name}" for entry in templates.iterdir() if entry.is_file()),
        ],
        "phasesweep.mcp": ["agent_prompt.md"],
    }
    assert len(shipped["phasesweep"]) > 1, "no packaged templates found"
    for package, relative_paths in shipped.items():
        patterns = [*declared["*"], *declared.get(package, [])]
        for relative in relative_paths:
            resource = importlib.resources.files(package).joinpath(relative)
            assert resource.is_file(), f"{package}: {relative} is not an importable resource"
            assert any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns), (
                f"{package}: {relative} matches no declared package-data pattern {patterns}"
            )


def test_readme_repository_links_are_relative_and_resolve() -> None:
    """README links and images address repository files through relative paths."""
    readme = PYPROJECT_PATH.parent / _pyproject()["project"]["readme"]
    targets = re.findall(r"\]\(([^)]+)\)", readme.read_text(encoding="utf-8"))
    assert targets
    for target in targets:
        parsed = urlsplit(target)
        if parsed.scheme:
            assert not target.startswith(
                (
                    "https://github.com/pszemraj/phasesweep/",
                    "https://raw.githubusercontent.com/pszemraj/phasesweep/",
                )
            ), target
        elif parsed.path:
            resolved = (readme.parent / parsed.path).resolve()
            assert resolved.is_relative_to(readme.parent), target
            assert resolved.is_file(), target


@pytest.mark.integration
@pytest.mark.parametrize(
    ("release_tag", "expected_version"),
    [("v0.4.0", "0.4.0"), ("v0.4.0.post1", "0.4.0.post1"), ("v1!0.4.0", "1!0.4.0")],
)
def test_release_version_uses_event_tag_with_competing_annotated_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release_tag: str, expected_version: str
) -> None:
    """The publishing build must select its release tag over an annotated RC."""
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_bytes(PYPROJECT_PATH.read_bytes())

    def run(*command: str) -> str:
        result = subprocess.run(
            command,
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return result.stdout.strip()

    git = (
        "git",
        "-c",
        "user.name=Release test",
        "-c",
        "user.email=release@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "tag.gpgsign=false",
    )
    run(*git, "init", "-q")
    run(*git, "add", "pyproject.toml")
    run(*git, "commit", "-qm", "Release fixture")
    run(*git, "tag", "-a", "v0.4.0rc1", "-m", "Release candidate")
    run(*git, "tag", release_tag)

    monkeypatch.delenv("SETUPTOOLS_SCM_OVERRIDES_FOR_PHASESWEEP", raising=False)
    assert run(sys.executable, "-m", "setuptools_scm") == "0.4.0rc1"

    workflow_path = PYPROJECT_PATH.parent / ".github" / "workflows" / "publish.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    build_step = next(
        step
        for step in workflow["jobs"]["build-distributions"]["steps"]
        if step["name"] == "Build distributions"
    )
    for name, value in build_step.get("env", {}).items():
        monkeypatch.setenv(name, value.replace("${{ github.event.release.tag_name }}", release_tag))

    assert run(sys.executable, "-m", "setuptools_scm") == expected_version


@pytest.mark.integration
def test_mcp_entrypoint_without_optional_sdk(tmp_path: Path) -> None:
    """Exercise the real entry point in a fresh process even with the dev extra installed."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from importlib.machinery import PathFinder
import sys
from importlib.metadata import entry_points
from unittest.mock import patch

find_spec = PathFinder.find_spec
def without_mcp(name, *args, **kwargs):
    return None if name == "mcp" or name.startswith("mcp.") else find_spec(name, *args, **kwargs)

with patch.object(PathFinder, "find_spec", side_effect=without_mcp):
    try:
        import mcp
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("negative test did not isolate the MCP SDK")
    entry = next(iter(entry_points(group="console_scripts", name="phasesweep-mcp")))
    sys.exit(entry.load()(["--catalog", sys.argv[1]]))
""",
            str(tmp_path / "unused-catalog.yaml"),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2, result.stderr
    assert "MCP support is not installed" in result.stderr
    assert "pip install" in result.stderr and "[mcp]" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_root_facade_exports_definition_objects() -> None:
    """The root facade keeps its supported API bound to the defining modules."""
    from phasesweep._metadata import __version__
    from phasesweep.config.io import load_experiment
    from phasesweep.config.models import Experiment
    from phasesweep.engine.read import experiment_status
    from phasesweep.engine.selection import NoFeasibleTrialError
    from phasesweep.engine.state import Winner
    from phasesweep.errors import ProcessCleanupUncertainError, UnsafeProcessCleanupError
    from phasesweep.reporting import report_objective

    expected = {
        "Experiment": Experiment,
        "NoFeasibleTrialError": NoFeasibleTrialError,
        "ProcessCleanupUncertainError": ProcessCleanupUncertainError,
        "UnsafeProcessCleanupError": UnsafeProcessCleanupError,
        "Winner": Winner,
        "__version__": __version__,
        "experiment_status": experiment_status,
        "load_experiment": load_experiment,
        "report_objective": report_objective,
        "run_experiment": run_experiment,
    }
    assert set(phasesweep.__all__) == expected.keys()
    assert all(getattr(phasesweep, name) is value for name, value in expected.items())


def test_published_generation_metadata_never_exposes_configured_secrets(
    tmp_path: Path,
) -> None:
    """A configured ``env`` value reaches disk only through the owner-only snapshot.

    ``config.snapshot.yaml`` exists so a published winner can be reproduced, so
    it necessarily holds ``env:`` values and is written ``0600``. Every other
    file in the generation namespace -- the summary, the lifecycle record, the
    winners, and ``reproducibility.json`` -- is ordinary umask-governed
    metadata an operator may share, and must carry digests rather than values.
    """
    secret = "s3cr3t-public-metadata-sentinel"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        trainer=write_constant_trainer(tmp_path),
        n_trials=1,
        env={"TRAINER_TOKEN": secret},
    )
    run_experiment(experiment)

    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    generation_dir = _generation_dir(experiment, generation_id)

    carriers = []
    for path in sorted(generation_dir.rglob("*")):
        if not path.is_file():
            continue
        if secret.encode() in path.read_bytes():
            carriers.append(path.relative_to(generation_dir).as_posix())

    assert carriers == ["config.snapshot.yaml"], carriers
    assert stat.S_IMODE((generation_dir / "config.snapshot.yaml").stat().st_mode) == 0o600


def test_deadline_exceeded_error_is_an_extractor_failure() -> None:
    """Extractor callers can distinguish deadline exhaustion from other failures."""
    from phasesweep.evidence.evaluation import DeadlineExceededError, ExtractorError

    assert issubclass(DeadlineExceededError, ExtractorError)
