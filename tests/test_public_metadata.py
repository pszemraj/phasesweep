"""Public packaging metadata and engine export drift checks (no build artifacts)."""

from __future__ import annotations

import importlib
import tomllib
from fnmatch import fnmatch
from importlib import resources
from pathlib import Path

import pytest
import yaml

import phasesweep.engine as engine
import phasesweep.evidence as evidence
from phasesweep.engine import errors as engine_errors
from phasesweep.evidence import evaluation as evidence_evaluation

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Runtime-required package data: package -> relative POSIX path that must ship in the wheel.
REQUIRED_PACKAGE_DATA = {
    "phasesweep": "templates/starter_experiment.yaml",
    "phasesweep.mcp": "agent_prompt.md",
}

CONSOLE_SCRIPTS = {
    "phasesweep": "phasesweep.cli:main",
    "phasesweep-mcp": "phasesweep.mcp.server:main",
}


def _pyproject() -> dict:
    if not PYPROJECT_PATH.is_file():
        pytest.skip(f"source pyproject.toml unavailable at {PYPROJECT_PATH}")
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def test_package_data_ships_runtime_required_files() -> None:
    """`phasesweep init` needs the starter template inside the wheel, not just the repo."""
    package_data = _pyproject()["tool"]["setuptools"]["package-data"]

    assert package_data.get("*") == ["py.typed"], package_data

    for package, relative_path in REQUIRED_PACKAGE_DATA.items():
        patterns = package_data.get(package)
        assert patterns, f"{package!r} declares no package-data; {relative_path} would be dropped"
        assert any(fnmatch(relative_path, pattern) for pattern in patterns), (
            f"no package-data pattern in {patterns} ships {package}/{relative_path}"
        )


def test_package_data_files_exist_in_source_tree() -> None:
    """A package-data pattern only helps if the declared file is actually checked in."""
    where = _pyproject()["tool"]["setuptools"]["packages"]["find"]["where"]
    assert where == ["src"], where
    src_root = PYPROJECT_PATH.parent / "src"

    for package, relative_path in REQUIRED_PACKAGE_DATA.items():
        expected = src_root.joinpath(*package.split("."), *relative_path.split("/"))
        assert expected.is_file(), f"declared package data missing from source tree: {expected}"


def test_console_scripts_declare_importable_targets() -> None:
    """Both entry points stay declared and keep pointing at real callables."""
    assert _pyproject()["project"]["scripts"] == CONSOLE_SCRIPTS

    for target in CONSOLE_SCRIPTS.values():
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute)), f"{target} is not callable"


def test_starter_template_resolves_the_way_the_cli_resolves_it() -> None:
    """Mirror `phasesweep.cli._starter_experiment_text` so packaging drift fails here first."""
    template = resources.files("phasesweep").joinpath("templates", "starter_experiment.yaml")
    assert template.is_file()

    text = template.read_text(encoding="utf-8")
    assert text.strip(), "packaged starter template is empty"

    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict), f"starter template is not a YAML mapping: {type(parsed)}"
    assert parsed.get("experiment") == "phasesweep_starter"
    assert parsed.get("phases"), "starter template declares no phases"

    # The CLI substitutes these before writing; a renamed placeholder would ship a
    # literal marker into the user's experiment.yaml.
    for placeholder in ("__PHASESWEEP_WORKDIR__", "__PHASESWEEP_STORAGE__"):
        assert placeholder in text, f"{placeholder} missing from packaged starter template"


def test_engine_exports_all_typed_preflight_errors() -> None:
    """Engine callers can catch every typed preflight failure from the public API."""
    names = (
        "ExperimentLockBusyError",
        "SamplerContinuationUnsupportedError",
        "StudyContextConflictError",
        "StudyFingerprintMismatchError",
        "StudySchemaMismatchError",
        "StudyStorageUnavailableError",
        "TrialTargetRegressionError",
    )

    assert set(names).issubset(engine.__all__)
    assert all(getattr(engine, name) is getattr(engine_errors, name) for name in names)


def test_evidence_exports_deadline_exceeded_error() -> None:
    """Custom extractor authors catch the deadline failure from the package namespace.

    ``DeadlineExceededError`` is the one extractor failure a caller may want
    to distinguish (the trial ran out of wallclock, the extractor itself is
    fine), so it must not require importing the private
    ``phasesweep.evidence.evaluation`` module path its sibling
    ``ExtractorError`` never did.
    """
    from phasesweep.evidence import DeadlineExceededError, ExtractorError

    assert DeadlineExceededError is evidence_evaluation.DeadlineExceededError
    assert issubclass(DeadlineExceededError, ExtractorError)
    assert "DeadlineExceededError" in evidence.__all__
