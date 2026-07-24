"""Public package metadata drift checks that do not create build artifacts."""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

import phasesweep.engine as engine
from phasesweep.engine import errors as engine_errors


def test_public_package_metadata() -> None:
    """Public metadata should advertise supported entry points and package data."""
    package = resources.files("phasesweep")
    assert package.joinpath("py.typed").is_file()
    assert package.joinpath("mcp", "agent_prompt.md").is_file()

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    project = pyproject["project"]
    classifiers = set(project["classifiers"])
    scripts = project["scripts"]
    package_data = pyproject["tool"]["setuptools"]["package-data"]

    assert project["license"] == "MIT"
    assert "Operating System :: POSIX" in classifiers
    assert "Programming Language :: Python :: 3.12" in classifiers
    assert "Typing :: Typed" in classifiers
    assert scripts == {
        "phasesweep": "phasesweep.cli:main",
        "phasesweep-mcp": "phasesweep.mcp.server:main",
    }
    assert package_data["*"] == ["py.typed"]
    assert package_data["phasesweep.mcp"] == ["agent_prompt.md"]


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
