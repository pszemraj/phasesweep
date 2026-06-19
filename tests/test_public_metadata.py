"""Public packaging metadata checks."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


def test_public_package_metadata() -> None:
    """Public metadata should advertise license, platform, and typing support."""
    marker = resources.files("phasesweep").joinpath("py.typed")
    assert marker.is_file()

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    project = pyproject["project"]
    classifiers = set(project["classifiers"])
    scripts = project["scripts"]

    assert project["license"] == "MIT"
    assert "Operating System :: POSIX" in classifiers
    assert "Programming Language :: Python :: 3.12" in classifiers
    assert "Typing :: Typed" in classifiers
    assert scripts["phasesweep"] == "phasesweep.cli:main"
    assert scripts["phasesweep-mcp"] == "phasesweep.mcp.server:main"
