"""Public packaging metadata checks."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


def test_package_includes_py_typed_marker() -> None:
    marker = resources.files("phasesweep").joinpath("py.typed")
    assert marker.is_file()


def test_public_metadata_sets_license_platform_and_typing() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    project = pyproject["project"]
    classifiers = set(project["classifiers"])

    assert project["license"] == "MIT"
    assert "Operating System :: POSIX" in classifiers
    assert "Programming Language :: Python :: 3.12" in classifiers
    assert "Typing :: Typed" in classifiers
