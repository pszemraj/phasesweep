"""The typing-escapes hook flags Any, typing.cast and type: ignore on added lines only."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_typing_escapes.py"


def _load_checker() -> ModuleType:
    """Import the hook script, which lives outside any package.

    :raises ImportError: The script cannot be loaded.
    :return ModuleType: The loaded script module.
    """
    spec = importlib.util.spec_from_file_location("check_typing_escapes", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_added_lines_reads_every_hunk_of_every_file() -> None:
    """Hunk headers give the new-file line ranges; a pure deletion adds nothing."""
    diff = (
        "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n"
        "@@ -3 +3,2 @@\n-x\n+y\n+z\n@@ -9,2 +10,0 @@\n-gone\n-gone\n"
        "diff --git a/tests/b.py b/tests/b.py\n--- /dev/null\n+++ b/tests/b.py\n"
        "@@ -0,0 +1 @@\n+new\n"
    )

    assert checker.added_lines(diff) == {"src/a.py": {3, 4}, "tests/b.py": {1}}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param("def f(x: Any) -> None: ...\n", [(1, "Any")], id="annotation"),
        pytest.param("x: dict[str, Any] = {}\n", [(1, "Any")], id="subscript"),
        pytest.param("import typing\nx: typing.Any = 1\n", [(2, "Any")], id="typing-attribute"),
        pytest.param("y = cast(int, x)\n", [(1, "typing.cast")], id="cast"),
        pytest.param(
            "import typing\ny = typing.cast(int, x)\n", [(2, "typing.cast")], id="qualified-cast"
        ),
        pytest.param(
            "y = f()  # type: ignore[attr-defined]\n", [(1, "# type: ignore")], id="ignore"
        ),
        pytest.param(
            'def f(x):\n    """Doc.\n\n    :param Any x: Value.\n    """\n',
            [(4, "Any in a docstring type")],
            id="docstring-param",
        ),
        pytest.param(
            'def f():\n    """Doc.\n\n    :return dict[str, Any]: Value.\n    """\n',
            [(4, "Any in a docstring type")],
            id="docstring-return",
        ),
        pytest.param("from typing import Any\n", [], id="import-alone"),
        pytest.param('"""Any value works here."""\n', [], id="prose"),
        pytest.param('y = "# type: ignore"\n', [], id="ignore-inside-string"),
        pytest.param("x: int = 1\n", [], id="clean"),
    ],
)
def test_find_escapes_flags_each_escape_on_added_lines(
    source: str, expected: list[tuple[int, str]]
) -> None:
    """Every escape is found by kind and line; prose, imports and strings are not escapes."""
    every_line = set(range(1, source.count("\n") + 2))

    assert checker.find_escapes(source, every_line) == expected


def test_find_escapes_ignores_lines_the_commit_does_not_add() -> None:
    """Pre-existing escapes on untouched lines are left for their own cleanup."""
    source = "def f(x: Any) -> None: ...\ndef g(y: Any) -> None: ...\n"

    assert checker.find_escapes(source, {2}) == [(2, "Any")]


def _git(repo: Path, *args: str) -> None:
    """Run git in a scratch repository.

    :param Path repo: Repository root.
    :param str args: Arguments after ``git``.
    :raises subprocess.CalledProcessError: git exits nonzero.
    """
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.mark.integration
def test_hook_refuses_a_staged_escape_and_passes_a_clean_commit(tmp_path: Path) -> None:
    """The hook reads the staged diff, so only what the commit adds decides the outcome."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "core.excludesFile", "/dev/null")
    module = tmp_path / "src" / "pkg.py"
    module.parent.mkdir()
    module.write_text("def f(x: Any) -> None: ...\n")
    _git(tmp_path, "add", "src/pkg.py")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "seed")

    module.write_text("def f(x: Any) -> None: ...\ndef g(y: int) -> None: ...\n")
    _git(tmp_path, "add", "src/pkg.py")
    clean = subprocess.run(
        ["python", str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert clean.returncode == 0, clean.stderr

    module.write_text("def f(x: Any) -> None: ...\ndef g(y: dict[str, Any]) -> None: ...\n")
    _git(tmp_path, "add", "src/pkg.py")
    refused = subprocess.run(
        ["python", str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert refused.returncode == 1
    assert "src/pkg.py:2: Any" in refused.stderr
    assert "src/pkg.py:1:" not in refused.stderr
    assert "TypedDict" in refused.stderr
