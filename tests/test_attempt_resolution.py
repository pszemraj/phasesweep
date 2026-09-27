"""Stale-attempt resolution: every discovery source signals through one procedure.

A stale attempt reaches recovery from the attempt registry, a stale RUNNING
trial, or a terminal trial that recorded cleanup uncertainty. Each normalizes
its attempt and resolves it through ``engine.attempts._resolve_attempt``, so no
source can signal a process group without the identity checks that procedure
applies. This module pins that statically with :mod:`ast`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import phasesweep


class _ScopedReferences(ast.NodeVisitor):
    """Record the innermost enclosing function of every reference to one name."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.scopes = ["<module>"]
        self.found: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.name:
            self.found.append(self.scopes[-1])

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == self.name:
            self.found.append(self.scopes[-1])
        self.generic_visit(node)


def _references(source_root: Path, name: str) -> list[tuple[str, str]]:
    """List every reference to ``name`` under ``source_root`` with its enclosing function.

    :param Path source_root: Package directory to scan.
    :param str name: Bare or attribute name to find.
    :return list[tuple[str, str]]: ``(module path, enclosing function)`` per reference.
    """
    references: list[tuple[str, str]] = []
    for path in sorted(source_root.rglob("*.py")):
        visitor = _ScopedReferences(name)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        module = path.relative_to(source_root).as_posix()
        references.extend((module, scope) for scope in visitor.found)
    return references


def test_only_resolve_attempt_signals_a_stale_process_group() -> None:
    """Every discovery source signals through one procedure, so none can skip its identity checks."""
    source_root = Path(phasesweep.__file__).parent

    assert _references(source_root, "cleanup_stale_trial_process") == [
        ("engine/attempts.py", "_resolve_attempt")
    ]


def test_reference_scan_sees_every_spelling(tmp_path: Path) -> None:
    """The scan counts bare, attribute, nested, and module-level references alike."""
    (tmp_path / "caller.py").write_text(
        "from reaper import signal_group\n"
        "import reaper\n"
        "def direct():\n"
        "    signal_group()\n"
        "def qualified():\n"
        "    def nested():\n"
        "        reaper.signal_group()\n"
        "handler = signal_group\n",
        encoding="utf-8",
    )

    assert _references(tmp_path, "signal_group") == [
        ("caller.py", "direct"),
        ("caller.py", "nested"),
        ("caller.py", "<module>"),
    ]
