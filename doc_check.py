"""Verify stdlib-only docstring coverage for source trees."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Issue:
    path: Path
    line: int
    message: str

    def render(self, root: Path) -> str:
        """Format the issue with a stable project-relative path."""
        try:
            display = self.path.relative_to(root)
        except ValueError:
            display = self.path
        return f"{display}:{self.line}: {self.message}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that public Python source objects have docstrings.",
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--check-lazy",
        action="store_true",
        help="also verify static __all__ entries resolve in each module",
    )
    return parser


def _python_files(paths: Sequence[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(
                candidate
                for candidate in path.rglob("*.py")
                if "__pycache__" not in candidate.parts
            )
        elif path.suffix == ".py":
            files.add(path)
    return sorted(files)


def _is_public_name(name: str) -> bool:
    return not name.startswith("_")


def _docstring_issues_for_body(
    path: Path,
    body: Sequence[ast.stmt],
    *,
    parent: str | None = None,
) -> list[_Issue]:
    issues: list[_Issue] = []
    for node in body:
        if not isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not _is_public_name(node.name):
            continue

        qualified_name = node.name if parent is None else f"{parent}.{node.name}"
        if ast.get_docstring(node) is None:
            issues.append(_Issue(path, node.lineno, f"{qualified_name} missing docstring"))

        if isinstance(node, ast.ClassDef):
            issues.extend(_docstring_issues_for_body(path, node.body, parent=qualified_name))
    return issues


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Tuple | ast.List):
        names: set[str] = set()
        for element in target.elts:
            names.update(_assigned_names(element))
        return names
    return set()


def _names_from_stmt(node: ast.stmt) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        names.add(node.name)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            names.update(_assigned_names(target))
    elif isinstance(node, ast.AnnAssign):
        names.update(_assigned_names(node.target))
    elif isinstance(node, ast.Import | ast.ImportFrom):
        for alias in node.names:
            names.add(alias.asname or alias.name.split(".", maxsplit=1)[0])
    elif isinstance(node, ast.Try):
        for child in [*node.body, *node.orelse, *node.finalbody]:
            names.update(_names_from_stmt(child))
        for handler in node.handlers:
            names.update(_names_from_body(handler.body))
    elif isinstance(node, ast.If):
        names.update(_names_from_body(node.body))
        names.update(_names_from_body(node.orelse))
    return names


def _names_from_body(body: Sequence[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for node in body:
        names.update(_names_from_stmt(node))
    return names


def _module_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        names.update(_names_from_stmt(node))
    return names


def _literal_all(node: ast.expr) -> tuple[list[str] | None, str | None]:
    try:
        value = ast.literal_eval(node)
    except (SyntaxError, ValueError):
        return None, "__all__ must be a static string sequence"

    if not isinstance(value, list | tuple | set):
        return None, "__all__ must be a static string sequence"

    exports: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None, "__all__ must contain only strings"
        exports.append(item)
    return exports, None


def _all_export_issues(path: Path, tree: ast.Module) -> list[_Issue]:
    issues: list[_Issue] = []
    names = _module_names(tree)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            continue

        exports, error = _literal_all(node.value)
        if error is not None:
            issues.append(_Issue(path, node.lineno, error))
            continue

        assert exports is not None
        for export in exports:
            if export not in names:
                issues.append(_Issue(path, node.lineno, f"__all__ exports unknown name {export!r}"))
    return issues


def _check_file(path: Path, *, check_lazy: bool) -> list[_Issue]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [_Issue(path, exc.lineno or 1, f"syntax error: {exc.msg}")]

    issues: list[_Issue] = []
    if ast.get_docstring(tree) is None:
        issues.append(_Issue(path, 1, "module missing docstring"))
    issues.extend(_docstring_issues_for_body(path, tree.body))
    if check_lazy:
        issues.extend(_all_export_issues(path, tree))
    return issues


def main(argv: Sequence[str] | None = None) -> int:
    """Run docstring checks from the command line."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    root = Path.cwd()

    issues: list[_Issue] = []
    for path in _python_files(args.paths):
        issues.extend(_check_file(path, check_lazy=args.check_lazy))

    if issues:
        for issue in sorted(issues, key=lambda item: (str(item.path), item.line, item.message)):
            print(issue.render(root))
        return 1

    print("No issues found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
