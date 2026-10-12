"""Refuse typing escapes on the lines a commit adds to src/ and tests/.

``Any`` says nothing about a value, ``typing.cast`` asserts a type without
checking it, and ``# type: ignore`` silences the checker. None of them may
appear on a line the staged commit adds. A line the commit edits counts as
added, so touching a line that already carries an escape means fixing it.
Untouched pre-existing occurrences are left for their own cleanup.

Run by the ``typing-escapes`` pre-commit hook. It reads the staged diff and the
staged file contents itself, so it takes no file arguments.
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tokenize

REMEDY = (
    "Narrow with isinstance and raise on the impossible branch, give a known shape a "
    "TypedDict, describe an external API with a Protocol, use JsonValue/JsonObject from "
    "phasesweep.runtime.json for open JSON, or object for a truly opaque value."
)

_HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
_FILE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
_DOC_TYPE = re.compile(r"^\s*:(?:param|return|rtype|type|var|ivar|cvar)\b([^:]*)")
_ANY_WORD = re.compile(r"\bAny\b")
_TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore")


def added_lines(diff: str) -> dict[str, set[int]]:
    """Map each file in a ``git diff -U0`` to the line numbers it adds.

    :param str diff: Unified diff text with zero context lines.
    :return dict[str, set[int]]: Added line numbers in the new file, by path.
    """
    result: dict[str, set[int]] = {}
    sections = _FILE.split(diff)
    # split() yields [preamble, path, body, path, body, ...].
    for path, body in zip(sections[1::2], sections[2::2], strict=True):
        lines = result.setdefault(path, set())
        for hunk in _HUNK.finditer(body):
            start = int(hunk.group(1))
            count = int(hunk.group(2)) if hunk.group(2) is not None else 1
            lines.update(range(start, start + count))
    return result


def _is_typing_name(node: ast.expr, name: str) -> bool:
    """Return whether ``node`` is ``name`` or ``typing.name``-style attribute access.

    :param ast.expr node: Expression to inspect.
    :param str name: Bare name to match, such as ``Any`` or ``cast``.
    :return bool: Whether the expression refers to that name.
    """
    if isinstance(node, ast.Name):
        return node.id == name
    return isinstance(node, ast.Attribute) and node.attr == name


def find_escapes(source: str, added: set[int]) -> list[tuple[int, str]]:
    """Find typing escapes on the given lines of one Python source.

    Imports are not flagged: an import only matters through a use, and every
    use is flagged.

    :param str source: Full text of the staged file.
    :param set[int] added: One-based line numbers the commit adds.
    :raises SyntaxError: ``source`` is not valid Python.
    :return list[tuple[int, str]]: ``(line, problem)`` pairs in line order.
    """
    found: set[tuple[int, str]] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and _is_typing_name(node.func, "cast"):
            found.add((node.lineno, "typing.cast"))
        elif (
            isinstance(node, ast.Name | ast.Attribute)
            and not isinstance(node.ctx, ast.Store)
            and _is_typing_name(node, "Any")
        ):
            found.add((node.lineno, "Any"))
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and _TYPE_IGNORE.search(token.string):
            found.add((token.start[0], "# type: ignore"))
    for number, line in enumerate(source.splitlines(), start=1):
        doc_type = _DOC_TYPE.match(line)
        if doc_type is not None and _ANY_WORD.search(doc_type.group(1)):
            found.add((number, "Any in a docstring type"))
    return sorted(item for item in found if item[0] in added)


def _git(*args: str) -> str:
    """Run a git command and return its standard output.

    :param str args: Arguments after ``git``.
    :raises subprocess.CalledProcessError: git exits nonzero.
    :return str: Captured standard output.
    """
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def main() -> int:
    """Check the staged diff and report every escape on an added line.

    :return int: 0 when no added line carries an escape, otherwise 1.
    """
    diff = _git(
        "diff", "--cached", "-U0", "--no-color", "--diff-filter=AMR", "--", "src/*.py", "tests/*.py"
    )
    problems: list[str] = []
    for path, lines in sorted(added_lines(diff).items()):
        if not lines:
            continue
        source = _git("show", f":{path}")
        problems.extend(f"{path}:{line}: {what}" for line, what in find_escapes(source, lines))
    if not problems:
        return 0
    print("check_typing_escapes: typing escapes on lines this commit adds:", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print(REMEDY, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
