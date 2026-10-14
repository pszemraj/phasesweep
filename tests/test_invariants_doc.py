"""Hold ``docs/invariants.md`` to the code and tests it cites.

The durability invariants name the symbols that hold each rule and the tests
that pin it. Nothing else notices when a holder is renamed, a test is deleted,
or a renumbering leaves a cross-reference pointing at a different rule, so
this module checks all three:

- every backticked ``module.symbol`` on a ``Held by`` line resolves by import
  and attribute walk (the doc drops the ``phasesweep.`` prefix);
- every ``tests/<file>.py::<name>`` the doc cites names a function, or a class
  and method, in that file, found by parsing it rather than importing it;
- every numbered cross-reference in ``src/``, ``tests/`` and ``docs/`` carries
  the title of the rule it means, either as ``("<heading text>")`` right after
  the number or as a link whose anchor is that heading's slug, and every link
  into ``invariants.md`` names a heading that exists.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "invariants.md"
REFERENCE_ROOTS = (ROOT / "src", ROOT / "tests", ROOT / "docs")
REFERENCE_SUFFIXES = frozenset({".py", ".md"})

HEADING = re.compile(r"^#### (\d+)\. (.+)$")
BACKTICKED = re.compile(r"`([^`]+)`")
DOTTED_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
TEST_REF = re.compile(r"(tests/[\w/]+\.py)::(\w+)(?:::(\w+))?(?:\[[^\]]*\])?")
#: Final segments that mark a dotted token as a file name, not a symbol.
FILE_SUFFIXES = frozenset(
    {"bak", "json", "jsonl", "lock", "md", "py", "toml", "txt", "yaml", "yml"}
)

#: Start of a numbered reference; the numbers it lists are parsed after it.
REFERENCE_START = re.compile(r"\binvariants?\s+(?=\d)", re.IGNORECASE)
NUMBER = re.compile(r"\d+")
QUOTED_TITLE = re.compile(r'\s*\("([^"]+)"\)')
LINKED_TITLE = re.compile(r"\]\([^)\s]*invariants\.md#([\w-]+)\)")
NEXT_NUMBER = re.compile(r"(?:\s*,\s*(?:and\s+)?|\s+and\s+)(?=\d)", re.IGNORECASE)
ANCHOR_LINK = re.compile(r"invariants\.md#([\w-]+)")
COMMENT_BREAK = re.compile(r"\s*\n\s*(?:#\s*)?")


@dataclass(frozen=True)
class Invariant:
    """One numbered entry of the invariants doc."""

    number: int
    title: str
    held_by: tuple[str, ...]
    tests: tuple[str, ...]


def _slug(heading: str) -> str:
    """Return the anchor GitHub derives from a Markdown heading.

    :param str heading: Heading text without its leading hashes.
    :return str: Lowercased slug with punctuation dropped and spaces as hyphens.
    """
    kept = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return kept.replace(" ", "-")


@cache
def _doc_lines() -> tuple[str, ...]:
    """Return the invariants doc as lines."""
    return tuple(DOC.read_text(encoding="utf-8").splitlines())


@cache
def _invariants() -> tuple[Invariant, ...]:
    """Parse every numbered invariant with its ``Held by`` and ``Test`` lines."""
    parsed: list[Invariant] = []
    current: tuple[int, str] | None = None
    held_by: list[str] = []
    tests: list[str] = []

    def close() -> None:
        if current is not None:
            parsed.append(Invariant(current[0], current[1], tuple(held_by), tuple(tests)))

    for line in _doc_lines():
        heading = HEADING.match(line)
        if heading is not None or line.startswith(("## ", "### ")):
            close()
            current = None if heading is None else (int(heading[1]), heading[2])
            held_by, tests = [], []
        elif current is not None and line.startswith("**Held by:**"):
            held_by.append(line)
        elif current is not None and line.startswith(("**Test:**", "**Tests:**")):
            tests.append(line)
    close()
    return tuple(parsed)


@cache
def _titles() -> dict[int, str]:
    """Map each invariant number to its heading title."""
    return {entry.number: entry.title for entry in _invariants()}


@cache
def _anchors() -> frozenset[str]:
    """Return the slug of every heading in the invariants doc."""
    return frozenset(_slug(line.lstrip("#")) for line in _doc_lines() if line.startswith("#"))


def _member(owner: object, name: str) -> object:
    """Return ``owner.name``, accepting an annotated instance field of a class.

    :param object owner: Module, class, or other object to look in.
    :param str name: Attribute name.
    :raises AttributeError: If neither the attribute nor a class annotation exists.
    :return object: The attribute, or the class itself for an annotated field.
    """
    if hasattr(owner, name):
        return getattr(owner, name)
    if isinstance(owner, type) and any(
        name in vars(klass).get("__annotations__", {}) for klass in owner.__mro__
    ):
        return owner
    raise AttributeError(f"{owner!r} has no attribute {name!r}")


def _resolve(dotted: str) -> None:
    """Import and walk a doc symbol such as ``engine.read.read_status``.

    :param str dotted: Symbol path relative to the ``phasesweep`` package.
    :raises ImportError: If the first segment is not a ``phasesweep`` module.
    :raises AttributeError: If a later segment does not exist.
    """
    first, *rest = dotted.split(".")
    qualified = f"phasesweep.{first}"
    owner: object = importlib.import_module(qualified)
    for part in rest:
        qualified = f"{qualified}.{part}"
        if (
            isinstance(owner, ModuleType)
            and hasattr(owner, "__path__")
            and importlib.util.find_spec(qualified) is not None
        ):
            owner = importlib.import_module(qualified)
        else:
            owner = _member(owner, part)


@cache
def _test_names(relative: str) -> frozenset[str]:
    """Return ``name`` and ``Class::method`` for every definition in a test file.

    :param str relative: Repository-relative path of the file.
    :return frozenset[str]: Top-level function and class names plus class methods.
    """
    path = ROOT / relative
    if not path.is_file():
        return frozenset()
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
            names.update(
                f"{node.name}::{item.name}"
                for item in node.body
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
            )
    return frozenset(names)


def _reference_files() -> list[Path]:
    """Return every Python and Markdown file that may cite an invariant.

    This module is left out: its parser probes are deliberately malformed references.
    """
    this_module = Path(__file__).resolve()
    return sorted(
        path
        for root in REFERENCE_ROOTS
        for path in root.rglob("*")
        if path.suffix in REFERENCE_SUFFIXES
        and path.is_file()
        and "__pycache__" not in path.parts
        and path.resolve() != this_module
    )


def _reference_errors(text: str) -> list[str]:
    """Return every numbered invariant reference in ``text`` that lacks its title.

    Line breaks and the comment markers that continue a Python comment are
    collapsed first, so a title may wrap.

    :param str text: File contents.
    :return list[str]: One message per number without a matching title.
    """
    flat = COMMENT_BREAK.sub(" ", text)
    errors: list[str] = []
    for start in REFERENCE_START.finditer(flat):
        position = start.end()
        while True:
            number_match = NUMBER.match(flat, position)
            if number_match is None:
                break
            number = int(number_match[0])
            position = number_match.end()
            expected = _titles().get(number)
            quoted = QUOTED_TITLE.match(flat, position)
            linked = LINKED_TITLE.match(flat, position)
            context = flat[max(start.start() - 30, 0) : position + 80]
            if expected is None:
                errors.append(f"invariant {number} does not exist: ...{context}...")
            elif quoted is not None:
                if quoted[1] != expected:
                    errors.append(
                        f"invariant {number} is titled {expected!r}, not {quoted[1]!r}: ...{context}..."
                    )
                position = quoted.end()
            elif linked is not None:
                if linked[1] != _slug(f"{number}. {expected}"):
                    errors.append(f"invariant {number} links to #{linked[1]}: ...{context}...")
                position = linked.end()
            else:
                errors.append(f'invariant {number} needs ("{expected}") after it: ...{context}...')
            following = NEXT_NUMBER.match(flat, position)
            if following is None:
                break
            position = following.end()
    return errors


def test_invariants_are_numbered_in_order_with_holders_and_tests() -> None:
    """Every entry is numbered 1..N in order and has one ``Held by`` and one ``Test`` line."""
    entries = _invariants()
    assert [entry.number for entry in entries] == list(range(1, len(entries) + 1))
    malformed = [
        entry.number for entry in entries if len(entry.held_by) != 1 or len(entry.tests) != 1
    ]
    assert malformed == []


def test_held_by_symbols_resolve() -> None:
    """Every backticked ``module.symbol`` on a ``Held by`` line imports and resolves."""
    unresolved: list[str] = []
    for line in (line for line in _doc_lines() if line.startswith("**Held by:**")):
        for token in BACKTICKED.findall(line):
            if DOTTED_NAME.fullmatch(token) is None or token.rsplit(".", 1)[1] in FILE_SUFFIXES:
                continue
            try:
                _resolve(token)
            except (ImportError, AttributeError) as error:
                unresolved.append(f"{token}: {error}")
    assert unresolved == []


def test_cited_tests_exist() -> None:
    """Every ``tests/<file>.py::<name>`` the doc cites is defined in that file."""
    cited = TEST_REF.findall(DOC.read_text(encoding="utf-8"))
    assert cited, "the parser found no cited tests"
    missing = [
        f"{path}::{name}" + (f"::{method}" if method else "")
        for path, name, method in cited
        if (f"{name}::{method}" if method else name) not in _test_names(path)
    ]
    assert missing == []


def test_test_lines_cite_only_tests() -> None:
    """Every backticked token on a ``Test`` line is a ``tests/<file>.py::<name>`` reference."""
    stray = [
        token
        for entry in _invariants()
        for line in entry.tests
        for token in BACKTICKED.findall(line)
        if TEST_REF.fullmatch(token) is None
    ]
    assert stray == []


def test_numbered_references_carry_their_titles() -> None:
    """A numbered reference names its rule's title, so renumbering cannot repoint it."""
    errors = [
        f"{path.relative_to(ROOT)}: {error}"
        for path in _reference_files()
        for error in _reference_errors(path.read_text(encoding="utf-8"))
    ]
    assert errors == []


def test_links_into_the_invariants_doc_name_real_headings() -> None:
    """Every ``invariants.md#anchor`` link names a heading that exists."""
    broken = [
        f"{path.relative_to(ROOT)}: #{anchor}"
        for path in [*_reference_files(), ROOT / "README.md"]
        for anchor in ANCHOR_LINK.findall(path.read_text(encoding="utf-8"))
        if anchor not in _anchors()
    ]
    assert broken == []


def test_reference_parser_rejects_a_wrong_or_missing_title() -> None:
    """The reference check fails a bare number, a wrong title, and a stale link anchor."""
    first = _titles()[1]
    second = _titles()[2]
    assert _reference_errors(f'see invariant 1 ("{first}")') == []
    assert _reference_errors(f'Invariants 1 ("{first}") and 2 ("{second}") hold.') == []
    assert _reference_errors(f"[invariant 1](invariants.md#{_slug('1. ' + first)})") == []
    assert _reference_errors(f'# invariant 1\n    # ("{first}")') == []
    assert len(_reference_errors("see invariant 1 for details")) == 1
    assert len(_reference_errors(f'invariants 1 ("{first}") and 2 ("{first}")')) == 1
    assert len(_reference_errors(f"[invariant 2](invariants.md#{_slug('1. ' + first)})")) == 1
