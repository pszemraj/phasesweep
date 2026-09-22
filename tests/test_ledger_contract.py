"""Static contract pinning every storage constructor to one ledger module.

The durability invariants say read paths must never create studies or build
file-backed storage, and that validation happens before any bind/claim/open.
Those rules only hold if there is exactly *one* module that may talk to
Optuna's storage constructors, so every other module has to go through helpers
that validate first. This module enforces that statically with :mod:`ast`, so a
new call site fails in CI before it can ever run.

The chokepoint is ``phasesweep/engine/ledger.py``. Two ratchet dicts
(:data:`_LEGACY_SITES`, :data:`_LEGACY_PRIVATE_IMPORTS`) record what is still
reachable from outside it. They are compared for *equality*, not containment:
adding a call site fails, and removing one fails until the ratchet is tightened
in the same commit. Both are empty: every storage constructor and every
storage-private helper is reached only through the ledger's public handle API,
so any entry added to either one is a new bypass that needs a stated reason.

Out of scope on purpose, because no static reader can follow them: dynamic
attribute access such as ``getattr(optuna, "create_study")``, and anything
reached through :mod:`importlib`. Both are absent from ``src/phasesweep``
today; the tier-guard and review process, not this test, keep them out.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
PACKAGE_ROOT = SRC / "phasesweep"

#: Module that is allowed to construct storage.
CHOKEPOINT = "phasesweep/engine/ledger.py"

#: Storage entry points no module outside :data:`CHOKEPOINT` may name.
BANNED_LEAVES: dict[str, frozenset[str]] = {
    "optuna": frozenset(
        {
            "create_study",
            "load_study",
            "delete_study",
            "copy_study",
            "get_storage",
            "RDBStorage",
            "JournalFileBackend",
            "JournalFileStorage",
        }
    ),
    "sqlite3": frozenset({"connect"}),
}

#: Every private helper :data:`CHOKEPOINT` defines. Each assumes its caller
#: already ran the validate-before-open order, so none may be imported outside
#: the ledger; the public API reaches them only after that order holds.
STORAGE_PRIVATE_NAMES = frozenset(
    {
        "_resolve_storage",
        "_build_phase_study",
        "_load_existing_phase_study",
        "_scan_ledger_format",
        "_validate_storage_versions",
        "_phase_trial_stats",
        "_phase_trial_stats_params",
        "_sqlite_phase_trial_stats",
        "_sqlite_study_exists",
        "_load_journal_study_snapshot",
        "_require_replay_matches_snapshot",
        "_journal_snapshot_storage",
        "_JournalSnapshot",
        "_trial_stats_from_rows",
        "_unavailable_phase_trial_stats",
        "_decoded_string_attr",
        "_describe_ledger",
    }
)

#: Banned storage entry points still reachable outside the chokepoint.
_LEGACY_SITES: dict[str, set[str]] = {}

#: Private ledger helpers still imported outside the chokepoint.
#:
#: Read paths hold a :class:`ValidatedLedger`, the write side holds a
#: :class:`ClaimedLedger`, and foreign-ledger recovery goes through
#: ``open_registry_study``; nothing outside the ledger reaches past its public
#: API.
_LEGACY_PRIVATE_IMPORTS: dict[str, set[str]] = {}


def _source_files() -> list[Path]:
    """Return every Python module shipped under ``src/phasesweep``.

    :return list[Path]: Sorted absolute paths of the package's ``.py`` files.
    """
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _relpath(path: Path) -> str:
    """Return a module path relative to ``src`` with POSIX separators.

    :param Path path: Absolute path to a packaged module.
    :return str: Path such as ``phasesweep/engine/ledger.py``.
    """
    return path.relative_to(SRC).as_posix()


def _module_name(relpath: str) -> str:
    """Return the dotted import name for a ``src``-relative module path.

    :param str relpath: Path such as ``phasesweep/engine/ledger.py``.
    :return str: Dotted module name such as ``phasesweep.engine.ledger``.
    """
    stem = relpath.removesuffix(".py")
    return stem.removesuffix("/__init__").replace("/", ".")


def _alias_table(tree: ast.AST) -> dict[str, str]:
    """Map every bound import alias in a module to its fully qualified target.

    The table is deliberately scope-insensitive: a function-local
    ``from optuna.storages.journal import JournalFileBackend`` binds the same
    name as a module-level one as far as this contract is concerned, and
    function-local imports are exactly how a call site would otherwise hide.

    :param ast.AST tree: Parsed module.
    :return dict[str, str]: Bound name -> dotted target it refers to.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    # ``import a.b.c`` binds ``a``; the chain resolver walks the rest.
                    root = alias.name.split(".")[0]
                    aliases[root] = root
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module is None:
                # Relative imports never reach ``optuna``/``sqlite3``.
                continue
            for alias in node.names:
                bound = alias.asname or alias.name
                aliases[bound] = f"{node.module}.{alias.name}"
    return aliases


def _dotted(node: ast.expr) -> str | None:
    """Flatten a ``Name``/``Attribute`` chain into a dotted string.

    :param ast.expr node: Expression to flatten.
    :return str | None: Dotted source text, or ``None`` when the chain is not
        a pure attribute access on a bare name (e.g. ``self.conn.connect``).
    """
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _resolve(dotted: str, aliases: dict[str, str]) -> str | None:
    """Rewrite a dotted reference through the module's import aliases.

    ``o.storages.RDBStorage`` under ``import optuna as o`` resolves to
    ``optuna.storages.RDBStorage``.

    :param str dotted: Dotted source text, as produced by :func:`_dotted`.
    :param dict[str, str] aliases: Alias table from :func:`_alias_table`.
    :return str | None: Fully qualified dotted name, or ``None`` when the root
        name was never imported in this module.
    """
    head, _, tail = dotted.partition(".")
    target = aliases.get(head)
    if target is None:
        return None
    return f"{target}.{tail}" if tail else target


def _is_banned(qualified: str) -> bool:
    """Return whether a fully qualified name is a banned storage entry point.

    :param str qualified: Fully qualified dotted name.
    :return bool: ``True`` when its root package and leaf are both banned.
    """
    parts = qualified.split(".")
    return parts[-1] in BANNED_LEAVES.get(parts[0], frozenset())


def banned_references(path: Path) -> set[str]:
    """Collect every banned storage entry point a module can reach.

    Both call targets (``optuna.create_study(...)``) and bare references
    (``factory = JournalFileBackend``) count, as does binding one by import,
    since any of the three puts the constructor within the module's reach.

    :param Path path: Module to scan.
    :return set[str]: Fully qualified banned names referenced by the module.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases = _alias_table(tree)
    found: set[str] = set()
    for bound, target in aliases.items():
        del bound
        if _is_banned(target):
            found.add(target)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name | ast.Attribute):
            continue
        dotted = _dotted(node)
        if dotted is None:
            continue
        qualified = _resolve(dotted, aliases)
        if qualified is not None and _is_banned(qualified):
            found.add(qualified)
    return found


def private_ledger_references(path: Path, ledger_module: str) -> set[str]:
    """Collect the storage-private helpers a module borrows from the ledger.

    Catches both ``from <ledger> import _x`` (including function-local ones)
    and attribute access such as ``optuna_module._x``.

    :param Path path: Module to scan.
    :param str ledger_module: Dotted name of the chokepoint module.
    :return set[str]: Storage-private names this module reaches for.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases = _alias_table(tree)
    found: set[str] = set()
    for target in aliases.values():
        module, _, leaf = target.rpartition(".")
        if module == ledger_module and leaf in STORAGE_PRIVATE_NAMES:
            found.add(leaf)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        dotted = _dotted(node)
        if dotted is None:
            continue
        qualified = _resolve(dotted, aliases)
        if qualified is None:
            continue
        module, _, leaf = qualified.rpartition(".")
        if module == ledger_module and leaf in STORAGE_PRIVATE_NAMES:
            found.add(leaf)
    return found


def test_storage_constructors_are_called_only_in_the_ledger_module() -> None:
    """Only the ledger module may reach Optuna/sqlite3 storage constructors."""
    chokepoint = SRC / CHOKEPOINT
    assert chokepoint.is_file(), (
        f"{CHOKEPOINT} does not exist; update CHOKEPOINT in this module when the "
        "ledger module is renamed or moved."
    )
    assert banned_references(chokepoint), (
        f"{CHOKEPOINT} no longer constructs any storage; the chokepoint moved and "
        "CHOKEPOINT here is stale."
    )

    found: dict[str, set[str]] = {}
    for path in _source_files():
        relpath = _relpath(path)
        if relpath == CHOKEPOINT:
            continue
        references = banned_references(path)
        if references:
            found[relpath] = references

    assert found == _LEGACY_SITES, (
        "Storage constructors outside the ledger module changed.\n"
        f"found:    {found}\n"
        f"expected: {_LEGACY_SITES}\n"
        "A new entry means a read path can build file-backed storage without "
        "validating first: route it through the ledger module instead. A removed "
        "entry means the ratchet must be tightened in the same commit."
    )


def test_ledger_private_names_are_not_imported_outside_the_ledger_module() -> None:
    """Storage-private ledger helpers stay private outside the ledger module."""
    chokepoint = SRC / CHOKEPOINT
    ledger_module = _module_name(CHOKEPOINT)
    tree = ast.parse(chokepoint.read_text(encoding="utf-8"), filename=str(chokepoint))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    missing = sorted(STORAGE_PRIVATE_NAMES - defined)
    assert not missing, (
        f"STORAGE_PRIVATE_NAMES lists {missing}, which {CHOKEPOINT} does not define; "
        "the list is stale."
    )

    found: dict[str, set[str]] = {}
    for path in _source_files():
        relpath = _relpath(path)
        if relpath == CHOKEPOINT:
            continue
        references = private_ledger_references(path, ledger_module)
        if references:
            found[relpath] = references

    assert found == _LEGACY_PRIVATE_IMPORTS, (
        "Private ledger helpers borrowed outside the ledger module changed.\n"
        f"found:    {found}\n"
        f"expected: {_LEGACY_PRIVATE_IMPORTS}\n"
        "Reaching past the ledger's public API skips the validate-before-open "
        "ordering the private helpers assume their caller already did."
    )


def test_ledger_public_api_returns_concrete_types() -> None:
    """Every name the ledger exports is defined and annotated with a real type."""
    chokepoint = SRC / CHOKEPOINT
    tree = ast.parse(chokepoint.read_text(encoding="utf-8"), filename=str(chokepoint))
    exported: list[str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            exported = [
                element.value
                for element in getattr(node.value, "elts", [])
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
    if exported is None:
        pytest.skip(
            f"{CHOKEPOINT} has no __all__, so its public API is undeclared and there "
            "is nothing for this test to check."
        )

    definitions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    problems: list[str] = []
    for name in exported:
        node = definitions.get(name)
        if node is None:
            problems.append(f"{name}: exported but not defined")
            continue
        if isinstance(node, ast.ClassDef):
            continue
        returns = node.returns
        if returns is None:
            problems.append(f"{name}: no return annotation")
        elif isinstance(returns, ast.Name) and returns.id == "Any":
            problems.append(f"{name}: returns Any instead of a concrete ledger type")
    assert not problems, "Ledger public API is not concretely typed: " + "; ".join(problems)
