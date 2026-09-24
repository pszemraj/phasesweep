"""Static contract pinning every storage constructor to one ledger module.

The durability invariants say read paths must never create studies or build
file-backed storage, and that validation happens before any bind/claim/open.
Those rules only hold if there is exactly *one* module that may talk to
Optuna's storage constructors, so every other module has to go through helpers
that validate first. This module enforces that statically with :mod:`ast`, so a
new call site fails in CI before it can ever run.

Most storage entry points are banned by name (:data:`BANNED_LEAVES`): calling,
referencing, or importing one all count. ``optuna.Study`` and
``sqlite3.Connection`` are banned only as calls (:data:`BANNED_CALLS`), because
the same classes annotate study and connection objects.

The chokepoint is ``phasesweep/engine/ledger.py``. Its private names are read
from its own source, so a new helper is private the moment it is defined. Two
ratchet dicts (:data:`_LEGACY_SITES`, :data:`_LEGACY_PRIVATE_IMPORTS`) record
what is still reachable from outside it. They are compared for *equality*, not
containment: adding a call site fails, and removing one fails until the ratchet
is tightened in the same commit. Both are empty: every storage constructor and
every storage-private helper is reached only through the ledger's public handle
API, so any entry added to either one is a new bypass that needs a stated
reason.

Optuna's journal is PhaseSweep's only durable backend, so SQLite and generic
SQLAlchemy-RDB constructors (:data:`BANNED_EVERYWHERE`) are refused even inside
the chokepoint itself; only the journal/Optuna entry points in
:data:`BANNED_LEAVES`/:data:`BANNED_CALLS` are allowed there.

Each detector also runs over synthetic source that must trip it, so a detector
that stops finding anything fails here instead of passing every real module.

Out of scope on purpose, because no static reader can follow them: dynamic
attribute access such as ``getattr(optuna, "create_study")``, and anything
reached through :mod:`importlib`. Both are absent from ``src/phasesweep``
today, and no automated gate keeps them out: ``tests/tiers.py`` scans only test
modules, so review is the only guard.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
PACKAGE_ROOT = SRC / "phasesweep"

#: Module that is allowed to construct storage.
CHOKEPOINT = "phasesweep/engine/ledger.py"

#: Storage entry points no module outside :data:`CHOKEPOINT` may name.
#: ``JournalStorage`` is banned outright, like ``RDBStorage``: over any backend
#: it is live storage, and only the chokepoint names it, even as a type. The
#: study listers take a storage URL and build that storage to answer, so they
#: create a missing SQLite file like any other constructor.
BANNED_LEAVES: dict[str, frozenset[str]] = {
    "optuna": frozenset(
        {
            "create_study",
            "load_study",
            "delete_study",
            "copy_study",
            "get_all_study_names",
            "get_all_study_summaries",
            "get_storage",
            "RDBStorage",
            "JournalFileBackend",
            "JournalFileStorage",
            "JournalStorage",
        }
    ),
    "sqlalchemy": frozenset({"create_engine", "engine_from_config"}),
    "sqlite3": frozenset({"connect"}),
}

#: Storage entry points no module outside :data:`CHOKEPOINT` may *call*.
#: ``optuna.Study(name, storage)`` resolves a storage URL through
#: ``get_storage`` internally, and ``sqlite3.Connection(path)`` opens the
#: database itself, but both classes also annotate objects throughout the
#: package, so only a call to either counts.
BANNED_CALLS: dict[str, frozenset[str]] = {
    "optuna": frozenset({"Study"}),
    "sqlite3": frozenset({"Connection"}),
}

#: Leaves among :data:`BANNED_LEAVES`/:data:`BANNED_CALLS` that are banned even
#: inside :data:`CHOKEPOINT` itself. Optuna's journal is the only durable
#: backend PhaseSweep has, so SQLite and generic SQLAlchemy-RDB storage have no
#: legitimate constructor anywhere in the package -- not even in the one
#: module allowed to build the journal's own live storage.
BANNED_EVERYWHERE: dict[str, frozenset[str]] = {
    "optuna": frozenset({"RDBStorage"}),
    "sqlalchemy": frozenset({"create_engine", "engine_from_config"}),
    "sqlite3": frozenset({"connect", "Connection"}),
}

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


def _package(path: Path, relpath: str | None) -> str | None:
    """Return the package a module's relative imports resolve against.

    :param Path path: Module on disk.
    :param str | None relpath: The module's ``src``-relative path, when it is
        not the path's own location under ``src``.
    :return str | None: Dotted package name, or ``None`` when the module sits
        outside ``src`` and no ``relpath`` places it in a package.
    """
    if relpath is None:
        if not path.is_relative_to(SRC):
            return None
        relpath = _relpath(path)
    module = _module_name(relpath)
    return module if relpath.endswith("/__init__.py") else module.rpartition(".")[0]


def _import_source(node: ast.ImportFrom, package: str | None) -> str | None:
    """Return the absolute module an ``ImportFrom`` reads, resolving relative levels.

    :param ast.ImportFrom node: Import statement.
    :param str | None package: Package of the importing module.
    :return str | None: Absolute dotted module, or ``None`` when a relative
        import cannot be anchored.
    """
    if not node.level:
        return node.module
    if not package:
        return None
    parts = package.split(".")
    if node.level - 1 >= len(parts):
        return None
    base = ".".join(parts[: len(parts) - (node.level - 1)])
    return f"{base}.{node.module}" if node.module else base


def _alias_table(tree: ast.AST, package: str | None) -> dict[str, str]:
    """Map every bound import alias in a module to its fully qualified target.

    The table is deliberately scope-insensitive: a function-local
    ``from optuna.storages.journal import JournalFileBackend`` binds the same
    name as a module-level one as far as this contract is concerned, and
    function-local imports are exactly how a call site would otherwise hide.
    Relative imports resolve against ``package``, so ``from .ledger import _x``
    names the same helper as its absolute spelling.

    :param ast.AST tree: Parsed module.
    :param str | None package: Package of the module, for relative imports.
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
            source = _import_source(node, package)
            if source is None:
                continue
            for alias in node.names:
                bound = alias.asname or alias.name
                aliases[bound] = f"{source}.{alias.name}"
    return aliases


def _parse(path: Path, relpath: str | None) -> tuple[ast.Module, dict[str, str]]:
    """Parse a module and build its import alias table.

    :param Path path: Module to parse.
    :param str | None relpath: ``src``-relative path anchoring relative imports.
    :return tuple[ast.Module, dict[str, str]]: Parsed tree and alias table.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tree, _alias_table(tree, _package(path, relpath))


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


def _is_banned(qualified: str, banned: dict[str, frozenset[str]] = BANNED_LEAVES) -> bool:
    """Return whether a fully qualified name is a banned storage entry point.

    :param str qualified: Fully qualified dotted name.
    :param dict[str, frozenset[str]] banned: Root package -> banned leaves.
    :return bool: ``True`` when its root package and leaf are both banned.
    """
    parts = qualified.split(".")
    return parts[-1] in banned.get(parts[0], frozenset())


def banned_references(path: Path, *, relpath: str | None = None) -> set[str]:
    """Collect every banned storage entry point a module can reach.

    For :data:`BANNED_LEAVES`, both call targets (``optuna.create_study(...)``)
    and bare references (``factory = JournalFileBackend``) count, as does
    binding one by import, since any of the three puts the constructor within
    the module's reach. For :data:`BANNED_CALLS` only a call counts, whatever
    import spelling names the class.

    :param Path path: Module to scan.
    :param str | None relpath: ``src``-relative path for a module that does
        not live under ``src``, anchoring its relative imports.
    :return set[str]: Fully qualified banned names referenced by the module.
    """
    tree, aliases = _parse(path, relpath)
    found = {target for target in aliases.values() if _is_banned(target)}
    for node in ast.walk(tree):
        is_call = isinstance(node, ast.Call)
        target_node = node.func if isinstance(node, ast.Call) else node
        if not isinstance(target_node, ast.Name | ast.Attribute):
            continue
        dotted = _dotted(target_node)
        if dotted is None:
            continue
        qualified = _resolve(dotted, aliases)
        if qualified is None:
            continue
        if _is_banned(qualified) or (is_call and _is_banned(qualified, BANNED_CALLS)):
            found.add(qualified)
    return found


def ledger_private_names(path: Path) -> frozenset[str]:
    """Return every private name a ledger module defines at module level.

    A private name is a ``_``-prefixed, non-dunder def, class, or assignment
    target, including one defined under a module-level ``if``, ``try``, or
    ``with``. Each assumes its caller already ran the validate-before-open
    order, so none may be imported outside the ledger. Names the module only
    imports belong to their defining module and are not listed.

    :param Path path: Ledger module to read.
    :return frozenset[str]: The module's private names.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    pending: list[ast.stmt] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                name.id
                for target in node.targets
                for name in ast.walk(target)
                if isinstance(name, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign | ast.AugAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.If):
            pending.extend([*node.body, *node.orelse])
        elif isinstance(node, ast.Try):
            pending.extend([*node.body, *node.orelse, *node.finalbody])
            for handler in node.handlers:
                pending.extend(handler.body)
        elif isinstance(node, ast.With):
            pending.extend(node.body)
    return frozenset(name for name in names if name.startswith("_") and not name.startswith("__"))


def private_ledger_references(
    path: Path,
    ledger_module: str,
    private_names: frozenset[str],
    *,
    relpath: str | None = None,
) -> set[str]:
    """Collect the storage-private helpers a module borrows from the ledger.

    Catches ``from <ledger> import _x`` under any absolute or relative spelling
    (including function-local ones) and attribute access such as
    ``ledger._x``.

    :param Path path: Module to scan.
    :param str ledger_module: Dotted name of the chokepoint module.
    :param frozenset[str] private_names: Names from :func:`ledger_private_names`.
    :param str | None relpath: ``src``-relative path for a module that does
        not live under ``src``, anchoring its relative imports.
    :return set[str]: Storage-private names this module reaches for.
    """
    tree, aliases = _parse(path, relpath)
    qualified_names = set(aliases.values())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            qualified = _resolve(dotted, aliases) if dotted is not None else None
            if qualified is not None:
                qualified_names.add(qualified)
    found: set[str] = set()
    for qualified in qualified_names:
        module, _, leaf = qualified.rpartition(".")
        if module == ledger_module and leaf in private_names:
            found.add(leaf)
    return found


def test_storage_constructors_are_called_only_in_the_ledger_module() -> None:
    """Only the ledger module may reach Optuna's journal constructors; SQLite/RDB ones nowhere."""
    chokepoint = SRC / CHOKEPOINT
    assert chokepoint.is_file(), (
        f"{CHOKEPOINT} does not exist; update CHOKEPOINT in this module when the "
        "ledger module is renamed or moved."
    )
    chokepoint_references = banned_references(chokepoint)
    assert chokepoint_references, (
        f"{CHOKEPOINT} no longer constructs any storage; the chokepoint moved and "
        "CHOKEPOINT here is stale."
    )
    sqlite_or_rdb_in_chokepoint = {
        qualified
        for qualified in chokepoint_references
        if qualified.rpartition(".")[2]
        in BANNED_EVERYWHERE.get(qualified.split(".", 1)[0], frozenset())
    }
    assert not sqlite_or_rdb_in_chokepoint, (
        f"{CHOKEPOINT} references SQLite/RDB storage: {sqlite_or_rdb_in_chokepoint}. "
        "The journal is PhaseSweep's only durable backend now, so SQLite/RDB "
        "constructors are banned everywhere, including the chokepoint itself."
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
    private_names = ledger_private_names(chokepoint)
    assert private_names, f"{CHOKEPOINT} defines no private names; the derivation broke."

    found: dict[str, set[str]] = {}
    for path in _source_files():
        relpath = _relpath(path)
        if relpath == CHOKEPOINT:
            continue
        references = private_ledger_references(path, ledger_module, private_names)
        if references:
            found[relpath] = references

    assert found == _LEGACY_PRIVATE_IMPORTS, (
        "Private ledger helpers borrowed outside the ledger module changed.\n"
        f"found:    {found}\n"
        f"expected: {_LEGACY_PRIVATE_IMPORTS}\n"
        "Reaching past the ledger's public API skips the validate-before-open "
        "ordering the private helpers assume their caller already did."
    )


def test_visitor_flags_storage_construction_but_not_type_annotations(tmp_path: Path) -> None:
    """Classes banned as calls stay usable as types; every other builder is banned outright.

    ``optuna.Study(name, storage)`` resolves a storage URL internally and
    ``sqlite3.Connection(path)`` opens the file, so the ban must see both
    calls, yet the same classes annotate study and connection objects all over
    the package. The visitor has to tell a call from a type, whichever import
    spelling reaches the class. Parsing a SQLite URL the way SQLAlchemy does
    opens nothing, so it stays allowed.
    """
    annotated = tmp_path / "annotated.py"
    annotated.write_text(
        "import optuna\n"
        "import sqlite3\n"
        "from optuna import Study\n"
        "from optuna.study import Study as Aliased\n"
        "from sqlalchemy.engine import make_url\n"
        "\n"
        "def keep(study: optuna.Study, other: Study, conn: sqlite3.Connection) -> optuna.study.Study:\n"
        "    x: Aliased = study\n"
        "    url = make_url('sqlite:///db')\n"
        "    url.get_dialect()().create_connect_args(url)\n"
        "    return isinstance(other, optuna.Study) and isinstance(conn, sqlite3.Connection) and x\n",
        encoding="utf-8",
    )
    constructing = tmp_path / "constructing.py"
    constructing.write_text(
        "import optuna\n"
        "import sqlalchemy\n"
        "import sqlite3\n"
        "from optuna.study import Study as Aliased\n"
        "from optuna.study import get_all_study_summaries\n"
        "from optuna.storages import JournalStorage\n"
        "from sqlite3 import Connection\n"
        "\n"
        "def build(url, backend, config):\n"
        "    optuna.Study('s', url)\n"
        "    optuna.study.Study('s', url)\n"
        "    Aliased('s', url)\n"
        "    optuna.get_all_study_names(storage=url)\n"
        "    get_all_study_summaries(storage=url)\n"
        "    sqlite3.Connection(url)\n"
        "    Connection(url)\n"
        "    sqlalchemy.create_engine(url)\n"
        "    sqlalchemy.engine_from_config(config)\n"
        "    return JournalStorage(backend)\n",
        encoding="utf-8",
    )

    assert banned_references(annotated) == set()
    assert banned_references(constructing) == {
        "optuna.Study",
        "optuna.study.Study",
        "optuna.get_all_study_names",
        "optuna.study.get_all_study_summaries",
        "optuna.storages.JournalStorage",
        "sqlite3.Connection",
        "sqlalchemy.create_engine",
        "sqlalchemy.engine_from_config",
    }


def test_ledger_private_names_are_derived_from_its_source(tmp_path: Path) -> None:
    """Every module-level private definition counts, and nothing public or imported does."""
    ledger = tmp_path / "ledger.py"
    ledger.write_text(
        "from phasesweep.engine.optuna import _borrowed\n"
        "__all__ = ['public']\n"
        "_SQL = 'SELECT 1'\n"
        "_TYPED: int = 1\n"
        "_A, _B = 1, 2\n"
        "def _helper(): ...\n"
        "async def _async_helper(): ...\n"
        "class _Snapshot: ...\n"
        "try:\n"
        "    def _guarded(): ...\n"
        "except ImportError:\n"
        "    def _fallback(): ...\n"
        "if True:\n"
        "    _CONDITIONAL = 1\n"
        "def public():\n"
        "    _local = 1\n"
        "    def _nested(): ...\n",
        encoding="utf-8",
    )

    assert ledger_private_names(ledger) == {
        "_SQL",
        "_TYPED",
        "_A",
        "_B",
        "_helper",
        "_async_helper",
        "_Snapshot",
        "_guarded",
        "_fallback",
        "_CONDITIONAL",
    }


#: Case id -> (importing module's ``src``-relative path, its source, helpers it borrows).
PRIVATE_IMPORT_PROBES: dict[str, tuple[str, str, set[str]]] = {
    "absolute": (
        "phasesweep/cli.py",
        "from phasesweep.engine.ledger import _helper\n",
        {"_helper"},
    ),
    "function-local-alias": (
        "phasesweep/cli.py",
        "def f():\n    from phasesweep.engine.ledger import _Snapshot as S\n    return S\n",
        {"_Snapshot"},
    ),
    "module-attribute": (
        "phasesweep/cli.py",
        "import phasesweep.engine.ledger as ledger\n\nledger._SQL\n",
        {"_SQL"},
    ),
    "sibling-relative": (
        "phasesweep/engine/read.py",
        "from .ledger import _helper\n",
        {"_helper"},
    ),
    "parent-relative": (
        "phasesweep/mcp/recovery.py",
        "from ..engine.ledger import _Snapshot\n",
        {"_Snapshot"},
    ),
    "package-relative-attribute": (
        "phasesweep/engine/read.py",
        "from . import ledger\n\nledger._SQL\n",
        {"_SQL"},
    ),
    "from-package-init": (
        "phasesweep/engine/__init__.py",
        "from .ledger import _helper\n",
        {"_helper"},
    ),
    "public-name": (
        "phasesweep/engine/read.py",
        "from .ledger import validate_ledger\n",
        set(),
    ),
    "other-package-ledger": (
        "phasesweep/mcp/recovery.py",
        "from .ledger import _helper\n",
        set(),
    ),
}


@pytest.mark.parametrize("case", sorted(PRIVATE_IMPORT_PROBES))
def test_private_import_detector_resolves_every_spelling(case: str, tmp_path: Path) -> None:
    """Each import spelling of a ledger private is caught; public and foreign names are not."""
    relpath, source, expected = PRIVATE_IMPORT_PROBES[case]
    module = tmp_path / "probe.py"
    module.write_text(source, encoding="utf-8")
    private_names = frozenset({"_helper", "_Snapshot", "_SQL"})

    found = private_ledger_references(
        module, "phasesweep.engine.ledger", private_names, relpath=relpath
    )

    assert found == expected


def _annotation_names(annotation: ast.expr) -> Iterator[str]:
    """Yield every name an annotation mentions, looking inside quoted forward references.

    :param ast.expr annotation: Annotation expression.
    :return Iterator[str]: Bare names and attribute leaves, in walk order.
    """
    for part in ast.walk(annotation):
        if isinstance(part, ast.Name):
            yield part.id
        elif isinstance(part, ast.Attribute):
            yield part.attr
        elif isinstance(part, ast.Constant) and isinstance(part.value, str):
            try:
                quoted = ast.parse(part.value, mode="eval").body
            except SyntaxError:
                continue  # a ``Literal`` string, not a type
            yield from _annotation_names(quoted)


def _return_type_problem(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Explain why a public ledger function's return annotation is not concrete.

    :param node: Function definition to check.
    :return str | None: The problem, or ``None`` when the annotation names only
        concrete types.
    """
    if node.returns is None:
        return "no return annotation"
    if any(name in ("Any", "object") for name in _annotation_names(node.returns)):
        return f"returns {ast.unparse(node.returns)}, which admits any object"
    return None


def test_ledger_public_api_returns_concrete_types() -> None:
    """Every name the ledger exports is defined and annotated with a real type."""
    chokepoint = SRC / CHOKEPOINT
    tree = ast.parse(chokepoint.read_text(encoding="utf-8"), filename=str(chokepoint))
    exported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            exported = [
                element.value
                for element in getattr(value, "elts", [])
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
    assert exported, (
        f"{CHOKEPOINT} declares no names in __all__, so its public API is undeclared; "
        "list the handles and openers other modules may use."
    )

    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    problems: list[str] = []
    for name in exported:
        node = definitions.get(name)
        if node is None:
            problems.append(f"{name}: exported but not a module-level function or class")
            continue
        if isinstance(node, ast.ClassDef):
            continue
        problem = _return_type_problem(node)
        if problem is not None:
            problems.append(f"{name}: {problem}")
    assert not problems, "Ledger public API is not concretely typed: " + "; ".join(problems)


@pytest.mark.parametrize(
    ("annotation", "concrete"),
    [
        ("optuna.Study | None", True),
        ("ClaimedLedger", True),
        ("'ClaimedLedger'", True),
        ("list['ClaimedLedger']", True),
        ("Literal['memory', 'on disk']", True),
        ("Any", False),
        ("typing.Any", False),
        ("Any | None", False),
        ("dict[str, Any]", False),
        ("object", False),
        ("'Any'", False),
        ("list['Any']", False),
    ],
)
def test_concrete_return_rule_rejects_any_object(annotation: str, concrete: bool) -> None:
    """``Any`` or ``object`` anywhere in a return annotation fails the public-API rule."""
    node = ast.parse(f"def f() -> {annotation}: ...").body[0]
    assert isinstance(node, ast.FunctionDef)
    assert (_return_type_problem(node) is None) is concrete
