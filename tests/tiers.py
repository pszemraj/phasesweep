"""Static classifier that decides which tests belong to the integration tier.

The fast review tier is ``pytest -m "not hardware and not integration"``. A test
belongs to the integration tier when it spawns real processes, waits on
wall-clock time, or drives a multi-step durable recovery workflow. The first two
properties are visible in the source, so this module detects them statically and
``tests/conftest.py`` fails collection when a test matches one of them without
carrying the marker. Detection is deliberately syntactic: it never imports the
module under inspection and never touches the filesystem beyond reading source.

Two rules fire here:

* **Rule (a), real processes** -- ``subprocess`` spawn entry points, ``os.killpg``,
  a ``start_new_session=`` keyword argument, or a call to the in-process MCP
  runner driver ``runner_main``.
* **Rule (b), wall-clock waits** -- a call whose leaf name is ``sleep``
  (``time.sleep``, ``asyncio.sleep``, a bare imported ``sleep``) or a string
  literal that embeds ``sleep(``, which is how the suite injects delays into
  generated trainer scripts. Monotonic clock reads such as ``time.monotonic``
  and ``time.perf_counter`` measure elapsed time without spending it and never
  qualify.

Both rules resolve transitively through same-module helper functions and
same-module fixtures, so a test that only calls a local ``_spawn_runner()``
helper is still classified by what that helper does.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

#: ``subprocess`` attributes that start a real child process.
SUBPROCESS_SPAWNERS = frozenset({"Popen", "run", "check_output", "check_call", "call"})

#: ``os`` attributes that signal direct process-group control.
OS_PROCESS_CONTROLS = frozenset({"killpg"})

#: Module attribute leaves that mean "this touches a real process", per head module.
_PROCESS_ATTRS: dict[str, frozenset[str]] = {
    "subprocess": SUBPROCESS_SPAWNERS,
    "os": OS_PROCESS_CONTROLS,
}

#: Bare call names that drive a full detached-run lifecycle in this suite.
PROCESS_DRIVER_CALLS = frozenset({"runner_main"})

#: Keyword arguments that only appear when a test becomes a session leader.
SESSION_KEYWORDS = frozenset({"start_new_session"})

#: Call leaf name that spends wall-clock time.
SLEEP_CALL = "sleep"

#: ``sleep(`` inside a string literal, i.e. a delay embedded in a generated script.
SLEEP_IN_SOURCE_STRING = re.compile(r"\bsleep\(")

_FunctionDef = ast.FunctionDef | ast.AsyncFunctionDef


def _decorator_leaf(node: ast.expr) -> str:
    """Return the trailing attribute or name of a decorator expression.

    :param ast.expr node: Decorator expression, possibly a call.
    :return: Trailing identifier, or the empty string when there is none.
    """
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _call_leaf(node: ast.Call) -> str:
    """Return the callee's trailing identifier for a call node.

    :param ast.Call node: Call expression to inspect.
    :return: Trailing identifier of the callee, or the empty string.
    """
    return _decorator_leaf(node.func)


def _attribute_head(node: ast.Attribute) -> str:
    """Return the base name of an attribute chain such as ``subprocess`` in ``subprocess.run``.

    :param ast.Attribute node: Attribute expression to inspect.
    :return: Name at the base of the chain, or the empty string when the base is
        not a plain name.
    """
    value = node.value
    while isinstance(value, ast.Attribute):
        value = value.value
    return value.id if isinstance(value, ast.Name) else ""


class _ModuleIndex:
    """Import aliases, function table, and fixture set for one parsed module."""

    def __init__(self, tree: ast.Module) -> None:
        """Index one parsed module so its test bodies can be resolved.

        :param ast.Module tree: Parsed module to index.
        """
        # Alias -> canonical module name, for `import subprocess as sp`.
        self.module_aliases: dict[str, str] = {}
        # Bare name -> "module.attr", for `from subprocess import Popen`.
        self.imported_names: dict[str, str] = {}
        self.functions: dict[str, _FunctionDef] = {}
        self.fixtures: set[str] = set()
        self._index_imports(tree)
        self._index_functions(tree)

    def _index_imports(self, tree: ast.Module) -> None:
        """Record every ``subprocess``/``os`` import in the module, at any nesting depth."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _PROCESS_ATTRS:
                        self.module_aliases[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                interesting = _PROCESS_ATTRS.get(node.module or "")
                if interesting is None:
                    continue
                for alias in node.names:
                    if alias.name in interesting:
                        self.imported_names[alias.asname or alias.name] = (
                            f"{node.module}.{alias.name}"
                        )

    def _index_functions(self, tree: ast.Module) -> None:
        """Record module-level functions and ``Test*`` class methods as resolvable helpers."""
        for node in tree.body:
            if isinstance(node, _FunctionDef):
                self._add_function(node)
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, _FunctionDef):
                        self._add_function(child)

    def _add_function(self, node: _FunctionDef) -> None:
        """Register one function definition, noting whether it is a pytest fixture."""
        self.functions.setdefault(node.name, node)
        if any(_decorator_leaf(deco) == "fixture" for deco in node.decorator_list):
            self.fixtures.add(node.name)

    def parameter_helpers(self, node: _FunctionDef) -> list[str]:
        """Return parameter names of ``node`` that name a fixture defined in this module.

        :param node: Function whose signature is inspected.
        :return: Names of same-module fixtures the function requests.
        """
        args = node.args
        names = [arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
        return [name for name in names if name in self.fixtures]


def _local_reason(node: ast.AST, index: _ModuleIndex) -> str | None:
    """Return why one syntax node makes a test an integration test, or ``None``.

    Only the node itself is considered; callers walk the tree and recurse through
    helpers.

    :param ast.AST node: Syntax node to classify.
    :param _ModuleIndex index: Import and function table for the enclosing module.
    :return: Human-readable reason, or ``None`` when the node is tier-neutral.
    """
    if isinstance(node, ast.Attribute):
        head = index.module_aliases.get(_attribute_head(node))
        if head is not None and node.attr in _PROCESS_ATTRS[head]:
            return f"references {head}.{node.attr}"
    elif isinstance(node, ast.Name):
        qualified = index.imported_names.get(node.id)
        if qualified is not None:
            return f"references {qualified}"
    elif isinstance(node, ast.keyword):
        if node.arg in SESSION_KEYWORDS:
            return f"passes {node.arg}="
    elif isinstance(node, ast.Call):
        leaf = _call_leaf(node)
        if leaf in PROCESS_DRIVER_CALLS:
            return f"calls {leaf}(), which drives a real detached run"
        if leaf == SLEEP_CALL:
            return "calls sleep(), a wall-clock wait"
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, str) and SLEEP_IN_SOURCE_STRING.search(node.value):
            return "embeds sleep( in a string literal, a wall-clock wait in generated source"
    return None


def _describe(reason: str, chain: tuple[str, ...]) -> str:
    """Attach the helper chain that led to a reason.

    :param str reason: Reason discovered in the helper body.
    :param tuple chain: Helper names traversed from the test to that body.
    :return: Reason string, annotated with the helper chain when there is one.
    """
    if not chain:
        return reason
    return f"{reason} (via {' -> '.join(chain)})"


def _scan_function(
    node: _FunctionDef,
    index: _ModuleIndex,
    seen: set[str],
    chain: tuple[str, ...],
) -> str | None:
    """Return the first integration reason found in ``node`` or its same-module helpers.

    :param node: Function definition to scan.
    :param _ModuleIndex index: Import and function table for the enclosing module.
    :param set seen: Function names already visited on this path, to stop recursion.
    :param tuple chain: Helper names traversed so far, for the reason string.
    :return: Human-readable reason, or ``None`` when nothing fires.
    """
    if node.name in seen:
        return None
    seen.add(node.name)

    pending: list[tuple[str, _FunctionDef]] = []
    for fixture_name in index.parameter_helpers(node):
        if fixture_name not in seen:
            pending.append((fixture_name, index.functions[fixture_name]))

    for child in ast.walk(node):
        reason = _local_reason(child, index)
        if reason is not None:
            return _describe(reason, chain)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            helper = index.functions.get(child.id)
            if helper is not None and child.id not in seen:
                pending.append((child.id, helper))

    for name, helper in pending:
        reason = _scan_function(helper, index, seen, (*chain, name))
        if reason is not None:
            return reason
    return None


def _test_functions(tree: ast.Module) -> list[_FunctionDef]:
    """Return every ``test_*`` function in a module, including ``Test*`` class methods.

    :param ast.Module tree: Parsed module.
    :return: Test function definitions in source order.
    """
    found: list[_FunctionDef] = []
    for node in tree.body:
        if isinstance(node, _FunctionDef) and node.name.startswith("test_"):
            found.append(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            found.extend(
                child
                for child in node.body
                if isinstance(child, _FunctionDef) and child.name.startswith("test_")
            )
    return found


def scan_module(source: str) -> dict[str, str]:
    """Return ``{test name: reason}`` for every test in ``source`` that belongs to the integration tier.

    Pure function: it parses the given text and performs no I/O. Tests are keyed
    by their bare function name, which is what pytest reports as
    ``item.originalname`` for parametrized cases.

    :param str source: Python source text of a test module.
    :return: Mapping of test function name to the reason it is an integration test.
        Tests that do not qualify are absent.
    :raises SyntaxError: If ``source`` is not parseable Python.
    """
    tree = ast.parse(source)
    index = _ModuleIndex(tree)
    flagged: dict[str, str] = {}
    for node in _test_functions(tree):
        reason = _scan_function(node, index, set(), ())
        if reason is not None:
            flagged[node.name] = reason
    return flagged


def flagged_tests(path: Path) -> dict[str, str]:
    """Return ``{test name: reason}`` for one test module on disk.

    :param Path path: Test module to read and scan.
    :return: Mapping of test function name to the reason it is an integration test.
    :raises OSError: If ``path`` cannot be read.
    :raises SyntaxError: If ``path`` does not contain parseable Python.
    """
    return scan_module(path.read_text(encoding="utf-8"))
