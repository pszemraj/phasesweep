"""Static classifier that decides which tests belong to the integration tier.

The fast review tier is ``pytest -m "not hardware and not integration"``. A test
belongs to the integration tier when it manages real processes, waits on
wall-clock time, drives a multi-step durable recovery workflow, or is otherwise
slow. This module enforces the part of that rule a test's own source shows: a
test that manages processes or waits on the clock *directly*, in its body or in
a same-module helper or fixture. ``tests/conftest.py`` fails collection when
such a test lacks the marker.

What the scanner does not see is deliberate. An engine run with a quick trainer,
such as ``run_experiment`` over an ``echo`` trainer, spawns its subprocess inside
the package; it stays in the fast tier, which relies on those runs for engine
coverage. Any other slow test is marked by the classification rule in
``docs/development.md``, not by this scanner. To keep such a test visible,
``tests/conftest.py`` lists unmarked tests whose call phase took at least
:data:`SLOW_CALL_SECONDS` in any run that leaves the integration tier out. The
list is a report, never a failure: timing-based failures flake on loaded hosts.

Detection is syntactic: it never imports the module under inspection and never
touches the filesystem beyond reading source. Two rules fire here:

* **Rule (a), real processes** -- a reference to a process primitive in
  :data:`PROCESS_PRIMITIVES` or the ``os.spawn*`` family, under any import
  spelling; a ``start_new_session=`` keyword argument; or a call to a shared
  helper in :data:`PROCESS_DRIVER_CALLS`.
* **Rule (b), wall-clock waits** -- a call whose leaf name is ``sleep``
  (``time.sleep``, ``asyncio.sleep``, a bare imported ``sleep``); a ``.wait(``
  given a timeout that is meant to run out, i.e. on a freshly built object no
  other code can set (``threading.Event().wait(5)``) or asserted to return false
  (``assert not ready.wait(5)``); a reference to a primitive in
  :data:`WALLCLOCK_PRIMITIVES` (``threading.Timer``, ``select.select``,
  ``signal.pause``); a call to a shared helper in :data:`WALLCLOCK_HELPER_CALLS`;
  or a string literal that embeds ``sleep(``, which is how the suite injects
  delays into generated trainer scripts. A handshake such as
  ``assert ready.wait(timeout=5)`` returns as soon as another thread sets the
  event, so its timeout only bounds a hang and does not qualify. Monotonic clock
  reads such as ``time.monotonic`` and ``time.perf_counter`` measure elapsed
  time without spending it and never qualify.

Both rules resolve transitively through same-module helper functions and
same-module fixtures, so a test that only calls a local ``_spawn_runner()``
helper, or only requests a local fixture that spawns, is still classified by
what that helper does. Helpers shared from ``tests/conftest.py``,
``tests/mcp_helpers.py`` and ``tests/ledger_fixtures.py`` live in another module,
so they are classified by name instead: :func:`scan_helpers` finds the ones that
spawn or wait, and ``tests/test_tier_guard.py`` fails until each is registered.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol

from _pytest.mark.expression import Expression

#: Root package -> attribute leaves that start or signal a real process. A
#: reference counts whatever path reaches the leaf, so ``multiprocessing.Pool``
#: and ``multiprocessing.pool.Pool`` both match.
PROCESS_PRIMITIVES: dict[str, frozenset[str]] = {
    "subprocess": frozenset(
        {"Popen", "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput"}
    ),
    "os": frozenset(
        {"killpg", "system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp"}
    ),
    "multiprocessing": frozenset({"Process", "Pool"}),
    "concurrent": frozenset({"ProcessPoolExecutor"}),
    "asyncio": frozenset({"create_subprocess_exec", "create_subprocess_shell"}),
    "pty": frozenset({"spawn", "fork"}),
}

#: ``os.spawnl`` through ``os.spawnvpe`` are one family, matched by prefix.
OS_SPAWN_PREFIX = "spawn"

#: Bare call names of shared helpers that manage a real process. ``runner_main``
#: drives a full detached-run lifecycle in pytest's own process. ``reaped_pid``
#: in conftest is deliberately absent: it runs ``true`` to completion before
#: returning, so the calling test neither manages a live process nor waits.
PROCESS_DRIVER_CALLS = frozenset({"runner_main"})

#: Bare call names of shared helpers that wait on the wall clock.
WALLCLOCK_HELPER_CALLS = frozenset({"wait_for_mcp_running_trial"})

#: Root package -> attribute leaves that block until a deadline or a signal.
WALLCLOCK_PRIMITIVES: dict[str, frozenset[str]] = {
    "threading": frozenset({"Timer"}),
    "select": frozenset({"select"}),
    "signal": frozenset({"pause"}),
}

#: Keyword arguments that only appear when a test becomes a session leader.
SESSION_KEYWORDS = frozenset({"start_new_session"})

#: Call leaf name that spends wall-clock time.
SLEEP_CALL = "sleep"

#: Method leaf that blocks until an event or, given a timeout, a deadline.
WAIT_METHOD = "wait"

#: ``sleep(`` inside a string literal, i.e. a delay embedded in a generated script.
SLEEP_IN_SOURCE_STRING = re.compile(r"\bsleep\(")

#: Call-phase seconds at which an unmarked test is slow enough to classify.
SLOW_CALL_SECONDS = 1.0

#: Markers that already keep a test out of the fast tier.
TIER_MARKERS = frozenset({"integration", "hardware"})

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


def _dotted(node: ast.expr) -> str | None:
    """Flatten a ``Name``/``Attribute`` chain such as ``os.path.join`` into dotted text.

    :param ast.expr node: Expression to flatten.
    :return: Dotted text, or ``None`` when the chain does not start at a bare name.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def is_process_primitive(qualified: str) -> bool:
    """Return whether a fully qualified name is a process primitive under rule (a).

    :param str qualified: Dotted name after import aliases are resolved.
    :return: ``True`` for a leaf in :data:`PROCESS_PRIMITIVES` or the ``os.spawn*`` family.
    """
    root, _, rest = qualified.partition(".")
    if not rest:
        return False
    leaf = rest.rpartition(".")[2]
    if leaf in PROCESS_PRIMITIVES.get(root, frozenset()):
        return True
    return root == "os" and rest.startswith(OS_SPAWN_PREFIX)


def is_wallclock_primitive(qualified: str) -> bool:
    """Return whether a fully qualified name is a blocking primitive under rule (b).

    :param str qualified: Dotted name after import aliases are resolved.
    :return: ``True`` for a leaf in :data:`WALLCLOCK_PRIMITIVES`.
    """
    root, _, rest = qualified.partition(".")
    return bool(rest) and rest.rpartition(".")[2] in WALLCLOCK_PRIMITIVES.get(root, frozenset())


def _registered_helper(name: str) -> str | None:
    """Describe a shared helper registered by name, or return ``None``.

    :param str name: Bare name a test calls or requests as a fixture.
    :return: What the helper does, or ``None`` when it is not registered.
    """
    if name in PROCESS_DRIVER_CALLS:
        return "a shared helper that manages a real process"
    if name in WALLCLOCK_HELPER_CALLS:
        return "a shared helper that waits on the wall clock"
    return None


def _parameter_names(node: _FunctionDef) -> list[str]:
    """Return the names of every parameter ``node`` declares.

    :param node: Function whose signature is inspected.
    :return: Positional-only, ordinary, and keyword-only parameter names.
    """
    args = node.args
    return [arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)]


def _timed_wait(node: ast.expr) -> bool:
    """Return whether an expression is a ``.wait(`` method call given a timeout.

    :param ast.expr node: Expression to inspect.
    :return: ``True`` for ``x.wait(t)`` or ``x.wait(timeout=t)``.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == WAIT_METHOD
    ):
        return False
    return bool(node.args) or any(keyword.arg == "timeout" for keyword in node.keywords)


def _wait_meant_to_run_out(node: ast.AST) -> bool:
    """Return whether a node is a timed ``.wait(`` that spends its whole timeout.

    That is a wait on an object built in the same expression, which no other
    code can set, or a wait asserted to return false. Any other timed wait is a
    handshake the peer ends early; its timeout only bounds a hang.

    :param ast.AST node: Syntax node to classify.
    :return: ``True`` for ``Event().wait(t)``, ``assert not x.wait(t)``, and
        ``x.wait(t) is False`` or ``== False``.
    """
    if isinstance(node, ast.Call):
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        return _timed_wait(node) and isinstance(receiver, ast.Call)
    if isinstance(node, ast.Assert):
        test = node.test
        return (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and _timed_wait(test.operand)
        )
    if isinstance(node, ast.Compare):
        return (
            _timed_wait(node.left)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Is | ast.Eq)
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value is False
        )
    return False


class _ModuleIndex:
    """Import aliases, function table, and fixture set for one parsed module."""

    def __init__(self, tree: ast.Module) -> None:
        """Index one parsed module so its test bodies can be resolved.

        :param ast.Module tree: Parsed module to index.
        """
        # Bound name -> the dotted target it names: ``sp`` -> ``subprocess``
        # for ``import subprocess as sp``, ``P`` -> ``subprocess.Popen`` for
        # ``from subprocess import Popen as P``.
        self.aliases: dict[str, str] = {}
        self.functions: dict[str, _FunctionDef] = {}
        self.fixtures: set[str] = set()
        self._index_imports(tree)
        self._index_functions(tree)

    def _index_imports(self, tree: ast.Module) -> None:
        """Record every absolute import in the module, at any nesting depth."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.aliases[alias.asname] = alias.name
                    else:
                        # ``import a.b`` binds ``a``; attribute chains walk the rest.
                        root = alias.name.partition(".")[0]
                        self.aliases[root] = root
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                # A relative import names a sibling test module, never a
                # process primitive, and helpers are only followed same-module.
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

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

    def resolve(self, node: ast.expr) -> str | None:
        """Return the fully qualified name a reference resolves to through imports.

        :param ast.expr node: ``Name`` or ``Attribute`` reference.
        :return: Dotted target, or ``None`` when its root name was never imported.
        """
        dotted = _dotted(node)
        if dotted is None:
            return None
        head, _, tail = dotted.partition(".")
        target = self.aliases.get(head)
        if target is None:
            return None
        return f"{target}.{tail}" if tail else target

    def parameter_helpers(self, node: _FunctionDef) -> list[str]:
        """Return parameter names of ``node`` that name a fixture defined in this module.

        :param node: Function whose signature is inspected.
        :return: Names of same-module fixtures the function requests.
        """
        return [name for name in _parameter_names(node) if name in self.fixtures]


def _local_reason(node: ast.AST, index: _ModuleIndex) -> str | None:
    """Return why one syntax node makes a test an integration test, or ``None``.

    Only the node itself is considered; callers walk the tree and recurse through
    helpers.

    :param ast.AST node: Syntax node to classify.
    :param _ModuleIndex index: Import and function table for the enclosing module.
    :return: Human-readable reason, or ``None`` when the node is tier-neutral.
    """
    if _wait_meant_to_run_out(node):
        return "waits out a .wait() timeout, a wall-clock wait"
    if isinstance(node, ast.Attribute | ast.Name):
        qualified = index.resolve(node)
        if qualified is not None and is_process_primitive(qualified):
            return f"references {qualified}"
        if qualified is not None and is_wallclock_primitive(qualified):
            return f"references {qualified}, a wall-clock wait"
    elif isinstance(node, ast.keyword):
        if node.arg in SESSION_KEYWORDS:
            return f"passes {node.arg}="
    elif isinstance(node, ast.Call):
        leaf = _call_leaf(node)
        registered = _registered_helper(leaf)
        if registered is not None:
            return f"calls {leaf}(), {registered}"
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

    # A registered shared fixture is reached by name, like a registered call.
    for name in _parameter_names(node):
        registered = _registered_helper(name)
        if registered is not None:
            return _describe(f"requests fixture {name}, {registered}", chain)

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


def scan_helpers(source: str) -> dict[str, str]:
    """Return ``{helper name: reason}`` for module-level helpers in ``source`` that spawn or wait.

    Test modules import these helpers from a shared module, where
    :func:`scan_module` cannot follow them, so each one found here has to be
    registered by name in :data:`PROCESS_DRIVER_CALLS` or
    :data:`WALLCLOCK_HELPER_CALLS`. Pure function, like :func:`scan_module`.

    :param str source: Python source text of a shared test helper module.
    :return: Mapping of helper name to the reason a test calling it is an integration test.
    :raises SyntaxError: If ``source`` is not parseable Python.
    """
    tree = ast.parse(source)
    index = _ModuleIndex(tree)
    flagged: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, _FunctionDef) and not node.name.startswith("test_"):
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


class CallReport(Protocol):
    """The fields of a ``pytest.TestReport`` the slow-test report reads."""

    nodeid: str
    when: str
    duration: float
    keywords: Mapping[str, object]


def excludes_integration(markexpr: str) -> bool:
    """Return whether a ``-m`` expression deselects a test marked only ``integration``.

    :param str markexpr: The run's effective marker expression, possibly empty.
    :return: ``True`` when the run leaves the integration tier out.
    """
    if not markexpr:
        return False
    return not Expression.compile(markexpr).evaluate(lambda name, **_: name == "integration")


def slow_unmarked(
    reports: Iterable[CallReport], threshold: float = SLOW_CALL_SECONDS
) -> list[tuple[str, float]]:
    """Return unmarked tests whose call phase took at least ``threshold`` seconds.

    :param Iterable[CallReport] reports: Test reports from one run, any phase.
    :param float threshold: Call-phase seconds at which a test is listed.
    :return: ``(node id, seconds)`` pairs, slowest first.
    """
    slow = [
        (report.nodeid, report.duration)
        for report in reports
        if report.when == "call"
        and report.duration >= threshold
        and TIER_MARKERS.isdisjoint(report.keywords)
    ]
    return sorted(slow, key=lambda pair: (-pair[1], pair[0]))
