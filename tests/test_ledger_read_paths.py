"""Every read path, against every golden ledger, writes nothing and refuses honestly.

Two durability invariants meet here. A read path never creates a study, never
constructs file-backed storage, and opens SQLite ``mode=ro`` (invariant 4); and
it validates the artifact-root binding and then the ledger format *before* any
bind, claim, or open, leaving the bytes untouched when it refuses (invariant 2).
The golden fixtures under ``tests/fixtures/ledgers`` supply the pre-cutover
shapes those refusals exist for, produced by real PhaseSweep runs -- including
two by the preserved 0.3.1 release itself -- rather than by a test that guesses
what old bytes look like.

``mcp-recover-inspect`` is held to the weaker half on purpose: recovery
preflight legitimately needs live ``Study`` objects, so it is not under the
constructor ban. Its invariant is "validate before open, and refuse a
pre-cutover ledger with the bytes unchanged", and today it bypasses the format
gate entirely. Those cells are strict xfails that flip with the ledger
chokepoint (M2).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from phasesweep.cli import main as cli_boundary
from phasesweep.engine import ArtifactRootConflictError, StudySchemaMismatchError
from phasesweep.engine.artifact_roots import ARTIFACT_ROOT_BINDING_SCHEMA_VERSION
from phasesweep.engine.read import read_status, read_winners
from phasesweep.engine.state import STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION
from phasesweep.mcp.recovery import RunRecoveryError, recover_run
from phasesweep.mcp.runs import RunStore
from phasesweep.mcp.snapshots import capture_result_snapshot
from tests.ledger_fixtures import (
    Materialized,
    _tree_bytes,
    copy_fixture,
    discover_ledger_fixtures,
    fixture_by_name,
    forbid_file_backed_storage,
    ledger_file,
    materialize,
)
from tests.mcp_helpers import make_run_handle

#: Every fixture this matrix requires. A missing entry means an incomplete
#: regeneration, not a smaller matrix, so it is asserted by name.
REQUIRED_FIXTURES = (
    "current-journal",
    "current-sqlite",
    "current-sqlite-optuna40-versioninfo",
    "precutover-binding2-sqlite",
    "precutover-mcp-state",
    "precutover-schema2-journal",
    "precutover-schema2-sqlite",
    "precutover-unmarked-tree-sqlite",
    "precutover-unstamped-journal",
    "precutover-unstamped-sqlite",
    "release-0.3.1-journal",
    "release-0.3.1-sqlite",
)

#: Read paths that must not construct file-backed storage at all.
PURE_READ_PATHS = (
    "engine-read_status",
    "engine-read_winners",
    "cli-status",
    "cli-show-winners",
    "mcp-snapshot",
)
#: Recovery preflight, which needs live studies and is only held to
#: bytes-unchanged plus its verdict.
RECOVERY_READ_PATH = "mcp-recover-inspect"
READ_PATHS = (*PURE_READ_PATHS, RECOVERY_READ_PATH)

#: Operator-facing phrases that identify a refusal on a text-only surface.
_SCHEMA_MISMATCH_PHRASE = "local storage ledger contains pre-cutover"
_ROOT_CONFLICT_PHRASES = ("uses unsupported pre-cutover", "contains pre-cutover PhaseSweep state")


def _classify_exception(exc: BaseException) -> str:
    """Map an engine refusal to its fixture verdict.

    :param BaseException exc: Exception a read path raised.
    :return str: ``"schema-mismatch"`` or ``"root-conflict"``.
    :raises BaseException: ``exc`` itself, when it is not a format refusal.
    """
    if isinstance(exc, StudySchemaMismatchError):
        return "schema-mismatch"
    if isinstance(exc, ArtifactRootConflictError):
        return "root-conflict"
    raise exc


def _classify_message(message: str) -> str:
    """Map an operator-facing diagnostic to its fixture verdict.

    :param str message: Rendered diagnostic from the CLI boundary or recovery.
    :return str: ``"schema-mismatch"``, ``"root-conflict"``, or ``"unclassified"``.
    """
    if _SCHEMA_MISMATCH_PHRASE in message:
        return "schema-mismatch"
    if any(phrase in message for phrase in _ROOT_CONFLICT_PHRASES):
        return "root-conflict"
    return f"unclassified: {message}"


def _engine_verdict(call: Callable[[], object]) -> str:
    """Run one in-process read path and report the verdict it reached.

    :param Callable[[], object] call: Read path invocation.
    :return str: ``"ok"`` when it returned, otherwise its refusal verdict.
    """
    try:
        call()
    except Exception as exc:
        return _classify_exception(exc)
    return "ok"


def _cli_verdict(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> str:
    """Run one CLI command through the real process boundary and classify it.

    ``CliRunner`` invokes the Click group directly and therefore never reaches
    the boundary that turns a refusal into an operator diagnostic and an exit
    status, which is exactly the verdict under test here (mirrors
    ``tests/test_cli.py::_invoke_cli_boundary``).

    :param list[str] argv: Arguments following the program name.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to set ``sys.argv``.
    :param pytest.CaptureFixture[str] capsys: Fixture capturing the diagnostic.
    :return str: ``"ok"`` on exit 0, otherwise the diagnostic's verdict.
    """
    monkeypatch.setattr(sys, "argv", ["phasesweep", *argv])
    monkeypatch.setattr(logging.getLogger(), "level", logging.INFO)
    with pytest.raises(SystemExit) as excinfo:
        cli_boundary()
    code = excinfo.value.code
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err, captured.err
    if code in (None, 0):
        return "ok"
    return _classify_message(captured.err)


def _recover_inspect_verdict(materialized: Materialized, tmp_path: Path) -> str:
    """Run MCP recovery preflight over a materialized fixture and classify it.

    The run store lives beside the fixture copy, so building it cannot show up
    as a change to the fixture's bytes. The handle names a long-dead PID with a
    mismatched start time and is marked cleanup-uncertain, which is the state
    that makes preflight load the phase studies (mirrors
    ``tests/test_mcp_server.py::test_operator_recovery_clears_no_status_cleanup_uncertainty``).

    :param Materialized materialized: Copied fixture and its config.
    :param Path tmp_path: Per-test temporary directory.
    :return str: ``"ok"`` when preflight reported, otherwise its verdict.
    """
    state_dir = tmp_path / "mcp-state"
    store = RunStore(state_dir)
    run_id = "golden-ledger-recover"
    config_bytes = materialized.config_path.read_bytes()
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=materialized.experiment.experiment,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config_bytes)
    store.mark_cleanup_uncertain(handle)
    messages: list[str] = []
    try:
        recover_run(state_dir, run_id, confirm=False, emit=messages.append)
    except RunRecoveryError as exc:
        return _classify_message(str(exc))
    except Exception as exc:
        return _classify_exception(exc)
    assert messages, "recovery preflight reported nothing"
    return "ok"


def _mode_cells() -> list[Any]:
    """Build the fixture x mode half of the matrix.

    :return list[Any]: One parameter set per readable fixture/mode pair.
    """
    return [
        pytest.param(fixture.name, mode, id=f"{fixture.name}-{mode}")
        for fixture in discover_ledger_fixtures()
        for mode in fixture.modes
    ]


def _matrix() -> list[Any]:
    """Build the fixture x mode x read-path matrix.

    :return list[pytest.param]: One parameter set per cell, with the recovery
        cells over pre-cutover ledgers marked as strict xfails.
    """
    cells: list[Any] = []
    for fixture in discover_ledger_fixtures():
        for mode in fixture.modes:
            for read_path in READ_PATHS:
                marks = []
                if read_path == RECOVERY_READ_PATH and fixture.expected(mode) != "ok":
                    marks.append(
                        pytest.mark.xfail(
                            strict=True,
                            reason=(
                                "recovery inspect bypasses the format gate; "
                                "flips with the ledger chokepoint (M2)"
                            ),
                        )
                    )
                cells.append(
                    pytest.param(
                        fixture.name,
                        mode,
                        read_path,
                        marks=marks,
                        id=f"{fixture.name}-{mode}-{read_path}",
                    )
                )
    return cells


@pytest.mark.parametrize(("fixture_name", "mode", "read_path"), _matrix())
def test_read_path_never_constructs_file_backed_storage_or_writes_bytes(
    fixture_name: str,
    mode: str,
    read_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A read over a golden ledger reaches the right verdict without writing."""
    fixture = fixture_by_name(fixture_name)
    expected = fixture.expected(mode)
    materialized = materialize(fixture_name, tmp_path, mode=mode)

    if read_path == RECOVERY_READ_PATH:
        verdict = _recover_inspect_verdict(materialized, tmp_path)
    else:
        attempts = forbid_file_backed_storage(monkeypatch)
        if read_path == "engine-read_status":
            verdict = _engine_verdict(lambda: read_status(materialized.experiment))
        elif read_path == "engine-read_winners":
            verdict = _engine_verdict(lambda: read_winners(materialized.experiment))
        elif read_path == "mcp-snapshot":
            verdict = _engine_verdict(lambda: capture_result_snapshot(materialized.experiment))
        elif read_path == "cli-status":
            verdict = _cli_verdict(["status", str(materialized.config_path)], monkeypatch, capsys)
        elif read_path == "cli-show-winners":
            verdict = _cli_verdict(
                ["show-winners", str(materialized.config_path)], monkeypatch, capsys
            )
        else:  # pragma: no cover - guarded by READ_PATHS
            raise AssertionError(f"unknown read path: {read_path}")
        assert attempts == [], f"{read_path} constructed file-backed storage: {attempts}"

    assert verdict == expected, f"{read_path} over {fixture_name} ({mode}) returned {verdict!r}"
    assert materialized.unchanged(), (
        f"{read_path} over {fixture_name} ({mode}) changed bytes: {materialized.changes()}"
    )


@pytest.mark.parametrize(("fixture_name", "mode"), _mode_cells())
def test_recovery_inspect_never_writes_to_a_golden_ledger(
    fixture_name: str, mode: str, tmp_path: Path
) -> None:
    """Recovery preflight leaves a golden ledger byte-identical, whatever it decides.

    The matrix above owns the verdict, and its pre-cutover recovery cells are
    strict xfails. A strict xfail only knows that the test failed, not which
    assertion failed, so the bytes half of the invariant would silently stop
    being enforced for exactly the fixtures it matters most for. This asserts
    it on its own, for every cell, and never xfails.
    """
    materialized = materialize(fixture_name, tmp_path, mode=mode)
    _recover_inspect_verdict(materialized, tmp_path)
    assert materialized.unchanged(), (
        f"recovery preflight over {fixture_name} ({mode}) changed bytes: {materialized.changes()}"
    )


@pytest.mark.parametrize("entry_point", ["open_existing", "constructor"])
def test_precutover_mcp_state_is_refused_without_writing(entry_point: str, tmp_path: Path) -> None:
    """An unmarked durable MCP run store is refused from either entry point.

    ``open_existing`` is the operator-recovery door and must stay purely
    observational; the constructor is the server's door and must not adopt
    0.3.1-era state by writing a current-format marker over it. Neither may
    change a byte on the way to refusing.
    """
    fixture = fixture_by_name("precutover-mcp-state")
    assert fixture.modes == (), "the MCP state fixture carries no ledger read modes"
    root = copy_fixture("precutover-mcp-state", tmp_path / "fixture")
    state_dir = root / "mcp_state"
    before = _tree_bytes(root)

    with pytest.raises(ValueError, match="MCP state directory"):
        if entry_point == "open_existing":
            RunStore.open_existing(state_dir)
        else:
            RunStore(state_dir)

    assert _tree_bytes(root) == before


@pytest.mark.parametrize("fixture_name", ["current-sqlite", "current-journal"])
def test_current_fixture_carries_this_release_format(fixture_name: str, tmp_path: Path) -> None:
    """The reference fixtures still declare the schema versions this release writes."""
    fixture = fixture_by_name(fixture_name)
    materialized = materialize(fixture_name, tmp_path, mode="tree")
    backend = str(fixture.manifest["backend"])
    ledger = ledger_file(materialized, backend)

    if backend == "sqlite":
        conn = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT value_json FROM study_user_attributes WHERE key = ?",
                (STUDY_SCHEMA_ATTR,),
            ).fetchall()
        finally:
            conn.close()
        stamps = [json.loads(value_json) for (value_json,) in rows]
    else:
        stamps = [
            record["user_attr"][STUDY_SCHEMA_ATTR]
            for record in (
                json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line
            )
            if record.get("op_code") == 2 and STUDY_SCHEMA_ATTR in record.get("user_attr", {})
        ]

    assert stamps == [STUDY_SCHEMA_VERSION], (
        f"{fixture_name} stamps {stamps}, not the current study schema "
        f"{STUDY_SCHEMA_VERSION}; regenerate tests/fixtures/ledgers"
    )
    assert fixture.manifest["binding_schema_version"] == ARTIFACT_ROOT_BINDING_SCHEMA_VERSION, (
        f"{fixture_name} records binding schema "
        f"{fixture.manifest['binding_schema_version']}, not the current "
        f"{ARTIFACT_ROOT_BINDING_SCHEMA_VERSION}; regenerate tests/fixtures/ledgers"
    )


@pytest.mark.parametrize("fixture_name", ["current-sqlite", "current-sqlite-optuna40-versioninfo"])
def test_sqlite_reader_never_touches_optuna_version_tables(
    fixture_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PhaseSweep's SQLite reads ignore Optuna's own version bookkeeping.

    Optuna's SQLite schema has been frozen across the whole supported range
    (``SCHEMA_VERSION = 12``, alembic head ``v3.2.0.a`` from 4.0 through 4.9):
    the only version-bearing bytes are ``version_info.library_version`` and
    ``alembic_version.version_num``. Reading either would make PhaseSweep's
    verdict depend on which Optuna wrote the ledger rather than on what the
    ledger contains, so the ``optuna40`` fixture -- whose recorded library
    version says 4.0.0 while its schema does not differ by one byte -- must
    read exactly like the reference fixture.
    """
    materialized = materialize(fixture_name, tmp_path, mode="tree")
    real_connect = sqlite3.connect
    statements: list[str] = []

    class RecordingConnection(sqlite3.Connection):
        """Connection that records every statement the reader issues."""

        def execute(self, sql: str, *params: Any) -> sqlite3.Cursor:
            """Record one statement, then run it.

            :param str sql: Statement the reader issued.
            :return sqlite3.Cursor: Cursor over the executed statement.
            """
            statements.append(sql)
            return super().execute(sql, *params)

        def executemany(self, sql: str, *params: Any) -> sqlite3.Cursor:
            """Record one repeated statement, then run it.

            :param str sql: Statement the reader issued.
            :return sqlite3.Cursor: Cursor over the executed statement.
            """
            statements.append(sql)
            return super().executemany(sql, *params)

    def recording_connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        """Open the database through the recording connection class.

        :param Any database: Database string or URI the reader passed.
        :return sqlite3.Connection: Recording connection to that database.
        """
        kwargs["factory"] = RecordingConnection
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", recording_connect)
    status = read_status(materialized.experiment)

    assert statements, "the status read issued no SQL"
    offending = [sql for sql in statements if "version_info" in sql or "alembic_version" in sql]
    assert not offending, f"{fixture_name} read Optuna version bookkeeping: {offending}"
    assert status["phases"][0]["trial_data_available"] is True


def test_required_ledger_fixtures_are_present() -> None:
    """Every fixture the matrix depends on is committed."""
    present = tuple(fixture.name for fixture in discover_ledger_fixtures())
    assert present == REQUIRED_FIXTURES


def test_every_ledger_fixture_is_documented() -> None:
    """Each fixture records its provenance and appears in the inventory README."""
    readme = (Path(__file__).resolve().parent / "fixtures" / "ledgers" / "README.md").read_text(
        encoding="utf-8"
    )
    for fixture in discover_ledger_fixtures():
        produced_by = fixture.manifest.get("produced_by")
        assert isinstance(produced_by, dict) and produced_by.get("phasesweep_git"), (
            f"{fixture.name} has no produced_by provenance"
        )
        assert fixture.name in readme, f"{fixture.name} is missing from the fixture README"
