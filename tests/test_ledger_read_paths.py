"""Every read path, against every golden ledger, writes nothing and refuses honestly.

Two durability invariants meet here. A read path never creates a study, never
constructs file-backed storage, and opens SQLite ``mode=ro`` (invariant 6); and
it validates the artifact-root binding and then the ledger format *before* any
bind, claim, or open, leaving the bytes untouched when it refuses (invariant 2).
The golden fixtures under ``tests/fixtures/ledgers`` supply the pre-cutover
shapes those refusals exist for, produced by real PhaseSweep runs -- including
two by the preserved 0.3.1 release itself -- rather than by a test that guesses
what old bytes look like.

``mcp-recover-inspect`` is held to the weaker half on purpose: recovery
preflight legitimately needs live ``Study`` objects, so it is not under the
constructor ban. Its invariant is "validate before open, and refuse a
pre-cutover ledger with the bytes unchanged", which it now satisfies by going
through the ledger handle like every other read path. Because it reaps through
those live studies, it also refuses any study bound to another artifact root.
That makes its ``ledger-only`` verdict over a current ledger a refusal where
every pure read says ``ok``.
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
import yaml

from phasesweep.cli import main as cli_boundary
from phasesweep.engine import ArtifactRootConflictError, StudySchemaMismatchError
from phasesweep.engine.artifact_roots import ARTIFACT_ROOT_BINDING_SCHEMA_VERSION
from phasesweep.engine.fingerprints import _experiment_semantic_fingerprint, _phase_fingerprint
from phasesweep.engine.read import read_status, read_winners
from phasesweep.engine.state import STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION
from phasesweep.errors import OperatorAction
from phasesweep.mcp.recovery import (
    RunRecoveryError,
    _load_recovery_studies,
    _RecoveryNeeds,
    recover_run,
)
from phasesweep.mcp.runs import RunStore, UnsupportedStateFormatError
from phasesweep.mcp.snapshots import capture_result_snapshot
from tests.conftest import reaped_pid
from tests.ledger_fixtures import (
    LEDGER_MODES,
    Materialized,
    copy_fixture,
    discover_ledger_fixtures,
    fixture_by_name,
    forbid_file_backed_storage,
    ledger_file,
    materialize,
    tree_changes,
    tree_snapshot,
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
_STUDY_OWNED_ELSEWHERE_PHRASE = "publishes into artifact root"

#: The verdict recovery owes a ``ledger-only`` cell that every pure read path
#: accepts. Recovery alone checks which artifact root owns each study it opens,
#: because it reaps through them. A ``ledger-only`` read points the ledger at a
#: workdir that never existed, so each study the ledger's own run bound belongs
#: to another root, and recovering from there must refuse to touch it.
_RECOVERY_LEDGER_ONLY_VERDICTS = {"ok": "study-owned-elsewhere"}


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
    :return str: ``"schema-mismatch"``, ``"root-conflict"``,
        ``"study-owned-elsewhere"``, or ``"unclassified"``.
    """
    if _SCHEMA_MISMATCH_PHRASE in message:
        return "schema-mismatch"
    if any(phrase in message for phrase in _ROOT_CONFLICT_PHRASES):
        return "root-conflict"
    if _STUDY_OWNED_ELSEWHERE_PHRASE in message:
        return "study-owned-elsewhere"
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


def _recover_inspect(materialized: Materialized, tmp_path: Path) -> tuple[str, dict[str, str]]:
    """Run MCP recovery preflight over a materialized fixture and classify it.

    The run store lives beside the fixture copy, so building it cannot show up
    as a change to the fixture's bytes; it is snapshotted on its own once
    built, because preflight must leave it untouched too. The handle names a
    long-dead PID with a mismatched start time and is marked cleanup-uncertain,
    which is the state that makes preflight load the phase studies (mirrors
    ``tests/test_mcp_server.py::test_operator_recovery_clears_no_status_cleanup_uncertainty``).

    :param Materialized materialized: Copied fixture and its config.
    :param Path tmp_path: Per-test temporary directory.
    :return tuple[str, dict[str, str]]: ``"ok"`` when preflight reported,
        otherwise its verdict; and how preflight changed the MCP state
        directory, empty when it did not.
    """
    state_dir = tmp_path / "mcp-state"
    store = RunStore(state_dir)
    run_id = "golden-ledger-recover"
    config_bytes = materialized.config_path.read_bytes()
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=materialized.experiment.experiment,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        pid=reaped_pid(),
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config_bytes)
    store.mark_cleanup_uncertain(handle)
    state_before = tree_snapshot(state_dir)
    messages: list[str] = []
    try:
        recover_run(state_dir, run_id, confirm=False, emit=messages.append)
    except RunRecoveryError as exc:
        verdict = _classify_message(str(exc))
    except Exception as exc:
        verdict = _classify_exception(exc)
    else:
        assert messages, "recovery preflight reported nothing"
        verdict = "ok"
    return verdict, tree_changes(state_before, tree_snapshot(state_dir))


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

    :return list[pytest.param]: One parameter set per cell.
    """
    return [
        pytest.param(
            fixture.name,
            mode,
            read_path,
            id=f"{fixture.name}-{mode}-{read_path}",
        )
        for fixture in discover_ledger_fixtures()
        for mode in fixture.modes
        for read_path in READ_PATHS
    ]


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
    state_changes: dict[str, str] = {}

    if read_path == RECOVERY_READ_PATH:
        if mode == "ledger-only":
            expected = _RECOVERY_LEDGER_ONLY_VERDICTS.get(expected, expected)
        verdict, state_changes = _recover_inspect(materialized, tmp_path)
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
    assert state_changes == {}, f"{read_path} changed the MCP state directory: {state_changes}"


@pytest.mark.parametrize(("fixture_name", "mode"), _mode_cells())
def test_recovery_inspect_never_writes_to_a_golden_ledger(
    fixture_name: str, mode: str, tmp_path: Path
) -> None:
    """Recovery preflight leaves a golden ledger and its run store untouched, whatever it decides.

    The matrix above owns the verdict. This isolates the bytes half so it is
    asserted on its own, for every cell, by a test that can only fail for that
    one reason.
    """
    materialized = materialize(fixture_name, tmp_path, mode=mode)
    _, state_changes = _recover_inspect(materialized, tmp_path)
    assert materialized.unchanged(), (
        f"recovery preflight over {fixture_name} ({mode}) changed bytes: {materialized.changes()}"
    )
    assert state_changes == {}, (
        f"recovery preflight over {fixture_name} ({mode}) changed the MCP state "
        f"directory: {state_changes}"
    )


@pytest.mark.parametrize(
    "fixture_name", ["precutover-schema2-sqlite", "precutover-unstamped-journal"]
)
def test_recovery_study_load_rewraps_the_engine_refusal(fixture_name: str, tmp_path: Path) -> None:
    """Recovery refuses a pre-cutover ledger in the engine's own words and remedy.

    The wrapper's job is to say which command the operator is running, not to
    re-explain the failure. So the original refusal text has to survive intact,
    and the remediation has to survive with it: ``rewrap`` carries the cause's
    ``action`` across the layer boundary so a routing caller still learns that
    the fix is the preserved prior release, not "run recover-run again".
    """
    materialized = materialize(fixture_name, tmp_path, mode="tree")
    needs = _RecoveryNeeds(
        terminal_status=None,
        stored_snapshot=None,
        prepared_publication_generation=None,
        cleanup_needed=False,
        terminal_cleanup_uncertain=False,
        ownership_storage_unavailable=False,
        snapshot_recovery_required=False,
        snapshot_unavailable=False,
        snapshot_finalize_needed=False,
    )

    with pytest.raises(RunRecoveryError) as excinfo:
        _load_recovery_studies(materialized.experiment, needs)

    assert _SCHEMA_MISMATCH_PHRASE in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, StudySchemaMismatchError)
    assert str(excinfo.value) == str(excinfo.value.__cause__)
    assert excinfo.value.actions == (OperatorAction.USE_PRIOR_RELEASE,)
    assert materialized.unchanged(), materialized.changes()


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
    before = tree_snapshot(root)

    # The format refusal, not the missing-layout ValueError whose message also
    # names an "MCP state directory": that one would mean the copy is broken.
    with pytest.raises(UnsupportedStateFormatError) as excinfo:
        if entry_point == "open_existing":
            RunStore.open_existing(state_dir)
        else:
            RunStore(state_dir)

    assert type(excinfo.value) is UnsupportedStateFormatError
    assert tree_changes(before, tree_snapshot(root)) == {}


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


def _fixtures_from_this_generator() -> list[str]:
    """Name every published fixture the working tree's generator produces.

    The ``release-*`` fixtures are the old release's own output: they carry its
    fingerprints, and every read path refuses them before comparing one.

    :return list[str]: Fixture names with a ledger and a current-generator origin.
    """
    return [
        fixture.name
        for fixture in discover_ledger_fixtures()
        if fixture.manifest["backend"] is not None and not fixture.name.startswith("release-")
    ]


@pytest.mark.parametrize("fixture_name", _fixtures_from_this_generator())
def test_fixture_fingerprints_do_not_depend_on_the_working_directory(
    fixture_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fixture's stored fingerprints match its config from any checkout path.

    An unset ``execution.cwd`` fingerprints the invocation directory, so a
    fixture generated that way matches its own config only when read from the
    generating checkout. Every other clone -- CI's included -- then sees a
    changed phase config wherever a read or resume compares fingerprints.
    """
    materialized = materialize(fixture_name, tmp_path, mode="tree")
    monkeypatch.chdir(tmp_path)
    experiment = materialized.experiment
    (phase,) = experiment.phases
    generations = materialized.root / "artifact_root" / experiment.experiment / "generations"
    (generation,) = sorted(generations.iterdir())

    winner = yaml.safe_load((generation / "phases" / phase.name / "winner.yaml").read_text())
    summary = yaml.safe_load((generation / "summary.yaml").read_text())
    assert winner["phase_fingerprint"] == _phase_fingerprint(experiment, phase, {})
    assert summary["config_fingerprint"] == _experiment_semantic_fingerprint(experiment)


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
    """Each fixture records its provenance and read modes and appears in the inventory README.

    A ``release-<version>-*`` fixture claims to be the old release's own
    output, so it must name exactly that tag: ``unknown``, a ``-dirty``
    checkout, or commits past the tag all mean the claim is unproven. A
    fixture with a ledger must declare a read mode, or the matrix silently
    drops it; one without a ledger has nothing to read.
    """
    readme = (Path(__file__).resolve().parent / "fixtures" / "ledgers" / "README.md").read_text(
        encoding="utf-8"
    )
    for fixture in discover_ledger_fixtures():
        produced_by = fixture.manifest.get("produced_by")
        assert isinstance(produced_by, dict) and produced_by.get("phasesweep_git"), (
            f"{fixture.name} has no produced_by provenance"
        )
        if fixture.name.startswith("release-"):
            version = fixture.name.removeprefix("release-").rpartition("-")[0]
            assert produced_by["phasesweep_git"] == f"v{version}", (
                f"{fixture.name} was produced by {produced_by['phasesweep_git']!r}, "
                f"not a clean checkout of v{version}"
            )
        assert fixture.name in readme, f"{fixture.name} is missing from the fixture README"

        modes = fixture.modes
        assert len(set(modes)) == len(modes) and set(modes) <= set(LEDGER_MODES), (
            f"{fixture.name} declares modes {modes}; each must be one of {LEDGER_MODES}, once"
        )
        assert set(fixture.manifest["expect"]) == set(modes), (
            f"{fixture.name} expects verdicts for {sorted(fixture.manifest['expect'])}, "
            f"but declares modes {modes}"
        )
        if fixture.manifest["backend"] is None:
            assert modes == (), f"{fixture.name} has no ledger, yet declares modes {modes}"
        else:
            assert modes, f"{fixture.name} has a ledger but declares no read mode"
