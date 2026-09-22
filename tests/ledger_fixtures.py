"""Discover, materialize, and police the golden ledger fixtures.

The fixtures under ``tests/fixtures/ledgers`` are real on-disk PhaseSweep
output, so a test that reads one has to answer two questions before it can
assert anything: where does the copy live, and what did it look like before the
read? :func:`materialize` answers both, and :func:`forbid_file_backed_storage`
turns the "read paths never construct file-backed storage" invariant into an
error at the moment it is violated rather than a diff noticed afterwards.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from phasesweep import load_experiment
from phasesweep.config import Experiment
from phasesweep.engine.artifact_roots import (
    _artifact_root_binding_payload,
    _artifact_root_identity,
)
from phasesweep.engine.paths import _artifact_root_binding_path
from phasesweep.engine.state import ARTIFACT_ROOT_ATTR
from tests.fixtures.make_ledger_fixtures import (
    LEDGER_FILENAME,
    experiment_payload,
    storage_url,
)

LEDGER_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "ledgers"

#: Read modes a fixture can be exercised under. ``tree`` keeps the fixture's own
#: artifact root, so the artifact-root binding participates; ``ledger-only``
#: points the same ledger at a workdir that has never existed, which is what a
#: fresh checkout reading an inherited ledger actually looks like.
LEDGER_MODES = ("tree", "ledger-only")


def _tree_bytes(root: Path) -> dict[str, bytes]:
    """Return the exact regular-file contents below ``root``.

    :param Path root: Directory to snapshot.
    :return dict[str, bytes]: Relative path to file contents, for every
        regular file in the tree.
    """
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@dataclass(frozen=True)
class LedgerFixture:
    """One committed golden ledger fixture and its manifest."""

    name: str
    path: Path
    manifest: dict[str, Any]

    @property
    def modes(self) -> tuple[str, ...]:
        """Return the read modes this fixture declares.

        :return tuple[str, ...]: Declared modes, possibly empty.
        """
        return tuple(self.manifest["modes"])

    def expected(self, mode: str) -> str:
        """Return the verdict this fixture owes a read path in one mode.

        :param str mode: ``"tree"`` or ``"ledger-only"``.
        :return str: ``"ok"``, ``"schema-mismatch"``, or ``"root-conflict"``.
        """
        return str(self.manifest["expect"][mode])


@dataclass(frozen=True)
class Materialized:
    """A fixture copied into a temporary tree, ready to be read."""

    root: Path
    experiment: Experiment
    config_path: Path
    ledger_dir: Path
    before: dict[str, bytes]

    def unchanged(self) -> bool:
        """Return whether the materialized tree still holds its original bytes.

        :return bool: ``True`` when no file below ``root`` was added, removed,
            or rewritten since materialization.
        """
        return _tree_bytes(self.root) == self.before

    def changes(self) -> dict[str, str]:
        """Describe how the materialized tree differs from its original bytes.

        :return dict[str, str]: Relative path to ``"added"`` / ``"removed"`` /
            ``"rewritten"`` for every file that differs.
        """
        after = _tree_bytes(self.root)
        diff = {path: "added" for path in after.keys() - self.before.keys()}
        diff.update({path: "removed" for path in self.before.keys() - after.keys()})
        diff.update(
            {
                path: "rewritten"
                for path in self.before.keys() & after.keys()
                if self.before[path] != after[path]
            }
        )
        return diff


def discover_ledger_fixtures() -> list[LedgerFixture]:
    """Load every committed golden ledger fixture, in name order.

    :return list[LedgerFixture]: Fixtures with their parsed manifests.
    :raises FileNotFoundError: A fixture directory carries no manifest.
    """
    fixtures: list[LedgerFixture] = []
    for path in sorted(LEDGER_FIXTURE_ROOT.iterdir()):
        if not path.is_dir():
            continue
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{path} has no manifest.json; regenerate the fixtures")
        fixtures.append(
            LedgerFixture(
                name=path.name,
                path=path,
                manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
            )
        )
    return fixtures


def fixture_by_name(name: str) -> LedgerFixture:
    """Return one committed fixture by name.

    :param str name: Fixture directory name.
    :return LedgerFixture: The matching fixture.
    :raises KeyError: No committed fixture carries that name.
    """
    for fixture in discover_ledger_fixtures():
        if fixture.name == name:
            return fixture
    raise KeyError(f"no golden ledger fixture named {name!r}")


def copy_fixture(name: str, dest: Path) -> Path:
    """Copy one committed fixture tree to a writable location.

    The MCP run store is a private namespace whose permissions git cannot
    carry, so it is restored here rather than left at the checkout's umask.

    :param str name: Fixture directory name.
    :param Path dest: Destination directory, which must not already exist.
    :return Path: The copied fixture root.
    """
    shutil.copytree(fixture_by_name(name).path, dest)
    mcp_state = dest / "mcp_state"
    if mcp_state.is_dir():
        mcp_state.chmod(0o700)
        for child in mcp_state.rglob("*"):
            if child.is_dir():
                child.chmod(0o700)
    return dest


def _write_config(
    config_path: Path, *, backend: str, ledger_dir: Path, workdir: Path
) -> Experiment:
    """Write and load the fixture experiment config for a materialized copy.

    The config comes from the same builder the generator used, with only the
    two paths substituted. Neither path is part of the experiment's semantic
    fingerprint, so a published fixture still reports
    ``published_config_matches_current``.

    :param Path config_path: Where to write the YAML.
    :param str backend: Ledger backend recorded in the manifest.
    :param Path ledger_dir: Directory holding the copied ledger file.
    :param Path workdir: Artifact-root parent this read should use.
    :return Experiment: Parsed experiment config.
    """
    payload = experiment_payload(
        storage_url=storage_url(backend, ledger_dir),
        workdir=workdir,
    )
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_experiment(config_path)


def _rebind(experiment: Experiment, schema_version: int | None) -> None:
    """Rewrite a copied artifact-root binding for its new absolute location.

    The binding records the absolute artifact root and a digest of the storage
    identity, so a copied tree's binding names the generation-time paths and
    would fail ownership validation for reasons that have nothing to do with
    the format boundary under test. Rewriting it restores the fixture's
    *intended* difference -- its ``schema_version`` -- and nothing else.

    :param Experiment experiment: Experiment owning the copied artifact root.
    :param int | None schema_version: Schema version the fixture records.
    """
    path = _artifact_root_binding_path(experiment)
    if not path.is_file():
        return
    payload = _artifact_root_binding_payload(experiment)
    payload["schema_version"] = schema_version
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _rebind_studies(experiment: Experiment, ledger: Path, backend: str) -> None:
    """Point a copied ledger's study root bindings at the copied artifact root.

    The reverse half of :func:`_rebind`. Each study records the absolute root
    it publishes into, so a copied study still names the generation-time tree,
    and a reader that checks per-study ownership would refuse the copy for a
    reason unrelated to the format boundary under test. Only that recorded
    value is rewritten. SQLite gets a single ``UPDATE``, and each journal op
    is re-encoded exactly as Optuna writes it, so everything else stays
    byte-identical.

    :param Experiment experiment: Experiment owning the copied artifact root.
    :param Path ledger: Copied ``study.db`` or ``study.journal``.
    :param str backend: ``"sqlite"`` or ``"journal"``.
    """
    root = _artifact_root_identity(experiment)
    if backend == "sqlite":
        conn = sqlite3.connect(ledger)
        try:
            with conn:
                conn.execute(
                    "UPDATE study_user_attributes SET value_json = ? WHERE key = ?",
                    (json.dumps(root), ARTIFACT_ROOT_ATTR),
                )
        finally:
            conn.close()
        return
    lines = ledger.read_text(encoding="utf-8").splitlines(keepends=True)
    rebound: list[str] = []
    for line in lines:
        if ARTIFACT_ROOT_ATTR in line:
            op = json.loads(line)
            attrs = op.get("user_attr")
            if "trial_id" not in op and isinstance(attrs, dict) and ARTIFACT_ROOT_ATTR in attrs:
                attrs[ARTIFACT_ROOT_ATTR] = root
                ending = line[len(line.rstrip("\n")) :]
                line = json.dumps(op, separators=(",", ":")) + ending
        rebound.append(line)
    ledger.write_text("".join(rebound), encoding="utf-8")


def materialize(name: str, tmp_path: Path, *, mode: str) -> Materialized:
    """Copy one fixture into ``tmp_path`` and build the config that reads it.

    ``tree`` mode copies the whole fixture, so the read sees the ledger *and*
    the artifact root that claimed it. ``ledger-only`` copies just the ledger
    and points the experiment at a workdir that does not exist, which is the
    case a binding cannot answer and the ledger format scan must.

    The config file and any MCP state a caller needs live beside ``root``, not
    inside it, so ``root`` holds fixture bytes and nothing else.

    :param str name: Fixture directory name.
    :param Path tmp_path: Per-test temporary directory.
    :param str mode: ``"tree"`` or ``"ledger-only"``.
    :return Materialized: Copied tree, its experiment, and its byte snapshot.
    :raises ValueError: ``mode`` is not a supported read mode.
    """
    if mode not in LEDGER_MODES:
        raise ValueError(f"unsupported ledger fixture mode: {mode!r}")
    fixture = fixture_by_name(name)
    backend = str(fixture.manifest["backend"])
    root = tmp_path / "fixture"
    if mode == "tree":
        copy_fixture(name, root)
        workdir = root / "artifact_root"
    else:
        root.mkdir()
        shutil.copytree(fixture.path / "ledger", root / "ledger")
        # Never created: a read path that materializes it has written.
        workdir = root / "unvisited_workdir"
    experiment = _write_config(
        tmp_path / "experiment.yaml",
        backend=backend,
        ledger_dir=root / "ledger",
        workdir=workdir,
    )
    if mode == "tree":
        _rebind(experiment, fixture.manifest["binding_schema_version"])
        _rebind_studies(experiment, root / "ledger" / LEDGER_FILENAME[backend], backend)
    return Materialized(
        root=root,
        experiment=experiment,
        config_path=tmp_path / "experiment.yaml",
        ledger_dir=root / "ledger",
        before=_tree_bytes(root),
    )


def ledger_file(materialized: Materialized, backend: str) -> Path:
    """Return the ledger file inside a materialized fixture.

    :param Materialized materialized: Copied fixture.
    :param str backend: Ledger backend recorded in the manifest.
    :return Path: Absolute path to ``study.db`` or ``study.journal``.
    """
    return materialized.ledger_dir / LEDGER_FILENAME[backend]


def forbid_file_backed_storage(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make any file-backed storage construction fail, and record every attempt.

    Durability invariant 4: a read path never creates a study and never builds
    file-backed storage, and its SQLite reads open the database ``mode=ro``.
    Three constructors can break that -- ``RDBStorage`` (which runs
    ``metadata.create_all`` and a version manager on open), ``JournalFileBackend``
    (which creates the journal if absent), and ``optuna.create_study`` -- plus
    any read-write ``sqlite3.connect``.

    ``optuna.load_study`` is deliberately *not* patched: the journal read path
    legitimately loads a study over an in-memory ``_JournalSnapshot``, which
    touches no file. ``JournalStorage`` itself is likewise allowed, because
    that is the wrapper the snapshot is handed to.

    The attempts are both raised *and* recorded, because several read paths
    swallow broad exceptions to stay tolerant of a live writer; the returned
    list survives that, so an assertion on it cannot be silenced.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to install the patches.
    :return list[str]: Attempted constructions, appended to as they happen.
    """
    from optuna.storages._rdb.storage import RDBStorage
    from optuna.storages.journal._file import JournalFileBackend

    attempts: list[str] = []

    def refuse(label: str) -> Any:
        """Build a replacement constructor that records and refuses every call.

        :param str label: Name reported in the recorded attempt.
        :return Any: Callable to install in place of the real constructor.
        """

        def constructor(*args: Any, **kwargs: Any) -> Any:
            """Record this construction attempt and refuse it.

            :raises AssertionError: Always; a read path must never get here.
            """
            detail = f"{label}({args[1:]!r}, {kwargs!r})" if args else f"{label}({kwargs!r})"
            attempts.append(detail)
            raise AssertionError(f"read path constructed file-backed storage: {detail}")

        return constructor

    monkeypatch.setattr(RDBStorage, "__init__", refuse("RDBStorage"))
    monkeypatch.setattr(JournalFileBackend, "__init__", refuse("JournalFileBackend"))

    import optuna

    monkeypatch.setattr(optuna, "create_study", refuse("optuna.create_study"))

    real_connect = sqlite3.connect

    def guarded_connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        """Allow only read-only URI connections, recording and refusing the rest.

        :param Any database: Database string or URI the caller passed.
        :return sqlite3.Connection: The real read-only connection.
        :raises AssertionError: The connection was not an explicit ``mode=ro`` URI.
        """
        if not kwargs.get("uri") or "mode=ro" not in str(database):
            detail = f"sqlite3.connect({database!r}, uri={kwargs.get('uri')!r})"
            attempts.append(detail)
            raise AssertionError(f"read path opened SQLite read-write: {detail}")
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", guarded_connect)
    return attempts
