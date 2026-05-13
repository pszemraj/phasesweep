"""Storage URL parsing and canonicalization. Backend detection, SQLAlchemy dialect folding, absolute vs. relative path preservation, and the same-host lock identity that equivalent URLs must share."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from phasesweep import load_experiment
from phasesweep.config import (
    Experiment,
    IntParam,
    JsonExtractor,
    Metric,
    Phase,
)
from phasesweep.orchestrator import (
    _resolve_storage,
)
from phasesweep.storage_urls import (
    _file_url_path,
    canonical_storage_identity,
    storage_backend,
)
from tests.conftest import make_experiment, write_yaml


@pytest.mark.parametrize(
    ("url", "expected_kind"),
    [
        # passthrough: SQLite URLs go to Optuna unchanged (no auto-remap)
        ("sqlite:///./runs/phases.db", "passthrough"),
        # passthrough: arbitrary RDB URLs go to Optuna unchanged
        ("postgresql://user:pass@host/db", "passthrough"),
        # in-memory: None means an Optuna InMemoryStorage (returned as None)
        (None, "none"),
    ],
)
def test_resolve_storage_passthrough_or_none(url: str | None, expected_kind: str) -> None:
    """Non-journal URLs are returned as-is (or None for in-memory)."""
    result = _resolve_storage(url)
    if expected_kind == "passthrough":
        assert result == url
    else:
        assert result is None


def test_resolve_storage_journal_url_creates_journal_storage(tmp_path):
    """Explicit journal:/// scheme constructs a JournalStorage (the only
    URL we actually translate, since Optuna has no journal:// understanding)."""
    from optuna.storages import JournalStorage

    result = _resolve_storage(f"journal:///{tmp_path}/phases.journal")
    assert isinstance(result, JournalStorage)


def test_validate_rejects_sqlite_with_parallel_n_jobs(tmp_path: Path) -> None:
    """Old auto-remap is gone; SQLite + n_jobs>1 must fail loudly at config-load."""
    p = write_yaml(
        tmp_path,
        f"""
        experiment: t
        storage: sqlite:///{tmp_path}/phases.db
        workdir: {tmp_path}/runs
        trial_command: "echo {{overrides}}"
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json, path: r.json, key: x }}
        phases:
          - name: p
            n_trials: 1
            n_jobs: 4
            allow_no_gpu_isolation: true
            search_space:
              x: {{ type: int, low: 0, high: 10 }}
        """,
    )
    with pytest.raises(ValidationError, match="SQLite serializes writers"):
        load_experiment(p)


def test_validate_accepts_sqlite_with_single_job(tmp_path: Path) -> None:
    """Sequential SQLite is fine; only n_jobs > 1 is rejected."""
    p = write_yaml(
        tmp_path,
        f"""
        experiment: t
        storage: sqlite:///{tmp_path}/phases.db
        workdir: {tmp_path}/runs
        trial_command: "echo {{overrides}}"
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json, path: r.json, key: x }}
        phases:
          - name: p
            n_trials: 1
            search_space:
              x: {{ type: int, low: 0, high: 10 }}
        """,
    )
    load_experiment(p)  # must not raise


def test_validate_accepts_journal_with_parallel(tmp_path: Path) -> None:
    """Explicit journal:/// is the right scheme for parallel sweeps."""
    p = write_yaml(
        tmp_path,
        f"""
        experiment: t
        storage: journal:///{tmp_path}/phases.journal
        workdir: {tmp_path}/runs
        trial_command: "echo {{overrides}}"
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json, path: r.json, key: x }}
        phases:
          - name: p
            n_trials: 1
            n_jobs: 4
            allow_no_gpu_isolation: true
            search_space:
              x: {{ type: int, low: 0, high: 10 }}
        """,
    )
    load_experiment(p)


def test_canonical_storage_identity_resolves_paths(tmp_path: Path) -> None:
    """File-based backends resolve to absolute paths so equivalent URL spellings
    (relative paths, ``..`` segments) produce one stable lock identity. None
    in returns None out (in-memory has no shared backend to collide on).

    SQLite-dialect collapse and RDB pass-through are pinned by sibling tests
    in this file (``test_storage_backend_collapses_sqlalchemy_dialects``,
    ``test_canonical_storage_identity_rdb_passes_through``); this test only
    pins the path-resolution and None-handling contract.
    """
    # SQLite path with `..` must be resolved away.
    sqlite_id = canonical_storage_identity(f"sqlite:///{tmp_path}/sub/../db.sqlite3")
    assert sqlite_id is not None
    assert ".." not in sqlite_id
    assert str(tmp_path / "db.sqlite3") in sqlite_id

    # Journal path resolves the same way.
    journal_id = canonical_storage_identity(f"journal:///{tmp_path}/study.journal")
    assert journal_id is not None
    assert str(tmp_path) in journal_id

    # None → None (in-memory study).
    assert canonical_storage_identity(None) is None


def test_sqlite_parallel_error_does_not_say_multi_host() -> None:
    """The validation error must not reintroduce the 'for multi-host' claim."""
    with pytest.raises(ValueError, match="single phasesweep orchestrator") as exc_info:
        Experiment(
            experiment="t",
            storage="sqlite:///test.db",
            trial_command="echo {overrides}",
            metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
            phases=[
                Phase(  # type: ignore[arg-type]
                    name="p",
                    n_trials=2,
                    n_jobs=2,
                    search_space={"x": IntParam(type="int", low=0, high=1)},
                )
            ],
        )
    assert "multi-host" not in str(exc_info.value).lower()


def test_storage_backend_collapses_sqlalchemy_dialects() -> None:
    """All SQLite dialects must reduce to the same logical backend name."""
    assert storage_backend("sqlite:///x.db") == "sqlite"
    assert storage_backend("sqlite+pysqlite:///x.db") == "sqlite"
    assert storage_backend("sqlite+pysqlcipher:///x.db") == "sqlite"
    assert storage_backend("postgresql+psycopg2://u@h/db") == "postgresql"
    assert storage_backend("journal:///x.journal") == "journal"
    assert storage_backend(None) is None


def test_sqlite_driver_url_rejected_with_parallel_jobs(tmp_path: Path) -> None:
    """Driver-qualified ``sqlite+pysqlite:///`` must trip the SQLite-parallel guard."""
    storage = f"sqlite+pysqlite:///{tmp_path / 'x.db'}"
    with pytest.raises(ValueError, match="SQLite"):
        make_experiment(workdir=tmp_path / "runs", storage=storage, n_jobs=2)


def test_canonical_storage_identity_rdb_passes_through() -> None:
    """RDB URLs are not rewritten — same-host advisory lock has no business
    second-guessing a remote DB URL."""
    url = "postgresql://u:p@host/db"
    assert canonical_storage_identity(url) == url


@pytest.mark.parametrize(
    ("url", "expected_path"),
    [
        ("sqlite:///relative.db", "relative.db"),
        ("sqlite:////tmp/absolute.db", "/tmp/absolute.db"),
        ("sqlite+pysqlite:///relative.db", "relative.db"),
        ("sqlite+pysqlite:////tmp/x.db", "/tmp/x.db"),
        ("sqlite://", ""),
        ("sqlite:///:memory:", ":memory:"),
        ("journal:///relative.journal", "relative.journal"),
        ("journal:////tmp/abs.journal", "/tmp/abs.journal"),
    ],
)
def test_file_url_path_preserves_absolute_paths(url: str, expected_path: str) -> None:
    """``_file_url_path`` must distinguish three-slash (relative) from
    four-slash (absolute) URLs. Pre-v0.5.10 it used ``lstrip("/")`` which
    destroyed the leading ``/`` on absolute paths."""
    assert _file_url_path(url) == expected_path


@pytest.mark.parametrize(
    ("scheme", "filename"),
    [("sqlite", "phases.db"), ("journal", "study.journal")],
)
def test_absolute_file_storage_identity_is_not_cwd_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheme: str,
    filename: str,
) -> None:
    """Absolute file-storage identities must not depend on caller cwd."""
    path = tmp_path / filename
    url = f"{scheme}:///{path}"

    cwd_a = tmp_path / "cwd_a"
    cwd_b = tmp_path / "cwd_b"
    cwd_a.mkdir()
    cwd_b.mkdir()

    monkeypatch.chdir(cwd_a)
    identity_a = canonical_storage_identity(url)

    monkeypatch.chdir(cwd_b)
    identity_b = canonical_storage_identity(url)

    expected = f"{scheme}:///" + str(path.resolve())
    assert identity_a == expected
    assert identity_b == expected


def test_plain_and_driver_sqlite_absolute_urls_collide(tmp_path: Path) -> None:
    """Dialect-folding must still work for absolute paths."""
    db = tmp_path / "phases.db"
    plain = f"sqlite:///{db}"
    driver = f"sqlite+pysqlite:///{db}"
    identity = canonical_storage_identity(plain)
    assert identity is not None
    assert identity.startswith("sqlite:///")
    assert identity == canonical_storage_identity(driver)
