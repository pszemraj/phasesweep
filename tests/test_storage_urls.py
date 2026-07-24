"""Storage URL parsing and canonicalization. Backend detection, SQLAlchemy dialect folding, absolute vs. relative path preservation, and the same-host lock identity that equivalent URLs must share."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from phasesweep import load_config, load_experiment
from phasesweep.config import (
    Experiment,
    IntParam,
    LogRegexExtractor,
    Metric,
    Phase,
    Suite,
)
from phasesweep.engine.optuna import _resolve_storage
from phasesweep.runtime.files import (
    canonical_storage_identity,
    file_url_path,
    storage_backend,
)
from tests.conftest import make_experiment, write_yaml


def test_resolve_storage_urls(tmp_path: Path) -> None:
    """Only journal:/// is translated; other URLs pass through to Optuna."""
    from optuna.storages import JournalStorage

    cases = [
        ("sqlite_passthrough", "sqlite:///./runs/phases.db", "passthrough"),
        ("rdb_passthrough", "postgresql://user:pass@host/db", "passthrough"),
        ("in_memory", None, "none"),
        ("journal", f"journal:///{tmp_path}/phases.journal", "journal"),
    ]

    for case, url, expected_kind in cases:
        result = _resolve_storage(url)
        if expected_kind == "passthrough":
            assert result == url, case
        elif expected_kind == "journal":
            assert isinstance(result, JournalStorage), case
        else:
            assert result is None, case


def _storage_policy_config(
    tmp_path: Path,
    *,
    storage: str,
    n_jobs: int,
    allow_unsafe_multihost: bool | None = None,
) -> Path:
    parallel = (
        f"""
            n_jobs: {n_jobs}
            allow_no_gpu_isolation: true"""
        if n_jobs > 1
        else ""
    )
    unsafe_multihost = (
        f"\n        allow_unsafe_multihost: {'true' if allow_unsafe_multihost else 'false'}"
        if allow_unsafe_multihost is not None
        else ""
    )
    return write_yaml(
        tmp_path,
        f"""
        experiment: t
        storage: {storage}{unsafe_multihost}
        provenance: {{revision: test-fixture-v1}}
        workdir: {tmp_path}/runs
        trial_command: "echo {{overrides}}"
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json_envelope, path: r.json, objective_name: x, split: test, policy: test }}
        phases:
          - name: p
            n_trials: 1{parallel}
            search_space:
              x: {{ type: int, low: 0, high: 10 }}
        """,
    )


@pytest.mark.parametrize(
    ("storage_template", "n_jobs", "raises"),
    [
        ("sqlite:///{tmp}/phases.db", 4, True),
        ("sqlite:///{tmp}/phases.db", 1, False),
        ("journal:///{tmp}/phases.journal", 4, False),
    ],
    ids=["sqlite_parallel_rejected", "sqlite_single_ok", "journal_parallel_ok"],
)
def test_validate_storage_parallel_policy(
    tmp_path: Path, storage_template: str, n_jobs: int, raises: bool
) -> None:
    """SQLite is sequential-only; explicit journal storage supports parallel sweeps."""
    p = _storage_policy_config(
        tmp_path,
        storage=storage_template.format(tmp=tmp_path),
        n_jobs=n_jobs,
    )
    if raises:
        with pytest.raises(ValidationError, match="SQLite serializes writers"):
            load_experiment(p)
    else:
        load_experiment(p)


@pytest.mark.parametrize(
    ("storage", "allow_unsafe_multihost", "raises"),
    [
        ("postgresql://user:pass@host/db", None, True),
        ("postgresql://user:pass@host/db", False, True),
        ("postgresql://user:pass@host/db", True, False),
        ("mysql+pymysql://user:pass@host/db", None, True),
        ("mysql+pymysql://user:pass@host/db", True, False),
        ("sqlite:///{tmp}/phases.db", None, False),
        ("sqlite:///{tmp}/phases.db", False, False),
        ("journal:///{tmp}/phases.journal", None, False),
    ],
    ids=[
        "postgres_default_rejected",
        "postgres_explicit_false_rejected",
        "postgres_acknowledged_ok",
        "mysql_default_rejected",
        "mysql_acknowledged_ok",
        "sqlite_unaffected_default",
        "sqlite_unaffected_explicit_false",
        "journal_unaffected",
    ],
)
def test_validate_storage_multihost_policy(
    tmp_path: Path,
    storage: str,
    allow_unsafe_multihost: bool | None,
    raises: bool,
) -> None:
    """A storage backend other than sqlite/journal requires an explicit
    allow_unsafe_multihost: true acknowledgement (review v0.5.14 / item D); sqlite and
    journal storage are unaffected regardless of the flag's value."""
    p = _storage_policy_config(
        tmp_path,
        storage=storage.format(tmp=tmp_path),
        n_jobs=1,
        allow_unsafe_multihost=allow_unsafe_multihost,
    )
    if raises:
        with pytest.raises(ValidationError, match="allow_unsafe_multihost"):
            load_experiment(p)
    else:
        load_experiment(p)


def test_multihost_storage_error_is_actionable() -> None:
    """The rejection must name the detected backend, state that PhaseSweep's
    coordination is single-host, and say how to acknowledge the risk."""
    with pytest.raises(ValueError) as exc_info:
        Experiment(
            experiment="t",
            storage="postgresql://user:pass@host/db",
            provenance={"revision": "test-fixture-v1"},
            trial_command="echo {overrides}",
            metric=Metric(
                extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
            ),
            phases=[
                Phase(  # type: ignore[arg-type]
                    name="p",
                    n_trials=2,
                    search_space={"x": IntParam(type="int", low=0, high=1)},
                )
            ],
        )
    message = str(exc_info.value)
    assert "postgresql" in message
    assert "host-local-filesystem" in message
    assert "allow_unsafe_multihost: true" in message
    assert "single host" in message


def test_multihost_storage_allowed_when_acknowledged() -> None:
    """Setting allow_unsafe_multihost: true permits shared RDB storage."""
    experiment = Experiment(
        experiment="t",
        storage="postgresql://user:pass@host/db",
        allow_unsafe_multihost=True,
        provenance={"revision": "test-fixture-v1"},
        trial_command="echo {overrides}",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
        phases=[
            Phase(  # type: ignore[arg-type]
                name="p",
                n_trials=2,
                search_space={"x": IntParam(type="int", low=0, high=1)},
            )
        ],
    )
    assert experiment.allow_unsafe_multihost is True
    assert experiment.storage == "postgresql://user:pass@host/db"


def test_suite_allow_unsafe_multihost_flows_from_defaults(tmp_path: Path) -> None:
    """``allow_unsafe_multihost`` flows from Suite defaults into each compiled
    study's Experiment exactly like ``storage`` and other defaulted fields
    (see ``Suite.experiment_for_study``); a study can still opt out and hit
    the same Experiment-level rejection as a standalone config."""
    config = load_config(
        write_yaml(
            tmp_path,
            """
            suite: multihost_suite
            defaults:
              storage: postgresql://user:pass@host/db
              allow_unsafe_multihost: true
              trial_command: "echo"
              provenance: {revision: default-v1}
              metric:
                name: x
                goal: minimize
                extractor: {type: log_regex, pattern: 'x=(?P<value>[0-9.]+)'}
            studies:
              - name: inherited
                phases: [{name: p, n_trials: 1}]
              - name: opted_out
                allow_unsafe_multihost: false
                phases: [{name: p, n_trials: 1}]
            """,
        )
    )

    assert isinstance(config, Suite)
    inherited_study, opted_out_study = config.studies

    inherited = config.experiment_for_study(inherited_study)
    assert inherited.storage == "postgresql://user:pass@host/db"
    assert inherited.allow_unsafe_multihost is True

    with pytest.raises(ValidationError, match="allow_unsafe_multihost"):
        config.experiment_for_study(opted_out_study)


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
            provenance={"revision": "test-fixture-v1"},
            trial_command="echo {overrides}",
            metric=Metric(
                extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
            ),
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


def test_file_url_path_preserves_absolute_paths() -> None:
    """``file_url_path`` must distinguish three-slash (relative) from
    four-slash (absolute) URLs. Pre-v0.5.10 it used ``lstrip("/")`` which
    destroyed the leading ``/`` on absolute paths."""
    cases = [
        ("sqlite:///relative.db", "relative.db"),
        ("sqlite:///relative.db?timeout=30", "relative.db"),
        ("sqlite:////tmp/absolute.db", "/tmp/absolute.db"),
        ("sqlite:////tmp/absolute.db?timeout=30#frag", "/tmp/absolute.db"),
        ("sqlite+pysqlite:///relative.db", "relative.db"),
        ("sqlite+pysqlite:////tmp/x.db", "/tmp/x.db"),
        ("sqlite://", ""),
        ("sqlite:///:memory:", ":memory:"),
        ("sqlite:///:memory:?cache=shared", ":memory:"),
        ("journal:///relative.journal", "relative.journal"),
        ("journal:////tmp/abs.journal", "/tmp/abs.journal"),
    ]

    for url, expected_path in cases:
        assert file_url_path(url) == expected_path, url


def test_absolute_file_storage_identity_is_not_cwd_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absolute file-storage identities must not depend on caller cwd."""
    cases = [("sqlite", "phases.db"), ("journal", "study.journal")]

    cwd_a = tmp_path / "cwd_a"
    cwd_b = tmp_path / "cwd_b"
    cwd_a.mkdir()
    cwd_b.mkdir()

    for scheme, filename in cases:
        path = tmp_path / filename
        url = f"{scheme}:///{path}"

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


def test_sqlite_uri_file_storage_identity_resolves_actual_path(tmp_path: Path) -> None:
    """SQLite URI filenames should lock the real DB, not a cwd-relative ``file:`` path."""
    db = tmp_path / "uri.db"
    uri_storage = f"sqlite:///file:{db}?mode=rwc&cache=shared&uri=true"
    plain_storage = f"sqlite:///{db}"

    assert canonical_storage_identity(uri_storage) == canonical_storage_identity(plain_storage)


def test_sqlite_uri_memory_storage_identity_is_in_memory() -> None:
    storage = "sqlite:///file:memdb1?mode=memory&cache=shared&uri=true"

    assert canonical_storage_identity(storage) == "sqlite:///:memory:"
