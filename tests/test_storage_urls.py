"""Storage URL parsing and canonicalization. Backend detection, SQLAlchemy dialect folding, absolute vs. relative path preservation, and the same-host lock identity that equivalent URLs must share."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from phasesweep import load_config, load_experiment
from phasesweep.config import (
    ExecutionContext,
    Experiment,
    IntParam,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
    Suite,
)
from phasesweep.engine import read_status, run_experiment
from phasesweep.engine.locking import _run_lock_paths
from phasesweep.engine.optuna import _resolve_storage
from phasesweep.engine.relocation import _plan_artifact_root_rebinds
from phasesweep.runtime.files import (
    canonical_storage_identity,
    file_url_path,
    sqlite_database_path,
    sqlite_uri_filename_path,
    storage_backend,
    storage_recovery_locator,
)
from tests.conftest import make_experiment, write_yaml


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_auto_storage_resolves_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_jobs: int
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment = make_experiment(
        workdir="relative", storage="auto", n_jobs=n_jobs, allow_no_gpu_isolation=True
    )
    url = experiment.resolved_storage
    assert url is not None
    assert experiment.storage == experiment.model_dump()["storage"] == "auto"
    assert "resolved_storage" not in experiment.model_dump()
    assert storage_backend(url) == ("journal" if n_jobs > 1 else "sqlite")
    filename = file_url_path(url) if n_jobs > 1 else sqlite_uri_filename_path(url)
    assert filename == str(
        tmp_path / "relative" / "t" / ("study.journal" if n_jobs > 1 else "study.db")
    )
    assert not (tmp_path / "relative").exists()
    explicit = experiment.model_copy(update={"storage": url})
    assert explicit.resolved_storage == url
    assert _run_lock_paths(experiment) == _run_lock_paths(explicit)


@pytest.mark.parametrize("invalid", ["provenance", "seed", "acknowledgement"])
def test_auto_storage_requires_persistent_contract(invalid: str) -> None:
    payload = make_experiment(storage="auto").model_dump()
    if invalid == "provenance":
        payload["provenance"] = {}
    else:
        payload["phases"][0]["sampler"] = (
            Sampler(type="random") if invalid == "seed" else Sampler(type="tpe", seed=0)
        ).model_dump()
    with pytest.raises(ValidationError, match="provenance|seed|acknowledge_nonresumable"):
        Experiment.model_validate(payload)


def test_suite_auto_storage_uses_compiled_names_and_overrides(tmp_path: Path) -> None:
    exp = make_experiment(workdir=tmp_path / "runs", storage="auto")
    defaults = exp.model_dump(mode="json")
    defaults.pop("experiment")
    phases = defaults.pop("phases")
    parallel = [{**phases[0], "name": "parallel", "n_jobs": 2, "allow_no_gpu_isolation": True}]
    suite = Suite.model_validate(
        {
            "suite": "suite",
            "defaults": defaults,
            "studies": [
                {"name": "first", "phases": phases},
                {"name": "second", "phases": phases + parallel},
                {"name": "memory", "phases": phases, "storage": None},
            ],
        }
    )
    first, second, memory = [suite.experiment_for_study(study) for study in suite.studies]
    assert (
        sqlite_database_path(first.resolved_storage)
        == tmp_path / "runs" / "suite__first" / "study.db"
    )
    assert file_url_path(second.resolved_storage) == str(
        tmp_path / "runs" / "suite__second" / "study.journal"
    )
    assert memory.resolved_storage is None
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_auto_storage_preserves_paths_across_run_resume_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_jobs: int
) -> None:
    import optuna

    root = tmp_path / "runs ? # %20 🚀"
    exp = make_experiment(
        workdir=root,
        storage="auto",
        n_jobs=n_jobs,
        allow_no_gpu_isolation=True,
        trial_command="echo x=0.5 {overrides}",
        execution=ExecutionContext(cwd=str(tmp_path), inherit_env="none"),
    )
    run_experiment(exp)
    database = root / exp.experiment / ("study.journal" if n_jobs > 1 else "study.db")
    assert database.is_file()
    assert (root / exp.experiment / ".gitignore").read_text() == "*\n"
    assert not (tmp_path / "runs ").exists()
    locator = storage_recovery_locator(exp.resolved_storage)
    monkeypatch.chdir(tmp_path)
    assert canonical_storage_identity(locator) == canonical_storage_identity(exp.resolved_storage)
    stored = optuna.load_study(study_name="t::p", storage=_resolve_storage(locator))
    assert len(stored.trials) == 2
    topup = exp.model_copy(update={"phases": [exp.phases[0].model_copy(update={"n_trials": 3})]})
    run_experiment(topup)
    assert len(stored.trials) == 3
    assert read_status(topup)["publication_integrity"] == "ok"


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("relocated", [False, True])
def test_auto_backend_change_refuses_existing_tree(
    tmp_path: Path, n_jobs: int, relocated: bool
) -> None:
    from phasesweep.engine import ArtifactRootConflictError, ArtifactRootRebindError

    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage="auto",
        n_jobs=n_jobs,
        n_trials=1,
        allow_no_gpu_isolation=True,
        trial_command="echo x=0.5 {overrides}",
    )
    run_experiment(exp)
    if relocated:
        moved_workdir = tmp_path / "moved"
        Path(exp.workdir).rename(moved_workdir)
        exp = exp.model_copy(update={"workdir": str(moved_workdir)})
    changed = exp.model_copy(
        update={"phases": [exp.phases[0].model_copy(update={"n_jobs": 3 - n_jobs})]}
    )
    with pytest.raises(ArtifactRootConflictError, match="n_jobs") as run_error:
        run_experiment(changed)
    with pytest.raises(ArtifactRootRebindError, match="n_jobs") as rebind_error:
        _plan_artifact_root_rebinds([changed])
    for error in (run_error, rebind_error):
        message = str(error.value)
        assert "auto" in message
        assert "study.db" in message
        assert "study.journal" in message
        assert "Restore" in message
        assert "new experiment name" in message
        assert "does not convert" in message
        assert "the next ordinary run binds" not in message
    new_database = (
        Path(file_url_path(changed.resolved_storage))
        if n_jobs == 1
        else sqlite_database_path(changed.resolved_storage)
    )
    assert not new_database.exists()


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
    allow_external_rdb_single_host: bool | None = None,
) -> Path:
    parallel = (
        f"""
            n_jobs: {n_jobs}
            allow_no_gpu_isolation: true"""
        if n_jobs > 1
        else ""
    )
    external_rdb_single_host = (
        f"\n        allow_external_rdb_single_host: "
        f"{'true' if allow_external_rdb_single_host else 'false'}"
        if allow_external_rdb_single_host is not None
        else ""
    )
    return write_yaml(
        tmp_path,
        f"""
        experiment: t
        storage: {storage}{external_rdb_single_host}
        provenance: {{revision: test-fixture-v1}}
        workdir: {tmp_path}/runs
        trial_command: "echo {{overrides}}"
        override_format: argparse
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json_envelope, path: r.json, objective_name: x, split: test, policy: test }}
        phases:
          - name: p
            n_trials: 1{parallel}
            sampler: {{ type: random, seed: 0 }}
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
    ("storage", "allow_external_rdb_single_host", "raises"),
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
def test_validate_storage_external_rdb_single_host_policy(
    tmp_path: Path,
    storage: str,
    allow_external_rdb_single_host: bool | None,
    raises: bool,
) -> None:
    """A storage backend other than sqlite/journal requires an explicit
    allow_external_rdb_single_host: true acknowledgement (review v0.5.15 / item E,
    renamed from allow_unsafe_multihost per review v0.5.14 / item D); sqlite and
    journal storage are unaffected regardless of the flag's value."""
    p = _storage_policy_config(
        tmp_path,
        storage=storage.format(tmp=tmp_path),
        n_jobs=1,
        allow_external_rdb_single_host=allow_external_rdb_single_host,
    )
    if raises:
        with pytest.raises(ValidationError, match="allow_external_rdb_single_host"):
            load_experiment(p)
    else:
        load_experiment(p)


def test_external_rdb_storage_error_is_actionable() -> None:
    """The rejection must name the detected backend, state that PhaseSweep's
    coordination is single-host, and say how to acknowledge the risk."""
    with pytest.raises(ValueError) as exc_info:
        Experiment(
            experiment="t",
            storage="postgresql://user:pass@host/db",
            provenance={"revision": "test-fixture-v1"},
            trial_command="echo {overrides}",
            override_format="argparse",
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
    assert "allow_external_rdb_single_host: true" in message
    assert "single host" in message


@pytest.mark.parametrize(
    ("storage", "secret"),
    [
        ("postgresql://user@host/db?password=TOPSECRET", "TOPSECRET"),
        ("postgresql://user@host/db?access_token=TOKEN-SECRET", "TOKEN-SECRET"),
        (
            "mssql+pyodbc:///?odbc_connect=DRIVER%3DODBC%3BPWD%3DSUPERSECRET%3BUID%3Duser",
            "SUPERSECRET",
        ),
    ],
    ids=["password", "access-token", "nested-odbc-connect"],
)
def test_rdb_query_credentials_never_appear_in_config_validation_errors(
    storage: str,
    secret: str,
) -> None:
    """Validation names the backend and required action without echoing its URL."""
    phase = Phase(  # type: ignore[arg-type]
        name="p",
        n_trials=1,
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )
    common = {
        "experiment": "credential-redaction",
        "storage": storage,
        "provenance": {"revision": "test-fixture-v1"},
        "trial_command": "echo {overrides}",
        "metric": Metric(
            extractor=LogRegexExtractor(
                type="log_regex",
                pattern=r"x=(?P<value>[0-9.eE+-]+)",
            )
        ),
        "phases": [phase],
    }

    with pytest.raises(ValueError) as policy_info:
        Experiment(**common)
    with pytest.raises(ValueError) as sampler_info:
        Experiment(**common, allow_external_rdb_single_host=True)

    assert secret not in str(policy_info.value)
    assert secret not in str(sampler_info.value)


def test_suite_allow_external_rdb_single_host_flows_from_defaults(tmp_path: Path) -> None:
    """``allow_external_rdb_single_host`` flows from Suite defaults into each compiled
    study's Experiment exactly like ``storage`` and other defaulted fields
    (see ``Suite.experiment_for_study``); a study can still opt out and hit
    the same Experiment-level rejection as a standalone config."""
    config = load_config(
        write_yaml(
            tmp_path,
            """
            suite: external_rdb_suite
            defaults:
              storage: postgresql://user:pass@host/db
              allow_external_rdb_single_host: true
              trial_command: "echo"
              override_format: argparse
              provenance: {revision: default-v1}
              metric:
                name: x
                goal: minimize
                extractor: {type: log_regex, pattern: 'x=(?P<value>[0-9.]+)'}
            studies:
              - name: inherited
                phases: [{name: p, n_trials: 1, sampler: {type: random, seed: 0}}]
              - name: opted_out
                allow_external_rdb_single_host: false
                phases: [{name: p, n_trials: 1, sampler: {type: random, seed: 0}}]
            """,
        )
    )

    assert isinstance(config, Suite)
    inherited_study, opted_out_study = config.studies

    inherited = config.experiment_for_study(inherited_study)
    assert inherited.storage == "postgresql://user:pass@host/db"
    assert inherited.allow_external_rdb_single_host is True

    with pytest.raises(ValidationError, match="allow_external_rdb_single_host"):
        config.experiment_for_study(opted_out_study)


def test_canonical_storage_identity_resolves_paths(tmp_path: Path) -> None:
    """File-based backends resolve to absolute paths so equivalent URL spellings
    (relative paths, ``..`` segments) produce one stable lock identity. None
    in returns None out (in-memory has no shared backend to collide on).

    SQLite-dialect collapse and RDB canonicalization are pinned by sibling
    tests in this file (``test_storage_backend_collapses_sqlalchemy_dialects``,
    ``test_equivalent_rdb_urls_share_one_identity``); this test only pins the
    path-resolution and None-handling contract.
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


@pytest.mark.parametrize(
    ("storage", "expected_backend", "expected_name"),
    [
        ("sqlite+pysqlite:///studies.db?timeout=30", "sqlite", "studies.db"),
        ("sqlite:///study#experiment.db", "sqlite", "study#experiment.db"),
        (
            "sqlite:///file:uri.db?mode=rwc&cache=shared&uri=true",
            "sqlite",
            "uri.db",
        ),
        ("journal:///studies.journal", "journal", "studies.journal"),
    ],
)
def test_storage_recovery_locator_freezes_relative_file_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage: str,
    expected_backend: str,
    expected_name: str,
) -> None:
    """Recovery locators retain the registration cwd and URL semantics."""
    registration_cwd = tmp_path / "registration"
    recovery_cwd = tmp_path / "recovery"
    registration_cwd.mkdir()
    recovery_cwd.mkdir()
    monkeypatch.chdir(registration_cwd)

    locator = storage_recovery_locator(storage)

    monkeypatch.chdir(recovery_cwd)
    assert locator is not None
    assert storage_backend(locator) == expected_backend
    assert Path(file_url_path(locator)).is_absolute() or "file:/" in file_url_path(locator)
    assert expected_name in locator
    assert str(registration_cwd) in locator
    if expected_backend == "sqlite" and "?" in storage:
        assert "timeout=30" in locator or "cache=shared" in locator


@pytest.mark.parametrize("scheme", ["sqlite", "sqlite+pysqlite"])
def test_literal_hash_sqlite_path_survives_status_resume_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheme: str
) -> None:
    import optuna

    monkeypatch.chdir(tmp_path)
    storage = f"{scheme}:///study#experiment.db"
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        n_trials=1,
        trial_command="echo x=0.5 {overrides}",
    )
    run_experiment(exp)
    assert (tmp_path / "study#experiment.db").is_file()
    assert read_status(exp)["phases"][0]["trials"] == {"COMPLETE": 1}
    topup = exp.model_copy(update={"phases": [exp.phases[0].model_copy(update={"n_trials": 2})]})
    run_experiment(topup)
    assert read_status(topup)["phases"][0]["published_study_unavailable"] is False
    locator = storage_recovery_locator(storage)
    monkeypatch.chdir(tmp_path.parent)
    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(locator))
    assert len(study.trials) == 2


def test_sqlite_parallel_error_does_not_say_multi_host() -> None:
    """The validation error must not reintroduce the 'for multi-host' claim."""
    with pytest.raises(ValueError, match="single phasesweep orchestrator") as exc_info:
        Experiment(
            experiment="t",
            storage="sqlite:///test.db",
            provenance={"revision": "test-fixture-v1"},
            trial_command="echo {overrides}",
            override_format="argparse",
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


@pytest.mark.parametrize(
    ("label", "left", "right"),
    [
        (
            "password rotation",
            "postgresql://sweep:old-secret@db.internal:5432/studies",
            "postgresql://sweep:new-secret@db.internal:5432/studies",
        ),
        (
            "authority user rotation",
            "postgresql://old-user:secret@db.internal/studies",
            "postgresql://new-user:secret@db.internal/studies",
        ),
        (
            "query password rotation",
            "postgresql://sweep@db.internal/studies?password=old-secret",
            "postgresql://sweep@db.internal/studies?PASSWORD=new-secret",
        ),
        (
            "access token rotation",
            "postgresql://sweep@db.internal/studies?access_token=old-token",
            "postgresql://sweep@db.internal/studies?ACCESS-TOKEN=new-token",
        ),
        (
            "SSL password rotation",
            "postgresql://sweep@db.internal/studies?sslpassword=old-secret",
            "postgresql://sweep@db.internal/studies?SSL_PASSWORD=new-secret",
        ),
        (
            "client secret rotation",
            "postgresql://sweep@db.internal/studies?client_secret=old-secret",
            "postgresql://sweep@db.internal/studies?CLIENT-SECRET=new-secret",
        ),
        (
            "nested ODBC credential rotation",
            "mssql+pyodbc:///?odbc_connect="
            "DRIVER%3D%7BODBC%3BDriver%7D%3BSERVER%3Ddb.internal%3B"
            "DATABASE%3Dstudies%3BUID%3Dold-user%3BPWD%3D%7Bold%3Bsecret%7D",
            "mssql+pyodbc:///?odbc_connect="
            "DRIVER%3D%7BODBC%3BDriver%7D%3BSERVER%3Ddb.internal%3B"
            "DATABASE%3Dstudies%3Buid%3Dnew-user%3Bpwd%3D%7Bnew%3Bsecret%7D",
        ),
        (
            "nested ODBC client-secret rotation",
            "mssql+pyodbc:///?odbc_connect="
            "SERVER%3Ddb.internal%3BDATABASE%3Dstudies%3BClientSecret%3Dold-secret",
            "mssql+pyodbc:///?odbc_connect="
            "SERVER%3Ddb.internal%3BDATABASE%3Dstudies%3Bclient_secret%3Dnew-secret",
        ),
        (
            "nested ODBC field order",
            "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb.internal%3BDATABASE%3Dstudies",
            "mssql+pyodbc:///?odbc_connect=DATABASE%3Dstudies%3BSERVER%3Ddb.internal",
        ),
        (
            "nested ODBC field-name case",
            "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb.internal%3BDATABASE%3Dstudies",
            "mssql+pyodbc:///?odbc_connect=server%3Ddb.internal%3Bdatabase%3Dstudies",
        ),
        (
            "nested ODBC connection options",
            "mssql+pyodbc:///?odbc_connect="
            "DRIVER%3D%7BODBC+Driver+17+for+SQL+Server%7D%3B"
            "SERVER%3Ddb.internal%3BDATABASE%3Dstudies%3BEncrypt%3Dno%3B"
            "TrustServerCertificate%3Dyes%3BTrusted_Connection%3Dyes%3B"
            "Integrated+Security%3DSSPI%3BConnection+Timeout%3D5%3B"
            "Query+Timeout%3D10%3BCommand+Timeout%3D20%3BTLSVersion%3D1.2",
            "mssql+pyodbc:///?odbc_connect="
            "DRIVER%3D%7BODBC+Driver+18+for+SQL+Server%7D%3B"
            "SERVER%3Ddb.internal%3BDATABASE%3Dstudies%3BEncrypt%3Dyes%3B"
            "TrustServerCertificate%3Dno%3BTrusted_Connection%3Dno%3B"
            "Integrated+Security%3Dfalse%3BConnection+Timeout%3D30%3B"
            "Query+Timeout%3D40%3BCommand+Timeout%3D50%3BTLSVersion%3D1.3",
        ),
        (
            "nested ODBC braced target values",
            "mssql+pyodbc:///?odbc_connect="
            "SERVER%3D%7Bdb%3Bnode%7D%3BDATABASE%3D%7Bstudies%7D%7Dprod%7D",
            "mssql+pyodbc:///?odbc_connect="
            "DATABASE%3D%7Bstudies%7D%7Dprod%7D%3BSERVER%3D%7Bdb%3Bnode%7D",
        ),
        (
            "nested ODBC redundant braces",
            "mssql+pyodbc:///?odbc_connect=SERVER%3D%7Bdb.internal%7D%3BDATABASE%3Dstudies",
            "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb.internal%3BDATABASE%3Dstudies",
        ),
        (
            "query order",
            "postgresql://sweep@db.internal/studies?a=1&b=2",
            "postgresql://sweep@db.internal/studies?b=2&a=1",
        ),
        (
            "connection-only options ignored",
            "postgresql://sweep@db.internal/studies",
            "postgresql://sweep@db.internal/studies"
            "?application_name=phasesweep&connect_timeout=10&sslmode=require"
            "&keepalives_idle=30&target_session_attrs=read-write",
        ),
        (
            "default vs explicit postgres port",
            "postgresql://sweep@db.internal/studies",
            "postgresql://sweep@db.internal:5432/studies",
        ),
        (
            "default vs explicit mysql port",
            "mysql://sweep@db.internal/studies",
            "mysql://sweep@db.internal:3306/studies",
        ),
        (
            "host case",
            "postgresql://sweep@DB.Internal/studies",
            "postgresql://sweep@db.internal/studies",
        ),
        (
            "psycopg driver spelling",
            "postgresql://sweep@db.internal/studies",
            "postgresql+psycopg://sweep@db.internal/studies",
        ),
        (
            "psycopg2 driver spelling",
            "postgresql+psycopg://sweep@db.internal/studies",
            "postgresql+psycopg2://sweep@db.internal/studies",
        ),
    ],
)
def test_equivalent_rdb_urls_share_one_identity(label: str, left: str, right: str) -> None:
    """Equivalent spellings of one external database must share one lock identity.

    Before v0.5.18 the raw URL string was hashed into the same-host lock path,
    so a rotated password or a reordered query silently split the lock and let
    two orchestrators write the same study (review v0.5.17 / blocker 5).
    """
    identity = canonical_storage_identity(left)
    assert identity is not None
    assert identity == canonical_storage_identity(right), label


@pytest.mark.parametrize(
    ("label", "left", "right"),
    [
        ("database", "postgresql://u@h/db_a", "postgresql://u@h/db_b"),
        ("host", "postgresql://u@host_a/db", "postgresql://u@host_b/db"),
        ("non-default port", "postgresql://u@h:5432/db", "postgresql://u@h:6432/db"),
        ("dialect family", "postgresql://u@h/db", "mysql://u@h/db"),
        (
            "unix socket vs tcp",
            "postgresql://u@/db?host=/var/run/postgresql",
            "postgresql://u@/db",
        ),
        (
            "different unix socket dir",
            "postgresql://u@/db?host=/var/run/postgresql",
            "postgresql://u@/db?host=/tmp",
        ),
        (
            "different PostgreSQL search_path",
            "postgresql://u@h/db?options=-csearch_path%3Dresearch_a",
            "postgresql://u@h/db?options=-csearch_path%3Dresearch_b",
        ),
        (
            "different schema option",
            "postgresql://u@h/db?schema=research_a",
            "postgresql://u@h/db?schema=research_b",
        ),
        (
            "different retained target option",
            "postgresql://u@h/db?cluster=primary",
            "postgresql://u@h/db?cluster=archive",
        ),
        (
            "different nested ODBC server",
            "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb-a%3BDATABASE%3Dstudies%3BPWD%3Dx",
            "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb-b%3BDATABASE%3Dstudies%3BPWD%3Dy",
        ),
        (
            "different nested ODBC database",
            "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb%3BDATABASE%3Dstudies-a%3BPWD%3Dx",
            "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb%3BDATABASE%3Dstudies-b%3BPWD%3Dy",
        ),
        (
            "different nested ODBC port",
            "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb%3BPORT%3D1433%3BDATABASE%3Dstudies",
            "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb%3BPORT%3D1434%3BDATABASE%3Dstudies",
        ),
        (
            "different nested ODBC DSN",
            "mssql+pyodbc:///?odbc_connect=DSN%3Dstudies-primary",
            "mssql+pyodbc:///?odbc_connect=DSN%3Dstudies-archive",
        ),
        (
            "different nested ODBC instance",
            "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb%3BINSTANCE%3Dprimary%3BDATABASE%3Dx",
            "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb%3BINSTANCE%3Darchive%3BDATABASE%3Dx",
        ),
        (
            "different nested ODBC socket",
            "mssql+pyodbc:///?odbc_connect=SOCKET%3D%2Fvar%2Frun%2Fdb-a%3BDATABASE%3Dx",
            "mssql+pyodbc:///?odbc_connect=SOCKET%3D%2Fvar%2Frun%2Fdb-b%3BDATABASE%3Dx",
        ),
        (
            "different nested ODBC schema",
            "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb%3BDATABASE%3Dx%3BSCHEMA%3Da",
            "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb%3BDATABASE%3Dx%3BSCHEMA%3Db",
        ),
    ],
)
def test_distinct_rdb_urls_keep_distinct_identities(label: str, left: str, right: str) -> None:
    """Canonicalization must not over-collide onto genuinely different targets."""
    assert canonical_storage_identity(left) != canonical_storage_identity(right), label


def test_duplicate_nested_odbc_target_fields_are_rejected() -> None:
    """Ambiguous duplicate targets fail config validation without leaking values."""
    storage = (
        "mssql+pyodbc:///?odbc_connect="
        "SERVER%3Ddb-a%3BDATABASE%3Dstudies%3Bserver%3Ddb-b%3BPWD%3DSUPERSECRET"
    )

    with pytest.raises(
        ValueError,
        match=(
            "Ambiguous ODBC storage target: duplicate field 'server' in "
            "odbc_connect; specify each target selector once"
        ),
    ):
        canonical_storage_identity(storage)

    phase = Phase(  # type: ignore[arg-type]
        name="p",
        n_trials=1,
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )
    with pytest.raises(ValidationError) as exc_info:
        Experiment(
            experiment="ambiguous-odbc",
            storage=storage,
            allow_external_rdb_single_host=True,
            provenance={"revision": "test-fixture-v1"},
            trial_command="echo {overrides}",
            metric=Metric(
                extractor=LogRegexExtractor(
                    type="log_regex",
                    pattern=r"x=(?P<value>[0-9.eE+-]+)",
                )
            ),
            phases=[phase],
        )

    assert exc_info.value.errors(include_input=False)[0]["loc"] == ("storage",)
    message = str(exc_info.value)
    assert "duplicate field 'server'" in message
    for private_value in ("db-a", "db-b", "studies", "SUPERSECRET"):
        assert private_value not in message


def test_duplicate_nested_odbc_target_fields_normalize_punctuation() -> None:
    """Punctuation variants of one target key remain an ambiguous duplicate."""
    storage = (
        "mssql+pyodbc:///?odbc_connect=INITIAL+CATALOG%3Dstudies-a%3BInitialCatalog%3Dstudies-b"
    )

    with pytest.raises(ValueError, match="duplicate field 'InitialCatalog'"):
        canonical_storage_identity(storage)


def test_malformed_nested_odbc_braces_are_rejected() -> None:
    """Malformed braced target values must not receive a misleading identity."""
    storage = "mssql+pyodbc:///?odbc_connect=SERVER%3D%7Bdb.internal%3BDATABASE%3Dx"

    with pytest.raises(ValueError, match="unterminated braced value in odbc_connect"):
        canonical_storage_identity(storage)


def test_malformed_nested_odbc_field_is_rejected() -> None:
    """Every non-empty ODBC field must use the structured ``key=value`` form."""
    storage = "mssql+pyodbc:///?odbc_connect=SERVER%3Ddb%3BBROKEN%3BDATABASE%3Dx"

    with pytest.raises(ValueError, match="field without '=' in odbc_connect"):
        canonical_storage_identity(storage)


def test_repeated_top_level_target_query_values_preserve_order() -> None:
    """Repeated target values retain failover order instead of sorting it away."""
    left = "postgresql://u@/db?host=db-a&host=db-b"
    right = "postgresql://u@/db?host=db-b&host=db-a"

    assert canonical_storage_identity(left) != canonical_storage_identity(right)


def test_rdb_identity_excludes_credentials_and_keeps_socket_path() -> None:
    """Credentials never reach identity; ``host=`` (unix socket) does."""
    identity = canonical_storage_identity(
        "postgresql://sweep:hunter2@/studies?host=/var/run/postgresql&connect_timeout=10"
    )

    assert identity is not None
    assert "hunter2" not in identity
    assert "sweep" not in identity
    assert "connect_timeout" not in identity
    assert "studies" in identity
    # The socket directory is identity-bearing, percent-encoded in the identity.
    assert "%2Fvar%2Frun%2Fpostgresql" in identity


def test_rdb_identity_is_deterministic_and_prefixed() -> None:
    """The emitted form is stable across calls and self-describing."""
    url = "postgresql+psycopg2://sweep:pw@DB.Internal/studies?application_name=x&b=2&a=1"

    identity = canonical_storage_identity(url)

    assert identity == "rdb://postgresql://db.internal:5432/studies?a=1&b=2"
    assert identity == canonical_storage_identity(url)


def test_unparseable_storage_identity_falls_back_to_raw_string() -> None:
    """Lock-path derivation must never crash on a URL SQLAlchemy cannot parse."""
    for storage in ("not a url", "://", "postgres_but_not_a_url"):
        assert canonical_storage_identity(storage) == storage


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


def test_sqlite_database_path_rejects_remote_file_uri_authorities(tmp_path: Path) -> None:
    remote = "sqlite:///file://db-host/tmp/phases.db?uri=true"
    local = f"sqlite:///file://localhost{tmp_path}/phases.db?uri=true"

    assert sqlite_database_path(remote) is None
    assert sqlite_database_path(local) == tmp_path / "phases.db"


def test_sqlite_uri_memory_storage_identity_is_in_memory() -> None:
    storage = "sqlite:///file:memdb1?mode=memory&cache=shared&uri=true"

    assert canonical_storage_identity(storage) == "sqlite:///:memory:"
