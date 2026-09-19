"""Storage URL parsing and canonicalization. Backend detection, SQLAlchemy dialect folding, absolute vs. relative path preservation, and the same-host lock identity that equivalent URLs must share."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from phasesweep import load_experiment
from phasesweep.config import (
    ExecutionContext,
    Experiment,
    IntParam,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
)
from phasesweep.engine import read_status, run_experiment
from phasesweep.engine.locking import _run_lock_paths
from phasesweep.engine.optuna import _resolve_storage
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
    from phasesweep.engine import ArtifactRootConflictError

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
    message = str(run_error.value)
    assert "auto" in message
    assert "study.db" in message
    assert "study.journal" in message
    assert "Restore" in message
    assert "new experiment name" in message
    new_database = (
        Path(file_url_path(changed.resolved_storage))
        if n_jobs == 1
        else sqlite_database_path(changed.resolved_storage)
    )
    assert not new_database.exists()


def test_resolve_storage_urls(tmp_path: Path) -> None:
    """Translate in-memory sentinels and journals while preserving RDB URLs."""
    import optuna
    from optuna.storages import JournalStorage

    cases = [
        ("sqlite_passthrough", "sqlite:///./runs/phases.db", "passthrough"),
        ("in_memory", None, "none"),
        ("in_memory_sentinel", ":memory:", "none"),
        ("in_memory_sqlite", "sqlite:///:memory:", "none"),
        ("in_memory_sqlite_empty", "sqlite://", "none"),
        (
            "in_memory_sqlite_uri",
            "sqlite:///file:phasesweep-memory?mode=memory&cache=shared&uri=true",
            "none",
        ),
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
            assert optuna.create_study(storage=result).trials == [], case


def test_bare_in_memory_storage_runs_without_persistent_preflight(tmp_path: Path) -> None:
    """The documented bare sentinel must reach a fresh in-memory study."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=":memory:",
        trial_command="echo x=1 {overrides}",
        n_trials=1,
        gpu_policy="none",
    )

    outcome = run_experiment(experiment)

    assert set(outcome) == {"p"}
    assert outcome["p"].trial_number == 0


def _storage_policy_config(
    tmp_path: Path,
    *,
    storage: str,
    n_jobs: int,
) -> Path:
    parallel = (
        f"""
            n_jobs: {n_jobs}
            allow_no_gpu_isolation: true"""
        if n_jobs > 1
        else ""
    )
    return write_yaml(
        tmp_path,
        f"""
        experiment: t
        storage: {storage}
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


@pytest.mark.parametrize("storage", ["postgresql://user:pass@host/db", "mysql://user@host/db"])
def test_external_storage_is_rejected_at_config_load(tmp_path: Path, storage: str) -> None:
    """Only local SQLite, Journal, in-memory, and auto storage are supported."""
    path = _storage_policy_config(tmp_path, storage=storage, n_jobs=1)

    with pytest.raises(
        ValidationError, match="storage must be in-memory, sqlite:///, journal:///, or auto"
    ):
        load_experiment(path)


def test_canonical_storage_identity_resolves_paths(tmp_path: Path) -> None:
    """File-based backends resolve to absolute paths so equivalent URL spellings
    (relative paths, ``..`` segments) produce one stable lock identity. None
    in returns None out (in-memory has no shared backend to collide on).

    This test pins only the path-resolution and None-handling contract.
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


def test_literal_tilde_sqlite_path_survives_status_resume_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit SQLite filenames keep SQLAlchemy's literal tilde semantics."""
    import optuna

    invocation = tmp_path / "invocation"
    home = tmp_path / "home"
    invocation.mkdir()
    home.mkdir()
    (invocation / "~").mkdir()
    monkeypatch.chdir(invocation)
    monkeypatch.setenv("HOME", str(home))
    storage = "sqlite:///~/study.db"
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        n_trials=1,
        trial_command="echo x=0.5 {overrides}",
    )

    run_experiment(exp)

    database = invocation / "~" / "study.db"
    assert database.is_file()
    assert not (home / "study.db").exists()
    assert sqlite_database_path(storage) == Path("~/study.db")
    assert read_status(exp)["phases"][0]["trials"] == {"COMPLETE": 1}

    topup = exp.model_copy(update={"phases": [exp.phases[0].model_copy(update={"n_trials": 2})]})
    run_experiment(topup)
    locator = storage_recovery_locator(storage)
    assert locator is not None
    assert canonical_storage_identity(locator) == canonical_storage_identity(storage)

    monkeypatch.chdir(tmp_path)
    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(locator))
    assert len(study.trials) == 2


def test_sqlite_parallel_error_does_not_say_multi_host() -> None:
    """The validation error must not reintroduce the 'for multi-host' claim."""
    with pytest.raises(ValueError, match="single-host parallel sweep") as exc_info:
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


def test_storage_backend_collapses_local_dialects() -> None:
    """All SQLite dialects must reduce to the same logical backend name."""
    assert storage_backend("sqlite:///x.db") == "sqlite"
    assert storage_backend("sqlite+pysqlite:///x.db") == "sqlite"
    assert storage_backend("sqlite+pysqlcipher:///x.db") == "sqlite"
    assert storage_backend("journal:///x.journal") == "journal"
    assert storage_backend(None) is None


def test_sqlite_driver_url_rejected_with_parallel_jobs(tmp_path: Path) -> None:
    """Driver-qualified ``sqlite+pysqlite:///`` must trip the SQLite-parallel guard."""
    storage = f"sqlite+pysqlite:///{tmp_path / 'x.db'}"
    with pytest.raises(ValueError, match="SQLite"):
        make_experiment(workdir=tmp_path / "runs", storage=storage, n_jobs=2)


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
