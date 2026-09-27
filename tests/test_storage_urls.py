"""Storage URL parsing and canonicalization: backend detection, absolute vs. relative path preservation, and the same-host lock identity that equivalent URLs must share."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from phasesweep import load_experiment
from phasesweep.config import (
    ExecutionContext,
    Experiment,
    Sampler,
)
from phasesweep.engine import read_status, run_experiment
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.locking import _run_lock_paths
from phasesweep.runtime.files import (
    canonical_storage_identity,
    file_url_path,
    storage_backend,
    storage_recovery_locator,
)
from tests.conftest import make_experiment, write_yaml


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_auto_storage_resolves_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_jobs: int
) -> None:
    """Auto storage resolves to a journal ledger regardless of n_jobs."""
    monkeypatch.chdir(tmp_path)
    experiment = make_experiment(
        workdir="relative", storage="auto", n_jobs=n_jobs, allow_no_gpu_isolation=True
    )
    url = experiment.resolved_storage
    assert url is not None
    assert experiment.storage == experiment.model_dump()["storage"] == "auto"
    assert "resolved_storage" not in experiment.model_dump()
    assert storage_backend(url) == "journal"
    assert file_url_path(url) == str(tmp_path / "relative" / "t" / "study.journal")
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
@pytest.mark.integration
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
    database = root / exp.experiment / "study.journal"
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


def test_resolve_storage_urls(tmp_path: Path) -> None:
    """None resolves to an in-memory study; a journal URL resolves to JournalStorage."""
    import optuna
    from optuna.storages import JournalStorage

    assert _resolve_storage(None) is None
    assert optuna.create_study(storage=_resolve_storage(None)).trials == []

    result = _resolve_storage(f"journal:///{tmp_path}/phases.journal")
    assert isinstance(result, JournalStorage)


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


def test_explicit_journal_storage_allows_parallel_jobs(tmp_path: Path) -> None:
    """Explicit journal storage supports n_jobs > 1 without a serialization guard."""
    p = _storage_policy_config(tmp_path, storage=f"journal:///{tmp_path}/phases.journal", n_jobs=4)

    load_experiment(p)


@pytest.mark.parametrize("storage", ["postgresql://user:pass@host/db", "mysql://user@host/db"])
def test_external_storage_is_rejected_at_config_load(tmp_path: Path, storage: str) -> None:
    """Only null, auto, and explicit journal:/// storage are supported."""
    path = _storage_policy_config(tmp_path, storage=storage, n_jobs=1)

    with pytest.raises(ValidationError, match="storage must be null, auto, or a journal:/// URL"):
        load_experiment(path)


@pytest.mark.parametrize(
    "storage",
    ["sqlite:///x.db", "sqlite:///:memory:", "sqlite://", ":memory:", "postgresql://h/db"],
)
def test_non_journal_storage_is_refused_at_config_load(tmp_path: Path, storage: str) -> None:
    """Every spelling but null, auto, and journal:/// is refused, with the remedy spelled out."""
    path = _storage_policy_config(tmp_path, storage=json.dumps(storage), n_jobs=1)

    with pytest.raises(ValidationError) as excinfo:
        load_experiment(path)

    message = str(excinfo.value)
    assert "storage must be null, auto, or a journal:/// URL" in message
    assert "journal:///path/to/study.journal" in message
    assert "storage: null" in message


def test_canonical_storage_identity_resolves_paths(tmp_path: Path) -> None:
    """File-based backends resolve to absolute paths so equivalent URL spellings
    (relative paths, ``..`` segments) produce one stable lock identity. None
    in returns None out (in-memory has no shared backend to collide on).

    This test pins only the path-resolution and None-handling contract.
    """
    # Journal path with `..` must be resolved away.
    journal_id = canonical_storage_identity(f"journal:///{tmp_path}/sub/../study.journal")
    assert journal_id is not None
    assert ".." not in journal_id
    assert str(tmp_path / "study.journal") in journal_id

    # None → None (in-memory study).
    assert canonical_storage_identity(None) is None


def test_storage_recovery_locator_freezes_relative_file_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery locators retain the registration cwd and URL semantics."""
    registration_cwd = tmp_path / "registration"
    recovery_cwd = tmp_path / "recovery"
    registration_cwd.mkdir()
    recovery_cwd.mkdir()
    monkeypatch.chdir(registration_cwd)

    locator = storage_recovery_locator("journal:///studies.journal")

    monkeypatch.chdir(recovery_cwd)
    assert locator is not None
    assert storage_backend(locator) == "journal"
    assert Path(file_url_path(locator)).is_absolute() or "file:/" in file_url_path(locator)
    assert "studies.journal" in locator
    assert str(registration_cwd) in locator


def test_storage_backend_collapses_dialect_suffixes() -> None:
    """A `+driver` suffix folds into its base scheme name; unset storage is None."""
    assert storage_backend("journal:///x.journal") == "journal"
    assert storage_backend("postgresql+psycopg2://h/db") == "postgresql"
    assert storage_backend(None) is None


def test_file_url_path_preserves_absolute_paths() -> None:
    """``file_url_path`` must distinguish three-slash (relative) from
    four-slash (absolute) URLs. Pre-v0.5.10 it used ``lstrip("/")`` which
    destroyed the leading ``/`` on absolute paths."""
    cases = [
        ("journal:///relative.journal", "relative.journal"),
        ("journal:///relative.journal?timeout=30", "relative.journal"),
        ("journal:////tmp/absolute.journal", "/tmp/absolute.journal"),
        ("journal:////tmp/absolute.journal?timeout=30#frag", "/tmp/absolute.journal"),
    ]

    for url, expected_path in cases:
        assert file_url_path(url) == expected_path, url


def test_absolute_file_storage_identity_is_not_cwd_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absolute file-storage identities must not depend on caller cwd."""
    cwd_a = tmp_path / "cwd_a"
    cwd_b = tmp_path / "cwd_b"
    cwd_a.mkdir()
    cwd_b.mkdir()

    path = tmp_path / "study.journal"
    url = f"journal:///{path}"

    monkeypatch.chdir(cwd_a)
    identity_a = canonical_storage_identity(url)

    monkeypatch.chdir(cwd_b)
    identity_b = canonical_storage_identity(url)

    expected = "journal:///" + str(path.resolve())
    assert identity_a == expected
    assert identity_b == expected
