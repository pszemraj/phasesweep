"""engine.read: permissive status/winner reads that never raise on a partial file."""

from __future__ import annotations

import hashlib
from pathlib import Path

import optuna
import pytest
import sqlalchemy
import yaml

import phasesweep.engine.optuna as engine_optuna
import phasesweep.engine.read as engine_read
from phasesweep import run_experiment
from phasesweep.config import (
    Experiment,
    FloatParam,
    IntParam,
    JsonEnvelopeExtractor,
    LogRegexExtractor,
    Metric,
    Phase,
    Promotion,
    Sampler,
    WandbExtractor,
)
from phasesweep.engine import read_status, read_winner, read_winners
from phasesweep.engine.guards import _plan_artifact_root_rebinds
from phasesweep.engine.run import experiment_status
from phasesweep.engine.state import (
    PUBLICATION_POINTER_SCHEMA_VERSION,
    _generation_path,
    _generation_record_path,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_path,
    _resolve_publication_pointer,
    _winner_path,
)
from tests.conftest import make_experiment, write_trainer


def _experiment(tmp_path: Path, *, storage: str | None = None) -> Experiment:
    return make_experiment(
        experiment="read_t",
        workdir=tmp_path / "wd",
        storage=storage,
        trial_command="python x.py {overrides}",
        override_format="argparse",
        metric=Metric(
            name="loss",
            goal="minimize",
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        ),
        phases=[
            Phase(
                name="p",
                n_trials=1,
                # Seeded random keeps this helper usable with persistent storage,
                # which rejects an unseeded stochastic sampler.
                sampler=Sampler(type="random", seed=0),
                search_space={"lr": FloatParam(type="float", low=1.0e-5, high=1.0e-2, log=True)},
            )
        ],
    )


def test_read_winner_parses_a_valid_file(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    path = _winner_path(exp, "p")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "phase": "p",
                "trial_number": 3,
                "metric": {"loss": 0.123, "goal": "minimize"},
                "params": {"lr": 0.001},
                "effective_overrides": {"lr": 0.001},
                "gates": [{"name": "g", "passed": True}],
                "completion": {"incomplete": False},
                "winner_source": {
                    "kind": "phase_trial",
                    "phase": "p",
                    "trial_number": 3,
                    "generation_id": "generation-test",
                    "attempt_id": "attempt-test",
                    "study": None,
                },
            }
        )
    )
    view = read_winner(exp, "p")
    assert view is not None
    assert view.trial_number == 3
    assert view.metric == 0.123
    assert view.gates_passed is True
    assert view.incomplete is False
    assert view.source is not None
    assert view.source.phase == "p"
    assert view.source.trial_number == 3


@pytest.mark.parametrize(
    "body",
    [
        '{"trial_number": 0, "metric": {"loss":',
        "phase: p\n",
        "- not\n- a\n- mapping\n",
        """\
phase: p
trial_number: 3
metric: {loss: 0.123, goal: minimize}
params: {lr: 0.001}
effective_overrides: {lr: 0.001}
completion: [not, a, mapping]
""",
    ],
    ids=["truncated", "missing_keys", "non_mapping", "bad_completion"],
)
def test_read_winner_tolerates_torn_or_malformed_file(tmp_path: Path, body: str) -> None:
    # Status reads stay permissive for legacy, hand-edited, or externally corrupted files.
    exp = _experiment(tmp_path)
    path = _winner_path(exp, "p")
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(body)
    assert read_winner(exp, "p") is None
    assert read_winners(exp) == []


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
def test_read_status_does_not_create_missing_storage(tmp_path: Path, backend: str) -> None:
    path = tmp_path / f"missing.{backend}"
    exp = _experiment(tmp_path, storage=f"{backend}:///{path}")

    status = read_status(exp)

    assert not path.exists()
    assert status["phases"][0]["trials"] == {}
    assert status["phases"][0]["trial_data_available"] is True
    assert status["phases"][0]["running_attempts"] == []
    assert status["metric"]["objective_evidence"] == {
        "kind": "log_regex",
        "attempt_location_scoped": True,
        "attempt_identity_bound": False,
        "source_identity_keyed": False,
        "objective_name_bound": False,
        "split_bound": False,
        "evaluation_policy_bound": False,
        "checkpoint_declared": False,
        "checkpoint_value_bound": False,
        "expected_step_declared": False,
        "expected_step_value_bound": False,
    }


def test_read_status_tolerates_uninitialized_sqlite_file(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    db.touch()
    exp = _experiment(tmp_path, storage=f"sqlite:///{db}")

    status = read_status(exp)

    assert status["phases"][0]["trials"] == {}
    assert status["phases"][0]["trial_data_available"] is False
    assert status["phases"][0]["running_attempts"] is None


@pytest.mark.parametrize("database_state", ["uninitialized", "absent-study", "empty-study"])
def test_external_status_does_not_initialize_or_change_schema(
    tmp_path, monkeypatch, database_state
):
    storage = f"sqlite:///{tmp_path / 'status.db'}"
    if database_state != "uninitialized":
        optuna.create_study(
            study_name="read_t::p" if database_state == "empty-study" else "unrelated",
            storage=storage,
        )
    engine = sqlalchemy.create_engine(storage)
    before = sqlalchemy.inspect(engine).get_table_names()
    writes = []

    @sqlalchemy.event.listens_for(engine, "before_cursor_execute")
    def observe_sql(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith(("CREATE", "INSERT", "UPDATE", "DELETE", "ALTER")):
            writes.append(statement)

    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *args, **kwargs: engine)
    experiment = _experiment(tmp_path).model_copy(
        update={
            "storage": "postgresql://localhost/status_placeholder",
            "allow_external_rdb_single_host": True,
        }
    )

    phase = read_status(experiment)["phases"][0]

    assert phase["trials"] == {}
    assert phase["trial_data_available"] is (database_state != "uninitialized")
    assert sqlalchemy.inspect(engine).get_table_names() == before
    assert writes == []
    engine.dispose()


def test_external_status_reads_counts_and_running_identities_in_one_statement(
    tmp_path, monkeypatch
):
    storage = f"sqlite:///{tmp_path / 'status.db'}"
    study = optuna.create_study(study_name="read_t::p", storage=storage)
    completed = study.ask()
    completed.set_user_attr("phasesweep_generation_id", "gen-1")
    completed.set_user_attr("phasesweep_attempt_id", "completed-attempt")
    study.tell(completed, 0.5)
    running = study.ask()
    running.set_user_attr("phasesweep_generation_id", "gen-1")
    running.set_user_attr("phasesweep_attempt_id", "attempt-1")
    study.ask()
    engine = sqlalchemy.create_engine(storage)
    statements = []

    @sqlalchemy.event.listens_for(engine, "before_cursor_execute")
    def observe_sql(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *args, **kwargs: engine)
    experiment = _experiment(tmp_path).model_copy(
        update={
            "storage": "postgresql://localhost/status_placeholder",
            "allow_external_rdb_single_host": True,
        }
    )

    stats = engine_optuna._phase_trial_stats(
        experiment, experiment.phases[0], engine_optuna._TrialRef(0, "gen-1", "completed-attempt")
    )

    assert stats.available
    assert stats.published_trial_available
    assert stats.counts == {"COMPLETE": 1, "RUNNING": 2}
    assert stats.generation_counts == {"gen-1": {"COMPLETE": 1, "RUNNING": 1}}
    assert sorted(stats.running_attempts, key=lambda ref: ref.trial_number) == [
        engine_optuna._TrialRef(1, "gen-1", "attempt-1"),
        engine_optuna._TrialRef(2, None, None),
    ]
    assert len(statements) == 1
    assert statements[0].lstrip().startswith("WITH")
    engine.dispose()


def test_read_status_uses_one_sqlite_snapshot_per_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "phases.db"
    storage = f"sqlite:///{db}"
    optuna.create_study(study_name="read_t::p", storage=storage).optimize(
        lambda trial: 1.0, n_trials=1
    )
    exp = _experiment(tmp_path, storage=storage)
    real_connect = engine_optuna.sqlite3.connect
    connections = 0

    def counting_connect(*args: object, **kwargs: object):
        nonlocal connections
        connections += 1
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(engine_optuna.sqlite3, "connect", counting_connect)

    status = read_status(exp)

    assert connections == 1
    assert status["phases"][0]["trials"] == {"COMPLETE": 1}
    assert status["phases"][0]["trial_data_available"] is True


def test_sqlite_status_query_aggregates_historical_rows_before_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "phases.db"
    storage = f"sqlite:///{db}"
    study = optuna.create_study(study_name="read_t::p", storage=storage)
    study.optimize(lambda trial: 1.0, n_trials=50)
    exp = _experiment(tmp_path, storage=storage)
    real_connect = engine_optuna.sqlite3.connect
    transferred_rows: list[int] = []

    class CursorProxy:
        def __init__(self, cursor) -> None:
            self._cursor = cursor

        def fetchall(self):
            rows = self._cursor.fetchall()
            transferred_rows.append(len(rows))
            return rows

    class ConnectionProxy:
        def __init__(self, connection) -> None:
            self._connection = connection

        def execute(self, *args: object, **kwargs: object):
            return CursorProxy(self._connection.execute(*args, **kwargs))

        def close(self) -> None:
            self._connection.close()

    def observed_connect(*args: object, **kwargs: object):
        return ConnectionProxy(real_connect(*args, **kwargs))

    monkeypatch.setattr(engine_optuna.sqlite3, "connect", observed_connect)

    status = read_status(exp)

    assert status["phases"][0]["trials"] == {"COMPLETE": 50}
    assert transferred_rows == [1]


@pytest.mark.parametrize(
    ("database_name", "storage_template"),
    [
        pytest.param("phases.db", "sqlite:///{db}?timeout=30", id="url-options"),
        pytest.param("uri.db", "sqlite:///file:{db}?mode=rwc&uri=true", id="uri-filename"),
    ],
)
def test_read_status_counts_sqlite_trials_with_storage_url_variants(
    tmp_path: Path, database_name: str, storage_template: str
) -> None:
    db = tmp_path / database_name
    storage = storage_template.format(db=db)
    optuna.create_study(study_name="read_t::p", storage=storage).optimize(
        lambda trial: 1.0, n_trials=1
    )
    exp = _experiment(tmp_path, storage=storage)

    status = read_status(exp)

    assert status["phases"][0]["trials"] == {"COMPLETE": 1}


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
def test_read_status_reports_running_attempts_from_the_counted_snapshot(
    tmp_path: Path, backend: str
) -> None:
    """Every backend reports RUNNING identities beside the counts they explain.

    The MCP terminal snapshot reconciles RUNNING rows against cleanup evidence
    and must not reread a study to learn which rows those are (PR #5 review /
    reviewer 2, blocker 6), so this comes from the same tolerant read.
    """
    path = tmp_path / f"phases.{backend}"
    exp = _experiment(tmp_path, storage=f"{backend}:///{path}")
    study = optuna.create_study(
        study_name="read_t::p",
        storage=engine_optuna._resolve_storage(exp.storage) or exp.storage,
    )
    study.optimize(lambda trial: 1.0, n_trials=1)
    running = study.ask()
    running.set_user_attr("phasesweep_generation_id", "gen-1")
    running.set_user_attr("phasesweep_attempt_id", "attempt-1")
    study.ask()

    phase = read_status(exp)["phases"][0]

    assert phase["trial_data_available"] is True
    assert phase["trials"] == {"COMPLETE": 1, "RUNNING": 2}
    assert phase["running_attempts"] == [
        {"trial_number": 1, "generation_id": "gen-1", "attempt_id": "attempt-1"},
        {"trial_number": 2, "generation_id": None, "attempt_id": None},
    ]


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
def test_read_status_reports_null_running_attempts_when_storage_is_unreadable(
    tmp_path: Path, backend: str
) -> None:
    """Unread trial data reports no RUNNING identities, not an empty list."""
    ledger = tmp_path / f"corrupt.{backend}"
    # Journal tolerates a torn final record; an earlier malformed record must fail.
    ledger.write_text("not a database or journal\nanother record\n", encoding="utf-8")
    exp = _experiment(tmp_path, storage=f"{backend}:///{ledger}")

    phase = read_status(exp)["phases"][0]

    assert phase["trial_data_available"] is False
    assert phase["running_attempts"] is None


@pytest.mark.parametrize("published", [False, True], ids=["unpublished", "published"])
@pytest.mark.parametrize("keep_prefix", [False, True], ids=["clobbered", "valid-prefix"])
@pytest.mark.parametrize("damage", ["garbage", "partial-json", "missing-newline", "operation"])
def test_journal_incomplete_or_invalid_snapshot_never_means_absent(
    tmp_path: Path, published: bool, keep_prefix: bool, damage: str
) -> None:
    from phasesweep.engine import ProcessCleanupUncertainError, StudyStorageUnavailableError
    from phasesweep.mcp.redaction import status_payload

    ledger = tmp_path / "study.journal"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{ledger}",
        n_trials=1,
        trial_command="echo x=0.5 {overrides}",
    )
    if published:
        run_experiment(experiment)
    else:
        optuna.create_study(
            study_name="t::p", storage=engine_optuna._resolve_storage(experiment.resolved_storage)
        )
    original = ledger.read_bytes()
    tail = {
        "garbage": b"not a journal record\n",
        "partial-json": b"{",
        "missing-newline": original.rstrip(b"\n").split(b"\n")[-1],
        "operation": b"{}\n",
    }[damage]
    damaged = (original if keep_prefix else b"") + tail
    ledger.write_bytes(damaged)
    root = tmp_path / "runs" / "t"
    root.mkdir(parents=True, exist_ok=True)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    status = read_status(experiment)
    mcp_status = status_payload(
        "exp", status, None, result_source="current_shared_study", elapsed_seconds=None
    )
    for payload in (status, experiment_status(experiment), mcp_status):
        phase = payload["phases"][0]
        assert phase["trial_data_available"] is False
        assert phase["published_study_unavailable"] is published
        assert not any(phase["trials"].values())
    assert status["phases"][0]["running_attempts"] is None
    with pytest.raises(StudyStorageUnavailableError):
        engine_optuna._load_existing_phase_study(experiment, experiment.phases[0])
    with pytest.raises(ProcessCleanupUncertainError):
        run_experiment(experiment)

    assert ledger.read_bytes() == damaged
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("change", ["append", "finish-partial", "truncate"])
def test_journal_status_uses_one_bounded_snapshot_during_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    ledger = tmp_path / "study.journal"
    experiment = _experiment(tmp_path, storage=f"journal:///{ledger}")
    study = optuna.create_study(
        study_name="read_t::p", storage=engine_optuna._resolve_storage(experiment.resolved_storage)
    )
    trial = study.ask()
    trial.set_user_attr("phasesweep_generation_id", "generation")
    trial.set_user_attr("phasesweep_attempt_id", "attempt")
    study.tell(trial, 0.5)
    complete = ledger.read_bytes()
    if change == "finish-partial":
        ledger.write_bytes(complete[:-1])
    expected = engine_optuna._TrialRef(0, "generation", "attempt")
    real_fstat = engine_optuna.os.fstat

    def change_after_capture(fd):
        captured = real_fstat(fd)
        if change == "append":
            with ledger.open("ab") as target:
                target.write(b"{")
        elif change == "finish-partial":
            ledger.write_bytes(complete)
        else:
            ledger.write_bytes(b"")
        return captured

    with monkeypatch.context() as patched:
        patched.setattr(engine_optuna.os, "fstat", change_after_capture)
        first = engine_optuna._phase_trial_stats(experiment, experiment.phases[0], expected)
    second = engine_optuna._phase_trial_stats(experiment, experiment.phases[0], expected)

    assert first.available is (change == "append")
    assert first.published_trial_available is (change == "append")
    assert first.counts == ({"COMPLETE": 1} if change == "append" else {})
    assert first.running_attempts == ([] if change == "append" else None)
    assert second.available is (change != "append")
    assert second.published_trial_available is (change == "finish-partial")


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sqlite", "journal"])
@pytest.mark.parametrize(
    "damage", ["missing-ledger", "missing-study", "empty-study", "corrupt", "stale-ledger"]
)
def test_published_status_distinguishes_absent_history_from_read_failure(
    tmp_path: Path, n_jobs: int, damage: str
) -> None:
    """The same status flags select the same remedy for both file backends."""
    from phasesweep.engine import (
        ArtifactRootRebindError,
        ProcessCleanupUncertainError,
        PublishedStudyMissingError,
    )
    from phasesweep.mcp.redaction import status_payload
    from phasesweep.mcp.snapshots import capture_result_snapshot

    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage="auto",
        n_jobs=n_jobs,
        n_trials=1,
        allow_no_gpu_isolation=True,
        trial_command="echo x=-{trial_id} {overrides}",
    )
    run_experiment(experiment)
    ledger = tmp_path / "runs" / "t" / ("study.db" if n_jobs == 1 else "study.journal")
    if damage == "stale-ledger":
        backup = ledger.read_bytes()
        experiment = experiment.model_copy(
            update={"phases": [experiment.phases[0].model_copy(update={"n_trials": 2})]}
        )
        assert run_experiment(experiment)["p"].trial_number == 1
        ledger.write_bytes(backup)
    elif damage == "missing-ledger":
        ledger.unlink()
    elif damage == "corrupt":
        ledger.write_text("not a database or journal\nanother record\n", encoding="utf-8")
    else:
        storage = engine_optuna._resolve_storage(experiment.resolved_storage)
        optuna.delete_study(study_name="t::p", storage=storage)
        if damage == "empty-study":
            optuna.create_study(study_name="t::p", storage=storage)
    ledger_before = ledger.read_bytes() if ledger.exists() else None
    generation_before = _generation_path(experiment).read_bytes()

    status = read_status(experiment)
    cli_status = experiment_status(experiment)
    mcp_status = status_payload(
        experiment_id="t",
        status=status,
        run=None,
        result_source="current_shared_study",
        elapsed_seconds=None,
    )
    snapshot = capture_result_snapshot(experiment)["status"]
    for payload in (status, cli_status, mcp_status, snapshot):
        assert payload["publication_integrity"] == "ok"
        phase = payload["phases"][0]
        assert phase["published_study_unavailable"] is True
        assert phase["trial_data_available"] is (damage != "corrupt")
        assert {state: count for state, count in phase["trials"].items() if count} == (
            {"COMPLETE": 1} if damage == "stale-ledger" else {}
        )
    assert status["phases"][0]["running_attempts"] == (None if damage == "corrupt" else [])
    assert (ledger.read_bytes() if ledger.exists() else None) == ledger_before
    expected_error = (
        ProcessCleanupUncertainError if damage == "corrupt" else PublishedStudyMissingError
    )
    with pytest.raises(expected_error):
        run_experiment(experiment)
    assert _generation_path(experiment).read_bytes() == generation_before
    if damage == "stale-ledger":
        with pytest.raises(ArtifactRootRebindError, match="published trial identity") as excinfo:
            _plan_artifact_root_rebinds([experiment])
        assert str(excinfo.value).endswith("Nothing was written.")
        assert ledger.read_bytes() == ledger_before
        assert _generation_path(experiment).read_bytes() == generation_before


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
@pytest.mark.parametrize("mismatch", [None, "number", "generation", "attempt", "state"])
def test_published_trial_status_requires_the_exact_completed_attempt(
    tmp_path: Path, backend: str, mismatch: str | None
) -> None:
    experiment = _experiment(tmp_path, storage=f"{backend}:///{tmp_path / 'ledger'}")
    study = optuna.create_study(
        study_name="read_t::p", storage=engine_optuna._resolve_storage(experiment.resolved_storage)
    )
    trial = study.ask()
    trial.set_user_attr("phasesweep_generation_id", "generation")
    trial.set_user_attr("phasesweep_attempt_id", "attempt")
    if mismatch != "state":
        study.tell(trial, 0.5)
    expected = engine_optuna._TrialRef(
        1 if mismatch == "number" else 0,
        "different" if mismatch == "generation" else "generation",
        "different" if mismatch == "attempt" else "attempt",
    )

    stats = engine_optuna._phase_trial_stats(experiment, experiment.phases[0], expected)

    assert stats.available
    assert stats.published_trial_available is (mismatch is None)
    assert stats.counts == {"RUNNING" if mismatch == "state" else "COMPLETE": 1}


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sqlite", "journal"])
def test_baseline_promotion_checks_local_candidate_and_allows_skipped_source_loss(
    tmp_path: Path, n_jobs: int
) -> None:
    from phasesweep.engine import PublishedStudyMissingError

    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage="auto",
        trial_command="echo x=-{trial_id}",
        phases=[
            Phase(
                name=name,
                n_trials=1,
                n_jobs=n_jobs,
                allow_no_gpu_isolation=True,
                sampler=Sampler(type="random", seed=0),
                promotion=(
                    Promotion(min_delta_vs="base", min_delta=100, on_fail="continue_baseline")
                    if name == "candidate"
                    else None
                ),
            )
            for name in ("base", "candidate")
        ],
    )
    run_experiment(experiment)
    ledger = tmp_path / "runs" / "t" / ("study.db" if n_jobs == 1 else "study.journal")
    backup = ledger.read_bytes()
    experiment = experiment.model_copy(
        update={
            "phases": [
                experiment.phases[0],
                experiment.phases[1].model_copy(update={"n_trials": 2}),
            ]
        }
    )
    winners = run_experiment(experiment)
    assert winners["candidate"].source.phase == "base"
    assert winners["candidate"].promotion["candidate_trial_number"] == 1
    assert all(
        not phase["published_study_unavailable"] for phase in read_status(experiment)["phases"]
    )

    optuna.delete_study(
        study_name="t::base", storage=engine_optuna._resolve_storage(experiment.resolved_storage)
    )
    resumed = run_experiment(experiment, from_phase="candidate")
    assert resumed["candidate"].attempt_id == winners["base"].attempt_id

    ledger.write_bytes(backup)
    generation_before = _generation_path(experiment).read_bytes()
    phases = read_status(experiment)["phases"]
    assert phases[0]["published_study_unavailable"] is False
    assert phases[1]["published_study_unavailable"] is True
    assert phases[1]["trial_data_available"] is True
    assert phases[1]["completed"] == 1
    with pytest.raises(
        PublishedStudyMissingError, match="phase 'candidate'.*published trial identity"
    ):
        run_experiment(experiment, from_phase="candidate")
    assert _generation_path(experiment).read_bytes() == generation_before


def _mark_generation_published(exp: Experiment, generation_id: str, phase_name: str) -> None:
    """Publish an immutable generation (summary + record + phase winner) on disk.

    Pointer validation (:func:`phasesweep.engine.state._last_successful_generation_id`)
    now reads back the generation's own immutable *summary*, not the
    lifecycle record's state, so a summary naming this owner and id is
    required for the pointer to resolve as published (review v0.5.15 /
    blocker 3). The record is written too, purely as the informational,
    post-commit artifact real publications also produce.
    """
    summary_path = _generation_summary_path(exp, generation_id)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        yaml.safe_dump({"experiment": exp.experiment, "generation_id": generation_id})
    )
    summary = summary_path.read_bytes()
    _last_successful_generation_path(exp).write_text(
        yaml.safe_dump(
            {
                "schema_version": PUBLICATION_POINTER_SCHEMA_VERSION,
                "experiment": exp.experiment,
                "generation_id": generation_id,
                "summary_size_bytes": len(summary),
                "summary_sha256": hashlib.sha256(summary).hexdigest(),
            }
        )
    )
    record_path = _generation_record_path(exp, generation_id)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        yaml.safe_dump(
            {"experiment": exp.experiment, "generation_id": generation_id, "state": "published"}
        )
    )
    winner_path = _generation_winner_path(exp, generation_id, phase_name)
    winner_path.parent.mkdir(parents=True, exist_ok=True)
    winner_path.write_text(
        yaml.safe_dump(
            {
                "phase": phase_name,
                "trial_number": 0,
                "metric": {"loss": 0.1, "goal": "minimize"},
                "params": {"lr": 0.001},
                "effective_overrides": {"lr": 0.001},
                "completion": {"incomplete": False},
                "generation_id": generation_id,
                "winner_source": {
                    "kind": "phase_trial",
                    "phase": phase_name,
                    "trial_number": 0,
                    "generation_id": generation_id,
                    "attempt_id": None,
                    "study": None,
                },
            }
        )
    )


def test_read_status_distinguishes_current_from_published_generation(tmp_path: Path) -> None:
    """A failed rerun's counts must never be mistaken for the older published winner's."""
    exp = _experiment(tmp_path)
    _mark_generation_published(exp, "generation-good", "p")
    # A newer generation has since started (and, in this scenario, failed) without
    # ever publishing - the mutable current-generation pointer moved on.
    _generation_path(exp).parent.mkdir(parents=True, exist_ok=True)
    _generation_path(exp).write_text(yaml.safe_dump({"generation_id": "generation-failed"}))

    status = read_status(exp)

    assert status["current_generation_id"] == "generation-failed"
    assert status["published_generation_id"] == "generation-good"
    assert status["current_generation_id"] != status["published_generation_id"]
    # In default (non-pinned) mode, represented_generation_id is the captured
    # published id, and winner/summary facts scope to it -- never the
    # failed/in-progress current one.
    assert status["represented_generation_id"] == "generation-good"
    assert status["is_published"] is True
    assert status["phases"][0]["winner_present"] is True
    assert status["summary_present"] is True


def test_read_status_explicit_generation_id_is_self_scoped(tmp_path: Path) -> None:
    """An explicit generation_id pins the represented identity to that one generation.

    ``current_generation_id`` and ``published_generation_id`` always report
    the *actual* pointers -- never forced to the pinned id (review v0.5.15 /
    blocker 3, defect 2: "pinned reads lie"). Here no current-pointer file
    exists at all, so ``current_generation_id`` is genuinely ``None`` even
    though the pinned generation itself is fully published.
    """
    exp = _experiment(tmp_path)
    _mark_generation_published(exp, "generation-good", "p")

    status = read_status(exp, generation_id="generation-good")

    assert status["current_generation_id"] is None
    assert status["published_generation_id"] == "generation-good"
    assert status["represented_generation_id"] == "generation-good"
    assert status["is_published"] is True
    assert status["phases"][0]["winner_present"] is True


def test_read_status_pinned_read_of_unpublished_generation_is_not_marked_published(
    tmp_path: Path,
) -> None:
    """Pinning a generation that never published reports is_published=False, not a lie."""
    exp = _experiment(tmp_path)
    _mark_generation_published(exp, "generation-good", "p")
    # A second, never-published generation has its own (unpublished) winner.
    other_winner = _generation_winner_path(exp, "generation-orphan", "p")
    other_winner.parent.mkdir(parents=True, exist_ok=True)
    other_winner.write_text(
        yaml.safe_dump(
            {
                "phase": "p",
                "trial_number": 1,
                "metric": {"loss": 0.2, "goal": "minimize"},
                "params": {"lr": 0.002},
                "effective_overrides": {"lr": 0.002},
                "completion": {"incomplete": False},
                "generation_id": "generation-orphan",
                "winner_source": {
                    "kind": "phase_trial",
                    "phase": "p",
                    "trial_number": 1,
                    "generation_id": "generation-orphan",
                    "attempt_id": None,
                    "study": None,
                },
            }
        )
    )

    status = read_status(exp, generation_id="generation-orphan")

    assert status["published_generation_id"] == "generation-good"
    assert status["represented_generation_id"] == "generation-orphan"
    assert status["is_published"] is False
    # The orphaned generation's own winner is still readable pinned.
    assert status["phases"][0]["winner_present"] is True


def test_read_status_legacy_workdir_without_generation_metadata_is_published(
    tmp_path: Path,
) -> None:
    """A pre-generation workdir's ``winner.yaml`` is the publication, so say so.

    Legacy layouts have no ``generation.yaml`` and therefore no pointer
    identity to compare, which used to report ``is_published: False`` right
    next to a real winner path -- an upgrading operator read "never
    published" about a published result. The three identity fields stay
    ``None`` (there is genuinely no generation id to report; none is
    fabricated) while ``is_published`` follows the winner actually on disk.
    """
    exp = _experiment(tmp_path)
    legacy_winner = _winner_path(exp, "p")
    legacy_winner.parent.mkdir(parents=True, exist_ok=True)
    legacy_winner.write_text(
        yaml.safe_dump(
            {
                "phase": "p",
                "trial_number": 2,
                "metric": {"loss": 0.3, "goal": "minimize"},
                "params": {"lr": 0.003},
                "effective_overrides": {"lr": 0.003},
                "completion": {"incomplete": False},
            }
        )
    )
    assert not _generation_path(exp).exists()

    status = read_status(exp)

    assert status["current_generation_id"] is None
    assert status["published_generation_id"] is None
    assert status["represented_generation_id"] is None
    assert status["is_published"] is True
    assert status["publication_integrity"] == "ok"
    assert status["phases"][0]["winner_present"] is True

    # The path-bearing CLI/suite view derives from the same read and must not
    # contradict its own winner path either.
    cli_status = experiment_status(exp)
    assert cli_status["is_published"] is True
    assert cli_status["phases"][0]["winner"] == str(legacy_winner)


def test_read_status_untouched_workdir_is_not_published(tmp_path: Path) -> None:
    """No generation metadata *and* no legacy winner is still "nothing published"."""
    exp = _experiment(tmp_path)

    status = read_status(exp)

    assert status["is_published"] is False
    assert status["phases"][0]["winner_present"] is False


def test_read_status_unpublished_generation_is_not_published(tmp_path: Path) -> None:
    """A generation-aware workdir that never published keeps ``is_published: False``.

    The legacy fallback must not leak into layouts that *do* carry generation
    metadata: once ``generation.yaml`` exists, a stale compatibility
    ``winner.yaml`` is not authoritative.
    """
    exp = _experiment(tmp_path)
    _generation_path(exp).parent.mkdir(parents=True, exist_ok=True)
    _generation_path(exp).write_text(yaml.safe_dump({"generation_id": "generation-running"}))
    legacy_winner = _winner_path(exp, "p")
    legacy_winner.parent.mkdir(parents=True, exist_ok=True)
    legacy_winner.write_text(
        yaml.safe_dump(
            {
                "phase": "p",
                "trial_number": 2,
                "metric": {"loss": 0.3, "goal": "minimize"},
                "params": {"lr": 0.003},
                "effective_overrides": {"lr": 0.003},
                "completion": {"incomplete": False},
            }
        )
    )

    status = read_status(exp)

    assert status["current_generation_id"] == "generation-running"
    assert status["published_generation_id"] is None
    assert status["is_published"] is False
    assert status["phases"][0]["winner_present"] is False


@pytest.mark.parametrize(
    "declared", [pytest.param(False, id="undeclared"), pytest.param(True, id="declared")]
)
def test_objective_evidence_assurance_json_envelope_checkpoint_binding(
    tmp_path: Path,
    declared: bool,
) -> None:
    """Checkpoint assurance must reflect whether values were declared."""
    exp = _experiment(tmp_path)
    identity = {"checkpoint": "ckpt-42", "expected_step": 1000} if declared else {}
    exp = exp.model_copy(
        update={
            "metric": Metric(
                name="loss",
                goal="minimize",
                extractor=JsonEnvelopeExtractor(
                    type="json_envelope",
                    path="result.json",
                    objective_name="loss",
                    split="test",
                    policy="test",
                    **identity,
                ),
            )
        }
    )

    status = read_status(exp)

    assert status["metric"]["objective_evidence"] == {
        "kind": "json_envelope",
        "attempt_location_scoped": True,
        "attempt_identity_bound": True,
        "source_identity_keyed": False,
        "objective_name_bound": True,
        "split_bound": True,
        "evaluation_policy_bound": True,
        "checkpoint_declared": declared,
        "checkpoint_value_bound": declared,
        "expected_step_declared": declared,
        "expected_step_value_bound": declared,
    }


@pytest.mark.parametrize(
    ("extractor", "expected_triple"),
    [
        pytest.param(
            JsonEnvelopeExtractor(
                type="json_envelope",
                path="result.json",
                objective_name="loss",
                split="test",
                policy="test",
            ),
            (True, True, False),
            id="json_envelope",
        ),
        pytest.param(
            LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
            (True, False, False),
            id="log_regex",
        ),
        pytest.param(
            WandbExtractor(type="wandb", entity="acme", project="proj", metric_key="eval/loss"),
            (True, False, True),
            id="wandb",
        ),
    ],
)
def test_objective_evidence_assurance_attempt_triple_by_kind(
    tmp_path: Path,
    extractor: JsonEnvelopeExtractor | LogRegexExtractor | WandbExtractor,
    expected_triple: tuple[bool, bool, bool],
) -> None:
    """Each extractor kind reports its own (location, identity, source-key) triple.

    ``json_envelope`` structurally echoes and cross-checks the attempt
    identity; ``wandb`` is keyed by an immutable run id that IS the attempt
    id; ``log_regex`` is merely read from an attempt-scoped location with
    nothing in its contents tying it to that attempt (review v0.5.15 / item C).
    """
    exp = _experiment(tmp_path)
    exp = exp.model_copy(
        update={"metric": Metric(name="loss", goal="minimize", extractor=extractor)}
    )

    status = read_status(exp)
    evidence = status["metric"]["objective_evidence"]

    assert (
        evidence["attempt_location_scoped"],
        evidence["attempt_identity_bound"],
        evidence["source_identity_keyed"],
    ) == expected_triple


# --------------------------------------------------------------------------
# Published results are rendered under their own historical semantics, never
# reinterpreted through whatever config is loaded today (review v0.5.16 /
# blocker 4).
# --------------------------------------------------------------------------

_DRIFT_TRAINER = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out")
parser.add_argument("--x", type=int, default=0)
args, _ = parser.parse_known_args()
print(f"x={args.x}")
"""


def _drift_experiment(tmp_path: Path, **overrides: object) -> Experiment:
    trainer = write_trainer(tmp_path / "trainer.py", _DRIFT_TRAINER)
    defaults: dict[str, object] = dict(
        experiment="drift_t",
        workdir=tmp_path / "wd",
        storage=f"sqlite:///{tmp_path / 'drift.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        metric=Metric(
            name="x",
            goal="minimize",
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        ),
        phases=[
            Phase(
                name="p",
                n_trials=1,
                comment="original hypothesis",
                sampler=Sampler(type="random", seed=0),
                search_space={"x": IntParam(type="int", low=0, high=10)},
            )
        ],
    )
    defaults.update(overrides)
    return make_experiment(**defaults)  # type: ignore[arg-type]


def test_published_result_is_not_reinterpreted_by_config_drift(tmp_path: Path) -> None:
    """Review v0.5.16 / blocker 4 reproduction: publish x/minimize, reload as y/maximize.

    Pre-fix, status labeled the official generation with the *current*
    metric name and inverted goal, and winner parsing silently dropped the
    actual winner because the current metric key was absent from the
    historical file.
    """
    published = _drift_experiment(tmp_path)
    run_experiment(published)

    drifted = _drift_experiment(
        tmp_path,
        metric=Metric(
            name="y",
            goal="maximize",
            extractor=LogRegexExtractor(type="log_regex", pattern=r"y=(?P<value>[0-9.eE+-]+)"),
        ),
        phases=[
            Phase(
                name="p",
                n_trials=1,
                comment="new hypothesis",
                sampler=Sampler(type="random", seed=0),
                search_space={"x": IntParam(type="int", low=100, high=110)},
            )
        ],
    )

    status = read_status(drifted)
    assert status["is_published"] is True
    assert status["metric"]["name"] == "x"
    assert status["metric"]["goal"] == "minimize"
    assert status["result_context"] == "represented_generation"
    assert status["published_config_matches_current"] is False
    assert status["phases"][0]["winner_present"] is True

    winners = read_winners(drifted)
    assert len(winners) == 1
    assert winners[0].metric_name == "x"
    assert winners[0].metric_goal == "minimize"


def test_run_control_edits_keep_published_config_current(tmp_path: Path) -> None:
    """A top-up or comment edit is not semantic drift for a published result."""
    published = _drift_experiment(tmp_path)
    run_experiment(published)

    topped_up = _drift_experiment(
        tmp_path,
        phases=[
            Phase(
                name="p",
                n_trials=2,
                comment="reworded documentation",
                sampler=Sampler(type="random", seed=0),
                search_space={"x": IntParam(type="int", low=0, high=10)},
            )
        ],
    )

    status = read_status(topped_up)
    assert status["published_config_matches_current"] is True
    assert status["metric"]["name"] == "x"


def test_read_status_without_any_publication_uses_current_config(tmp_path: Path) -> None:
    """With nothing published there is no historical context to render."""
    experiment = _drift_experiment(tmp_path)
    status = read_status(experiment)
    assert status["result_context"] == "current_config"
    assert status["published_config_matches_current"] is None
    assert status["result_phase_plan"] == ["p"]
    assert status["metric"]["name"] == "x"


def test_published_result_keeps_its_own_phase_plan_after_a_rename(tmp_path: Path) -> None:
    """A phase renamed after publication must not hide the published winner.

    ``phases`` stays the *current* config's progress view -- a run would have
    to produce a winner for ``q`` -- while ``result_phase_plan`` reports the
    plan the publication actually used, so a reader can enumerate it instead
    of concluding the publication is empty (review v0.5.16 / blocker 4).
    """
    published = _drift_experiment(tmp_path)
    run_experiment(published)

    renamed = _drift_experiment(
        tmp_path,
        phases=[
            Phase(
                name="q",
                n_trials=1,
                comment="renamed phase",
                sampler=Sampler(type="random", seed=0),
                search_space={"x": IntParam(type="int", low=0, high=10)},
            )
        ],
    )

    status = read_status(renamed)
    assert status["is_published"] is True
    assert status["result_phase_plan"] == ["p"]
    assert status["published_config_matches_current"] is False
    assert [phase["phase"] for phase in status["phases"]] == ["q"]
    assert status["phases"][0]["winner_present"] is False

    generation_id = status["represented_generation_id"]
    # The declared-phase default is unchanged: the CLI resume/progress readers
    # that depend on it keep asking about the phases configured today.
    assert read_winners(renamed, generation_id=generation_id) == []

    (winner,) = read_winners(
        renamed,
        generation_id=generation_id,
        phase_names=status["result_phase_plan"],
    )
    assert winner.phase == "p"
    assert (winner.metric_name, winner.metric_goal) == ("x", "minimize")


def test_read_status_reuses_the_pointer_authenticated_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A status snapshot never reopens summary bytes after authenticating its pointer."""
    experiment = _drift_experiment(tmp_path)
    run_experiment(experiment)
    publication = _resolve_publication_pointer(experiment)
    assert publication.state == "ok"
    generation_id = publication.generation_id
    assert generation_id is not None

    summary_path = _generation_summary_path(experiment, generation_id)
    summary_path.write_bytes(summary_path.read_bytes() + b"\n# changed after resolution\n")
    monkeypatch.setattr(engine_read, "_resolve_publication_pointer", lambda _exp: publication)

    def unexpected_reread(_path: Path | None):
        raise AssertionError("authenticated summary was reopened")

    monkeypatch.setattr(engine_read, "_read_summary_payload", unexpected_reread)

    status = read_status(experiment)
    assert status["publication_integrity"] == "ok"
    assert status["result_phase_plan"] == ["p"]
    assert status["metric"]["name"] == "x"
    assert _resolve_publication_pointer(experiment).state == "failed"


def test_result_phase_plan_falls_back_to_the_current_config_for_a_planless_summary(
    tmp_path: Path,
) -> None:
    """A summary that records no plan leaves the current config describing it.

    Pre-manifest layouts published a summary with no ``phase_plan`` at all;
    the only phase names such a tree can be read under are the configured
    ones, and that fallback must survive.
    """
    exp = _experiment(tmp_path)
    _mark_generation_published(exp, "generation-planless", "p")

    status = read_status(exp)

    assert status["is_published"] is True
    assert status["result_context"] == "current_config"
    assert status["result_phase_plan"] == ["p"]


@pytest.mark.parametrize("unsafe_name", ["../escape", "", "phases/p"])
def test_read_winners_refuses_phase_names_that_are_not_path_components(
    tmp_path: Path, unsafe_name: str
) -> None:
    """Explicit plans become path segments, so they are validated like config names.

    Plans parsed off disk are filtered before they reach here, so this guards
    the remaining way an unsafe name could arrive: a caller passing one.
    """
    exp = _experiment(tmp_path)

    with pytest.raises(ValueError, match="phase name"):
        read_winners(exp, phase_names=[unsafe_name])
