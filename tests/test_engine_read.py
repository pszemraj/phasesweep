"""engine.read: permissive status/winner reads that never raise on a partial file."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import optuna
import pytest

import phasesweep.engine.ledger as engine_ledger
import phasesweep.engine.optuna as engine_optuna
import phasesweep.engine.read as engine_read
from phasesweep import run_experiment
from phasesweep.config import (
    Experiment,
    FloatParam,
    IntParam,
    JsonEnvelopeExtractor,
    JsonExtractor,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
    WandbExtractor,
)
from phasesweep.engine import read_status, read_winners
from phasesweep.engine.paths import (
    _generation_path,
    _generation_summary_path,
)
from phasesweep.engine.publication import _resolve_publication_pointer
from phasesweep.engine.run import experiment_status
from tests.conftest import make_experiment, mark_current_format, write_trainer
from tests.ledger_fixtures import tree_snapshot


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


def test_read_status_aggregates_each_sqlite_phase_in_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "phases.db"
    storage = f"sqlite:///{db}"
    study = optuna.create_study(study_name="read_t::p", storage=storage)
    # More than one trial, so a per-trial transfer cannot pass as the single
    # aggregated row asserted below.
    study.optimize(lambda trial: 1.0, n_trials=3)
    exp = _experiment(tmp_path, storage=storage)
    mark_current_format(exp, study)
    real_connect = engine_ledger.sqlite3.connect
    connections = 0
    transferred: list[tuple[str, int]] = []

    class CursorProxy:
        def __init__(self, cursor, sql: str) -> None:
            self._cursor = cursor
            self._sql = sql

        def fetchall(self):
            rows = self._cursor.fetchall()
            transferred.append((self._sql[:40], len(rows)))
            return rows

    class ConnectionProxy:
        def __init__(self, connection) -> None:
            self._connection = connection

        def execute(self, sql: str, *args: object, **kwargs: object):
            return CursorProxy(self._connection.execute(sql, *args, **kwargs), sql)

        def close(self) -> None:
            self._connection.close()

    def observed_connect(*args: object, **kwargs: object):
        nonlocal connections
        connections += 1
        return ConnectionProxy(real_connect(*args, **kwargs))

    monkeypatch.setattr(engine_ledger.sqlite3, "connect", observed_connect)

    status = read_status(exp)

    # Connection 1 is the ledger-format scan ``validate_ledger`` runs before any
    # phase is read. Each phase then adds exactly one: the trial-stats snapshot
    # ``read_phase_trial_stats`` opens, so the counts and the RUNNING
    # identities that explain them come from a single snapshot.
    assert connections == 1 + len(exp.phases)
    assert status["phases"][0]["trials"] == {"COMPLETE": 3}
    assert status["phases"][0]["trial_data_available"] is True
    # The format scan reads the ledger first and is not what this pins down.
    # The statement under test is the one naming the ``phase_trials`` CTE: it
    # must aggregate server-side and hand back a single row, not one per trial.
    aggregated = [rows for sql, rows in transferred if "phase_trials" in sql]
    assert aggregated == [1], transferred


@pytest.mark.parametrize(
    ("database_name", "storage_template"),
    [
        pytest.param("phases.db", "sqlite:///{db}?timeout=30", id="url-options"),
        pytest.param("study#experiment.db", "sqlite:///{db}", id="literal-hash"),
        pytest.param("study#experiment.db", "sqlite+pysqlite:///{db}", id="driver-literal-hash"),
        pytest.param("uri.db", "sqlite:///file:{db}?mode=rwc&uri=true", id="uri-filename"),
    ],
)
def test_read_status_counts_sqlite_trials_with_storage_url_variants(
    tmp_path: Path, database_name: str, storage_template: str
) -> None:
    db = tmp_path / database_name
    storage = storage_template.format(db=db)
    study = optuna.create_study(study_name="read_t::p", storage=storage)
    study.optimize(lambda trial: 1.0, n_trials=1)
    exp = _experiment(tmp_path, storage=storage)
    mark_current_format(exp, study)

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
        storage=engine_ledger._resolve_storage(exp.storage) or exp.storage,
    )
    study.optimize(lambda trial: 1.0, n_trials=1)
    running = study.ask()
    running.set_user_attr("phasesweep_generation_id", "gen-1")
    running.set_user_attr("phasesweep_attempt_id", "attempt-1")
    study.ask()
    mark_current_format(exp, study)

    phase = read_status(exp)["phases"][0]

    assert phase["trial_data_available"] is True
    assert phase["trials"] == {"COMPLETE": 1, "RUNNING": 2}
    assert phase["running_attempts"] == [
        {"trial_number": 1, "generation_id": "gen-1", "attempt_id": "attempt-1"},
        {"trial_number": 2, "generation_id": None, "attempt_id": None},
    ]


@pytest.mark.parametrize(
    ("backend", "cause"),
    [("sqlite", "DatabaseError"), ("journal", "JSONDecodeError")],
)
def test_read_status_reports_null_running_attempts_when_storage_is_unreadable(
    tmp_path: Path, backend: str, cause: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Unread trial data reports no RUNNING identities, not an empty list."""
    ledger = tmp_path / f"corrupt.{backend}"
    exp = _experiment(tmp_path, storage=f"{backend}:///{ledger}")
    study = optuna.create_study(
        study_name="read_t::p",
        storage=engine_ledger._resolve_storage(exp.resolved_storage),
    )
    mark_current_format(exp, study)
    # Journal tolerates a torn final record; an earlier malformed record must fail.
    ledger.write_text("not a database or journal\nanother record\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.ledger"):
        phase = read_status(exp)["phases"][0]

    assert phase["trial_data_available"] is False
    assert phase["running_attempts"] is None
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "phasesweep.engine.ledger"
    ]
    assert len(warnings) == 1
    assert str(ledger) in warnings[0]
    assert cause in warnings[0]


def _journal_experiment(tmp_path: Path, *, published: bool) -> tuple[Experiment, Path]:
    """Return an experiment whose journal holds a current study, run or only created.

    :param Path tmp_path: Test-owned directory for the tree and the journal.
    :param bool published: Run the experiment once, so the journal holds a
        published trial, instead of only creating and stamping its study.
    :return tuple[Experiment, Path]: The experiment and its journal file.
    """
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
        study = optuna.create_study(
            study_name="t::p", storage=engine_ledger._resolve_storage(experiment.resolved_storage)
        )
        mark_current_format(experiment, study)
    return experiment, ledger


def _status_phases(experiment: Experiment, status: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the first phase of every status surface a journal read feeds."""
    from phasesweep.mcp.redaction import status_payload

    mcp_status = status_payload(
        "exp",
        status,
        {"run_id": "mcp-run"},
        result_source="current_shared_study",
        elapsed_seconds=None,
    )
    return [payload["phases"][0] for payload in (status, experiment_status(experiment), mcp_status)]


@pytest.mark.parametrize("published", [False, True], ids=["unpublished", "published"])
@pytest.mark.parametrize("keep_prefix", [False, True], ids=["clobbered", "valid-prefix"])
@pytest.mark.parametrize("damage", ["operation", "middle-garbage"])
def test_journal_malformed_record_before_its_end_never_means_absent(
    tmp_path: Path, published: bool, keep_prefix: bool, damage: str
) -> None:
    """A record Optuna cannot replay, or a bad line another line follows, is refused everywhere."""
    from phasesweep.engine import ProcessCleanupUncertainError, StudyStorageUnavailableError

    experiment, ledger = _journal_experiment(tmp_path, published=published)
    original = ledger.read_bytes()
    last_record = original.rstrip(b"\n").split(b"\n")[-1] + b"\n"
    tail = {
        "operation": b"{}\n",
        "middle-garbage": b"not a journal record\n" + last_record,
    }[damage]
    damaged = (original if keep_prefix else b"") + tail
    ledger.write_bytes(damaged)
    root = tmp_path / "runs" / "t"
    root.mkdir(parents=True, exist_ok=True)
    before = tree_snapshot(root)

    status = read_status(experiment)
    for phase in _status_phases(experiment, status):
        assert phase["trial_data_available"] is False
        assert phase["published_study_unavailable"] is published
        assert not any(phase["trials"].values())
    assert status["phases"][0]["running_attempts"] is None
    with pytest.raises(StudyStorageUnavailableError):
        engine_ledger._load_existing_phase_study(experiment, experiment.phases[0])
    with pytest.raises(ProcessCleanupUncertainError):
        run_experiment(experiment)

    assert ledger.read_bytes() == damaged
    assert tree_snapshot(root) == before


@pytest.mark.parametrize("published", [False, True], ids=["unpublished", "published"])
@pytest.mark.parametrize("keep_prefix", [False, True], ids=["clobbered", "valid-prefix"])
@pytest.mark.parametrize("damage", ["garbage", "partial-json", "missing-newline"])
def test_journal_partial_final_record_reads_as_optuna_does_and_blocks_writes(
    tmp_path: Path, published: bool, keep_prefix: bool, damage: str
) -> None:
    """Reads skip a bad final line exactly as Optuna does; a run refuses to append after it.

    Optuna's reader skips a final line that lacks its newline or does not
    decode, so status reports the complete records a live load would see. Its
    writer would glue the next record onto that line, so the run refuses
    before any live open, names the byte to truncate the journal at, and
    writes nothing.
    """
    from phasesweep.engine import IncompleteJournalRecordError, ProcessCleanupUncertainError

    experiment, ledger = _journal_experiment(tmp_path, published=published)
    original = ledger.read_bytes()
    complete = _status_phases(experiment, read_status(experiment))
    tail = {
        "garbage": b"not a journal record\n",
        "partial-json": b"{",
        "missing-newline": original.rstrip(b"\n").split(b"\n")[-1],
    }[damage]
    prefix = original if keep_prefix else b""
    ledger.write_bytes(prefix + tail)
    root = tmp_path / "runs" / "t"
    root.mkdir(parents=True, exist_ok=True)
    before = tree_snapshot(root)

    for phase, undamaged in zip(
        _status_phases(experiment, read_status(experiment)), complete, strict=True
    ):
        assert phase["trial_data_available"] is True
        if keep_prefix:
            assert phase["trials"] == undamaged["trials"]
            assert phase["published_study_unavailable"] is False
        else:
            # Nothing before the bad line: Optuna reads no study at all.
            assert not any(phase["trials"].values())
            assert phase["published_study_unavailable"] is published
    loaded = engine_ledger._load_existing_phase_study(experiment, experiment.phases[0])
    assert (loaded is not None) is keep_prefix
    with pytest.raises(ProcessCleanupUncertainError) as refused:
        run_experiment(experiment)

    cause = refused.value.__cause__
    assert isinstance(cause, IncompleteJournalRecordError)
    assert f"truncate -s {len(prefix)} {ledger}" in str(cause)
    assert ledger.read_bytes() == prefix + tail
    assert tree_snapshot(root) == before


@pytest.mark.parametrize("change", ["append", "finish-partial", "truncate"])
def test_journal_status_uses_one_bounded_snapshot_during_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    ledger = tmp_path / "study.journal"
    experiment = _experiment(tmp_path, storage=f"journal:///{ledger}")
    study = optuna.create_study(
        study_name="read_t::p", storage=engine_ledger._resolve_storage(experiment.resolved_storage)
    )
    # Stamped first, as the engine does, so the journal's last record is the
    # trial's completion: a partial copy of it reads as a RUNNING trial.
    mark_current_format(experiment, study)
    trial = study.ask()
    trial.set_user_attr("phasesweep_generation_id", "generation")
    trial.set_user_attr("phasesweep_attempt_id", "attempt")
    study.tell(trial, 0.5)
    complete = ledger.read_bytes()
    expected = engine_optuna._TrialRef(0, "generation", "attempt")
    # The handle's format scan also captures the journal. Take it on the whole
    # journal, before the damage and the patch, so the handle is verified and
    # the one capture the file changes under is the phase snapshot's. A handle
    # scanned over the partial record would report every read unavailable.
    handle = engine_ledger.validate_ledger(experiment)
    assert handle.format_verified
    if change == "finish-partial":
        ledger.write_bytes(complete[:-1])
    real_fstat = engine_ledger.os.fstat

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
        patched.setattr(engine_ledger.os, "fstat", change_after_capture)
        first = engine_ledger.read_phase_trial_stats(handle, experiment.phases[0], expected)
    second = engine_ledger.read_phase_trial_stats(handle, experiment.phases[0], expected)

    # The first read sees only the bytes present when it opened the file: all
    # of them, the journal less its final newline (a partial last record,
    # skipped as Optuna skips it), or a file that shrank under it.
    assert (
        first.available,
        first.counts,
        first.running_attempts,
        first.published_trial_available,
    ) == {
        "append": (True, {"COMPLETE": 1}, [], True),
        "finish-partial": (True, {"RUNNING": 1}, [expected], False),
        "truncate": (False, {}, None, False),
    }[change]
    # The second read sees the finished change; a partial record appended
    # after the complete trial is skipped the same way.
    assert second.available is True
    assert second.published_trial_available is (change != "truncate")


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sqlite", "journal"])
@pytest.mark.parametrize(
    "damage", ["missing-ledger", "missing-study", "empty-study", "corrupt", "stale-ledger"]
)
@pytest.mark.integration
def test_published_status_distinguishes_absent_history_from_read_failure(
    tmp_path: Path, n_jobs: int, damage: str
) -> None:
    """The same status flags select the same remedy for both file backends."""
    from phasesweep.engine import (
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
        storage = engine_ledger._resolve_storage(experiment.resolved_storage)
        optuna.delete_study(study_name="t::p", storage=storage)
        if damage == "empty-study":
            mark_current_format(experiment, optuna.create_study(study_name="t::p", storage=storage))
    ledger_before = ledger.read_bytes() if ledger.exists() else None
    generation_before = _generation_path(experiment).read_bytes()

    status = read_status(experiment)
    cli_status = experiment_status(experiment)
    mcp_status = status_payload(
        experiment_id="t",
        status=status,
        run={"run_id": "mcp-run"},
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


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
@pytest.mark.parametrize("mismatch", [None, "number", "generation", "attempt", "state"])
def test_published_trial_status_requires_the_exact_completed_attempt(
    tmp_path: Path, backend: str, mismatch: str | None
) -> None:
    experiment = _experiment(tmp_path, storage=f"{backend}:///{tmp_path / 'ledger'}")
    study = optuna.create_study(
        study_name="read_t::p", storage=engine_ledger._resolve_storage(experiment.resolved_storage)
    )
    trial = study.ask()
    trial.set_user_attr("phasesweep_generation_id", "generation")
    trial.set_user_attr("phasesweep_attempt_id", "attempt")
    if mismatch != "state":
        study.tell(trial, 0.5)
    mark_current_format(experiment, study)
    expected = engine_optuna._TrialRef(
        1 if mismatch == "number" else 0,
        "different" if mismatch == "generation" else "generation",
        "different" if mismatch == "attempt" else "attempt",
    )

    stats = engine_ledger.read_phase_trial_stats(
        engine_ledger.validate_ledger(experiment), experiment.phases[0], expected
    )

    assert stats.available
    assert stats.published_trial_available is (mismatch is None)
    assert stats.counts == {"RUNNING" if mismatch == "state" else "COMPLETE": 1}


def test_read_status_untouched_workdir_is_not_published(tmp_path: Path) -> None:
    """No generation metadata *and* no legacy winner is still "nothing published"."""
    exp = _experiment(tmp_path)

    status = read_status(exp)

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
            WandbExtractor(type="wandb", entity="e", project="p", metric_key="eval/loss"),
            (False, False, True),
            id="wandb",
        ),
        pytest.param(
            JsonExtractor(type="json", path="result.json", key="eval.loss"),
            (True, False, False),
            id="json",
        ),
    ],
)
def test_objective_evidence_assurance_attempt_triple_by_kind(
    tmp_path: Path,
    extractor: JsonEnvelopeExtractor | JsonExtractor | LogRegexExtractor | WandbExtractor,
    expected_triple: tuple[bool, bool, bool],
) -> None:
    """Each extractor kind reports its own (location, identity, source-key) triple.

    ``json_envelope`` structurally echoes and cross-checks the attempt
    identity; ``log_regex`` is merely read from an attempt-scoped location
    with nothing in its contents tying it to that attempt.
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


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
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
