"""Terminal result snapshots: how the runner captures, finalizes, and freezes a run's results."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import optuna
import pytest
import yaml

import phasesweep.engine.generation as generation_ops
import phasesweep.engine.ledger as engine_ledger
import phasesweep.engine.optuna as engine_optuna
from phasesweep.config.io import load_experiment
from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.errors import StudyStorageUnavailableError
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.locking import _experiment_lock
from phasesweep.engine.paths import (
    _experiment_dir,
    _generation_path,
    _generation_summary_path,
    _generations_dir,
    _last_successful_generation_path,
    _summary_path,
    _winner_path,
)
from phasesweep.engine.repeats import RepeatMember, spread
from phasesweep.engine.run import TerminalReport, run_experiment
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    TRIAL_OUTCOME_ATTR,
    TRIAL_OUTCOME_SCHEMA_VERSION,
    Winner,
    WinnerSource,
)
from phasesweep.mcp import runner as mcp_runner
from phasesweep.mcp.snapshots import RunResultSnapshot
from tests.conftest import (
    make_experiment,
    mark_current_format,
    single_trial_confound,
    write_constant_trainer,
    write_trainer,
)
from tests.ledger_fixtures import published_generation_id
from tests.mcp_helpers import SEEDED_RANDOM, write_slow_mcp_config

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the detached runner relies on POSIX process groups + /proc liveness",
)


def test_runner_persists_terminal_evidence_before_snapshot_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_path = tmp_path / "status.json"
    observed: dict[str, object] = {}

    def fail_finalization(*_args: object, **_kwargs: object) -> dict[str, object]:
        observed.update(json.loads(status_path.read_text()))
        raise RuntimeError("snapshot serialization failed")

    monkeypatch.setattr(mcp_runner, "finalize_result_snapshot", fail_finalization)
    mcp_runner._write_status(
        status_path,
        {
            "run_id": "r0",
            "returncode": 143,
            "error_class": "cancelled",
            "cleanup_confirmed": True,
        },
        result_snapshot={},
        result_snapshot_error=None,
    )

    assert observed["error_class"] == "cancelled"
    assert observed["cleanup_confirmed"] is True
    assert "ended_at" in observed
    assert observed["result_snapshot_state"] == "pending"
    assert observed["result_snapshot"] == {}
    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "failed"
    assert final["result_snapshot_error"] == "RuntimeError"
    assert "result_snapshot" not in final


@pytest.mark.integration
def test_runner_defers_shutdown_until_snapshot_finalization_is_durable(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    child_code = """
import sys
import time
from pathlib import Path

import phasesweep.mcp.runner as runner

runner.install_signal_handlers()

def slow_finalization(snapshot):
    time.sleep(0.5)
    return snapshot

runner.finalize_result_snapshot = slow_finalization
runner._write_status(
    Path(sys.argv[1]),
    {
        "run_id": "r-signal",
        "returncode": 0,
        "error_class": None,
        "cleanup_confirmed": True,
    },
    result_snapshot={"captured": True},
    result_snapshot_error=None,
)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code, str(status_path)],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if status_path.is_file():
                status = json.loads(status_path.read_text())
                if status.get("result_snapshot_state") == "pending":
                    break
            if proc.poll() is not None:
                raise AssertionError(f"finalizer exited early with {proc.returncode}")
            time.sleep(0.01)
        else:
            raise AssertionError("runner did not persist the pending snapshot state")

        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)

    assert proc.returncode == 143
    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "complete"
    assert final["result_snapshot"] == {"captured": True}


def test_runner_finalizes_pre_captured_terminal_snapshot(tmp_path: Path) -> None:
    config = write_slow_mcp_config(tmp_path)
    status_path = tmp_path / "status.json"
    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)
    snapshot = mcp_runner.capture_result_snapshot(experiment)

    mcp_runner._write_status(
        status_path,
        {
            "run_id": "r0",
            "returncode": 0,
            "error_class": None,
            "cleanup_confirmed": True,
        },
        result_snapshot=snapshot,
        result_snapshot_error=None,
    )

    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "complete"
    assert final["result_snapshot"]["status"]["phases"][0]["phase"] == "p"


@pytest.mark.integration
def test_terminal_snapshot_is_captured_before_experiment_lock_release(tmp_path: Path) -> None:
    experiment = make_experiment(
        persistent=tmp_path, trainer=write_constant_trainer(tmp_path), n_trials=1
    )
    captured: dict[str, object] = {}

    def capture_locked(report: TerminalReport) -> None:
        assert report.primary_error is None
        with (
            pytest.raises(RuntimeError, match="Another phasesweep process"),
            _experiment_lock(experiment),
        ):
            pass
        captured.update(
            mcp_runner.capture_result_snapshot(
                experiment,
                generation_id=report.generation_id,
            )
        )

    run_experiment(experiment, terminal_callback=capture_locked)

    top_up = experiment.model_copy(
        update={"phases": [experiment.phases[0].model_copy(update={"n_trials": 2})]}
    )
    run_experiment(top_up)

    captured_phase = captured["status"]["phases"][0]  # type: ignore[index]
    assert captured_phase["n_trials"] == 1
    assert captured_phase["completed"] == 1
    assert captured_phase["generation_trials"] == {"COMPLETE": 1}
    current_phase = mcp_runner.capture_result_snapshot(top_up)["status"]["phases"][0]
    assert current_phase["n_trials"] == 2
    assert current_phase["completed"] == 2
    assert current_phase["generation_trials"] == {"COMPLETE": 1}


@pytest.mark.integration
def test_terminal_snapshot_reads_partial_winners_from_failed_generation(tmp_path: Path) -> None:
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import os
        import sys

        if os.environ["PHASESWEEP_PHASE"] == "a":
            print("x=0.5")
        else:
            sys.exit(2)
        """,
    )
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'studies.journal'}",
        trial_command=f"{sys.executable} {trainer} {{overrides}}",
        phases=[
            Phase(name="a", n_trials=1, sampler=SEEDED_RANDOM, search_space={}),
            Phase(
                name="b",
                n_trials=1,
                max_consecutive_failures=1,
                sampler=SEEDED_RANDOM,
                search_space={},
            ),
        ],
    )
    captured: dict[str, object] = {}

    def capture_failed(report: TerminalReport) -> None:
        assert isinstance(report.primary_error, NoFeasibleTrialError)
        captured.update(
            mcp_runner.capture_result_snapshot(
                experiment,
                generation_id=report.generation_id,
            )
        )

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment, terminal_callback=capture_failed)

    phases = captured["status"]["phases"]  # type: ignore[index]
    assert phases[0]["winner_present"] is True
    assert phases[1]["winner_present"] is False
    (winner,) = captured["winners"]  # type: ignore[index]
    assert winner["phase"] == "a"
    assert winner["source"] == {
        "kind": "phase_trial",
        "phase": "a",
        "trial_number": winner["trial_number"],
        "generation_id": winner["generation_id"],
        "attempt_id": winner["attempt_id"],
    }


def test_terminal_snapshot_tolerates_missing_lifecycle_record(tmp_path: Path) -> None:
    """A missing per-generation record must not fail the snapshot capture.

    The engine writes that record as an optional post-commit diagnostic; a
    capture that hard-required it would contradict engine-defined success
    whenever the best-effort write had failed.
    """
    experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1)
    mark_current_format(experiment)
    generation_path = _generation_path(experiment)
    generation_path.parent.mkdir(parents=True, exist_ok=True)
    generation_path.write_text("generation_id: prior-generation\n")

    snapshot = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id="pinned-generation",
    )

    assert snapshot["status"]["represented_generation_id"] == "pinned-generation"
    assert snapshot["status"]["is_published"] is False


def test_snapshot_finalization_keeps_prior_attempt_out_of_generation_counts(
    tmp_path: Path,
) -> None:
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'studies.journal'}",
        phases=[Phase(name="p", n_trials=1, sampler=SEEDED_RANDOM, search_space={})],
    )
    journal = _resolve_storage(experiment.resolved_storage)
    study = optuna.create_study(study_name="t::p", storage=journal, direction="minimize")
    mark_current_format(experiment, study)
    trial = study.ask()
    trial.set_user_attr("phasesweep_generation_id", "old-generation")
    trial.set_user_attr("phasesweep_attempt_id", "old-attempt")

    captured = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id="current-generation",
    )
    unowned = mcp_runner.finalize_result_snapshot(captured)

    assert unowned == captured
    assert unowned["status"]["phases"][0]["trials"]["RUNNING"] == 1

    finalized = mcp_runner.finalize_result_snapshot(
        captured,
        confirmed_attempt_ids={"old-attempt"},
    )

    phase = finalized["status"]["phases"][0]
    assert phase["trials"]["RUNNING"] == 0
    assert phase["trials"]["INTERRUPTED"] == 1
    assert phase["generation_trials"] == {}
    assert study.get_trials(deepcopy=False)[0].state == optuna.trial.TrialState.RUNNING


@pytest.mark.parametrize("storage_kind", ["none", "corrupt-journal"])
def test_terminal_snapshot_freezes_unavailable_trial_data_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage_kind: str,
) -> None:
    """Unavailable trial storage freezes explicit flags without a redundant study read.

    ``running_attempts`` is ``None`` rather than ``[]`` here: nothing was read,
    so an empty list would claim there are no RUNNING rows.
    """
    storage = None
    if storage_kind == "corrupt-journal":
        ledger = tmp_path / "studies.journal"
        # A bad line that another line follows is corruption, not a final record.
        ledger.write_bytes(b"not a journal record\n" * 2)
        storage = f"journal:///{ledger}"
    experiment = make_experiment(workdir=tmp_path / "runs", storage=storage, n_trials=1)

    def fail_redundant_read(*args: object, **kwargs: object) -> None:
        raise OSError("storage remains unavailable")

    # Patched on Optuna itself, so the guard holds no matter how PhaseSweep
    # imports its own study helpers: any study load at all fails the capture.
    monkeypatch.setattr(optuna, "load_study", fail_redundant_read)

    if storage_kind == "corrupt-journal":
        with pytest.raises(StudyStorageUnavailableError, match="could not be completely read"):
            mcp_runner.capture_result_snapshot(experiment)
        assert not _experiment_dir(experiment).exists()
        return

    snapshot = mcp_runner.capture_result_snapshot(experiment)

    phase = snapshot["status"]["phases"][0]
    assert phase["trial_data_available"] is False
    assert phase["running_attempts"] is None


@pytest.mark.parametrize("pointer_commits", [True, False], ids=["commit", "abort"])
@pytest.mark.integration
def test_publication_snapshot_rebinds_unavailable_trial_data_only_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pointer_commits: bool,
) -> None:
    """A pre-commit storage failure describes the new publication only after commit."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'studies.journal'}",
        trial_command="echo x=0.5 {overrides}",
        n_trials=1,
    )
    generation_id = f"storage-unavailable-{'commit' if pointer_commits else 'abort'}"
    capture_reads = 0
    original_stats = engine_ledger._trial_stats

    def transient_capture_failure(
        ledger: engine_ledger.ValidatedLedger,
        published_trials: Mapping[str, engine_optuna._TrialRef | None],
    ) -> dict[str, engine_optuna._PhaseTrialStats]:
        nonlocal capture_reads
        if _generation_summary_path(ledger.experiment, generation_id).is_file():
            capture_reads += 1

            def locked_read(*_args: object, **_kwargs: object) -> None:
                raise OSError("journal storage is locked")

            with monkeypatch.context() as capture_patch:
                capture_patch.setattr(engine_ledger, "_journal_snapshot_storage", locked_read)
                return original_stats(ledger, published_trials)

        return original_stats(ledger, published_trials)

    monkeypatch.setattr(engine_ledger, "_trial_stats", transient_capture_failure)
    if not pointer_commits:
        pointer_path = _last_successful_generation_path(experiment)
        original_write = generation_ops.artifact_io._write_yaml_atomic

        def fail_pointer_write(path: Path, payload: object) -> None:
            if path == pointer_path:
                raise OSError("simulated publication abort")
            original_write(path, payload)

        monkeypatch.setattr(generation_ops.artifact_io, "_write_yaml_atomic", fail_pointer_write)

    hook = mcp_runner._RunnerPublicationHook(
        tmp_path / "status.json",
        {
            "run_id": generation_id,
            "returncode": 0,
            "error_class": None,
            "cleanup_confirmed": True,
            "failure": None,
        },
    )
    if pointer_commits:
        run_experiment(experiment, generation_id=generation_id, publication_hook=hook)
    else:
        with pytest.raises(OSError, match="simulated publication abort"):
            run_experiment(experiment, generation_id=generation_id, publication_hook=hook)

    assert capture_reads == 1
    assert hook.snapshot is not None
    phase = hook.snapshot["status"]["phases"][0]
    assert phase["trial_data_available"] is False
    assert phase["running_attempts"] is None
    assert phase["published_study_unavailable"] is pointer_commits
    assert hook.snapshot["status"]["is_published"] is pointer_commits


def test_published_snapshot_rebinds_selected_phase_flags_from_summary(tmp_path: Path) -> None:
    """Commit covers winners, carried flags, and omitted phases."""
    phase_names = ("carried", "new", "old-only")
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        phases=[
            Phase(name=name, n_trials=1, sampler=SEEDED_RANDOM, search_space={})
            for name in phase_names
        ],
    )
    generation_id = "candidate-generation"
    mark_current_format(experiment)
    published_summary = {
        "experiment": experiment.experiment,
        "generation_id": generation_id,
        "phases": [
            {
                "name": "carried",
                "trial_number": 2,
                "generation_id": "prior-generation",
                "attempt_id": "carried-attempt",
            },
            {
                "name": "new",
                "trial_number": 0,
                "generation_id": generation_id,
                "attempt_id": "new-attempt",
            },
        ],
    }
    summary_path = _generation_summary_path(experiment, generation_id)
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(yaml.safe_dump(published_summary))
    snapshot = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id=generation_id,
        engine_winners={},
    )
    phases = {phase["phase"]: phase for phase in snapshot["status"]["phases"]}
    phases["carried"]["trial_data_available"] = True
    phases["carried"]["running_attempts"] = []
    phases["carried"]["published_study_unavailable"] = True
    phases["old-only"]["published_study_unavailable"] = True
    committed = mcp_runner.mark_result_snapshot_published(
        snapshot,
        generation_id=generation_id,
        published_summary=published_summary,
    )

    committed_phases = {phase["phase"]: phase for phase in committed["status"]["phases"]}
    assert committed_phases["new"]["published_study_unavailable"] is True
    assert committed_phases["carried"]["published_study_unavailable"] is True
    assert committed_phases["old-only"]["published_study_unavailable"] is False


@pytest.mark.integration
def test_terminal_snapshot_survives_a_post_engine_study_load_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful run's result is frozen even if studies became unloadable.

    The capture collects RUNNING rows from the tolerant ``read_status`` read
    instead of rereading every phase study afterwards. A reread would have no
    tolerance for failure: one transient lock in that millisecond-wide window
    would fail the capture, and a terminal snapshot that was never captured is
    unrecoverable by design -- so a completed run's frozen result would be
    lost to storage flakiness that the tolerant status read itself shrugs off.

    The one tolerant read replays a journal snapshot through ``optuna.load_study``,
    so exactly one load succeeds and any reload after it fails.
    """
    experiment = make_experiment(
        persistent=tmp_path, trainer=write_constant_trainer(tmp_path), n_trials=1
    )
    reports: list[TerminalReport] = []

    run_experiment(experiment, terminal_callback=reports.append)

    (report,) = reports
    assert report.primary_error is None
    assert report.winners

    loads = 0
    real_load_study = optuna.load_study

    def fail_on_reload(*args: object, **kwargs: object) -> optuna.Study:
        nonlocal loads
        loads += 1
        if loads > 1:
            raise StudyStorageUnavailableError("database is locked")
        return real_load_study(*args, **kwargs)

    # Patched on Optuna itself rather than on a PhaseSweep helper, so no
    # import binding can hide a study reload from this test. Every remaining
    # storage read must be the tolerant one read_status performs.
    monkeypatch.setattr(optuna, "load_study", fail_on_reload)

    snapshot = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id=report.generation_id,
        engine_winners=report.winners,
    )

    assert loads == 1
    phase = snapshot["status"]["phases"][0]
    assert phase["trial_data_available"] is True
    assert phase["running_attempts"] == []
    assert phase["completed"] == 1
    (frozen,) = snapshot["winners"]
    assert frozen["phase"] == "p"
    assert frozen["metric"] == report.winners["p"].metric
    assert frozen["trial_number"] == report.winners["p"].trial_number

    status_path = tmp_path / "status.json"
    mcp_runner._write_status(
        status_path,
        {"run_id": "r0", "returncode": 0, "error_class": None, "cleanup_confirmed": True},
        result_snapshot=snapshot,
        result_snapshot_error=None,
    )

    assert json.loads(status_path.read_text())["result_snapshot_state"] == "complete"


def test_terminal_snapshot_reports_running_attempts_from_the_status_read(
    tmp_path: Path,
) -> None:
    """RUNNING identities come from the one tolerant read, unchanged in shape."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'studies.journal'}",
        phases=[
            Phase(
                name="p",
                n_trials=2,
                sampler=SEEDED_RANDOM,
                search_space={"x": {"type": "int", "low": 0, "high": 9}},
            )
        ],
    )
    journal = _resolve_storage(experiment.resolved_storage)
    study = optuna.create_study(study_name="t::p", storage=journal, direction="minimize")
    mark_current_format(experiment, study)
    identified = study.ask()
    identified.set_user_attr("phasesweep_generation_id", "current-generation")
    identified.set_user_attr("phasesweep_attempt_id", "current-attempt")
    # Its objective recorded a failure before the orchestrator died, so recovery keeps it.
    identified.set_user_attr(
        TRIAL_OUTCOME_ATTR,
        {"schema_version": TRIAL_OUTCOME_SCHEMA_VERSION, "sequence": 1, "outcome": "failure"},
    )
    # A RUNNING row written before attempt identity existed: reported with
    # null identity, never dropped, so the counts and the list agree. It
    # recorded nothing, so recovery records it as interrupted.
    study.ask()

    snapshot = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id="current-generation",
    )

    phase = snapshot["status"]["phases"][0]
    assert phase["trials"]["RUNNING"] == 2
    assert phase["trial_data_available"] is True
    assert phase["running_attempts"] == [
        {
            "trial_number": 0,
            "generation_id": "current-generation",
            "attempt_id": "current-attempt",
            "recovers_as": "FAIL",
        },
        {
            "trial_number": 1,
            "generation_id": None,
            "attempt_id": None,
            "recovers_as": "INTERRUPTED",
        },
    ]

    finalized = mcp_runner.finalize_result_snapshot(
        snapshot,
        confirmed_attempt_ids={"current-attempt"},
    )

    finalized_phase = finalized["status"]["phases"][0]
    assert finalized_phase["trials"] == {"RUNNING": 1, "FAIL": 1}
    assert finalized_phase["generation_trials"] == {"RUNNING": 0, "FAIL": 1}
    assert finalized_phase["running_attempts"] == [
        {"trial_number": 1, "generation_id": None, "attempt_id": None, "recovers_as": "INTERRUPTED"}
    ]

    fully_finalized = mcp_runner.finalize_result_snapshot(
        snapshot,
        confirmed_attempt_ids={"current-attempt", "allocated-attempt"},
        confirmed_attempt_locations={
            "allocated-attempt": ("p", 1, "prior-generation"),
        },
    )

    fully_finalized_phase = fully_finalized["status"]["phases"][0]
    assert fully_finalized_phase["trials"] == {"RUNNING": 0, "FAIL": 1, "INTERRUPTED": 1}
    assert fully_finalized_phase["generation_trials"] == {"RUNNING": 0, "FAIL": 1}
    assert fully_finalized_phase["running_attempts"] == []


def test_snapshot_freezes_engine_winners_without_rereading_files(tmp_path: Path) -> None:
    """Engine-supplied winners are frozen verbatim, with no second file read."""
    experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1)
    winner = Winner(
        trial_number=4,
        params={"x": 3},
        effective_overrides={"x": 3},
        metric=0.25,
        gates=[{"type": "g", "passed": True}],
        completion={"incomplete": False},
        generation_id="engine-generation",
        attempt_id="engine-attempt",
        source=WinnerSource(
            kind="phase_trial",
            phase="p",
            trial_number=4,
            generation_id="engine-generation",
            attempt_id="engine-attempt",
        ),
        confound=single_trial_confound(4),
    )

    snapshot = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id="engine-generation",
        engine_winners={"p": winner},
    )

    # No winner file exists anywhere on disk; the frozen winner is exactly
    # the engine's own in-memory outcome.
    (frozen,) = snapshot["winners"]
    assert frozen["phase"] == "p"
    assert frozen["trial_number"] == 4
    assert frozen["metric"] == 0.25
    assert frozen["gates_passed"] is True
    assert frozen["incomplete"] is False
    assert frozen["generation_id"] == "engine-generation"
    assert frozen["source"] == {
        "kind": "phase_trial",
        "phase": "p",
        "trial_number": 4,
        "generation_id": "engine-generation",
        "attempt_id": "engine-attempt",
    }


def test_snapshot_freezes_observed_values(tmp_path: Path) -> None:
    """A winner's report-only values are frozen with it and come back out of the stored view."""
    experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1)
    winner = Winner(
        trial_number=0,
        params={"x": 3},
        effective_overrides={"x": 3},
        metric=0.25,
        completion={"incomplete": False},
        generation_id="engine-generation",
        attempt_id="engine-attempt",
        source=WinnerSource(
            kind="phase_trial",
            phase="p",
            trial_number=0,
            generation_id="engine-generation",
            attempt_id="engine-attempt",
        ),
        confound=single_trial_confound(0),
        observed={"peak": 7.5, "steps": 3.0},
    )

    snapshot = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id="engine-generation",
        engine_winners={"p": winner},
    )

    (frozen,) = snapshot["winners"]
    assert frozen["observed"] == {"peak": 7.5, "steps": 3.0}
    assert frozen["repeats"] is None
    stored = RunResultSnapshot.model_validate(snapshot)
    assert [view.observed for view in stored.winner_views()] == [{"peak": 7.5, "steps": 3.0}]


def test_snapshot_freezes_the_repeats_summary_without_config_strings(tmp_path: Path) -> None:
    """A group winner's spread is frozen; the repeat key and member identities are not."""
    experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1)

    def member(number: int, seed: int, metric: float) -> RepeatMember:
        return {
            "trial_number": number,
            "repeat": seed,
            "metric": metric,
            "generation_id": "engine-generation",
            "attempt_id": f"attempt-{number}",
            "objective_provenance": {"schema_version": 1},
            "trainer_input": {"schema_version": 1},
            "constraints": {},
            "observed": {},
        }

    winner = Winner(
        trial_number=5,
        params={"x": 3},
        effective_overrides={"x": 3},
        metric=0.3,
        completion={"incomplete": False},
        generation_id="engine-generation",
        attempt_id="attempt-5",
        source=WinnerSource(
            kind="phase_trial",
            phase="p",
            trial_number=5,
            generation_id="engine-generation",
            attempt_id="attempt-5",
        ),
        confound=single_trial_confound(5),
        repeats={
            "key": "trainer.secret_seed_name",
            "reduce": "median",
            "spread": spread([0.2, 0.3, 0.7]),
            "members": [member(3, 0, 0.2), member(4, 1, 0.3), member(5, 2, 0.7)],
        },
    )

    snapshot = mcp_runner.capture_result_snapshot(
        experiment,
        generation_id="engine-generation",
        engine_winners={"p": winner},
    )

    (frozen,) = snapshot["winners"]
    expected = {
        "reduce": "median",
        "n": 3,
        "mean": pytest.approx(0.4),
        "sd": pytest.approx(statistics.stdev([0.2, 0.3, 0.7])),
        "min": 0.2,
        "max": 0.7,
        "member_trials": [3, 4, 5],
    }
    assert frozen["repeats"] == expected
    assert "secret_seed_name" not in json.dumps(snapshot)
    (view,) = RunResultSnapshot.model_validate(snapshot).winner_views()
    assert view.repeats == expected


@pytest.mark.parametrize("from_phase", [None, "b"])
@pytest.mark.integration
def test_failed_fingerprint_preflight_preserves_published_results(
    tmp_path: Path,
    from_phase: str | None,
) -> None:
    """A failed preflight advances the current pointer but never the published one.

    The current-generation pointer legitimately moves to this new (failed)
    invocation -- a new invocation always overwrites it starting from
    "preflighting", and every outcome path must drive it to a terminal state
    -- but the convenience root projections and, critically, the last-success
    pointer stay exactly as the prior successful publication left them.
    """
    phases = [
        Phase(
            name="a", n_trials=1, fixed_overrides={"k": 1}, sampler=SEEDED_RANDOM, search_space={}
        ),
        Phase(name="b", n_trials=1, inherits=["a"], sampler=SEEDED_RANDOM, search_space={}),
    ]
    experiment = make_experiment(
        persistent=tmp_path, trainer=write_constant_trainer(tmp_path), phases=phases
    )
    run_experiment(experiment)
    first_generation = published_generation_id(experiment)
    assert first_generation is not None

    # The convenience root projections must stay untouched by a failed resume;
    # the current-generation pointer is deliberately excluded here since it
    # legitimately advances even on a preflight failure (see docstring).
    protected_paths = [
        _summary_path(experiment),
        *(_winner_path(experiment, phase.name) for phase in phases),
    ]
    before = {path: path.read_bytes() for path in protected_paths}
    generations_before = set(_generations_dir(experiment).iterdir())
    changed = experiment.model_copy(
        update={
            "phases": [
                phases[0].model_copy(update={"fixed_overrides": {"k": 2}}),
                phases[1],
            ]
        }
    )
    callback_generations: list[str] = []
    captured: list[dict[str, object]] = []

    def capture_failed_resume(report: TerminalReport) -> None:
        assert isinstance(report.primary_error, RuntimeError)
        callback_generations.append(report.generation_id)
        captured.append(
            mcp_runner.capture_result_snapshot(
                changed,
                generation_id=report.generation_id,
            )
        )

    with pytest.raises(RuntimeError, match="different phase config"):
        run_experiment(
            changed,
            from_phase=from_phase,
            terminal_callback=capture_failed_resume,
        )

    assert len(callback_generations) == 1
    assert len(captured) == 1
    # current_generation_id is always the actual mutable pointer, which this
    # failed preflight attempt does legitimately claim (a new invocation
    # overwrites it with "preflighting" before it knows whether preflight
    # will succeed). published_generation_id is always the actual validated
    # last-success pointer -- the *first* (successful) run here -- never
    # forced to equal the pinned generation_id, even though this pinned
    # capture's own represented_generation_id is that failed generation.
    assert captured[0]["status"]["current_generation_id"] == callback_generations[0]  # type: ignore[index]
    assert captured[0]["status"]["published_generation_id"] == first_generation  # type: ignore[index]
    assert captured[0]["status"]["represented_generation_id"] == callback_generations[0]  # type: ignore[index]
    assert captured[0]["status"]["is_published"] is False  # type: ignore[index]
    assert captured[0]["status"]["summary_present"] is False  # type: ignore[index]
    assert all(
        phase["generation_trials"] == {}
        for phase in captured[0]["status"]["phases"]  # type: ignore[index]
    )
    assert captured[0]["winners"] == []
    # Convenience root projections are untouched, and the published pointer
    # still resolves to the prior successful generation.
    assert {path: path.read_bytes() for path in protected_paths} == before
    assert published_generation_id(experiment) == first_generation
    failed_generations = set(_generations_dir(experiment).iterdir()) - generations_before
    assert len(failed_generations) == 1
    assert "state: failed" in (failed_generations.pop() / "generation.yaml").read_text()
    # The current pointer legitimately advanced to this failed attempt.
    current_pointer = yaml.safe_load(_generation_path(experiment).read_text())
    assert current_pointer["generation_id"] == callback_generations[0]
    assert current_pointer["state"] == "failed"
