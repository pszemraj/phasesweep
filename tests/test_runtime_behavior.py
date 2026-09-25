"""Runtime semantics: NaN/inf propagation through extractors, parallel trial behavior, sampler configuration at study-creation time, max_consecutive_failures abort, and Optuna logging suppression."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import optuna
import pytest
import yaml
from pydantic import ValidationError

from phasesweep import run_experiment
from phasesweep.config import (
    CategoricalParam,
    Constraint,
    Experiment,
    FloatParam,
    IntParam,
    JsonEnvelopeExtractor,
    JsonExtractor,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
    WandbSummaryRequiredGate,
)
from phasesweep.engine import (
    PhaseSweepError,
    ProcessCleanupUncertainError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TerminalReport,
    read_status,
    read_winner,
)
from phasesweep.engine.ledger import (
    _resolve_storage,
    claim_ledger,
    open_phase_study,
    validate_ledger,
)
from phasesweep.engine.optuna import _build_sampler, _suggest
from phasesweep.engine.paths import (
    _attempts_dir,
    _generation_path,
    _last_successful_generation_path,
    _summary_path,
    _winner_path,
)
from phasesweep.engine.phase import CsvSnapshotThrottle
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    PHASE_ABORT_ATTR,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_ENV_NAMES_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_TARGET_ATTR,
)
from phasesweep.engine.study_policy import _load_phase_policy_state
from phasesweep.engine.trial import ExecutedTrial, TrialExecutionError, extract_trial_result
from phasesweep.evidence import TrialContext
from phasesweep.runtime.process import ProcessResult, write_attempt_lifecycle
from phasesweep.runtime.reaper import PROCESS_IDENTITY_FILE
from tests.conftest import (
    copy_fake_train,
    make_experiment,
    make_trial_context,
    patch_rejected_trial_user_attr,
    requires_nonroot,
    write_constant_trainer,
    write_flag_gated_trainer,
    write_param_echo_trainer,
    write_trainer,
)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        b"not-json",
        b"\xff",
        b'{"other": 1}',
        b'{"loss": true}',
        b'{"loss": "0.2"}',
        b'{"loss": null}',
        b'{"loss": []}',
        b'{"loss": {}}',
        b'{"loss": NaN}',
        b'{"loss": 1e400}',
        json.dumps({"loss": 10**400}).encode(),
        pytest.param("unreadable", marks=requires_nonroot),
    ],
)
def test_json_primary_invalid_evidence_fails_trial(tmp_path, payload):
    source = tmp_path / "r.json"
    if payload == "unreadable":
        source.write_text('{"loss": 0.2}')
        source.chmod(0)
    elif payload is not None:
        source.write_bytes(payload)
    experiment = make_experiment(
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="loss"))
    )
    executed = ExecutedTrial(
        ctx=make_trial_context(tmp_path),
        process=ProcessResult(return_code=0, timed_out=False, pid=123, duration_seconds=0.1),
    )
    try:
        result = extract_trial_result(experiment=experiment, executed=executed)
        assert result.metric is None
        assert not result.feasible
        assert result.failure_reason.startswith("metric extractor")
    finally:
        if source.exists():
            source.chmod(0o600)


@pytest.mark.parametrize(("value", "state"), [(3, "COMPLETE"), (10**400, "FAIL")])
@pytest.mark.integration
def test_invalid_remote_constraint_fails_but_measured_violation_is_complete(
    tmp_path,
    wandb_worker_sdk,
    value,
    state,
):
    from phasesweep.config import WandbExtractor

    wandb_worker_sdk(f"""
        class Api:
            def __init__(self, **kwargs): pass
            def run(self, path):
                return type("Run", (), {{"state": "finished", "summary_metrics": {{"memory": {value!r}}}}})()
    """)
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'study.journal'}",
        trial_command="echo {overrides} x=0.25",
        n_trials=1,
        constraints=[
            Constraint(
                name="memory",
                max=1,
                extractor=WandbExtractor(
                    type="wandb", entity="e", project="p", metric_key="memory", timeout_seconds=5
                ),
            )
        ],
    )
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)
    trial = optuna.load_study(
        study_name="t::p", storage=_resolve_storage(experiment.resolved_storage)
    ).trials[0]
    assert trial.state.name == state
    if state == "COMPLETE":
        assert trial.value == 0.25
    else:
        assert "numeric evidence" in json.dumps(trial.user_attrs)


def test_run_experiment_revalidates_programmatically_copied_config(tmp_path: Path) -> None:
    """The library boundary rejects model_copy updates that skipped validation."""
    experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1)
    phase = experiment.phases[0].model_copy(update={"n_trials": True})
    copied = experiment.model_copy(update={"phases": [phase]})

    with pytest.raises(ValidationError, match="numbers, not booleans"):
        run_experiment(copied)

    assert not (tmp_path / "runs").exists()


def test_csv_snapshot_throttle_debounces_full_rewrites() -> None:
    throttle = CsvSnapshotThrottle(min_trials=10, min_seconds=30.0)

    assert throttle.should_write(finished=1, now=100.0)
    throttle.mark_written(finished=1, now=100.0)
    assert not throttle.should_write(finished=9, now=120.0)
    assert throttle.should_write(finished=11, now=120.0)
    throttle.mark_written(finished=11, now=120.0)
    assert not throttle.should_write(finished=12, now=149.9)
    assert throttle.should_write(finished=12, now=150.0)


@pytest.mark.integration
def test_seeded_random_sequence_is_stable_across_top_up_batches(tmp_path: Path) -> None:
    """Seeded random draws depend on durable trial identity, not process lifetime."""
    phase = Phase(
        name="p",
        n_trials=4,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": CategoricalParam(type="categorical", choices=list(range(100)))},
    )

    def sampled(storage: Path, batches: list[int]) -> list[int]:
        resolved = _resolve_storage(f"journal:///{storage}")
        for n_trials in batches:
            study = optuna.create_study(
                study_name="stable-random::p",
                storage=resolved,
                sampler=_build_sampler(phase.sampler, phase.search_space),
                load_if_exists=True,
            )
            study.optimize(
                lambda trial: float(trial.suggest_categorical("x", list(range(100)))),
                n_trials=n_trials,
            )
        loaded = optuna.load_study(
            study_name="stable-random::p",
            storage=resolved,
        )
        return [int(trial.params["x"]) for trial in loaded.trials]

    expected = sampled(tmp_path / "single.journal", [4])
    assert expected == sampled(tmp_path / "ones.journal", [1, 1, 1, 1])
    assert expected == sampled(tmp_path / "uneven.journal", [1, 3])
    assert expected == sampled(tmp_path / "mixed.journal", [2, 1, 1])


def test_grid_top_up_does_not_repeat_stored_assignments(tmp_path: Path) -> None:
    """A reconstructed GridSampler continues through its stored grid assignments."""
    search_space = {"x": CategoricalParam(type="categorical", choices=[1, 2, 3, 4])}
    sampler = Sampler(type="grid", seed=0)
    storage = _resolve_storage(f"journal:///{tmp_path / 'grid.journal'}")

    for _ in range(2):
        study = optuna.create_study(
            study_name="stable-grid::p",
            storage=storage,
            sampler=_build_sampler(sampler, search_space),
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: float(trial.suggest_categorical("x", [1, 2, 3, 4])), n_trials=2
        )

    loaded = optuna.load_study(study_name="stable-grid::p", storage=storage)
    assert len(loaded.trials) == 4
    assert {trial.params["x"] for trial in loaded.trials} == {1, 2, 3, 4}


@pytest.mark.parametrize(
    ("sampler", "search_space", "n_trials", "expected_type"),
    [
        pytest.param(
            Sampler(type="random", seed=0),
            {"x": CategoricalParam(type="categorical", choices=[1, 2])},
            1,
            "_TrialNumberRandomSampler",
            id="random",
        ),
        pytest.param(
            Sampler(type="grid", seed=0),
            {"x": CategoricalParam(type="categorical", choices=[1, 2])},
            2,
            "GridSampler",
            id="grid",
        ),
        # tpe/cmaes are the point of these two cases, so they carry the
        # acknowledgement persistent storage requires for a non-resumable sampler.
        pytest.param(
            Sampler(type="tpe", seed=0, n_startup_trials=3, acknowledge_nonresumable=True),
            {"x": CategoricalParam(type="categorical", choices=[1, 2])},
            1,
            "TPESampler",
            id="tpe",
        ),
        pytest.param(
            Sampler(type="cmaes", seed=0, acknowledge_nonresumable=True),
            {"x": IntParam(type="int", low=1, high=2)},
            1,
            "CmaEsSampler",
            id="cmaes",
        ),
    ],
)
def test_persistent_execution_reattaches_configured_sampler_and_pruner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sampler: Sampler,
    search_space: dict,
    n_trials: int,
    expected_type: str,
) -> None:
    """A validation-only load cannot supply Optuna defaults to execution."""

    class OptimizeObserved(RuntimeError):
        pass

    phase = Phase(
        name="p",
        n_trials=n_trials,
        n_jobs=2 if sampler.type == "tpe" else 1,
        gpu_policy="none" if sampler.type == "tpe" else "single_per_trial",
        allow_no_gpu_isolation=sampler.type == "tpe",
        allow_partial_grid=sampler.type == "grid",
        sampler=sampler,
        search_space=search_space,
    )
    storage = f"journal:///{tmp_path / f'{sampler.type}.journal'}"
    exp = make_experiment(
        experiment=f"sampler_{sampler.type}",
        storage=storage,
        workdir=tmp_path / "runs",
        phases=[phase],
    )
    study = open_phase_study(claim_ledger(validate_ledger(exp)), phase)
    study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    observed: dict[str, object] = {}

    def inspect_optimize(study: optuna.Study, objective, **kwargs) -> None:
        observed["sampler"] = type(study.sampler).__name__
        observed["pruner"] = type(study.pruner).__name__
        observed["n_startup_trials"] = getattr(study.sampler, "_n_startup_trials", None)
        observed["constant_liar"] = getattr(study.sampler, "_constant_liar", None)
        raise OptimizeObserved

    monkeypatch.setattr(optuna.Study, "optimize", inspect_optimize)
    with pytest.raises(OptimizeObserved):
        run_experiment(exp)

    assert observed["sampler"] == expected_type
    assert observed["pruner"] == "NopPruner"
    if sampler.type == "tpe":
        assert observed["n_startup_trials"] == 3
        assert observed["constant_liar"] is True


@pytest.mark.integration
def test_parallel_trials_e2e(tmp_path):
    """Run a phase with n_jobs=4 on the synthetic trainer. Exercises:
    - JournalFileStorage via explicit journal:/// URL (review v0.5.2 / blocker 6)
    - constant_liar on TPE
    - concurrent subprocess execution
    - no database-locked errors
    """
    trainer = copy_fake_train(tmp_path)

    journal_path = tmp_path / "phases.journal"
    exp = make_experiment(
        experiment="parallel_test",
        workdir=tmp_path / "runs",
        storage=f"journal:///{journal_path}",
        trial_command=f"python {trainer} {{overrides}}",
        metric=Metric(
            name="eval_loss",
            extractor=JsonEnvelopeExtractor(
                type="json_envelope",
                objective_name="eval_loss",
                split="validation",
                policy="synthetic",
            ),
        ),
        name="lr_sweep",
        n_trials=8,
        n_jobs=4,
        allow_no_gpu_isolation=True,
        sampler=Sampler(type="tpe", seed=42, acknowledge_nonresumable=True),
        search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-2, log=True)},
    )
    winners = run_experiment(exp)

    assert "lr_sweep" in winners
    assert 1e-5 <= winners["lr_sweep"].params["lr"] <= 1e-2

    # Verify all 8 trials actually ran (trial directories exist).
    # v0.5.7: outputs are now namespaced as <workdir>/<experiment>/<phase>/.
    runs_dir = tmp_path / "runs" / "parallel_test" / "lr_sweep"
    trial_dirs = sorted(runs_dir.glob("trial_*"))
    assert len(trial_dirs) == 8

    assert journal_path.exists(), "JournalFileStorage file should exist"


@pytest.mark.integration
def test_failed_trials_marked_fail_not_complete(tmp_path):
    """Process crashes should produce FAIL trials, not COMPLETE with inf."""
    db_path = tmp_path / "phases.journal"
    exp = make_experiment(
        experiment="fail_state_test",
        workdir=tmp_path / "runs",
        storage=f"journal:///{db_path}",
        trial_command="false {overrides}",
        name="a",
        n_trials=3,
        max_consecutive_failures=10,
        search_space={"x": FloatParam(type="float", low=0, high=1)},
    )
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(exp)

    study = optuna.load_study(
        study_name="fail_state_test::a", storage=_resolve_storage(f"journal:///{db_path}")
    )
    for trial in study.get_trials():
        # Every trial should be FAIL, not COMPLETE.
        assert trial.state == optuna.trial.TrialState.FAIL, (
            f"Trial {trial.number} is {trial.state.name}, expected FAIL"
        )


@pytest.mark.integration
def test_repeated_in_memory_run_cannot_reuse_stale_trial_and_preserves_last_good_results(
    tmp_path: Path,
) -> None:
    success = write_trainer(
        tmp_path / "success.py",
        """
        import json, pathlib, sys
        pathlib.Path(sys.argv[1]).write_text(json.dumps({"x": 0.123}))
        print("x=0.123")
        """,
    )
    no_result = write_trainer(tmp_path / "no_result.py", "pass")
    workdir = tmp_path / "runs"
    first = make_experiment(
        workdir=workdir,
        trial_command=f"python {success} {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )

    winner = run_experiment(first)["p"]
    assert winner.metric == pytest.approx(0.123)
    phase_dir = workdir / "t" / "p"
    first_trial_dir = next(phase_dir.glob("trial_*"))
    assert (first_trial_dir / "r.json").is_file()
    protected = {
        path: path.read_bytes()
        for path in (
            _winner_path(first, "p"),
            _summary_path(first),
            _last_successful_generation_path(first),
        )
    }

    second = first.model_copy(
        update={"trial_command": f"python {no_result} {{trial_dir}}/r.json {{overrides}}"}
    )
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(second)

    published_winner = read_winner(second, "p")
    status = read_status(second)
    assert published_winner is not None
    # The failed rerun (second) claimed a new current generation, but the
    # winner published on disk still comes from the first, successful one -
    # the two identities must be reported distinctly, never conflated.
    assert status["current_generation_id"] != published_winner.generation_id
    assert status["published_generation_id"] == published_winner.generation_id
    assert status["phases"][0]["winner_present"] is True

    trial_dirs = sorted(phase_dir.glob("trial_*"))
    assert len(trial_dirs) == 2
    assert first_trial_dir in trial_dirs
    second_trial_dir = next(path for path in trial_dirs if path != first_trial_dir)
    assert not (second_trial_dir / "r.json").exists()
    assert {path: path.read_bytes() for path in protected} == protected


@pytest.mark.integration
def test_existing_tree_preflights_missing_reached_phase_before_claim_or_topup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newly reached W&B phase must validate before an existing phase can top up."""
    trainer = write_constant_trainer(tmp_path)
    local = Phase(name="local", n_trials=1, sampler=Sampler(type="random", seed=0))
    initial = make_experiment(persistent=tmp_path, trainer=trainer, phases=[local])
    run_experiment(initial)
    generation_before = _generation_path(initial).read_bytes()

    remote = Phase(
        name="remote",
        n_trials=1,
        sampler=Sampler(type="random", seed=0),
        gates=[
            WandbSummaryRequiredGate(
                type="wandb_summary_required",
                entity="entity",
                project="project",
                keys=["complete"],
            )
        ],
    )
    expanded = initial.model_copy(
        update={"phases": [local.model_copy(update={"n_trials": 2}), remote]}
    )
    monkeypatch.setenv("WANDB_MODE", "offline")

    with pytest.raises(PhaseSweepError, match="requires online"):
        run_experiment(expanded)

    assert _generation_path(initial).read_bytes() == generation_before
    study = optuna.load_study(study_name="t::local", storage=_resolve_storage(initial.storage))
    assert len(study.get_trials(deepcopy=False)) == 1


def test_terminal_callback_reports_success_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = make_experiment(workdir=tmp_path / "runs")
    captured: list[TerminalReport] = []

    def preflight(_ledger, *, cleanup_report, from_phase):
        del from_phase
        cleanup_report.uncertain_attempt_ids.add("attempt-uncertain")
        return {}

    monkeypatch.setattr("phasesweep.engine.guards._preflight_existing_studies", preflight)
    monkeypatch.setattr(
        "phasesweep.engine.run._run_experiment_inner",
        lambda *_args, **_kwargs: {},
    )

    assert run_experiment(experiment, terminal_callback=captured.append) == {}
    assert len(captured) == 1
    report = captured[0]
    assert report.primary_error is None
    assert report.failure_stage is None
    assert report.uncertain_attempt_ids == frozenset({"attempt-uncertain"})


def test_terminal_callback_preserves_failure_when_callback_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = make_experiment(workdir=tmp_path / "runs")
    primary_error = NoFeasibleTrialError("phase failed")
    captured: list[TerminalReport] = []

    class CallbackError(RuntimeError):
        pass

    def fail_run(*_args, **_kwargs):
        raise primary_error

    def fail_callback(report: TerminalReport) -> None:
        captured.append(report)
        raise CallbackError("snapshot failed")

    monkeypatch.setattr(
        "phasesweep.engine.guards._preflight_existing_studies",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr("phasesweep.engine.run._run_experiment_inner", fail_run)

    with pytest.raises(NoFeasibleTrialError) as exc_info:
        run_experiment(experiment, terminal_callback=fail_callback)

    assert exc_info.value is primary_error
    assert len(captured) == 1
    assert captured[0].primary_error is primary_error
    assert captured[0].failure_stage == "execution"


def test_terminal_callback_failure_cannot_fail_published_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reporting callback is a diagnostic consumer, never an outcome authority.

    By the time the callback runs, a successful generation is already
    published as the last successful result; raising would present that
    committed success as a caller-visible failure (review v0.5.14 / item C).
    """
    experiment = make_experiment(workdir=tmp_path / "runs")

    class CallbackError(RuntimeError):
        pass

    def fail_callback(_report: TerminalReport) -> None:
        raise CallbackError("snapshot failed")

    monkeypatch.setattr(
        "phasesweep.engine.guards._preflight_existing_studies",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "phasesweep.engine.run._run_experiment_inner",
        lambda *_args, **_kwargs: {},
    )

    with caplog.at_level(logging.ERROR, logger="phasesweep.engine.run"):
        assert run_experiment(experiment, terminal_callback=fail_callback) == {}

    assert any("terminal callback failed" in record.message for record in caplog.records)


@pytest.mark.integration
def test_constraint_extractor_failure_marks_trial_fail(tmp_path):
    """Missing constraint output -> TrialState.FAIL, not COMPLETE+infeasible."""
    # Write metric only — constraint extractor will fail to find param_bytes.
    trainer = write_constant_trainer(tmp_path, key="eval_loss", value=1.0)

    db = tmp_path / "phases.journal"
    exp = make_experiment(
        experiment="c2",
        workdir=tmp_path / "runs",
        storage=f"journal:///{db}",
        trial_command=f"python {trainer} --out {{trial_dir}}/result.json {{overrides}}",
        metric=Metric(
            name="eval_loss",
            extractor=LogRegexExtractor(
                type="log_regex", pattern=r"eval_loss=(?P<value>[0-9.eE+-]+)"
            ),
        ),
        constraints=[
            Constraint(
                name="param_bytes",
                extractor=JsonExtractor(type="json", path="result.json", key="param_bytes"),
                max=1000,
            )
        ],
        name="a",
        n_trials=2,
        max_consecutive_failures=10,
        search_space={"x": FloatParam(type="float", low=0, high=1)},
    )

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(exp)

    study = optuna.load_study(study_name="c2::a", storage=_resolve_storage(f"journal:///{db}"))
    for trial in study.get_trials():
        assert trial.state == optuna.trial.TrialState.FAIL, (
            f"Trial {trial.number} is {trial.state.name}; expected FAIL because "
            "constraint extractor could not find param_bytes."
        )
        # The failure_reason user_attr should mention the constraint name.
        assert "param_bytes" in trial.user_attrs.get("phasesweep_failure_reason", "")


@pytest.mark.parametrize(
    ("extracted_values", "failure_reason"),
    [
        pytest.param(
            [float("inf")],
            "metric extractor returned non-finite value: inf",
            id="inf_metric",
        ),
        pytest.param(
            [1.0, float("nan")],
            "constraint extractor 'param_bytes' returned non-finite value: nan",
            id="nan_constraint",
        ),
    ],
)
def test_non_finite_extracted_value_returns_failed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extracted_values: list[float],
    failure_reason: str,
) -> None:
    """Non-finite extractor returns reach both engine-level finite-value guards."""
    experiment = make_experiment(
        constraints=[
            Constraint(
                name="param_bytes",
                extractor=JsonExtractor(type="json", path="result.json", key="param_bytes"),
                max=1000,
            )
        ]
    )
    values = iter(extracted_values)
    monkeypatch.setattr(
        "phasesweep.engine.trial.run_extractor",
        lambda *_args, **_kwargs: next(values),
    )
    executed = ExecutedTrial(
        ctx=TrialContext(
            experiment="t",
            phase="p",
            trial_id=0,
            generation_id="generation-test",
            attempt_id="attempt-test",
            overrides_sha256="0" * 64,
            trial_dir=tmp_path,
            run_name="t-p-0-attempt-test",
            return_code=0,
            duration_seconds=0.1,
        ),
        process=ProcessResult(return_code=0, timed_out=False, pid=123, duration_seconds=0.1),
    )

    result = extract_trial_result(experiment=experiment, executed=executed)

    assert result.metric is None
    assert result.feasible is False
    assert result.failure_reason == failure_reason


@pytest.mark.integration
def test_abort_after_gpu_acquire_prevents_queued_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With n_jobs > GPU pool, queued trials must re-check abort after acquiring.

    Setup: single GPU, 4 parallel jobs, all trials fail, max_consecutive_failures=1.
    Without the post-acquire recheck, ~4 trials launch before abort fires;
    with the recheck only the first 1-2 actually launch subprocesses.
    """
    trainer = write_trainer(
        tmp_path,
        """
        import sys, time
        # Simulate slow failing trial so peers queue behind us.
        time.sleep(0.2)
        sys.exit(1)
        """,
    )
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_uuid_map",
        lambda: {"0": "GPU-test-0"},
    )
    exp = make_experiment(
        experiment="abort_recheck",
        workdir=tmp_path / "runs",
        trial_command=f"python {trainer} {{overrides}}",
        n_trials=16,
        n_jobs=4,
        gpu_ids=[0],  # only one slot — n_jobs=4 will queue
        max_consecutive_failures=1,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": FloatParam(type="float", low=0, high=1)},
    )

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(exp)

    # Count actual subprocess launches (each launch creates a stdout.log).
    # With the post-acquire recheck, the first failure aborts queued threads
    # before they launch their subprocess. Without it, all 4 would launch.
    # Derive the directory from the experiment to track the
    # ``<workdir>/<experiment>/<phase>/`` layout (review v0.5.12 caught the
    # earlier hard-coded ``<workdir>/<phase>/`` path silently matching nothing
    # and rendering the assertion vacuous).
    runs_dir = Path(exp.workdir).expanduser().resolve() / exp.experiment / exp.phases[0].name
    trial_dirs = sorted(runs_dir.glob("trial_*"))
    launched = [d for d in trial_dirs if (d / "stdout.log").exists()]
    assert 1 <= len(launched) <= 2, (
        f"With max_consecutive_failures=1 and a 1-slot GPU pool, expected at "
        f"least 1 and at most 2 trial launches (1 failing + at most 1 in-flight "
        f"before abort propagates); "
        f"got {len(launched)}. Launched: {[d.name for d in launched]}. "
        "Queued threads ignored the post-acquire abort flag."
    )


def test_optuna_logging_verbosity_tracks_cli_verbose_flag() -> None:
    """Default CLI output quiets Optuna; ``-v`` restores Optuna INFO logs."""
    from phasesweep.cli import _configure_logging

    cases = [
        ("default_quiet", optuna.logging.INFO, False, optuna.logging.WARNING),
        ("verbose", optuna.logging.WARNING, True, optuna.logging.INFO),
    ]

    # Logging state is process-global, and pytest's capture handlers sit on the root logger.
    root, saved_verbosity = logging.getLogger(), optuna.logging.get_verbosity()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        for case, initial, verbose, expected in cases:
            optuna.logging.set_verbosity(initial)
            _configure_logging(verbose=verbose)
            assert optuna.logging.get_verbosity() == expected, case
    finally:
        optuna.logging.set_verbosity(saved_verbosity)
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_runtime_platform_guard_feature_checks_and_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real execution needs POSIX process groups and flock; dry-run does not."""
    from phasesweep.runtime import files as runtime_files

    # macOS reports ``sys.platform == 'darwin'`` but still has POSIX features;
    # the guard is intentionally feature-based instead of matching platform names.
    assert runtime_files._supports_posix_runtime_features(
        os_name="posix",
        has_killpg=True,
        has_fcntl=True,
    )
    assert not runtime_files._supports_posix_runtime_features(
        os_name="nt",
        has_killpg=True,
        has_fcntl=True,
    )
    assert not runtime_files._supports_posix_runtime_features(
        os_name="posix",
        has_killpg=False,
        has_fcntl=True,
    )
    assert not runtime_files._supports_posix_runtime_features(
        os_name="posix",
        has_killpg=True,
        has_fcntl=False,
    )

    exp = make_experiment(
        experiment="platform_check",
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path}/platform.journal",
        n_trials=1,
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )

    monkeypatch.setattr(runtime_files, "_supports_posix_runtime_features", lambda: False)

    with pytest.raises(RuntimeError, match="requires a POSIX platform"):
        run_experiment(exp)

    # Dry-run remains available because it launches no subprocesses and takes no locks.
    winners = run_experiment(exp, dry_run=True)
    assert set(winners) == {"p"}


@pytest.mark.integration
def test_max_consecutive_failures_aborts_phase(tmp_path):
    """Trial command always fails -> phase aborts before running n_trials."""
    storage = f"journal:///{tmp_path}/fail.journal"
    exp = make_experiment(
        experiment="failtest",
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command="false {overrides}",
        name="a",
        n_trials=100,
        max_consecutive_failures=3,
        search_space={"x": FloatParam(type="float", low=0, high=1)},
    )
    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(exp)
    # Verify only a small number of trials actually executed before the abort.
    # We can't predict exactly how many because Optuna may have a few in flight,
    # but it should be << 100.
    study = optuna.load_study(study_name="failtest::a", storage=_resolve_storage(storage))
    n = len(study.get_trials(deepcopy=False))
    assert n < 30, f"expected early abort, got {n} trials"


@pytest.mark.integration
def test_aborted_phase_is_not_published_by_identical_noop_retry(tmp_path: Path) -> None:
    """A durable abort survives restart: the identical no-op re-run stays failed.

    First invocation: one success, then failures reach the threshold with every
    trial terminal. Before review v0.5.17 / blocker 1, a second identical
    invocation saw ``remaining == 0`` and published the lone COMPLETE trial as
    a completed phase — converting a recorded failure into a success with no
    new work.
    """
    marker = tmp_path / "succeeded_once"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import pathlib, sys
        marker = pathlib.Path({str(marker)!r})
        if marker.exists():
            sys.exit(1)
        marker.touch()
        print("x=0.25")
        """,
    )
    db = tmp_path / "abort.journal"
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{db}",
        trial_command=f"python {trainer} {{overrides}}",
        override_format="argparse",
        n_trials=3,
        max_consecutive_failures=2,
        sampler={"type": "random", "seed": 7},
    )

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(exp)

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    record = study.user_attrs[PHASE_ABORT_ATTR]
    assert record["policy"] == "max_consecutive_failures"
    assert record["consecutive_failures"] == 2

    # Identical retry: no new work is schedulable (3/3 terminal trials), so the
    # durable abort must keep the phase failed instead of publishing.
    with pytest.raises(NoFeasibleTrialError, match="previously aborted"):
        run_experiment(exp)
    assert not _last_successful_generation_path(exp).exists()


@pytest.mark.integration
def test_abort_recovery_target_with_no_remaining_slots_is_schema_mismatch(
    tmp_path: Path,
) -> None:
    """Contradictory durable abort/trial counts are operator-visible study state."""
    trainer = write_trainer(tmp_path / "trainer.py", "raise SystemExit(1)")
    storage = f"journal:///{tmp_path / 'abort.journal'}"

    def experiment(n_trials: int) -> Experiment:
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=storage,
            trial_command=f"python {trainer} {{overrides}}",
            override_format="argparse",
            n_trials=n_trials,
            max_consecutive_failures=2,
            sampler={"type": "random", "seed": 7},
        )

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(experiment(3))

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(storage))
    cohort_digest = study.trials[0].user_attrs[TRAINER_ENV_DIGEST_ATTR]
    for sequence in (3, 4):
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.FAIL,
                user_attrs={
                    TRAINER_ENV_DIGEST_ATTR: cohort_digest,
                    TRIAL_OUTCOME_ATTR: {
                        "schema_version": 1,
                        "sequence": sequence,
                        "outcome": "failure",
                        "cause": "simulated external terminal row",
                    },
                },
            )
        )
    study.set_user_attr(TRIAL_TARGET_ATTR, 4)

    with pytest.raises(StudySchemaMismatchError, match="no remaining trial slots"):
        run_experiment(experiment(4))


def test_parallel_abort_stamps_environment_before_pruning(tmp_path, monkeypatch):
    """A scheduled worker entering after a peer abort still permits a later top-up."""
    experiment = make_experiment(
        workdir=tmp_path / "work",
        storage=f"journal:///{tmp_path / 'study.journal'}",
        n_trials=2,
        n_jobs=2,
        gpu_policy="none",
        allow_no_gpu_isolation=True,
        max_consecutive_failures=1,
    )
    first_finished = threading.Event()
    second_entered = threading.Event()
    original_optimize = optuna.Study.optimize
    launched = []

    def optimize(study, objective, **kwargs):
        def scheduled_objective(trial):
            if trial.number == 0:
                assert second_entered.wait(10)
                try:
                    return objective(trial)
                finally:
                    first_finished.set()
            second_entered.set()
            assert first_finished.wait(10)
            return objective(trial)

        original_optimize(study, scheduled_objective, **kwargs)

    def failed_launch(**kwargs):
        launched.append(kwargs["trial_id"])
        raise TrialExecutionError("simulated trainer failure; no process launched")

    monkeypatch.setattr("phasesweep.engine.phase.launch_trial", failed_launch)
    with monkeypatch.context() as patch:
        patch.setattr(optuna.Study, "optimize", optimize)
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(experiment)

    study = optuna.load_study(
        study_name="t::p", storage=_resolve_storage(experiment.resolved_storage)
    )
    previous = study.trials
    assert [trial.state.name for trial in previous] == ["FAIL", "PRUNED"]
    for trial in previous:
        assert trial.user_attrs[TRAINER_ENV_DIGEST_ATTR]
        assert TRAINER_ENV_NAMES_ATTR in trial.user_attrs

    updated = experiment.model_copy(
        update={"phases": [experiment.phases[0].model_copy(update={"n_trials": 3})]}
    )
    # The supported retry must reach new work, without a schema/migration error.
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(updated)
    assert launched == [0, 2]
    assert len(study.trials) == 3
    assert [trial.user_attrs for trial in study.trials[:2]] == [
        trial.user_attrs for trial in previous
    ]


@pytest.mark.integration
def test_topup_after_abort_runs_new_work_and_clears_durable_abort(tmp_path: Path) -> None:
    """Raising n_trials after an abort is the explicit resume path.

    The top-up schedules genuinely new attempts; reaching a successful winner
    selection consumes the durable abort record (review v0.5.17 / blocker 1).
    """
    flag = tmp_path / "resume_enabled"
    trainer = write_flag_gated_trainer(tmp_path, flag)
    db = tmp_path / "abort.journal"

    def _exp(n_trials: int):
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=f"journal:///{db}",
            trial_command=f"python {trainer} {{overrides}}",
            override_format="argparse",
            n_trials=n_trials,
            max_consecutive_failures=2,
            sampler={"type": "random", "seed": 7},
        )

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(_exp(3))

    flag.touch()
    winners = run_experiment(_exp(6))

    assert winners["p"].metric == pytest.approx(0.5)
    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    assert study.user_attrs.get(PHASE_ABORT_ATTR) is None


@pytest.mark.integration
def test_supported_topup_preserves_consecutive_failure_streak(tmp_path: Path) -> None:
    """A random-sampler top-up continues the durable failure streak."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import os, sys
        if os.environ["PHASESWEEP_TRIAL_ID"] == "0":
            print("x=0.5")
        else:
            sys.exit(1)
        """,
    )
    db = tmp_path / "topup.journal"

    def _exp(n_trials: int) -> Experiment:
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=f"journal:///{db}",
            trial_command=f"python {trainer} {{overrides}}",
            override_format="argparse",
            n_trials=n_trials,
            max_consecutive_failures=3,
            sampler={"type": "random", "seed": 7},
        )

    first = run_experiment(_exp(3))
    assert first["p"].metric == pytest.approx(0.5)
    published_before = _last_successful_generation_path(_exp(3)).read_text()

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(_exp(4))

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    assert [trial.state.name for trial in study.trials] == ["COMPLETE", "FAIL", "FAIL", "FAIL"]
    assert [trial.user_attrs[TRIAL_OUTCOME_ATTR]["sequence"] for trial in study.trials] == [
        1,
        2,
        3,
        4,
    ]
    assert study.user_attrs[PHASE_ABORT_ATTR]["consecutive_failures"] == 3
    assert _last_successful_generation_path(_exp(4)).read_text() == published_before


@pytest.mark.integration
def test_outcome_ledger_recovers_when_abort_marker_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed marker write cannot make an identical retry publish."""
    trainer = write_trainer(tmp_path / "trainer.py", "raise SystemExit(1)")
    db = tmp_path / "abort.journal"
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{db}",
        trial_command=f"python {trainer} {{overrides}}",
        override_format="argparse",
        n_trials=2,
        max_consecutive_failures=2,
        sampler={"type": "random", "seed": 7},
    )

    real_set_user_attr = optuna.Study.set_user_attr
    failed = {"once": False}

    def fail_first_abort_marker(study: optuna.Study, key: str, value: object) -> None:
        if key == PHASE_ABORT_ATTR and isinstance(value, dict) and not failed["once"]:
            failed["once"] = True
            raise RuntimeError("injected abort-marker write failure")
        real_set_user_attr(study, key, value)

    monkeypatch.setattr(optuna.Study, "set_user_attr", fail_first_abort_marker)
    with pytest.raises(StudyStorageUnavailableError, match="abort marker could not be persisted"):
        run_experiment(exp)

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    assert study.user_attrs.get(PHASE_ABORT_ATTR) is None
    assert [trial.user_attrs[TRIAL_OUTCOME_ATTR]["outcome"] for trial in study.trials] == [
        "failure",
        "failure",
    ]

    monkeypatch.setattr(optuna.Study, "set_user_attr", real_set_user_attr)
    with pytest.raises(NoFeasibleTrialError, match="previously aborted"):
        run_experiment(exp)
    assert not _last_successful_generation_path(exp).exists()


def _outcome_write_experiment(
    tmp_path: Path,
    *,
    trainer_body: str,
    storage: str,
    constraints: list[Constraint] | None = None,
    n_jobs: int = 1,
) -> Experiment:
    """Build the two-trial experiment the outcome-write-failure tests share."""
    trainer = write_trainer(tmp_path / "trainer.py", trainer_body)
    extra: dict[str, Any] = {}
    if n_jobs > 1:
        extra = {"n_jobs": n_jobs, "gpu_policy": "none", "allow_no_gpu_isolation": True}
    return make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} {{overrides}}",
        override_format="argparse",
        n_trials=2,
        constraints=constraints,
        max_consecutive_failures=5,
        sampler={"type": "random", "seed": 7},
        **extra,
    )


@pytest.mark.integration
def test_transient_trial_outcome_write_failure_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One flaky outcome write must not cost the phase a trial.

    The bounded retry is what keeps the fail-closed reaction to an *unwritable*
    ledger from firing on a momentarily busy one (PR #5 review / reviewer 2
    pass 2, blocker 4).
    """
    db = tmp_path / "retry.journal"
    exp = _outcome_write_experiment(
        tmp_path,
        trainer_body='print("x=0.5")',
        storage=f"journal:///{db}",
    )

    real_set_user_attr = optuna.Trial.set_user_attr
    injected = {"n": 0}

    def fail_first_outcome_write(trial: optuna.Trial, key: str, value: object) -> None:
        if key == TRIAL_OUTCOME_ATTR and injected["n"] == 0:
            injected["n"] += 1
            raise RuntimeError("injected transient outcome write failure")
        real_set_user_attr(trial, key, value)

    monkeypatch.setattr(optuna.Trial, "set_user_attr", fail_first_outcome_write)
    winners = run_experiment(exp)

    assert injected["n"] == 1
    assert winners["p"].metric == pytest.approx(0.5)
    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    assert [trial.state.name for trial in study.trials] == ["COMPLETE", "COMPLETE"]
    assert [trial.user_attrs[TRIAL_OUTCOME_ATTR]["sequence"] for trial in study.trials] == [1, 2]
    assert all(
        trial.user_attrs[TRIAL_OUTCOME_ATTR]["outcome"] == "success" for trial in study.trials
    )


@pytest.mark.integration
def test_outcome_retry_backoff_does_not_hold_completion_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flaky worker's backoff must not block a peer's durable outcome write."""
    journal = tmp_path / "retry-parallel.journal"
    exp = _outcome_write_experiment(
        tmp_path,
        trainer_body="""
        import os, time
        if os.environ["PHASESWEEP_TRIAL_ID"] == "1":
            time.sleep(0.2)
        print("x=0.5")
        """,
        storage=f"journal:///{journal}",
        n_jobs=2,
    )
    real_set_user_attr = optuna.Trial.set_user_attr
    first_failed = threading.Event()
    successful_writes: list[int] = []

    def fail_trial_zero_once(trial: optuna.Trial, key: str, value: object) -> None:
        if key == TRIAL_OUTCOME_ATTR and trial.number == 0 and not first_failed.is_set():
            first_failed.set()
            raise RuntimeError("injected transient outcome write failure")
        if key == TRIAL_OUTCOME_ATTR:
            successful_writes.append(trial.number)
        real_set_user_attr(trial, key, value)

    monkeypatch.setattr(optuna.Trial, "set_user_attr", fail_trial_zero_once)
    monkeypatch.setattr("phasesweep.engine.phase._OUTCOME_WRITE_RETRY_DELAYS", (0.5, 0.5))

    winners = run_experiment(exp)

    assert winners["p"].metric == pytest.approx(0.5)
    study = optuna.load_study(
        study_name="t::p",
        storage=_resolve_storage(f"journal:///{journal}"),
    )
    sequences = sorted(trial.user_attrs[TRIAL_OUTCOME_ATTR]["sequence"] for trial in study.trials)
    assert first_failed.is_set()
    assert successful_writes[:2] == [1, 0]
    assert sequences == [1, 2]


@pytest.mark.parametrize(
    ("trainer_body", "constraints"),
    [
        pytest.param('print("x=0.5")', None, id="successful_trial"),
        pytest.param(
            """
            import os, sys
            if os.environ["PHASESWEEP_TRIAL_ID"] == "0":
                sys.exit(1)
            print("x=0.5")
            """,
            None,
            id="trainer_failure_trial",
        ),
        pytest.param(
            """
            import os
            print("bytes=5000" if os.environ["PHASESWEEP_TRIAL_ID"] == "0" else "bytes=100")
            print("x=0.5")
            """,
            [
                Constraint(
                    name="param_bytes",
                    extractor=LogRegexExtractor(
                        type="log_regex",
                        pattern=r"bytes=(?P<value>[0-9.]+)",
                    ),
                    max=1000,
                )
            ],
            id="infeasible_trial",
        ),
    ],
)
@pytest.mark.integration
def test_persistent_outcome_write_failure_leaves_trial_running_until_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_body: str,
    constraints: list[Constraint] | None,
) -> None:
    """An unwritable outcome ledger must never produce a terminal row without one.

    Optuna commits the trial for *any* ``Exception`` leaving the objective and
    then refuses user-attr writes on the finished trial, so a FAIL recorded
    without its outcome attr would wedge the study permanently: every later
    invocation rejects it in ``_load_phase_policy_state``. The objective
    therefore aborts outside Optuna's catch set, leaving the trial RUNNING with
    its registry entry intact, and the phase reports a retryable storage
    outage. The next run recovers it through the ordinary stale-attempt
    protocol, which writes the outcome *before* the terminal transition
    (PR #5 review / reviewer 2 pass 2, blocker 4).
    """
    db = tmp_path / "outcome.journal"
    exp = _outcome_write_experiment(
        tmp_path,
        trainer_body=trainer_body,
        storage=f"journal:///{db}",
        constraints=constraints,
    )

    real_set_user_attr = patch_rejected_trial_user_attr(
        monkeypatch,
        TRIAL_OUTCOME_ATTR,
        "injected outcome write failure",
    )
    with pytest.raises(StudyStorageUnavailableError, match="deliberately left RUNNING"):
        run_experiment(exp)

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    trials = study.get_trials(deepcopy=False)
    assert [trial.state for trial in trials] == [optuna.trial.TrialState.RUNNING]
    assert not [
        trial
        for trial in trials
        if trial.state.is_finished() and TRIAL_OUTCOME_ATTR not in trial.user_attrs
    ]
    # The durable attempt record is what the next run recovers through.
    assert list((tmp_path / "runs" / "t" / "attempts").glob("*.json"))
    assert not _last_successful_generation_path(exp).exists()

    monkeypatch.setattr(optuna.Trial, "set_user_attr", real_set_user_attr)
    winners = run_experiment(exp)

    assert winners["p"].metric == pytest.approx(0.5)
    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    trials = study.get_trials(deepcopy=False)
    assert [trial.state for trial in trials] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert trials[0].user_attrs[TRIAL_OUTCOME_ATTR] == {
        "schema_version": 1,
        "sequence": 1,
        "outcome": "failure",
        "cause": "stale RUNNING trial recovered after its orchestrator stopped",
    }
    # The whole point: the ledger still validates, so the study is reusable.
    assert _load_phase_policy_state(study).max_sequence == 2


@pytest.mark.integration
def test_parallel_outcome_write_failure_surfaces_without_orphan_terminal_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The n_jobs>1 path surfaces the same abort through the fatal-abort slot.

    Optuna's threaded loop never calls ``result()`` on the last submitted
    futures, so a worker's abort is silently dropped there; only the
    orchestrator-owned fatal slot re-raised after ``study.optimize`` makes it
    visible (PR #5 review / reviewer 2 pass 2, blocker 4).
    """
    journal = tmp_path / "outcome.journal"
    exp = _outcome_write_experiment(
        tmp_path,
        trainer_body='print("x=0.5")',
        storage=f"journal:///{journal}",
        n_jobs=2,
    )

    patch_rejected_trial_user_attr(
        monkeypatch,
        TRIAL_OUTCOME_ATTR,
        "injected outcome write failure",
    )
    with pytest.raises(StudyStorageUnavailableError, match="deliberately left RUNNING"):
        run_experiment(exp)

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{journal}"))
    trials = study.get_trials(deepcopy=False)
    assert not [
        trial
        for trial in trials
        if trial.state.is_finished() and TRIAL_OUTCOME_ATTR not in trial.user_attrs
    ]
    assert any(trial.state == optuna.trial.TrialState.RUNNING for trial in trials)
    assert not _last_successful_generation_path(exp).exists()


@pytest.mark.parametrize("cleanup_attr_persists", [True, False])
@pytest.mark.parametrize("identity_persists", [True, False])
@pytest.mark.integration
def test_unsafe_cleanup_blocks_topup_until_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_attr_persists: bool,
    identity_persists: bool,
) -> None:
    """A larger target cannot acknowledge a possibly live trainer.

    The fatal outcome policy is the fallback authority when the redundant
    cleanup attribute write fails. In both cases the active-attempt locator
    remains until ordinary preflight positively confirms cleanup and commits
    the study recovery ledger.
    """
    import phasesweep.engine.trial as trial_mod

    db = tmp_path / "abort.journal"

    def _exp(n_trials: int) -> Experiment:
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=f"journal:///{db}",
            # This fixture must produce no metric, regardless of override rendering.
            trial_command="true {overrides}",
            n_trials=n_trials,
            sampler={"type": "random", "seed": 7},
        )

    real_run_supervised = trial_mod.run_supervised
    real_set_user_attr = optuna.Trial.set_user_attr
    calls = {"n": 0}
    cleanup_safe = {"value": False}

    def uncertain_on_second(*args: object, **kwargs: object) -> ProcessResult:
        calls["n"] += 1
        result = real_run_supervised(*args, **kwargs)
        if calls["n"] == 2:
            result.cleanup_confirmed = False
            # ``real_run_supervised`` already observed a clean exit before this
            # fault injection. Restore the durable pre-exit shape a genuinely
            # unconfirmed cleanup leaves behind so preflight must consult the
            # process identity and our cleanup verdict.
            trial_dir = Path(str(kwargs["trial_dir"]))
            write_attempt_lifecycle(
                trial_dir,
                attempt_id=str(kwargs["attempt_id"]),
                state="allocated" if identity_persists else "launching",
            )
            if not identity_persists:
                (trial_dir / PROCESS_IDENTITY_FILE).unlink()
        return result

    def maybe_refuse_cleanup_attr(
        trial: optuna.Trial,
        key: str,
        value: object,
    ) -> None:
        if not cleanup_attr_persists and key == CLEANUP_CONFIRMED_ATTR:
            raise RuntimeError("injected cleanup attribute write failure")
        real_set_user_attr(trial, key, value)

    monkeypatch.setattr("phasesweep.engine.trial.run_supervised", uncertain_on_second)
    monkeypatch.setattr(optuna.Trial, "set_user_attr", maybe_refuse_cleanup_attr)
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process",
        lambda _identity: cleanup_safe["value"],
    )
    monkeypatch.setattr(
        "phasesweep.engine.cleanup.cleanup_stale_trial_process",
        lambda _identity: cleanup_safe["value"],
    )
    with pytest.raises(ProcessCleanupUncertainError, match="cleanup could not be confirmed"):
        run_experiment(_exp(2))

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    record = study.user_attrs[PHASE_ABORT_ATTR]
    assert record["policy"] == "unsafe_process_cleanup"
    assert "cleanup could not be confirmed" in record["cause"]
    unsafe_trial = next(
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.user_attrs[TRIAL_OUTCOME_ATTR]["outcome"] == "fatal"
    )
    assert unsafe_trial.user_attrs[TRIAL_OUTCOME_ATTR]["policy"] == "unsafe_process_cleanup"
    if cleanup_attr_persists:
        assert unsafe_trial.user_attrs[CLEANUP_CONFIRMED_ATTR] is False
    else:
        assert CLEANUP_CONFIRMED_ATTR not in unsafe_trial.user_attrs
    assert list(_attempts_dir(_exp(2)).glob("*.json"))

    # A larger target is not cleanup authority and launches no new trial.
    before = len(study.trials)
    with pytest.raises(ProcessCleanupUncertainError):
        run_experiment(_exp(3))
    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    assert len(study.trials) == before
    assert list(_attempts_dir(_exp(3)).glob("*.json"))
    assert CLEANUP_RECOVERED_TRIALS_ATTR not in study.user_attrs

    # Positive cleanup is durably consumed before the registry is retired and
    # the ordinary fatal-abort top-up path becomes available.
    cleanup_safe["value"] = True
    if not identity_persists:
        write_attempt_lifecycle(
            Path(str(unsafe_trial.user_attrs[TRIAL_DIR_ATTR])),
            attempt_id=str(unsafe_trial.user_attrs[ATTEMPT_ID_ATTR]),
            state="exited",
            return_code=0,
            cleanup_confirmed=True,
        )
    monkeypatch.setattr("phasesweep.engine.trial.run_supervised", real_run_supervised)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_exp(3))

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    assert len(study.trials) == before + 1
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [unsafe_trial.number]
    assert list(_attempts_dir(_exp(3)).glob("*.json")) == []


@pytest.mark.integration
def test_stale_abort_record_cleared_before_selection_survives_selection_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The durable abort clear must precede winner selection.

    A top-up that died mid-selection used to leave the durable abort record
    behind; the next identical invocation then reported "previously aborted"
    and — for stateful samplers — rejected the advised raise-n_trials remedy
    too, wedging the phase (review v0.5.17 gap hunt). Selection is
    deterministic from durable trial data, so clearing first is safe: the
    replay re-derives the same winner.
    """
    flag = tmp_path / "resume_enabled"
    trainer = write_flag_gated_trainer(tmp_path, flag)
    db = tmp_path / "abort.journal"

    def _exp(n_trials: int):
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=f"journal:///{db}",
            trial_command=f"python {trainer} {{overrides}}",
            override_format="argparse",
            n_trials=n_trials,
            max_consecutive_failures=2,
            sampler={"type": "random", "seed": 7},
        )

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(_exp(3))

    flag.touch()
    import phasesweep.engine.phase as phase_mod

    real_select = phase_mod._select_phase_winner

    def crash_in_selection(*args: object, **kwargs: object):
        raise RuntimeError("simulated crash during winner selection")

    monkeypatch.setattr("phasesweep.engine.phase._select_phase_winner", crash_in_selection)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_experiment(_exp(6))

    # The durable abort was consumed before selection started...
    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    assert study.user_attrs.get(PHASE_ABORT_ATTR) is None

    # ...so the identical replay republishes deterministically instead of
    # wedging on "previously aborted".
    monkeypatch.setattr("phasesweep.engine.phase._select_phase_winner", real_select)
    winners = run_experiment(_exp(6))
    assert winners["p"].metric == pytest.approx(0.5)


@pytest.mark.integration
def test_parallel_failure_threshold_uses_completion_order(tmp_path: Path) -> None:
    """Two fast failures trip the abort before a slow later success can reset it.

    Before review v0.5.17 / blocker 7, the threshold was evaluated by a
    post-trial callback reading a mutable aggregate counter, so a success
    finishing between two failures' callbacks could erase them. The decision
    now happens in the objective's completion-order critical section: with the
    success still sleeping, the abort must be tripped at completion sequence 2.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import os, sys, time
        if os.environ["PHASESWEEP_TRIAL_ID"] == "2":
            time.sleep(8)
            print("x=0.5")
        else:
            sys.exit(1)
        """,
    )
    db = tmp_path / "parallel.journal"
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{db}",
        trial_command=f"python {trainer} {{overrides}}",
        override_format="argparse",
        n_trials=3,
        n_jobs=3,
        gpu_policy="none",
        allow_no_gpu_isolation=True,
        max_consecutive_failures=2,
        sampler={"type": "random", "seed": 7},
    )

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(exp)

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    record = study.user_attrs[PHASE_ABORT_ATTR]
    # The abort was decided when the second failure recorded its outcome —
    # before the sleeping success could reset the consecutive counter.
    assert record["completion_sequence"] == 2
    assert record["consecutive_failures"] == 2
    states = sorted(t.state.name for t in study.get_trials(deepcopy=False))
    assert states == ["COMPLETE", "FAIL", "FAIL"]


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.integration
def test_unexpected_objective_error_is_phase_fatal_and_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_jobs: int
) -> None:
    """A worker bug cannot be swallowed by Optuna or published on retry."""
    trainer = write_trainer(tmp_path / "trainer.py", 'print("x=0.5")')
    journal = tmp_path / "fatal.journal"
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{journal}",
        trial_command=f"python {trainer} {{overrides}}",
        override_format="argparse",
        n_trials=2,
        n_jobs=n_jobs,
        gpu_policy="none",
        allow_no_gpu_isolation=True,
        sampler={"type": "random", "seed": 7},
    )
    real_extract = extract_trial_result

    def fail_second_extraction(**kwargs: object):
        executed = kwargs["executed"]
        if isinstance(executed, ExecutedTrial) and executed.ctx.trial_id == 1:
            raise RuntimeError("injected objective implementation bug")
        return real_extract(**kwargs)

    monkeypatch.setattr(
        "phasesweep.engine.phase.extract_trial_result",
        fail_second_extraction,
    )
    with pytest.raises(RuntimeError, match="injected objective implementation bug"):
        run_experiment(exp)

    study = optuna.load_study(
        study_name="t::p",
        storage=_resolve_storage(f"journal:///{journal}"),
    )
    assert sorted(trial.state.name for trial in study.trials) == ["COMPLETE", "FAIL"]
    fatal = next(
        trial.user_attrs[TRIAL_OUTCOME_ATTR]
        for trial in study.trials
        if trial.user_attrs[TRIAL_OUTCOME_ATTR]["outcome"] == "fatal"
    )
    assert fatal["cause"] == "RuntimeError: injected objective implementation bug"
    assert study.user_attrs[PHASE_ABORT_ATTR]["policy"] == "unexpected_objective_exception"
    assert not _last_successful_generation_path(exp).exists()
    assert list(_attempts_dir(exp).glob("*.json")) == []

    monkeypatch.setattr(
        "phasesweep.engine.phase.extract_trial_result",
        real_extract,
    )
    with pytest.raises(NoFeasibleTrialError, match="previously aborted"):
        run_experiment(exp)
    assert not _last_successful_generation_path(exp).exists()


@pytest.mark.integration
def test_noop_rerun_skips_gpu_discovery_and_target_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed phase republishes from a host that cannot launch work.

    Reading or republishing a finished GPU experiment from a CPU-only login
    node must not require GPU discovery or mutate the durable accepted target
    (review v0.5.14 / blocker 4).
    """
    trainer = write_param_echo_trainer(tmp_path)
    experiment = make_experiment(
        persistent=tmp_path, trainer=trainer, n_trials=1, sampler=Sampler(type="random", seed=0)
    )
    first = run_experiment(experiment)

    def _no_gpu_create(**_kwargs: object) -> None:
        raise RuntimeError("simulated: no GPUs detected on this host")

    monkeypatch.setattr("phasesweep.engine.phase.GpuPool.create", _no_gpu_create)
    rerun = run_experiment(experiment)

    assert rerun["p"].trial_number == first["p"].trial_number
    assert rerun["p"].metric == first["p"].metric
    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(experiment.storage))
    assert study.user_attrs[TRIAL_TARGET_ATTR] == 1
    assert len(study.trials) == 1


@pytest.mark.integration
def test_failed_gpu_topup_preserves_accepted_target_and_old_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A top-up that cannot launch leaves the prior accepted target usable.

    The larger target must only be durably accepted after launch prerequisites
    (GPU discovery, wallclock budget) pass; otherwise a transient local GPU
    problem permanently strands the study above its last working config
    (review v0.5.14 / blocker 4).
    """
    trainer = write_param_echo_trainer(tmp_path)
    phase = Phase(
        name="p",
        n_trials=1,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": IntParam(type="int", low=0, high=10)},
    )
    experiment = make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase])
    first = run_experiment(experiment)

    def _no_gpu_create(**_kwargs: object) -> None:
        raise RuntimeError("simulated: no GPUs detected on this host")

    monkeypatch.setattr("phasesweep.engine.phase.GpuPool.create", _no_gpu_create)
    top_up = experiment.model_copy(update={"phases": [phase.model_copy(update={"n_trials": 2})]})
    with pytest.raises(RuntimeError, match="no GPUs detected"):
        run_experiment(top_up)

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(experiment.storage))
    assert study.user_attrs[TRIAL_TARGET_ATTR] == 1
    assert len(study.trials) == 1

    rerun = run_experiment(experiment)
    assert rerun["p"].metric == first["p"].metric


# A pairwise-unequal mixed-type choice set. `True` is not equal to 0, 1.5, or
# "one", so config validation accepts it and Optuna's `choices.index(value)`
# lookup resolves each sampled value to its own index. That is the property
# `CategoricalParam` now enforces (PR #5 review / reviewer 2, blocker 1): a
# choice set containing 1 alongside True would collapse both onto one index the
# moment Optuna recorded `FrozenTrial.params`.
_FIDELITY_CHOICES: list[Any] = [0, 1.5, "one", True]

# Trainer losses keyed by the rendered argparse token for each choice, chosen so
# the bool wins. A bool is the value most likely to be silently downgraded to
# int on a round trip, so it is the winner every downstream surface must report.
_FIDELITY_LOSSES = {"0": 3.0, "1.5": 2.0, "one": 1.0, "true": 0.0}


def _assert_identical_scalar(actual: Any, expected: Any, *, surface: str) -> None:
    """Assert two scalars match in both value and concrete type.

    Plain ``==`` is exactly the comparison that cannot see this bug: a winner
    reporting ``1`` for a trial that ran ``True`` compares equal to the truth.

    :param Any actual: Value read back from the surface under test.
    :param Any expected: Value the engine actually sampled.
    :param str surface: Human-readable name of the surface, for the failure message.
    """
    assert type(actual) is type(expected) and actual == expected, (
        f"{surface}: expected {expected!r} ({type(expected).__name__}), "
        f"got {actual!r} ({type(actual).__name__})"
    )


def _resolved_overrides_by_trial(phase_dir: Path) -> dict[int, dict[str, Any]]:
    """Read every ``overrides_resolved.json`` under a phase dir, keyed by trial number.

    :param Path phase_dir: ``<workdir>/<experiment>/<phase>``.
    :return dict[int, dict[str, Any]]: Trial number -> resolved override payload.
    """
    resolved: dict[int, dict[str, Any]] = {}
    for path in sorted(phase_dir.glob("trial_*/overrides_resolved.json")):
        number = int(path.parent.name.split("__")[0].removeprefix("trial_"))
        resolved[number] = json.loads(path.read_text())
    return resolved


@pytest.mark.integration
def test_categorical_value_keeps_its_type_across_every_persisted_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sampled categorical must be byte-identical from trainer input to child phase.

    Differential regression for PR #5 review / reviewer 2, blocker 1. Optuna
    records a sampled categorical as ``choices.index(value)``, an ``==`` lookup,
    so a choice set holding Python-equal-but-distinct values published a winner
    naming a value the trial never ran. The validator now rejects such sets;
    this proves the surviving guarantee holds end to end for an accepted set:
    the live suggestion, ``FrozenTrial.params``, ``overrides_resolved.json``,
    ``winner.yaml``, and the dependent phase's inherited overrides all carry the
    same value *and* the same type.
    """
    trainer = write_trainer(
        tmp_path,
        f"""
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--x", required=True)
        args, _ = ap.parse_known_args()
        print("x=" + str({_FIDELITY_LOSSES!r}[args.x]))
        """,
    )
    storage_url = f"journal:///{tmp_path / 'fidelity.journal'}"
    exp = make_experiment(
        experiment="categorical_fidelity",
        workdir=tmp_path / "runs",
        storage=storage_url,
        trial_command=f"python {trainer} {{overrides}}",
        phases=[
            Phase(
                name="pick",
                n_trials=len(_FIDELITY_CHOICES),
                sampler=Sampler(type="grid", seed=0),
                search_space={
                    "x": CategoricalParam(type="categorical", choices=list(_FIDELITY_CHOICES))
                },
            ),
            Phase(
                name="child",
                inherits=["pick"],
                n_trials=1,
                sampler=Sampler(type="random", seed=0),
                search_space={},
            ),
        ],
    )

    # (a) The live value handed to _composed_overrides, i.e. the trainer input.
    live: dict[int, Any] = {}
    real_suggest = _suggest

    def recording_suggest(trial: optuna.Trial, name: str, param: Any) -> Any:
        value = real_suggest(trial, name, param)
        live[trial.number] = value
        return value

    monkeypatch.setattr("phasesweep.engine.phase._suggest", recording_suggest)

    winners = run_experiment(exp)

    # The grid covered every choice exactly once, each with its declared type.
    assert sorted(((type(v).__name__, v) for v in live.values()), key=repr) == sorted(
        ((type(c).__name__, c) for c in _FIDELITY_CHOICES), key=repr
    )

    # (b) FrozenTrial.params, read back out of persistent storage.
    study = optuna.load_study(
        study_name="categorical_fidelity::pick", storage=_resolve_storage(storage_url)
    )
    assert len(study.trials) == len(_FIDELITY_CHOICES)
    for trial in study.trials:
        _assert_identical_scalar(
            trial.params["x"], live[trial.number], surface=f"FrozenTrial.params[{trial.number}]"
        )

    # (c) The per-trial resolved-overrides audit artifact.
    phase_dir = tmp_path / "runs" / "categorical_fidelity" / "pick"
    resolved = _resolved_overrides_by_trial(phase_dir)
    assert set(resolved) == set(live)
    for number, value in live.items():
        _assert_identical_scalar(
            resolved[number]["x"], value, surface=f"overrides_resolved.json[{number}]"
        )

    # (d) The published winner: the bool-valued trial won, and says so.
    winner_doc = yaml.safe_load(_winner_path(exp, "pick").read_text())
    _assert_identical_scalar(
        live[winner_doc["trial_number"]], True, surface="live value of the winning trial"
    )
    _assert_identical_scalar(winner_doc["params"]["x"], True, surface="winner.yaml params")
    _assert_identical_scalar(
        winner_doc["effective_overrides"]["x"], True, surface="winner.yaml effective_overrides"
    )
    _assert_identical_scalar(winners["pick"].params["x"], True, surface="in-memory winner params")

    # (e) What the dependent phase actually inherited and re-published.
    child_resolved = _resolved_overrides_by_trial(tmp_path / "runs" / "categorical_fidelity/child")
    assert child_resolved
    for number, payload in child_resolved.items():
        _assert_identical_scalar(
            payload["x"], True, surface=f"inherited overrides_resolved.json[{number}]"
        )
    child_doc = yaml.safe_load(_winner_path(exp, "child").read_text())
    _assert_identical_scalar(
        child_doc["effective_overrides"]["x"], True, surface="child winner.yaml"
    )
    _assert_identical_scalar(
        winners["child"].effective_overrides["x"], True, surface="in-memory child winner"
    )


@pytest.mark.parametrize(
    "sampler",
    [
        pytest.param(Sampler(type="grid", seed=0), id="grid"),
        pytest.param(Sampler(type="random", seed=0), id="random"),
        pytest.param(
            Sampler(type="tpe", seed=0, acknowledge_nonresumable=True),
            id="tpe",
        ),
    ],
)
def test_categorical_suggestion_round_trips_through_storage_for_every_sampler(
    tmp_path: Path, sampler: Sampler
) -> None:
    """Every sampler's suggestion survives Optuna persistence with its type intact.

    The end-to-end fidelity test above pins grid, the only sampler that visits
    every choice deterministically. This narrower check runs the real
    ``_suggest`` under random and TPE too and compares the value the objective
    received against the value persistent storage hands back
    (PR #5 review / reviewer 2, blocker 1).
    """
    param = CategoricalParam(type="categorical", choices=list(_FIDELITY_CHOICES))
    storage_url = _resolve_storage(f"journal:///{tmp_path / 'round-trip.journal'}")
    study_name = f"round-trip::{sampler.type}"
    live: dict[int, Any] = {}

    def objective(trial: optuna.Trial) -> float:
        live[trial.number] = _suggest(trial, "x", param)
        return 0.0

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=_build_sampler(sampler, {"x": param}),
    )
    study.optimize(objective, n_trials=len(_FIDELITY_CHOICES))

    loaded = optuna.load_study(study_name=study_name, storage=storage_url)
    assert len(loaded.trials) == len(_FIDELITY_CHOICES)
    for trial in loaded.trials:
        _assert_identical_scalar(
            trial.params["x"],
            live[trial.number],
            surface=f"{sampler.type} FrozenTrial.params[{trial.number}]",
        )
