"""Runtime semantics: NaN/inf propagation through extractors, parallel trial behavior, sampler configuration at study-creation time, max_consecutive_failures abort, and Optuna logging suppression."""

from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import optuna
import pytest
import yaml

from phasesweep import load_config, load_experiment, run_experiment, run_suite
from phasesweep.config import (
    CategoricalParam,
    Constraint,
    Experiment,
    IntParam,
    JsonExtractor,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
    WandbExtractor,
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
from phasesweep.engine.artifacts import _load_winner
from phasesweep.engine.optuna import (
    _build_sampler,
    _create_phase_study,
    _resolve_storage,
    _suggest,
)
from phasesweep.engine.paths import (
    _attempts_dir,
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
    PHASE_DECISION_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_ENV_NAMES_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_TARGET_ATTR,
)
from phasesweep.engine.study_policy import _load_phase_policy_state
from phasesweep.engine.trial import (
    ExecutedTrial,
    TrialExecutionError,
    extract_trial_result,
)
from phasesweep.errors import UnsafeProcessCleanupError
from phasesweep.evidence import TrialContext
from phasesweep.runtime import process as runtime_process
from phasesweep.runtime.process import (
    PhaseSweepShutdown,
    ProcessResult,
    ShutdownCleanupReport,
    SignalOwnershipUnavailableError,
    install_signal_handlers,
    signal_handler_scope,
    write_attempt_lifecycle,
)
from tests.conftest import (
    copy_fake_train,
    make_experiment,
    patch_rejected_trial_user_attr,
    write_constant_trainer,
    write_trainer,
    write_yaml,
)


def test_csv_snapshot_throttle_debounces_full_rewrites() -> None:
    throttle = CsvSnapshotThrottle(min_trials=10, min_seconds=30.0)

    assert throttle.should_write(finished=1, now=100.0)
    throttle.mark_written(finished=1, now=100.0)
    assert not throttle.should_write(finished=9, now=120.0)
    assert throttle.should_write(finished=11, now=120.0)
    throttle.mark_written(finished=11, now=120.0)
    assert not throttle.should_write(finished=12, now=149.9)
    assert throttle.should_write(finished=12, now=150.0)


def test_seeded_random_sequence_is_stable_across_top_up_batches(tmp_path: Path) -> None:
    """Seeded random draws depend on durable trial identity, not process lifetime."""
    phase = Phase(
        name="p",
        n_trials=4,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": CategoricalParam(type="categorical", choices=list(range(100)))},
    )

    def sampled(storage: Path, batches: list[int]) -> list[int]:
        for n_trials in batches:
            study = optuna.create_study(
                study_name="stable-random::p",
                storage=f"sqlite:///{storage}",
                sampler=_build_sampler(phase.sampler, phase.search_space),
                load_if_exists=True,
            )
            study.optimize(
                lambda trial: float(trial.suggest_categorical("x", list(range(100)))),
                n_trials=n_trials,
            )
        loaded = optuna.load_study(
            study_name="stable-random::p",
            storage=f"sqlite:///{storage}",
        )
        return [int(trial.params["x"]) for trial in loaded.trials]

    expected = sampled(tmp_path / "single.db", [4])
    assert expected == sampled(tmp_path / "ones.db", [1, 1, 1, 1])
    assert expected == sampled(tmp_path / "uneven.db", [1, 3])
    assert expected == sampled(tmp_path / "mixed.db", [2, 1, 1])


def test_grid_top_up_does_not_repeat_stored_assignments(tmp_path: Path) -> None:
    """A reconstructed GridSampler continues through its stored grid assignments."""
    search_space = {"x": CategoricalParam(type="categorical", choices=[1, 2, 3, 4])}
    sampler = Sampler(type="grid", seed=0)
    storage = f"sqlite:///{tmp_path / 'grid.db'}"

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
    storage = (
        f"journal:///{tmp_path / 'tpe.journal'}"
        if sampler.type == "tpe"
        else f"sqlite:///{tmp_path / f'{sampler.type}.db'}"
    )
    exp = make_experiment(
        experiment=f"sampler_{sampler.type}",
        storage=storage,
        workdir=tmp_path / "runs",
        phases=[phase],
    )
    _create_phase_study(exp, phase)
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


def _sleeping_score_experiment(
    tmp_path: Path,
    *,
    experiment: str,
    n_trials: int = 3,
    timeout_seconds_per_phase: float | None = None,
    timeout_seconds_per_run: float | None = None,
    allow_incomplete_on_timeout: bool = False,
    sleep_seconds: float = 0.5,
) -> Experiment:
    trainer = write_trainer(
        tmp_path,
        f"""
        import argparse, json, time
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        time.sleep({sleep_seconds})
        with open(args.out, "w") as f:
            json.dump({{"x": 1.0}}, f)
        print("x=1.0")
        """,
    )
    return Experiment(
        experiment=experiment,
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
        timeout_seconds_per_run=timeout_seconds_per_run,
        phases=[
            Phase(
                name="p",
                n_trials=n_trials,
                timeout_seconds_per_phase=timeout_seconds_per_phase,
                allow_incomplete_on_timeout=allow_incomplete_on_timeout,
                search_space={},
            )
        ],
    )


def test_parallel_trials_e2e(tmp_path):
    """Run a phase with n_jobs=4 on the synthetic trainer. Exercises:
    - JournalFileStorage via explicit journal:/// URL (review v0.5.2 / blocker 6)
    - constant_liar on TPE
    - concurrent subprocess execution
    - no database-locked errors
    """
    trainer = copy_fake_train(tmp_path)

    journal_path = tmp_path / "phases.journal"
    yaml_text = f"""
experiment: parallel_test
storage: journal:///{journal_path}
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} {{overrides}}"
override_format: argparse
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json_envelope, path: result.json, objective_name: eval_loss, split: validation, policy: synthetic }}
phases:
  - name: lr_sweep
    n_trials: 8
    n_jobs: 4
    allow_no_gpu_isolation: true
    sampler: {{ type: tpe, seed: 42, acknowledge_nonresumable: true }}
    search_space:
      lr: {{ type: float, low: 1e-5, high: 1e-2, log: true }}
"""
    yaml_path = tmp_path / "exp.yaml"
    yaml_path.write_text(yaml_text)

    exp = load_experiment(yaml_path)
    winners = run_experiment(exp)

    assert "lr_sweep" in winners
    assert 1e-5 <= winners["lr_sweep"].params["lr"] <= 1e-2

    # Verify all 8 trials actually ran (trial directories exist).
    # v0.5.7: outputs are now namespaced as <workdir>/<experiment>/<phase>/.
    runs_dir = tmp_path / "runs" / "parallel_test" / "lr_sweep"
    trial_dirs = sorted(runs_dir.glob("trial_*"))
    assert len(trial_dirs) == 8

    assert journal_path.exists(), "JournalFileStorage file should exist"


def test_failed_trials_marked_fail_not_complete(tmp_path):
    """Process crashes should produce FAIL trials, not COMPLETE with inf."""
    db_path = tmp_path / "phases.db"
    yaml_text = f"""
experiment: fail_state_test
storage: sqlite:///{db_path}
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path / "runs"}
trial_command: "false {{overrides}}"
override_format: argparse
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json_envelope, path: result.json, objective_name: eval_loss, split: validation, policy: synthetic }}
phases:
  - name: a
    n_trials: 3
    max_consecutive_failures: 10
    sampler: {{ type: random, seed: 0 }}
    search_space: {{ x: {{ type: float, low: 0, high: 1 }} }}
"""
    yaml_path = tmp_path / "exp.yaml"
    yaml_path.write_text(yaml_text)
    exp = load_experiment(yaml_path)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(exp)

    study = optuna.load_study(study_name="fail_state_test::a", storage=f"sqlite:///{db_path}")
    for trial in study.get_trials():
        # Every trial should be FAIL, not COMPLETE.
        assert trial.state == optuna.trial.TrialState.FAIL, (
            f"Trial {trial.number} is {trial.state.name}, expected FAIL"
        )


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


def test_terminal_callback_reports_success_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = make_experiment(workdir=tmp_path / "runs")
    captured: list[TerminalReport] = []

    def preflight(_experiment, *, cleanup_report, from_phase, preloaded_studies=None):
        del from_phase, preloaded_studies
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


def test_constraint_extractor_failure_marks_trial_fail(tmp_path):
    """Missing constraint output -> TrialState.FAIL, not COMPLETE+infeasible."""
    trainer = tmp_path / "trainer.py"
    write_trainer(
        trainer,
        """
        import json, sys, argparse
        ap = argparse.ArgumentParser()
        ap.add_argument('--out', required=True)
        args, _ = ap.parse_known_args()
        # Write metric only — constraint extractor will fail to find param_bytes.
        with open(args.out, 'w') as f:
            json.dump({'eval_loss': 1.0}, f)
        print('eval_loss=1.0')
        """,
    )

    db = tmp_path / "phases.db"
    yaml_text = f"""
experiment: c2
storage: sqlite:///{db}
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} --out {{trial_dir}}/result.json {{overrides}}"
override_format: argparse
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: log_regex, pattern: 'eval_loss=(?P<value>[0-9.eE+-]+)' }}
constraints:
  - name: param_bytes
    extractor: {{ type: json, path: result.json, key: param_bytes }}
    max: 1000
phases:
  - name: a
    n_trials: 2
    max_consecutive_failures: 10
    sampler: {{ type: random, seed: 0 }}
    search_space: {{ x: {{ type: float, low: 0, high: 1 }} }}
"""
    p = tmp_path / "exp.yaml"
    p.write_text(yaml_text)
    exp = load_experiment(p)
    from phasesweep.engine.selection import NoFeasibleTrialError

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(exp)

    study = optuna.load_study(study_name="c2::a", storage=f"sqlite:///{db}")
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
                extractor=JsonExtractor(
                    type="json",
                    path="result.json",
                    key="param_bytes",
                ),
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
        process=ProcessResult(
            return_code=0,
            timed_out=False,
            pid=123,
            duration_seconds=0.1,
        ),
    )

    result = extract_trial_result(experiment=experiment, executed=executed)

    assert result.metric is None
    assert result.feasible is False
    assert result.failure_reason == failure_reason


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
    yaml_text = f"""
experiment: abort_recheck
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} {{overrides}}"
override_format: argparse
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json_envelope, objective_name: eval_loss, split: test, policy: test }}
phases:
  - name: p
    n_trials: 16
    n_jobs: 4
    gpu_ids: [0]   # only one slot — n_jobs=4 will queue
    max_consecutive_failures: 1
    sampler: {{ type: random, seed: 0 }}
    search_space: {{ x: {{ type: float, low: 0, high: 1 }} }}
"""
    p = tmp_path / "exp.yaml"
    p.write_text(yaml_text)
    exp = load_experiment(p)

    from phasesweep.engine.selection import NoFeasibleTrialError

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

    for case, initial, verbose, expected in cases:
        optuna.logging.set_verbosity(initial)
        _configure_logging(verbose=verbose)
        assert optuna.logging.get_verbosity() == expected, case

    # Reset for any later tests.
    logging.getLogger().handlers.clear()


def test_runtime_rejects_unexplained_trial_budget_shortfall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sampler that stops early cannot publish a falsely complete winner."""
    exp = _sleeping_score_experiment(tmp_path, experiment="budget_shortfall", n_trials=1)
    monkeypatch.setattr(optuna.Study, "optimize", lambda self, objective, **kwargs: None)

    with pytest.raises(RuntimeError, match="stopped after 0/1 terminal trials") as exc_info:
        run_experiment(exp)
    assert not isinstance(exc_info.value, PhaseSweepError)


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

    body = f"""
experiment: platform_check
storage: sqlite:///{tmp_path}/platform.db
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path}/runs
trial_command: "echo {{overrides}}"
override_format: argparse
metric:
  name: x
  goal: minimize
  extractor: {{ type: json_envelope, objective_name: x, split: test, policy: test }}
phases:
  - name: p
    n_trials: 1
    sampler: {{ type: random, seed: 0 }}
    search_space: {{ x: {{ type: int, low: 0, high: 1 }} }}
"""
    exp = load_experiment(write_yaml(tmp_path, body))

    monkeypatch.setattr(runtime_files, "_supports_posix_runtime_features", lambda: False)

    with pytest.raises(RuntimeError, match="requires a POSIX platform"):
        run_experiment(exp)

    # Dry-run remains available because it launches no subprocesses and takes no locks.
    winners = run_experiment(exp, dry_run=True)
    assert set(winners) == {"p"}


def test_max_consecutive_failures_aborts_phase(tmp_path):
    """Trial command always fails -> phase aborts before running n_trials."""
    body = f"""
experiment: failtest
storage: sqlite:///{tmp_path}/fail.db
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path}/runs
trial_command: "false {{overrides}}"
override_format: argparse
metric:
  name: loss
  goal: minimize
  extractor: {{ type: json_envelope, objective_name: loss, split: test, policy: test }}
phases:
  - name: a
    n_trials: 100
    max_consecutive_failures: 3
    sampler: {{ type: random, seed: 0 }}
    search_space: {{ x: {{ type: float, low: 0, high: 1 }} }}
"""
    exp = load_experiment(write_yaml(tmp_path, body))
    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(exp)
    # Verify only a small number of trials actually executed before the abort.
    # We can't predict exactly how many because Optuna may have a few in flight,
    # but it should be << 100.
    import sqlite3

    conn = sqlite3.connect(tmp_path / "fail.db")
    n = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    conn.close()
    assert n < 30, f"expected early abort, got {n} trials"


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
    db = tmp_path / "abort.db"
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{db}",
        trial_command=f"python {trainer} {{overrides}}",
        override_format="argparse",
        n_trials=3,
        max_consecutive_failures=2,
        sampler={"type": "random", "seed": 7},
    )

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(exp)

    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
    record = study.user_attrs[PHASE_ABORT_ATTR]
    assert record["policy"] == "max_consecutive_failures"
    assert record["consecutive_failures"] == 2

    # Identical retry: no new work is schedulable (3/3 terminal trials), so the
    # durable abort must keep the phase failed instead of publishing.
    with pytest.raises(NoFeasibleTrialError, match="previously aborted"):
        run_experiment(exp)
    assert not _last_successful_generation_path(exp).exists()


def test_abort_recovery_target_with_no_remaining_slots_is_schema_mismatch(
    tmp_path: Path,
) -> None:
    """Contradictory durable abort/trial counts are operator-visible study state."""
    trainer = write_trainer(tmp_path / "trainer.py", "raise SystemExit(1)")
    storage = f"sqlite:///{tmp_path / 'abort.db'}"

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

    study = optuna.load_study(study_name="t::p", storage=storage)
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


def test_topup_after_abort_runs_new_work_and_clears_durable_abort(tmp_path: Path) -> None:
    """Raising n_trials after an abort is the explicit resume path.

    The top-up schedules genuinely new attempts; reaching a successful winner
    selection consumes the durable abort record (review v0.5.17 / blocker 1).
    """
    flag = tmp_path / "resume_enabled"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import pathlib, sys
        if not pathlib.Path({str(flag)!r}).exists():
            sys.exit(1)
        print("x=0.5")
        """,
    )
    db = tmp_path / "abort.db"

    def _exp(n_trials: int):
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=f"sqlite:///{db}",
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
    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
    assert study.user_attrs.get(PHASE_ABORT_ATTR) is None


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
    db = tmp_path / "topup.db"

    def _exp(n_trials: int) -> Experiment:
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=f"sqlite:///{db}",
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

    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
    assert [trial.state.name for trial in study.trials] == ["COMPLETE", "FAIL", "FAIL", "FAIL"]
    assert [trial.user_attrs[TRIAL_OUTCOME_ATTR]["sequence"] for trial in study.trials] == [
        1,
        2,
        3,
        4,
    ]
    assert study.user_attrs[PHASE_ABORT_ATTR]["consecutive_failures"] == 3
    assert _last_successful_generation_path(_exp(4)).read_text() == published_before


def test_outcome_ledger_recovers_when_abort_marker_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed marker write cannot make an identical retry publish."""
    trainer = write_trainer(tmp_path / "trainer.py", "raise SystemExit(1)")
    db = tmp_path / "abort.db"
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{db}",
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

    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
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


def test_transient_trial_outcome_write_failure_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One flaky outcome write must not cost the phase a trial.

    The bounded retry is what keeps the fail-closed reaction to an *unwritable*
    ledger from firing on a momentarily busy one (PR #5 review / reviewer 2
    pass 2, blocker 4).
    """
    db = tmp_path / "retry.db"
    exp = _outcome_write_experiment(
        tmp_path,
        trainer_body='print("x=0.5")',
        storage=f"sqlite:///{db}",
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
    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
    assert [trial.state.name for trial in study.trials] == ["COMPLETE", "COMPLETE"]
    assert [trial.user_attrs[TRIAL_OUTCOME_ATTR]["sequence"] for trial in study.trials] == [1, 2]
    assert all(
        trial.user_attrs[TRIAL_OUTCOME_ATTR]["outcome"] == "success" for trial in study.trials
    )


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
    db = tmp_path / "outcome.db"
    exp = _outcome_write_experiment(
        tmp_path,
        trainer_body=trainer_body,
        storage=f"sqlite:///{db}",
        constraints=constraints,
    )

    real_set_user_attr = patch_rejected_trial_user_attr(
        monkeypatch,
        TRIAL_OUTCOME_ATTR,
        "injected outcome write failure",
    )
    with pytest.raises(StudyStorageUnavailableError, match="deliberately left RUNNING"):
        run_experiment(exp)

    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
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
    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
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

    db = tmp_path / "abort.db"

    def _exp(n_trials: int) -> Experiment:
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=f"sqlite:///{db}",
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
                (trial_dir / runtime_process.PROCESS_IDENTITY_FILE).unlink()
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

    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
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
    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
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

    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
    assert len(study.trials) == before + 1
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [unsafe_trial.number]
    assert list(_attempts_dir(_exp(3)).glob("*.json")) == []


@pytest.mark.parametrize("wandb_source", ["metric", "gate"])
def test_wandb_cleanup_uncertainty_blocks_topup_until_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wandb_source: str,
) -> None:
    """An uncertain W&B worker remains recoverable through the trial root."""
    db = tmp_path / "wandb-abort.db"
    trainer = write_trainer(tmp_path / "noop.py", "print('x=0.5')")

    def _exp(n_trials: int) -> Experiment:
        wandb = {
            "entity": "team",
            "project": "project",
            "poll_seconds": 0.01,
            "timeout_seconds": 1.0,
        }
        metric = Metric(extractor=WandbExtractor(type="wandb", metric_key="eval/loss", **wandb))
        if wandb_source == "metric":
            gates = []
        else:
            gates = [
                WandbSummaryRequiredGate(type="wandb_summary_required", keys=["gate"], **wandb)
            ]
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=f"sqlite:///{db}",
            trial_command=f"python {trainer} {{overrides}}",
            override_format="argparse",
            metric=metric,
            phases=[
                Phase(
                    name="p",
                    n_trials=n_trials,
                    gpu_policy="none",
                    sampler=Sampler(type="random", seed=7),
                    gates=gates,
                )
            ],
        )

    real_worker_supervisor = runtime_process.run_supervised
    cleanup_safe = {"value": False}
    worker_calls = {"count": 0}
    reports: list[TerminalReport] = []
    failure_call = 1 if wandb_source == "metric" else 2

    def supervise_wandb_worker(command: str, **kwargs: Any) -> ProcessResult:
        worker_calls["count"] += 1
        response_path = Path(shlex.split(command)[-1])
        result = real_worker_supervisor("true", **kwargs)
        if worker_calls["count"] == failure_call:
            result.cleanup_confirmed = False
            trial_dir = Path(kwargs["trial_dir"])
            write_attempt_lifecycle(
                trial_dir,
                attempt_id=str(kwargs["attempt_id"]),
                state="launching",
            )
        else:
            response_path.write_text(
                json.dumps(
                    {
                        "status": "summary",
                        "summary": {"eval/loss": 0.5, "gate": "present"},
                    }
                ),
                encoding="utf-8",
            )
        return result

    # ``poll_wandb_summary`` imports this name dynamically. The trainer keeps
    # the original alias from ``engine.trial``, so it still has a real
    # supervised process of its own before the W&B worker takes over the
    # durable identity in the same trial directory.
    monkeypatch.setattr("phasesweep.runtime.process.run_supervised", supervise_wandb_worker)
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process",
        lambda _identity: cleanup_safe["value"],
    )
    monkeypatch.setattr(
        "phasesweep.engine.cleanup.cleanup_stale_trial_process",
        lambda _identity: cleanup_safe["value"],
    )

    with pytest.raises(UnsafeProcessCleanupError, match="W&B polling worker cleanup"):
        run_experiment(_exp(1), terminal_callback=reports.append)

    # The gate case completes the metric poll before its own worker takes over
    # the same durable identity and leaves cleanup uncertain.
    assert worker_calls["count"] == failure_call
    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
    unsafe_trial = study.get_trials(deepcopy=False)[0]
    unsafe_attempt_id = unsafe_trial.user_attrs[ATTEMPT_ID_ATTR]
    unsafe_trial_dir = Path(str(unsafe_trial.user_attrs[TRIAL_DIR_ATTR]))
    assert isinstance(unsafe_attempt_id, str)
    assert len(reports) == 1
    assert reports[0].cleanup_confirmed is False
    assert reports[0].uncertain_attempt_ids == frozenset({unsafe_attempt_id})
    assert unsafe_trial.user_attrs[CLEANUP_CONFIRMED_ATTR] is False
    assert unsafe_trial.user_attrs[TRIAL_OUTCOME_ATTR]["policy"] == "unsafe_process_cleanup"
    assert study.user_attrs[PHASE_ABORT_ATTR]["policy"] == "unsafe_process_cleanup"
    assert (unsafe_trial_dir / runtime_process.PROCESS_IDENTITY_FILE).is_file()
    active_attempts = list(_attempts_dir(_exp(1)).glob("*.json"))
    assert len(active_attempts) == 1
    active_attempt = json.loads(active_attempts[0].read_text(encoding="utf-8"))
    assert active_attempt["attempt_id"] == unsafe_attempt_id

    # Raising the target is not cleanup authority, so the second invocation
    # must stop before creating a new trial or worker.
    before = len(study.trials)
    with pytest.raises(ProcessCleanupUncertainError):
        run_experiment(_exp(2))
    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
    assert len(study.trials) == before
    assert worker_calls["count"] == failure_call
    assert list(_attempts_dir(_exp(2)).glob("*.json")) == active_attempts

    cleanup_safe["value"] = True
    winners = run_experiment(_exp(2))

    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
    assert winners["p"].metric == pytest.approx(0.5)
    assert worker_calls["count"] == (2 if wandb_source == "metric" else 4)
    assert [trial.state for trial in study.get_trials(deepcopy=False)] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [unsafe_trial.number]
    assert list(_attempts_dir(_exp(2)).glob("*.json")) == []


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
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import pathlib, sys
        if not pathlib.Path({str(flag)!r}).exists():
            sys.exit(1)
        print("x=0.5")
        """,
    )
    db = tmp_path / "abort.db"

    def _exp(n_trials: int):
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=f"sqlite:///{db}",
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
    study = optuna.load_study(study_name="t::p", storage=f"sqlite:///{db}")
    assert study.user_attrs.get(PHASE_ABORT_ATTR) is None

    # ...so the identical replay republishes deterministically instead of
    # wedging on "previously aborted".
    monkeypatch.setattr("phasesweep.engine.phase._select_phase_winner", real_select)
    winners = run_experiment(_exp(6))
    assert winners["p"].metric == pytest.approx(0.5)


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


def test_parallel_unexpected_objective_error_is_phase_fatal_and_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        n_jobs=2,
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

    monkeypatch.setattr(
        "phasesweep.engine.phase.extract_trial_result",
        real_extract,
    )
    with pytest.raises(NoFeasibleTrialError, match="previously aborted"):
        run_experiment(exp)
    assert not _last_successful_generation_path(exp).exists()


def test_phase_timeout_refuses_incomplete_winner(tmp_path: Path) -> None:
    """A phase wallclock timeout must not bless the best partial trial by default."""
    exp = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout",
        timeout_seconds_per_phase=0.2,
    )

    with pytest.raises(
        TimeoutError,
        match=r"timed out via phase guard .*Refusing to select a winner",
    ):
        run_experiment(exp)


def test_phase_timeout_preempts_active_trial(tmp_path: Path) -> None:
    """A phase wallclock timeout is a hard subprocess deadline, not only an Optuna scheduler timeout."""
    # The trainer would sleep 30s per trial; timing margins are deliberately
    # enormous on both sides so a loaded CI host cannot flip the verdict: a
    # preempted run finishes in well under 10s, a non-preempted one needs 30+.
    exp = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout_hard",
        timeout_seconds_per_phase=0.05,
        sleep_seconds=30.0,
    )

    started = time.monotonic()
    with pytest.raises(
        TimeoutError,
        match=r"timed out via phase guard .*Refusing to select a winner",
    ):
        run_experiment(exp)
    elapsed = time.monotonic() - started

    assert elapsed < 10.0
    # Causal marker: had the trial run to completion, it would have written
    # its result file.
    phase_dir = tmp_path / "runs" / "phase_timeout_hard" / "p"
    assert list(phase_dir.glob("trial_00000__*/r.json")) == []


def test_incomplete_timeout_can_be_explicitly_accepted(tmp_path: Path) -> None:
    # Two-sided timing margin: one 1s trial must finish well within the 6s
    # budget even on a loaded host, while all ten (>= 10s of sleeping alone)
    # can never finish inside it.
    exp = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout_allowed",
        n_trials=10,
        timeout_seconds_per_phase=6.0,
        allow_incomplete_on_timeout=True,
        sleep_seconds=1.0,
    )

    winners = run_experiment(exp)

    completion = winners["p"].completion
    assert completion["requested_trials"] == 10
    assert 1 <= completion["completed_trials"] < completion["requested_trials"]
    assert completion["completed_trials"] <= completion["finished_trials"]
    assert completion["finished_trials"] <= completion["requested_trials"]
    assert completion["incomplete"] is True
    assert completion["reason"] == "timeout"
    assert completion["timeout_scope"] == "phase"


@pytest.mark.parametrize("allow_incomplete_on_timeout", [False, True])
def test_timeout_after_all_terminal_trials_is_complete_enough(
    tmp_path: Path,
    allow_incomplete_on_timeout: bool,
) -> None:
    """A timeout guard should not reject a phase once every requested trial is terminal."""
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json, os, time
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        if os.environ["PHASESWEEP_TRIAL_ID"] == "0":
            with open(args.out, "w") as f:
                json.dump({"x": 1.0}, f)
            print("x=1.0")
        else:
            time.sleep(30.0)
        """,
    )
    exp = Experiment(
        experiment="phase_timeout_all_terminal",
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
        phases=[
            Phase(
                name="p",
                n_trials=2,
                timeout_seconds_per_phase=8.0,
                allow_incomplete_on_timeout=allow_incomplete_on_timeout,
                search_space={},
            )
        ],
    )

    winners = run_experiment(exp)

    completion = winners["p"].completion
    assert completion["requested_trials"] == 2
    assert completion["completed_trials"] == 1
    assert completion["finished_trials"] == 2
    assert completion["incomplete"] is False

    current = exp.model_copy(
        update={
            "phases": [
                exp.phases[0].model_copy(update={"allow_incomplete_on_timeout": False}),
            ],
        }
    )
    loaded = _load_winner(current, current.phases[0], {})
    assert loaded.completion["incomplete"] is False


def test_timeout_winner_is_not_masked_by_consecutive_failure_abort(tmp_path: Path) -> None:
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json, os, time
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        if os.environ["PHASESWEEP_TRIAL_ID"] == "0":
            with open(args.out, "w") as f:
                json.dump({"x": 1.0}, f)
            print("x=1.0")
        else:
            time.sleep(30.0)
        """,
    )
    exp = Experiment(
        experiment="phase_timeout_allowed_abort_counter",
        workdir=str(tmp_path / "runs"),
        storage=f"sqlite:///{tmp_path / 'timeout-counter.db'}",
        provenance={"revision": "test-fixture-v1"},
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
        phases=[
            Phase(
                name="p",
                n_trials=3,
                max_consecutive_failures=1,
                timeout_seconds_per_phase=8.0,
                allow_incomplete_on_timeout=True,
                sampler=Sampler(type="random", seed=7),
                search_space={},
            )
        ],
    )

    winners = run_experiment(exp)

    assert winners["p"].trial_number == 0
    completion = winners["p"].completion
    assert completion["requested_trials"] == 3
    assert 1 <= completion["completed_trials"] < completion["requested_trials"]
    assert (
        completion["completed_trials"]
        <= completion["finished_trials"]
        < completion["requested_trials"]
    )
    assert completion["incomplete"] is True
    assert completion["reason"] == "timeout"
    assert completion["timeout_scope"] == "phase"
    study = optuna.load_study(
        study_name="phase_timeout_allowed_abort_counter::p",
        storage=exp.storage,
    )
    deadline_trials = [
        trial
        for trial in study.trials
        if trial.user_attrs.get(TRIAL_OUTCOME_ATTR, {}).get("outcome") == "cancelled"
    ]
    assert len(deadline_trials) == 1
    assert study.user_attrs.get(PHASE_ABORT_ATTR) is None
    assert _load_phase_policy_state(study).consecutive_failures == 0


@pytest.mark.parametrize("clock_elapses", [True, False])
def test_scheduler_deadline_decides_partial_winner_versus_failure_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock_elapses: bool
) -> None:
    """Only an elapsed clock lets a failure-aborted phase publish a partial winner.

    The failing trial exits on its own before its wallclock-capped subprocess
    timeout fires, so no per-trial cause sets ``deadline_exhausted``, and the
    same trial trips ``max_consecutive_failures``. When the phase clock also
    elapses while that trial is in flight, only the scheduler-level check can
    observe the deadline, and timeout precedence must still publish the earlier
    successful trial. With budget left, nothing is relabelled: the streak abort
    stands and the phase fails.
    """
    import types

    import phasesweep.engine.phase as phase_mod

    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json, os, sys
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        if os.environ["PHASESWEEP_TRIAL_ID"] == "0":
            with open(args.out, "w") as f:
                json.dump({"x": 1.0}, f)
            print("x=1.0")
        else:
            sys.exit(1)
        """,
    )
    exp = Experiment(
        experiment="phase_scheduler_deadline_with_abort",
        workdir=str(tmp_path / "runs"),
        storage=f"sqlite:///{tmp_path / 'decision.db'}",
        provenance={"revision": "test-fixture-v1"},
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
        phases=[
            Phase(
                name="p",
                n_trials=3,
                max_consecutive_failures=1,
                timeout_seconds_per_phase=600.0,
                allow_incomplete_on_timeout=True,
                gpu_policy="none",
                allow_no_gpu_isolation=True,
                sampler=Sampler(
                    type="tpe",
                    seed=7,
                    n_startup_trials=10,
                    acknowledge_nonresumable=True,
                ),
                search_space={"x": IntParam(type="int", low=1, high=3)},
            ),
            Phase(
                name="child",
                inherits=["p"],
                n_trials=1,
                gpu_policy="none",
                allow_no_gpu_isolation=True,
                sampler=Sampler(type="random", seed=7),
                search_space={},
            ),
        ],
    )

    # Only the phase module's clock is virtualised: the real clock still bounds
    # Optuna's own timeout, GPU leases, and extraction, so no other code path
    # can attribute a deadline here.
    offset = {"seconds": 0.0}
    real_monotonic = time.monotonic
    monkeypatch.setattr(
        phase_mod,
        "time",
        types.SimpleNamespace(
            monotonic=lambda: real_monotonic() + offset["seconds"],
            sleep=lambda _seconds: None,
        ),
    )
    real_launch_trial = phase_mod.launch_trial
    launched_trials: list[int] = []

    def launch_and_burn_the_budget(**kwargs: object) -> object:
        launched_trials.append(int(kwargs["trial_id"]))
        executed = real_launch_trial(**kwargs)
        if clock_elapses and kwargs["trial_id"] == 1:
            offset["seconds"] = 10_000.0
        return executed

    monkeypatch.setattr(phase_mod, "launch_trial", launch_and_burn_the_budget)

    if not clock_elapses:
        with pytest.raises(NoFeasibleTrialError, match="aborted after 1 consecutive failures"):
            run_experiment(exp)
        return

    real_select = phase_mod._select_phase_winner
    crashed = {"once": False}

    def crash_first_selection(*args: object, **kwargs: object):
        if not crashed["once"]:
            crashed["once"] = True
            raise RuntimeError("simulated crash after partial-timeout decision")
        return real_select(*args, **kwargs)

    monkeypatch.setattr(phase_mod, "_select_phase_winner", crash_first_selection)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_experiment(exp)

    study = optuna.load_study(
        study_name="phase_scheduler_deadline_with_abort::p",
        storage=exp.storage,
    )
    assert study.user_attrs[PHASE_DECISION_ATTR]["decision"] == "accepted_partial_timeout"
    trial_count = len(study.trials)

    winners = run_experiment(exp)

    study = optuna.load_study(
        study_name="phase_scheduler_deadline_with_abort::p",
        storage=exp.storage,
    )
    assert len(study.trials) == trial_count

    assert winners["p"].trial_number == 0
    completion = winners["p"].completion
    assert completion["requested_trials"] == 3
    assert completion["finished_trials"] == 2
    assert completion["completed_trials"] == 1
    assert completion["incomplete"] is True
    assert completion["reason"] == "timeout"
    assert completion["timeout_scope"] == "phase"

    # The child study now binds the parent's published winner. A third
    # identical run must still replay both studies without treating the
    # parent's accepted partial target as a top-up, and without asking TPE to
    # reconstruct continuation state for suggestions that will never launch.
    launch_count = len(launched_trials)
    replayed = run_experiment(exp)
    assert replayed["p"].trial_number == winners["p"].trial_number
    assert replayed["child"].trial_number == winners["child"].trial_number
    assert len(launched_trials) == launch_count


def test_refused_partial_timeout_consumes_simultaneous_failure_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raising the timeout must remain a valid retry after timeout wins a race."""
    import types

    import phasesweep.engine.phase as phase_mod

    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json, os, sys
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        if os.environ["PHASESWEEP_TRIAL_ID"] == "0":
            with open(args.out, "w") as f:
                json.dump({"x": 1.0}, f)
            print("x=1.0")
        else:
            sys.exit(1)
        """,
    )
    exp = Experiment(
        experiment="refused_partial_timeout_abort",
        workdir=str(tmp_path / "runs"),
        storage=f"sqlite:///{tmp_path / 'refused.db'}",
        provenance={"revision": "test-fixture-v1"},
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
        phases=[
            Phase(
                name="p",
                n_trials=3,
                max_consecutive_failures=1,
                timeout_seconds_per_phase=600.0,
                gpu_policy="none",
                allow_no_gpu_isolation=True,
                sampler=Sampler(type="random", seed=7),
                search_space={},
            )
        ],
    )

    offset = {"seconds": 0.0}
    real_monotonic = time.monotonic
    monkeypatch.setattr(
        phase_mod,
        "time",
        types.SimpleNamespace(
            monotonic=lambda: real_monotonic() + offset["seconds"],
            sleep=lambda _seconds: None,
        ),
    )
    real_launch_trial = phase_mod.launch_trial

    def launch_and_expire(**kwargs: object) -> object:
        executed = real_launch_trial(**kwargs)
        if kwargs["trial_id"] == 1:
            offset["seconds"] = 10_000.0
        return executed

    monkeypatch.setattr(phase_mod, "launch_trial", launch_and_expire)
    with pytest.raises(TimeoutError, match="Refusing to select a winner"):
        run_experiment(exp)

    study = optuna.load_study(
        study_name="refused_partial_timeout_abort::p",
        storage=exp.storage,
    )
    assert study.user_attrs.get(PHASE_ABORT_ATTR) is None
    assert _load_phase_policy_state(study).consecutive_failures == 0

    # The operator's documented remedy is now viable: with a fresh/larger
    # budget, the remaining attempt can run instead of the stale failure abort
    # rejecting the invocation during phase startup.
    offset["seconds"] = 0.0
    write_constant_trainer(tmp_path)
    monkeypatch.setattr(phase_mod, "launch_trial", real_launch_trial)
    winners = run_experiment(exp)
    assert winners["p"].metric == pytest.approx(0.5)


def test_shutdown_during_objective_does_not_persist_fatal_phase_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An orchestrator signal is cancellation, not an objective implementation bug."""
    import phasesweep.engine.phase as phase_mod

    trainer = write_constant_trainer(tmp_path)
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'shutdown.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=2,
        gpu_policy="none",
        allow_no_gpu_isolation=True,
        max_consecutive_failures=1,
        sampler={"type": "random", "seed": 7},
    )
    real_launch_trial = phase_mod.launch_trial
    interrupted = {"once": False}

    def interrupt_first_launch(**kwargs: object) -> object:
        if not interrupted["once"]:
            interrupted["once"] = True
            report = ShutdownCleanupReport(
                signum=signal.SIGTERM,
                cleanup_confirmed=True,
                child_pgids=(),
            )
            raise PhaseSweepShutdown(signal.SIGTERM, report)
        return real_launch_trial(**kwargs)

    monkeypatch.setattr(phase_mod, "launch_trial", interrupt_first_launch)
    with pytest.raises(PhaseSweepShutdown):
        run_experiment(exp)

    study = optuna.load_study(study_name="t::p", storage=exp.storage)
    assert study.user_attrs.get(PHASE_ABORT_ATTR) is None
    assert study.trials[0].user_attrs[TRIAL_OUTCOME_ATTR]["outcome"] == "cancelled"

    winners = run_experiment(exp)
    assert winners["p"].metric == pytest.approx(0.5)
    study = optuna.load_study(study_name="t::p", storage=exp.storage)
    assert study.user_attrs.get(PHASE_ABORT_ATTR) is None


@pytest.mark.parametrize("lease_timeout", [True, False])
def test_gpu_lease_timeout_type_decides_partial_winner_versus_fatal_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease_timeout: bool
) -> None:
    """Only ``GpuLeaseTimeoutError`` may relabel a lease wait as wallclock exhaustion.

    Phase attribution narrows on the dedicated subclass. A plain
    ``TimeoutError`` escaping the lock layer is an infrastructure failure: it
    must surface as a fatal abort, never let ``allow_incomplete_on_timeout``
    publish a partial winner off broken infrastructure. This pins the
    ``except GpuLeaseTimeoutError`` contract in ``engine/phase.py`` — widening
    it to ``TimeoutError`` or raising the plain superclass from the pool fails
    one side of this parametrization.
    """
    import contextlib

    import phasesweep.engine.phase as phase_mod
    from phasesweep.runtime.gpu import GpuAssignment, GpuLeaseTimeoutError

    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        with open(args.out, "w") as f:
            json.dump({"x": 1.0}, f)
        print("x=1.0")
        """,
    )
    exp = Experiment(
        experiment="gpu_lease_timeout_attribution",
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
        phases=[
            Phase(
                name="p",
                n_trials=3,
                max_consecutive_failures=1,
                timeout_seconds_per_phase=600.0,
                allow_incomplete_on_timeout=True,
                search_space={},
            )
        ],
    )

    exc_type = GpuLeaseTimeoutError if lease_timeout else TimeoutError

    class _FailingSecondLeasePool:
        def __init__(self) -> None:
            self.calls = 0

        @contextlib.contextmanager
        def acquire(self, *, deadline: float | None = None):
            self.calls += 1
            if self.calls > 1:
                raise exc_type("GPU lease wait exhausted")
            yield GpuAssignment(visible_devices=None, lease_fds=())

    monkeypatch.setattr(
        phase_mod.GpuPool, "create", classmethod(lambda cls, **kwargs: _FailingSecondLeasePool())
    )

    if not lease_timeout:
        with pytest.raises(TimeoutError, match="GPU lease wait exhausted"):
            run_experiment(exp)
        return

    winners = run_experiment(exp)

    assert winners["p"].trial_number == 0
    completion = winners["p"].completion
    assert completion["incomplete"] is True
    assert completion["reason"] == "timeout"


def test_incomplete_timeout_winner_requires_current_opt_in_on_resume(tmp_path: Path) -> None:
    accepted = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout_resume_guard",
        n_trials=10,
        timeout_seconds_per_phase=6.0,
        allow_incomplete_on_timeout=True,
        sleep_seconds=1.0,
    )
    run_experiment(accepted)

    current = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout_resume_guard",
        n_trials=10,
        timeout_seconds_per_phase=6.0,
        sleep_seconds=1.0,
    )
    with pytest.raises(RuntimeError, match="incomplete phase result"):
        _load_winner(current, current.phases[0], {})


def test_run_timeout_refuses_incomplete_winner(tmp_path: Path) -> None:
    exp = _sleeping_score_experiment(
        tmp_path,
        experiment="run_timeout",
        timeout_seconds_per_run=0.2,
    )

    with pytest.raises(TimeoutError, match="run guard"):
        run_experiment(exp)


def test_noop_rerun_skips_gpu_discovery_and_target_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed phase republishes from a host that cannot launch work.

    Reading or republishing a finished GPU experiment from a CPU-only login
    node must not require GPU discovery or mutate the durable accepted target
    (review v0.5.14 / blocker 4).
    """
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json
        p = argparse.ArgumentParser()
        p.add_argument("--out")
        p.add_argument("--x", type=int, default=0)
        a, _ = p.parse_known_args()
        print(f"x={a.x}")
        """,
    )
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
        sampler=Sampler(type="random", seed=0),
    )
    first = run_experiment(experiment)

    def _no_gpu_create(**_kwargs: object) -> None:
        raise RuntimeError("simulated: no GPUs detected on this host")

    monkeypatch.setattr("phasesweep.engine.phase.GpuPool.create", _no_gpu_create)
    rerun = run_experiment(experiment)

    assert rerun["p"].trial_number == first["p"].trial_number
    assert rerun["p"].metric == first["p"].metric
    study = optuna.load_study(study_name="t::p", storage=storage)
    assert study.user_attrs[TRIAL_TARGET_ATTR] == 1
    assert len(study.trials) == 1


def test_failed_gpu_topup_preserves_accepted_target_and_old_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A top-up that cannot launch leaves the prior accepted target usable.

    The larger target must only be durably accepted after launch prerequisites
    (GPU discovery, wallclock budget) pass; otherwise a transient local GPU
    problem permanently strands the study above its last working config
    (review v0.5.14 / blocker 4).
    """
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json
        p = argparse.ArgumentParser()
        p.add_argument("--out")
        p.add_argument("--x", type=int, default=0)
        a, _ = p.parse_known_args()
        print(f"x={a.x}")
        """,
    )
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    phase = Phase(
        name="p",
        n_trials=1,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": IntParam(type="int", low=0, high=10)},
    )
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        phases=[phase],
    )
    first = run_experiment(experiment)

    def _no_gpu_create(**_kwargs: object) -> None:
        raise RuntimeError("simulated: no GPUs detected on this host")

    monkeypatch.setattr("phasesweep.engine.phase.GpuPool.create", _no_gpu_create)
    top_up = experiment.model_copy(update={"phases": [phase.model_copy(update={"n_trials": 2})]})
    with pytest.raises(RuntimeError, match="no GPUs detected"):
        run_experiment(top_up)

    study = optuna.load_study(study_name="t::p", storage=storage)
    assert study.user_attrs[TRIAL_TARGET_ATTR] == 1
    assert len(study.trials) == 1

    rerun = run_experiment(experiment)
    assert rerun["p"].metric == first["p"].metric


def test_signal_handler_scope_restores_host_signal_state_on_success_and_failure(
    tmp_path: Path,
) -> None:
    """run_experiment restores the host's prior signal handlers and mask on every exit path.

    A library that leaves its own SIGTERM/SIGINT/SIGHUP handlers and unblocked
    mask installed after returning steals the embedding process's own
    shutdown handling permanently (review v0.5.14 / blocker 6). This must be
    undone whether the run succeeds or raises.
    """

    def host_handler(_signum: int, _frame: object) -> None:
        raise AssertionError("host handler should never fire during this test")

    def assert_host_state_active() -> None:
        for sig in runtime_process._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is host_handler
        current_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        assert set(runtime_process._SHUTDOWN_SIGNALS) <= current_mask

    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_process._SHUTDOWN_SIGNALS}
    prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    try:
        for sig in runtime_process._SHUTDOWN_SIGNALS:
            signal.signal(sig, host_handler)
        signal.pthread_sigmask(signal.SIG_BLOCK, set(runtime_process._SHUTDOWN_SIGNALS))

        trainer = write_constant_trainer(tmp_path)
        experiment = make_experiment(
            workdir=tmp_path / "runs",
            trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
            override_format="argparse",
            n_trials=1,
        )
        run_experiment(experiment)
        assert_host_state_active()

        failing_trainer = write_trainer(tmp_path / "failing.py", "raise SystemExit(1)")
        failing_experiment = make_experiment(
            experiment="fails",
            workdir=tmp_path / "runs",
            trial_command=f"python {failing_trainer} --out {{trial_dir}}/r.json {{overrides}}",
            override_format="argparse",
            n_trials=1,
            max_consecutive_failures=1,
        )
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(failing_experiment)
        assert_host_state_active()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_signal_handler_scope_delivers_pending_signal_to_host_handler_after_restore() -> None:
    """A signal pending at scope-exit must reach the restored HOST handler, not
    ``_shutdown_handler`` mid-restoration (review v0.5.15 / blocker 2A).

    Pre-fix, the mask was restored before the host handlers, so a pending
    SIGTERM fired while ``_shutdown_handler`` was still installed for it,
    raising ``PhaseSweepShutdown`` out of the cleanup path and leaving some
    host handlers unrestored. The fixed order is: block, then restore
    handlers, then restore the mask.
    """
    if not hasattr(signal, "pthread_sigmask"):
        pytest.skip("pthread_sigmask not available")

    received: list[int] = []

    def host_handler(signum: int, _frame: object) -> None:
        received.append(signum)

    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_process._SHUTDOWN_SIGNALS}
    prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    try:
        for sig in runtime_process._SHUTDOWN_SIGNALS:
            signal.signal(sig, host_handler)

        with signal_handler_scope():
            # Block SIGTERM ourselves so the kill below queues instead of
            # firing immediately (the scope's own entry unblocks it).
            signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGTERM,))
            os.kill(os.getpid(), signal.SIGTERM)
            assert received == [], "signal fired before scope exit"

        # Scope exit order: block -> restore handlers -> restore mask. The
        # pending SIGTERM is delivered on the final unblock, by which point
        # the HOST handler (not phasesweep's) is installed. CPython only
        # invokes the Python-level handler at the next eval-breaker check,
        # not necessarily synchronously with the unblocking call, so poll
        # briefly instead of asserting immediately.
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.001)
        assert received == [signal.SIGTERM]
        assert signal.getsignal(signal.SIGTERM) is host_handler
        assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == prior_mask
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_signal_handler_scope_continues_restoring_after_one_signal_signal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failed ``signal.signal`` restore must not abort the rest of the loop.

    Every other prior handler must still be restored, and the first
    restoration error surfaces once the scope body itself did not already
    raise (review v0.5.15 / blocker 2A).
    """
    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_process._SHUTDOWN_SIGNALS}
    try:

        def make_sentinel(tag: int):
            def handler(signum: int, _frame: object) -> None:
                return None

            handler.__name__ = f"sentinel_{tag}"
            return handler

        sentinel_handlers = {sig: make_sentinel(sig) for sig in runtime_process._SHUTDOWN_SIGNALS}
        for sig, handler in sentinel_handlers.items():
            signal.signal(sig, handler)

        first_signal = runtime_process._SHUTDOWN_SIGNALS[0]
        real_signal = signal.signal
        failed_once: set[int] = set()

        def flaky_signal(signalnum: int, handler: object) -> object:
            if signalnum == first_signal and signalnum not in failed_once:
                failed_once.add(signalnum)
                raise OSError("simulated restore failure")
            return real_signal(signalnum, handler)

        with pytest.raises(OSError, match="simulated restore failure"), signal_handler_scope():
            # Patch only after the scope's own entry-time installation
            # (which uses the real signal.signal) has already happened.
            monkeypatch.setattr(signal, "signal", flaky_signal)

        for sig in runtime_process._SHUTDOWN_SIGNALS:
            if sig == first_signal:
                # Restoration failed for this one; phasesweep's handler is
                # still installed until manual cleanup below.
                assert signal.getsignal(sig) is runtime_process._shutdown_handler
                continue
            assert signal.getsignal(sig) is sentinel_handlers[sig]
    finally:
        monkeypatch.undo()
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_run_suite_installs_signal_handlers_once_for_all_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A multi-study suite installs shutdown handlers once, not once per component.

    Each component experiment enters its own nested ``signal_handler_scope()``
    call; because the suite's outer scope already owns the shutdown signals,
    every nested entry must be a reentrant no-op (review v0.5.14 / blocker 6)
    — exactly one install and one restore for the whole suite, never one pair
    per study.
    """
    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_process._SHUTDOWN_SIGNALS}
    try:
        # Clean slate: nothing already owns the shutdown signals here,
        # regardless of what an earlier test in this session left installed.
        # install_signal_handlers() now checks OS ground truth, so resetting
        # the actual handlers is sufficient to make it see "not installed".
        for sig in runtime_process._SHUTDOWN_SIGNALS:
            signal.signal(sig, signal.SIG_DFL)

        signal_calls: list[int] = []
        original_signal = signal.signal

        def counting_signal(signalnum: int, handler: object) -> object:
            signal_calls.append(signalnum)
            return original_signal(signalnum, handler)

        monkeypatch.setattr(runtime_process.signal, "signal", counting_signal)

        trainer = write_constant_trainer(tmp_path)
        config = load_config(
            write_yaml(
                tmp_path,
                f"""
                suite: nesting_suite
                defaults:
                  workdir: {tmp_path}/runs
                  trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
                  override_format: argparse
                  metric:
                    name: x
                    goal: minimize
                    extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
                studies:
                  - name: one
                    phases:
                      - name: p
                        n_trials: 1
                        search_space: {{}}
                  - name: two
                    phases:
                      - name: p
                        n_trials: 1
                        search_space: {{}}
                """,
            )
        )

        run_suite(config)

        # One install (len(_SHUTDOWN_SIGNALS) calls) and one restore (another
        # len(_SHUTDOWN_SIGNALS) calls) for the whole suite — never doubled by
        # the two nested per-study scopes.
        assert len(signal_calls) == 2 * len(runtime_process._SHUTDOWN_SIGNALS)
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def _worker_errors(operation: Callable[[], None]) -> list[BaseException]:
    """Run one operation on a worker and return captured failures."""
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            operation()
        except BaseException as exc:  # noqa: BLE001 - returned for the main thread to assert on
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    return errors


def _enter_signal_handler_scope() -> None:
    """Enter and leave one signal-handler scope."""
    with signal_handler_scope():
        pass


def test_signal_handler_scope_raises_off_main_thread_without_prior_install() -> None:
    """Off the main thread, with nothing already owning shutdown signals, the scope refuses.

    ``signal.signal`` only works on the main thread, so a scope entered from a
    worker thread with no enclosing install cannot safely take ownership; it
    must raise a typed error instead of silently running unprotected.
    """
    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_process._SHUTDOWN_SIGNALS}
    try:
        for sig in runtime_process._SHUTDOWN_SIGNALS:
            if signal.getsignal(sig) is runtime_process._shutdown_handler:
                signal.signal(sig, signal.SIG_DFL)

        errors = _worker_errors(_enter_signal_handler_scope)

        assert len(errors) == 1
        assert isinstance(errors[0], SignalOwnershipUnavailableError)
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_signal_handler_scope_is_noop_once_process_lifetime_install_owns_signals() -> None:
    """A process-lifetime ``install_signal_handlers()`` call is never undone by a nested scope.

    Entry points (CLI, MCP server) install shutdown handlers once for the
    whole process. A later ``signal_handler_scope()`` — even from a worker
    thread, where taking ownership from scratch would be impossible — must
    see that ownership is already established and do nothing, on entry or
    exit.
    """
    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_process._SHUTDOWN_SIGNALS}
    try:
        install_signal_handlers()
        for sig in runtime_process._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is runtime_process._shutdown_handler

        errors = _worker_errors(_enter_signal_handler_scope)

        assert errors == []
        # Entry-point ownership persists: the nested scope did not tear it down.
        for sig in runtime_process._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is runtime_process._shutdown_handler
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_install_signal_handlers_inside_open_scope_survives_that_scope_exit() -> None:
    """An ``install_signal_handlers()`` call inside an open scope must outlive the scope.

    ``install_signal_handlers()`` recognizes an already-installed handler set
    as its own idempotent path, so calling it while a
    ``signal_handler_scope()`` is open took process-lifetime ownership on the
    strength of the *scope's* installation. The scope then restored the host's
    handlers on exit while ownership stayed claimed, so every later scope
    no-opped with nothing installed and child process groups leaked on
    shutdown. The scope now hands its installation over instead of restoring.
    """

    def host_handler(_signum: int, _frame: object) -> None:
        raise AssertionError("host handler should never fire during this test")

    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_process._SHUTDOWN_SIGNALS}
    prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    try:
        for sig in runtime_process._SHUTDOWN_SIGNALS:
            signal.signal(sig, host_handler)
        signal.pthread_sigmask(signal.SIG_BLOCK, set(runtime_process._SHUTDOWN_SIGNALS))

        with signal_handler_scope():
            install_signal_handlers()

        for sig in runtime_process._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is runtime_process._shutdown_handler
        assert not set(runtime_process._SHUTDOWN_SIGNALS) & signal.pthread_sigmask(
            signal.SIG_BLOCK, set()
        )

        # The ownership claim is now truthful, so a later no-op scope is safe.
        with signal_handler_scope():
            pass
        for sig in runtime_process._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is runtime_process._shutdown_handler
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_worker_thread_install_cannot_steal_scope_ownership() -> None:
    """A worker-thread ``install_signal_handlers()`` raises and takes nothing.

    During an open main-thread ``signal_handler_scope()`` every shutdown
    handler already points at ``_shutdown_handler``, so a worker thread used
    to take the idempotent fast path, flip ``_process_lifetime_owner``, and
    silently convert the scope's temporary installation into permanent
    process ownership — the scope exit then skipped restoring the host's
    handlers (review v0.5.16 / blocker 5). The install must now reject
    off-main-thread callers before touching any ownership state.
    """

    def host_handler(_signum: int, _frame: object) -> None:
        raise AssertionError("host handler should never fire during this test")

    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_process._SHUTDOWN_SIGNALS}
    try:
        for sig in runtime_process._SHUTDOWN_SIGNALS:
            signal.signal(sig, host_handler)

        with signal_handler_scope():
            errors = _worker_errors(install_signal_handlers)
            assert not runtime_process._process_lifetime_owner

        assert len(errors) == 1
        assert isinstance(errors[0], SignalOwnershipUnavailableError)
        assert not runtime_process._process_lifetime_owner
        # The scope's exit restored the host's handlers because no legitimate
        # process-lifetime handover happened.
        for sig in runtime_process._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is host_handler
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_absorb_shutdown_signals_reports_signal_and_defers_it_to_next_checkpoint() -> None:
    """A shutdown inside an absorb window is reported, not raised — then honored later.

    The publication transaction uses this to win its race against a shutdown
    signal deterministically (review v0.5.16 / blocker 1): the window exit
    reports the absorbed signal on the yielded object instead of raising, and
    the next ``defer_shutdown_signals()`` exit (e.g. the next trial launch)
    still delivers the shutdown before new work starts.
    """
    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_process._SHUTDOWN_SIGNALS}
    try:
        install_signal_handlers()

        with runtime_process.absorb_shutdown_signals() as absorbed:
            os.kill(os.getpid(), signal.SIGTERM)
            # Give an unblocked sibling thread's delivery path (if any) a
            # chance to run the Python-level handler; either delivery route
            # must end up recorded, never raised, inside the window.
            time.sleep(0.05)

        assert absorbed.signum == signal.SIGTERM

        # The absorbed signal is still pending: the next deferral checkpoint
        # delivers it before any new work could start.
        with (
            pytest.raises(runtime_process.PhaseSweepShutdown) as exc_info,
            runtime_process.defer_shutdown_signals(),
        ):
            pass
        assert exc_info.value.signum == signal.SIGTERM
    finally:
        runtime_process._deferred_shutdown_signum = None
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_service_pending_shutdown_is_noop_without_absorbed_signal() -> None:
    """The explicit checkpoint does nothing when no shutdown was absorbed."""
    assert runtime_process.service_pending_shutdown() is None


def test_stale_process_lifetime_claim_is_reasserted_on_scope_entry() -> None:
    """A scope entered under a stale ownership claim reinstalls the OS handlers.

    ``_process_lifetime_owner`` is a Python-side boolean; another library can
    re-bind a shutdown signal after the entry point installed. A later
    main-thread scope must notice the divergence and reinstall phasesweep's
    handler so the run does not silently execute without child-group cleanup.
    """
    prior_handlers = {sig: signal.getsignal(sig) for sig in runtime_process._SHUTDOWN_SIGNALS}
    try:
        install_signal_handlers()
        interloper_sig = runtime_process._SHUTDOWN_SIGNALS[0]
        signal.signal(interloper_sig, signal.SIG_IGN)

        with signal_handler_scope():
            assert signal.getsignal(interloper_sig) is runtime_process._shutdown_handler
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def test_extraction_past_deadline_fails_the_trial(tmp_path: Path) -> None:
    """A successful trainer whose evidence lands past the deadline cannot count.

    ``timeout_seconds_per_phase``/``timeout_seconds_per_run`` bound the whole
    trial; before review v0.5.17 / blocker 8 extraction ran with no deadline
    at all, so a run could exceed both configured limits and still publish a
    complete winner.
    """
    experiment = make_experiment(workdir=tmp_path)
    (tmp_path / "r.json").write_text('{"x": 1.0}')
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
        process=ProcessResult(
            return_code=0,
            timed_out=False,
            pid=123,
            duration_seconds=0.1,
        ),
    )

    result = extract_trial_result(
        experiment=experiment,
        executed=executed,
        deadline=time.monotonic() - 1.0,
    )

    assert result.metric is None
    assert result.feasible is False
    assert result.failure_reason is not None
    assert "wallclock deadline exceeded" in result.failure_reason
    assert result.deadline_exhausted is True


def test_process_failure_after_deadline_is_not_relabelled(tmp_path: Path) -> None:
    """An elapsed clock is not causal when the trainer already failed."""
    experiment = make_experiment(workdir=tmp_path)
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
            return_code=1,
            duration_seconds=0.1,
        ),
        process=ProcessResult(
            return_code=1,
            timed_out=False,
            pid=123,
            duration_seconds=0.1,
            failure_reason="trainer failed",
        ),
    )

    result = extract_trial_result(
        experiment=experiment,
        executed=executed,
        deadline=time.monotonic() - 1.0,
    )

    assert result.failure_reason == "trainer failed"
    assert result.deadline_exhausted is False


@pytest.mark.parametrize(
    ("capped_by_deadline", "timed_out", "expected"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_trainer_timeout_attribution_follows_wallclock_cap(
    tmp_path: Path, capped_by_deadline: bool, timed_out: bool, expected: bool
) -> None:
    """A killed trainer is deadline-attributed only when its cap was the deadline."""
    experiment = make_experiment(workdir=tmp_path)
    failure = "timed out after 1.0s" if timed_out else "trainer failed"
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
            return_code=1,
            duration_seconds=0.1,
        ),
        process=ProcessResult(
            return_code=1,
            timed_out=timed_out,
            pid=123,
            duration_seconds=0.1,
            failure_reason=failure,
        ),
    )

    result = extract_trial_result(
        experiment=experiment,
        executed=executed,
        trainer_timeout_is_deadline=capped_by_deadline,
    )

    assert result.failure_reason == failure
    assert result.deadline_exhausted is expected


def test_elapsed_phase_clock_does_not_relabel_completed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal ordinary failure stays NoFeasible after the phase clock advances."""
    import types

    import phasesweep.engine.phase as phase_mod

    real_now = time.monotonic()
    calls = iter([real_now, real_now, real_now + 1_000.0])
    monkeypatch.setattr(
        phase_mod,
        "time",
        types.SimpleNamespace(
            monotonic=lambda: next(calls, real_now + 1_000.0),
            sleep=lambda _seconds: None,
        ),
    )
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        trial_command="false {overrides}",
        override_format="argparse",
        n_trials=1,
        max_consecutive_failures=10,
        timeout_seconds_per_phase=100.0,
    )

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)


def test_unrelated_launch_timeout_is_not_relabelled_as_phase_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only GPU-lease timeout is deadline policy; launch timeouts stay fatal."""
    storage = f"sqlite:///{tmp_path / 'timeout-cause.db'}"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        n_trials=1,
        gpu_policy="none",
        timeout_seconds_per_phase=100.0,
    )

    def unrelated_timeout(**_kwargs: object) -> None:
        raise TimeoutError("injected non-deadline launch timeout")

    monkeypatch.setattr("phasesweep.engine.phase.launch_trial", unrelated_timeout)

    with pytest.raises(TimeoutError, match="injected non-deadline launch timeout"):
        run_experiment(experiment)

    study = optuna.load_study(study_name="t::p", storage=storage)
    assert study.user_attrs[PHASE_ABORT_ATTR]["policy"] == "unexpected_objective_exception"


def test_slow_extraction_cannot_publish_complete_past_phase_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewer repro (review v0.5.17 / blocker 8): quick trainer, slow extraction.

    The trainer finishes inside the 0.5s budget but extraction takes 3s. The
    run used to publish ``incomplete: false`` more than 2x past both limits;
    it must now report a timeout and publish nothing.
    """
    trainer = write_trainer(tmp_path, "print('x=1.0')")
    exp = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f"python {trainer} {{overrides}}",
        override_format="argparse",
        n_trials=1,
        timeout_seconds_per_phase=0.5,
    )

    import phasesweep.engine.trial as trial_mod

    real_extractor = trial_mod.run_extractor

    def slow_extractor(*args, **kwargs):
        time.sleep(3.0)
        return real_extractor(*args, **kwargs)

    monkeypatch.setattr("phasesweep.engine.trial.run_extractor", slow_extractor)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        run_experiment(exp)
    elapsed = time.monotonic() - started

    assert elapsed < 20.0
    assert not _last_successful_generation_path(exp).exists()


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
    storage_url = f"sqlite:///{tmp_path / 'fidelity.db'}"
    yaml_path = write_yaml(
        tmp_path,
        f"""
        experiment: categorical_fidelity
        storage: {storage_url}
        provenance: {{revision: test-fixture-v1}}
        workdir: {tmp_path / "runs"}
        trial_command: "python {trainer} {{overrides}}"
        override_format: argparse
        metric:
          name: eval_loss
          goal: minimize
          extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
        phases:
          - name: pick
            n_trials: {len(_FIDELITY_CHOICES)}
            sampler: {{ type: grid, seed: 0 }}
            search_space:
              x: {{ type: categorical, choices: [0, 1.5, "one", true] }}
          - name: child
            inherits: [pick]
            n_trials: 1
            sampler: {{ type: random, seed: 0 }}
            search_space: {{}}
        """,
    )

    # (a) The live value handed to _composed_overrides, i.e. the trainer input.
    live: dict[int, Any] = {}
    real_suggest = _suggest

    def recording_suggest(trial: optuna.Trial, name: str, param: Any) -> Any:
        value = real_suggest(trial, name, param)
        live[trial.number] = value
        return value

    monkeypatch.setattr("phasesweep.engine.phase._suggest", recording_suggest)

    exp = load_experiment(yaml_path)
    winners = run_experiment(exp)

    # The grid covered every choice exactly once, each with its declared type.
    assert sorted(((type(v).__name__, v) for v in live.values()), key=repr) == sorted(
        ((type(c).__name__, c) for c in _FIDELITY_CHOICES), key=repr
    )

    # (b) FrozenTrial.params, read back out of persistent storage.
    study = optuna.load_study(study_name="categorical_fidelity::pick", storage=storage_url)
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
    storage_url = f"sqlite:///{tmp_path / 'round-trip.db'}"
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
