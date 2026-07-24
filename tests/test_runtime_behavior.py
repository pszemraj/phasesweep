"""Runtime semantics: NaN/inf propagation through extractors, parallel trial behavior, sampler configuration at study-creation time, max_consecutive_failures abort, and Optuna logging suppression."""

from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path

import optuna
import pytest

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
)
from phasesweep.engine import TerminalReport, read_status, read_winner
from phasesweep.engine.optuna import _build_sampler, _create_phase_study
from phasesweep.engine.phase import CsvSnapshotThrottle
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    TRIAL_TARGET_ATTR,
    _last_successful_generation_path,
    _load_winner,
    _summary_path,
    _winner_path,
)
from phasesweep.engine.trial import ExecutedTrial, extract_trial_result
from phasesweep.evidence import TrialContext
from phasesweep.runtime import process as runtime_process
from phasesweep.runtime.process import (
    ProcessResult,
    SignalOwnershipUnavailableError,
    install_signal_handlers,
    signal_handler_scope,
)
from tests.conftest import (
    copy_fake_train,
    make_experiment,
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
        pytest.param(
            Sampler(type="tpe", seed=0, n_startup_trials=3),
            {"x": CategoricalParam(type="categorical", choices=[1, 2])},
            1,
            "TPESampler",
            id="tpe",
        ),
        pytest.param(
            Sampler(type="cmaes", seed=0),
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
) -> Experiment:
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json, time
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        time.sleep(0.5)
        with open(args.out, "w") as f:
            json.dump({"x": 1.0}, f)
        print("x=1.0")
        """,
    )
    return Experiment(
        experiment=experiment,
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
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
trial_command: "python {trainer} --out {{trial_dir}}/result.json {{overrides}}"
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json_envelope, path: result.json, objective_name: eval_loss, split: validation, policy: synthetic }}
phases:
  - name: lr_sweep
    n_trials: 8
    n_jobs: 4
    allow_no_gpu_isolation: true
    sampler: {{ type: tpe, seed: 42 }}
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
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json_envelope, path: result.json, objective_name: eval_loss, split: validation, policy: synthetic }}
phases:
  - name: a
    n_trials: 3
    max_consecutive_failures: 10
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

    def preflight(_experiment, *, cleanup_report):
        cleanup_report.uncertain_attempt_ids.add("attempt-uncertain")
        return {}

    monkeypatch.setattr("phasesweep.engine.run._preflight_existing_studies", preflight)
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
        "phasesweep.engine.run._preflight_existing_studies",
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
        "phasesweep.engine.run._preflight_existing_studies",
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


def test_abort_after_gpu_acquire_prevents_queued_trials(tmp_path: Path) -> None:
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
    yaml_text = f"""
experiment: abort_recheck
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} {{overrides}}"
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
    assert len(launched) <= 2, (
        f"With max_consecutive_failures=1 and a 1-slot GPU pool, expected at "
        f"most 2 trial launches (1 failing + 1 in-flight before abort propagates); "
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

    with pytest.raises(RuntimeError, match="stopped after 0/1 terminal trials"):
        run_experiment(exp)


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
metric:
  name: x
  goal: minimize
  extractor: {{ type: json_envelope, objective_name: x, split: test, policy: test }}
phases:
  - name: p
    n_trials: 1
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
metric:
  name: loss
  goal: minimize
  extractor: {{ type: json_envelope, objective_name: loss, split: test, policy: test }}
phases:
  - name: a
    n_trials: 100
    max_consecutive_failures: 3
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
    exp = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout_hard",
        timeout_seconds_per_phase=0.05,
    )

    started = time.monotonic()
    with pytest.raises(
        TimeoutError,
        match=r"timed out via phase guard .*Refusing to select a winner",
    ):
        run_experiment(exp)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    phase_dir = tmp_path / "runs" / "phase_timeout_hard" / "p"
    assert list(phase_dir.glob("trial_00000__*/r.json")) == []


def test_incomplete_timeout_can_be_explicitly_accepted(tmp_path: Path) -> None:
    exp = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout_allowed",
        n_trials=10,
        timeout_seconds_per_phase=3.0,
        allow_incomplete_on_timeout=True,
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
            time.sleep(5.0)
        """,
    )
    exp = Experiment(
        experiment="phase_timeout_all_terminal",
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
        phases=[
            Phase(
                name="p",
                n_trials=2,
                timeout_seconds_per_phase=3.0,
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
            time.sleep(10.0)
        """,
    )
    exp = Experiment(
        experiment="phase_timeout_allowed_abort_counter",
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
        phases=[
            Phase(
                name="p",
                n_trials=3,
                max_consecutive_failures=1,
                timeout_seconds_per_phase=3.0,
                allow_incomplete_on_timeout=True,
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


def test_incomplete_timeout_winner_requires_current_opt_in_on_resume(tmp_path: Path) -> None:
    accepted = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout_resume_guard",
        n_trials=10,
        timeout_seconds_per_phase=3.0,
        allow_incomplete_on_timeout=True,
    )
    run_experiment(accepted)

    current = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout_resume_guard",
        n_trials=10,
        timeout_seconds_per_phase=3.0,
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
            n_trials=1,
        )
        run_experiment(experiment)
        assert_host_state_active()

        failing_trainer = write_trainer(tmp_path / "failing.py", "raise SystemExit(1)")
        failing_experiment = make_experiment(
            experiment="fails",
            workdir=tmp_path / "runs",
            trial_command=f"python {failing_trainer} --out {{trial_dir}}/r.json {{overrides}}",
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

        errors: list[BaseException] = []

        def worker() -> None:
            try:
                with signal_handler_scope():
                    pass
            except BaseException as exc:  # noqa: BLE001 - captured for the main thread to assert on
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

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

        errors: list[BaseException] = []

        def worker() -> None:
            try:
                with signal_handler_scope():
                    pass
            except BaseException as exc:  # noqa: BLE001 - captured for the main thread to assert on
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert errors == []
        # Entry-point ownership persists: the nested scope did not tear it down.
        for sig in runtime_process._SHUTDOWN_SIGNALS:
            assert signal.getsignal(sig) is runtime_process._shutdown_handler
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)
