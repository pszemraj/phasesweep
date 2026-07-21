"""Runtime semantics: NaN/inf propagation through extractors, parallel trial behavior, sampler configuration at study-creation time, max_consecutive_failures abort, and Optuna logging suppression."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import optuna
import pytest

from phasesweep import load_experiment, run_experiment
from phasesweep.config import (
    CategoricalParam,
    Experiment,
    IntParam,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
)
from phasesweep.engine.optuna import _build_sampler, _create_phase_study
from phasesweep.engine.phase import CsvSnapshotThrottle
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    _last_successful_generation_path,
    _load_winner,
    _summary_path,
    _winner_path,
)
from tests.conftest import copy_fake_train, make_experiment, write_trainer, write_yaml


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

    assert sampled(tmp_path / "single.db", [4]) == sampled(tmp_path / "batched.db", [1, 1, 1, 1])


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
        allow_partial_grid=sampler.type == "grid",
        sampler=sampler,
        search_space=search_space,
    )
    exp = make_experiment(
        experiment=f"sampler_{sampler.type}",
        storage=f"sqlite:///{tmp_path / f'{sampler.type}.db'}",
        workdir=tmp_path / "runs",
        phases=[phase],
    )
    _create_phase_study(exp, phase)
    observed: dict[str, object] = {}

    def inspect_optimize(study: optuna.Study, objective, **kwargs) -> None:
        observed["sampler"] = type(study.sampler).__name__
        observed["pruner"] = type(study.pruner).__name__
        observed["n_startup_trials"] = getattr(study.sampler, "_n_startup_trials", None)
        raise OptimizeObserved

    monkeypatch.setattr(optuna.Study, "optimize", inspect_optimize)
    with pytest.raises(OptimizeObserved):
        run_experiment(exp)

    assert observed["sampler"] == expected_type
    assert observed["pruner"] == "NopPruner"
    if sampler.type == "tpe":
        assert observed["n_startup_trials"] == 3


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
    - JournalFileStorage via explicit journal:/// URL (B3, v0.5.2 / blocker 6)
    - constant_liar on TPE (B4)
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

    trial_dirs = sorted(phase_dir.glob("trial_*"))
    assert len(trial_dirs) == 2
    assert first_trial_dir in trial_dirs
    second_trial_dir = next(path for path in trial_dirs if path != first_trial_dir)
    assert not (second_trial_dir / "r.json").exists()
    assert {path: path.read_bytes() for path in protected} == protected


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
    ("trainer_payload", "metric_log", "constraints_yaml", "exp_name"),
    [
        # Constraint extractor returns NaN
        (
            "{'eval_loss': 1.0, 'param_bytes': float('nan')}",
            "1.0",
            "constraints:\n  - name: param_bytes\n    extractor: { type: json, path: result.json, key: param_bytes }\n    max: 1000\n",
            "nan_c",
        ),
        # Metric extractor returns +inf
        (
            "{'eval_loss': float('inf')}",
            "inf",
            "",
            "inf_m",
        ),
    ],
    ids=["nan_constraint", "inf_metric"],
)
def test_non_finite_extracted_value_marks_trial_fail(
    tmp_path,
    trainer_payload: str,
    metric_log: str,
    constraints_yaml: str,
    exp_name: str,
):
    """A non-finite metric or constraint value is a malformed trial — FAIL,
    not COMPLETE-with-sentinel. Both the metric path and the constraint path
    must surface the failure the same way; one parametrized test exercises
    both routes through the same end-to-end pipeline."""
    trainer = tmp_path / "trainer.py"
    write_trainer(
        trainer,
        f"""
        import json, sys, argparse, math
        ap = argparse.ArgumentParser()
        ap.add_argument('--out', required=True)
        args, _ = ap.parse_known_args()
        with open(args.out, 'w') as f:
            json.dump({trainer_payload}, f)
        print('eval_loss={metric_log}')
        """,
    )

    db = tmp_path / "phases.db"
    yaml_text = f"""
experiment: {exp_name}
storage: sqlite:///{db}
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} --out {{trial_dir}}/result.json {{overrides}}"
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: log_regex, pattern: 'eval_loss=(?P<value>[0-9.eE+-]+)' }}
{constraints_yaml}
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

    study = optuna.load_study(study_name=f"{exp_name}::a", storage=f"sqlite:///{db}")
    for trial in study.get_trials():
        assert trial.state == optuna.trial.TrialState.FAIL


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
