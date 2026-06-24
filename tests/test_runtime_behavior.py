"""Runtime semantics: NaN/inf propagation through extractors, parallel trial behavior, sampler configuration at study-creation time, max_consecutive_failures abort, and Optuna logging suppression."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import optuna
import pytest

from phasesweep import load_experiment, run_experiment
from phasesweep.config import Experiment, JsonExtractor, Metric, Phase
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import _load_winner
from tests.conftest import copy_fake_train, write_trainer, write_yaml


def _sleeping_score_experiment(
    tmp_path: Path,
    *,
    experiment: str,
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
        time.sleep(0.15)
        with open(args.out, "w") as f:
            json.dump({"x": 1.0}, f)
        """,
    )
    return Experiment(
        experiment=experiment,
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
        timeout_seconds_per_run=timeout_seconds_per_run,
        phases=[
            Phase(
                name="p",
                n_trials=3,
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
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} --out {{trial_dir}}/result.json {{overrides}}"
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json, path: result.json, key: eval_loss }}
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
workdir: {tmp_path / "runs"}
trial_command: "false {{overrides}}"
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json, path: result.json, key: eval_loss }}
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
        """,
    )

    db = tmp_path / "phases.db"
    yaml_text = f"""
experiment: c2
storage: sqlite:///{db}
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} --out {{trial_dir}}/result.json {{overrides}}"
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json, path: result.json, key: eval_loss }}
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
    ("trainer_payload", "constraints_yaml", "exp_name"),
    [
        # Constraint extractor returns NaN
        (
            "{'eval_loss': 1.0, 'param_bytes': float('nan')}",
            "constraints:\n  - name: param_bytes\n    extractor: { type: json, path: result.json, key: param_bytes }\n    max: 1000\n",
            "nan_c",
        ),
        # Metric extractor returns +inf
        (
            "{'eval_loss': float('inf')}",
            "",
            "inf_m",
        ),
    ],
    ids=["nan_constraint", "inf_metric"],
)
def test_non_finite_extracted_value_marks_trial_fail(
    tmp_path, trainer_payload: str, constraints_yaml: str, exp_name: str
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
        """,
    )

    db = tmp_path / "phases.db"
    yaml_text = f"""
experiment: {exp_name}
storage: sqlite:///{db}
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} --out {{trial_dir}}/result.json {{overrides}}"
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json, path: result.json, key: eval_loss }}
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
  extractor: {{ type: json, path: r.json, key: eval_loss }}
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
workdir: {tmp_path}/runs
trial_command: "echo {{overrides}}"
metric:
  name: x
  goal: minimize
  extractor: {{ type: json, path: r.json, key: x }}
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
workdir: {tmp_path}/runs
trial_command: "false {{overrides}}"
metric:
  name: loss
  goal: minimize
  extractor: {{ type: json, path: r.json, key: loss }}
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
    assert not (tmp_path / "runs" / "phase_timeout_hard" / "p" / "trial_00000" / "r.json").exists()


def test_incomplete_timeout_can_be_explicitly_accepted(tmp_path: Path) -> None:
    exp = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout_allowed",
        timeout_seconds_per_phase=0.2,
        allow_incomplete_on_timeout=True,
    )

    winners = run_experiment(exp)

    completion = winners["p"].completion
    assert completion["requested_trials"] == 3
    assert 1 <= completion["completed_trials"] < completion["requested_trials"]
    assert completion["completed_trials"] <= completion["finished_trials"]
    assert completion["finished_trials"] <= completion["requested_trials"]
    assert completion["incomplete"] is True
    assert completion["reason"] == "timeout"
    assert completion["timeout_scope"] == "phase"


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
        else:
            time.sleep(1.0)
        """,
    )
    exp = Experiment(
        experiment="phase_timeout_allowed_abort_counter",
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
        phases=[
            Phase(
                name="p",
                n_trials=3,
                max_consecutive_failures=1,
                timeout_seconds_per_phase=0.5,
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
        timeout_seconds_per_phase=0.2,
        allow_incomplete_on_timeout=True,
    )
    run_experiment(accepted)

    current = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout_resume_guard",
        timeout_seconds_per_phase=0.2,
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
