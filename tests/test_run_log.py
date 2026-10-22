"""``run.log`` carries one TRIAL line per finished trial, the process-free progress stream."""

from __future__ import annotations

import csv
import logging
import re
import subprocess
import sys
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Constraint, Phase
from phasesweep.config.search import CategoricalParam, IntParam, Repeats, Sampler
from phasesweep.engine import phase as phase_ops
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.paths import _experiment_dir, _phase_dir
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import FAILURE_REASON_ATTR, FEASIBLE_ATTR, TRIAL_OUTCOME_ATTR
from phasesweep.engine.trial import _environment_identity
from phasesweep.evidence.models import LogRegexExtractor
from phasesweep.runtime.gpu import GpuPool
from tests.conftest import _bounded_metric, make_experiment, trial_outcome, write_trainer


def _trial_lines(experiment_log: Path, phase: str) -> list[str]:
    """Return every ``run.log`` line the phase logged for a finished trial, in order."""
    marker = f"phase={phase} TRIAL "
    return [line for line in experiment_log.read_text().splitlines() if marker in line]


def _field(line: str, pattern: str) -> str:
    """Return the first group of ``pattern`` in ``line``, failing when it is absent."""
    match = re.search(pattern, line)
    assert match is not None, f"{pattern!r} not in {line!r}"
    return match[1]


def _x_trainer(tmp_path: Path) -> Path:
    """Write a trainer that logs ``x=<10.5 * --x>`` and exits nonzero for ``--x 2``."""
    return write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse
        import sys

        parser = argparse.ArgumentParser()
        parser.add_argument("--x", type=int, default=0)
        args, _ = parser.parse_known_args()
        if args.x == 2:
            print("trainer exploded\\nsecond line", file=sys.stderr)
            sys.exit(3)
        print(f"x={args.x * 10.5}")
        """,
    )


def test_run_log_records_one_line_per_finished_trial(tmp_path: Path) -> None:
    """A feasible, a failed and an infeasible trial each get one line with their numbers."""
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=_x_trainer(tmp_path),
        metric=_bounded_metric(max=50),
        name="p",
        n_trials=4,
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[1, 2, 5, 4])},
    )

    run_experiment(experiment)

    lines = _trial_lines(_experiment_dir(experiment) / "run.log", "p")
    assert [_field(line, r"TRIAL trial=(\d+)") for line in lines] == ["0", "1", "2", "3"]
    assert [_field(line, r"\((\d+)/4 finished\)") for line in lines] == ["1", "2", "3", "4"]
    by_state = {_field(line, r"state=(\w+)"): line for line in lines}
    assert sorted(by_state) == ["COMPLETE", "FAIL", "INFEASIBLE"]

    failed = by_state["FAIL"]
    assert "state=FAIL value=none" in failed
    assert failed.endswith(" reason=non-zero exit code 3; stderr: second line")
    infeasible = by_state["INFEASIBLE"]
    assert "state=INFEASIBLE value=52.5" in infeasible
    assert "reason=metric or evidence constraints were infeasible" in infeasible
    feasible = [line for line in lines if "state=COMPLETE" in line]
    assert len(feasible) == 2
    assert all(re.search(r"state=COMPLETE value=\d", line) for line in feasible)
    assert all("reason=" not in line for line in feasible)
    assert all("outcome=" not in line for line in lines)
    assert all(re.search(r"duration=\d+\.\ds", line) for line in lines)
    assert all("repeat=" not in line for line in lines)


@pytest.mark.parametrize("source", ["metric", "constraint"])
@pytest.mark.parametrize(("bound", "limit", "value"), [("max", 50, 52.5), ("min", 50, 10.5)])
def test_infeasible_reason_names_measured_value_and_violated_bound(
    tmp_path: Path, source: str, bound: str, limit: int, value: float
) -> None:
    metric = _bounded_metric(**({bound: limit} if source == "metric" else {}))
    constraint_name = "memory"
    constraints = (
        []
        if source == "metric"
        else [
            Constraint(
                name=constraint_name,
                extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.]+)"),
                **{bound: limit},
            )
        ]
    )
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=_x_trainer(tmp_path),
        metric=metric,
        constraints=constraints,
        n_trials=2,
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[1, 5])},
    )

    run_experiment(experiment)

    study = optuna.load_study(
        study_name="t::p", storage=_resolve_storage(experiment.resolved_storage)
    )
    rejected = next(trial for trial in study.trials if trial.value == value)
    assert rejected.state == optuna.trial.TrialState.COMPLETE
    cause = rejected.user_attrs[TRIAL_OUTCOME_ATTR]["cause"]
    assert (metric.name if source == "metric" else constraint_name) in cause
    assert str(value) in cause
    assert bound in cause
    assert str(limit) in cause
    lines = _trial_lines(_experiment_dir(experiment) / "run.log", "p")
    assert any(line.endswith(f" reason={cause}") for line in lines)


def test_run_log_names_the_repeat_value_of_each_trial(tmp_path: Path) -> None:
    """A repeats phase shows which seed every trial ran."""
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=_x_trainer(tmp_path),
        phases=[
            Phase(
                name="p",
                n_trials=4,
                sampler=Sampler(type="random", seed=0),
                search_space={"x": IntParam(type="int", low=3, high=9)},
                repeats=Repeats(key="seed", values=[0, 1]),
            )
        ],
    )

    run_experiment(experiment)

    lines = _trial_lines(_experiment_dir(experiment) / "run.log", "p")
    assert [_field(line, r"repeat=seed=(\d+)") for line in lines] == ["0", "1", "0", "1"]


def _stderr_cause_trainer(tmp_path: Path) -> Path:
    """Write a trainer that succeeds for ``--x 4`` and otherwise fails with a stderr line per value."""
    return write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse
        import sys

        parser = argparse.ArgumentParser()
        parser.add_argument("--x", type=int, default=0)
        args, _ = parser.parse_known_args()
        if args.x == 4:
            print("x=4.0")
            sys.exit(0)
        print("loading model", file=sys.stderr)
        if args.x == 1:
            print(
                "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
                file=sys.stderr,
            )
        else:
            print("ModuleNotFoundError: No module named 'm" + str(args.x) + "'", file=sys.stderr)
        sys.exit(1)
        """,
    )


def test_failed_trials_carry_their_stderr_cause_and_the_hint_only_for_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One cause string reaches the outcome record, trials.csv and the TRIAL line; the hint is for imports."""
    monkeypatch.setattr("phasesweep.engine.progress._WARNED", set())
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=_stderr_cause_trainer(tmp_path),
        metric=_bounded_metric(),
        n_trials=4,
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[1, 2, 3, 4])},
    )

    run_experiment(experiment)

    study = optuna.load_study(
        study_name="t::p", storage=_resolve_storage(experiment.resolved_storage)
    )
    x_by_trial = {trial.number: trial.params["x"] for trial in study.trials}
    causes = {
        1: "non-zero exit code 1; stderr: torch.OutOfMemoryError: CUDA out of memory. "
        "Tried to allocate 2.00 GiB",
        2: "non-zero exit code 1; stderr: ModuleNotFoundError: No module named 'm2'",
        3: "non-zero exit code 1; stderr: ModuleNotFoundError: No module named 'm3'",
    }
    log_path = _experiment_dir(experiment) / "run.log"
    lines = {int(_field(line, r"TRIAL trial=(\d+)")): line for line in _trial_lines(log_path, "p")}
    with (_phase_dir(experiment, "p") / "trials.csv").open(newline="") as handle:
        rows = {int(row["number"]): row for row in csv.DictReader(handle)}
    for trial in study.trials:
        x = x_by_trial[trial.number]
        if x == 4:
            assert FAILURE_REASON_ATTR not in trial.user_attrs
            assert " reason=" not in lines[trial.number]
            continue
        assert trial.user_attrs[FAILURE_REASON_ATTR] == causes[x]
        assert trial.user_attrs[TRIAL_OUTCOME_ATTR]["cause"] == causes[x]
        assert rows[trial.number][f"user_attr:{FAILURE_REASON_ATTR}"] == causes[x]
        assert lines[trial.number].endswith(f" reason={causes[x]}")

    hints = [line for line in log_path.read_text().splitlines() if "execution.inherit_env" in line]
    first_import_failure = min(number for number, x in x_by_trial.items() if x in {2, 3})
    assert len(hints) == 1
    assert f"[p/trial_{first_import_failure}] the trainer exited with code 1 under" in hints[0]
    assert "No module named 'm" in hints[0]


def test_no_feasible_headline_covers_the_causes_of_infeasible_trials_too(tmp_path: Path) -> None:
    """The headline promises a cause for every failed or infeasible trial, and each row keeps its own."""
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=_x_trainer(tmp_path),
        metric=_bounded_metric(max=1.0),
        n_trials=2,
        max_consecutive_failures=5,
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[1, 2])},
    )

    with pytest.raises(NoFeasibleTrialError, match="records why each failed or infeasible trial"):
        run_experiment(experiment)

    with (_phase_dir(experiment, "p") / "trials.csv").open(newline="") as handle:
        rows = {row["state"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"FAIL", "INFEASIBLE"}
    assert "non-zero exit code 3" in rows["FAIL"][f"user_attr:{FAILURE_REASON_ATTR}"]
    # An infeasible trial finished without an execution failure, so its cause is in its outcome.
    assert rows["INFEASIBLE"][f"user_attr:{FAILURE_REASON_ATTR}"] == ""
    assert "infeasible: metric" in rows["INFEASIBLE"][f"user_attr:{TRIAL_OUTCOME_ATTR}"]
    assert "> max=1.0" in rows["INFEASIBLE"][f"user_attr:{TRIAL_OUTCOME_ATTR}"]


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.integration
def test_cli_failed_trial_tracebacks_require_verbose(tmp_path: Path, verbose: bool) -> None:
    """Normal CLI failures keep the TRIAL cause; verbose restores Optuna diagnostics."""
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=_x_trainer(tmp_path),
        n_trials=1,
        max_consecutive_failures=1,
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[2])},
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(experiment.model_dump_json())

    result = subprocess.run(
        [sys.executable, "-m", "phasesweep", "run", str(config_path), *(["-v"] if verbose else [])],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=20,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "TRIAL trial=0 state=FAIL" in result.stderr
    assert "reason=non-zero exit code 3; stderr: second line" in result.stderr
    if verbose:
        assert "A new study created" in result.stderr
        assert "Trial 0 failed" in result.stderr
        assert "Traceback (most recent call last):" in result.stderr
    else:
        assert "Trial 0 failed" not in result.stderr
        assert "Traceback (most recent call last):" not in result.stderr


def _execution(tmp_path: Path) -> phase_ops._PhaseExecution:
    """Build a phase execution over an empty in-memory study."""
    experiment = make_experiment(workdir=tmp_path / "runs", n_trials=6, gpu_policy="none")
    phase = experiment.phases[0]
    return phase_ops._PhaseExecution(
        experiment=experiment,
        phase=phase,
        study=optuna.create_study(),
        inherited_winners={},
        generation_id="generation",
        environment_identity=_environment_identity(experiment, phase.name),
        gpu_pool=GpuPool.for_phase(phase),
        optimize_deadline=None,
        timeout_source=None,
    )


def _finish(run: phase_ops._PhaseExecution, outcome: str) -> int:
    """Record ``outcome`` for a new trial of ``run``'s study, as the objective does; return its number."""
    trial = run.study.ask()
    trial.set_user_attr(FEASIBLE_ATTR, outcome == "success")
    trial.set_user_attr(
        TRIAL_OUTCOME_ATTR, trial_outcome(run.completion_sequence + 1, outcome, cause="boom")
    )
    run.advance(trial.number, outcome)
    return trial.number


def test_trial_lines_carry_their_own_completion_ordinal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Parallel trials that finish together each log the ordinal their outcome took, not a shared count."""
    run = _execution(tmp_path)
    numbers = [_finish(run, "success"), _finish(run, "failure"), _finish(run, "success")]
    # Under parallel jobs every trial is terminal before the first callback reads the study.
    run.study.tell(numbers[0], 0.5)
    run.study.tell(numbers[1], state=optuna.trial.TrialState.FAIL)
    run.study.tell(numbers[2], 0.5)
    caplog.set_level(logging.INFO)

    for number in numbers:
        phase_ops._settle_finished_trial(run, run.study, run.study.trials[number])

    ordinals = [
        _field(record.getMessage(), r"\((\d+/6) finished\)")
        for record in caplog.records
        if " TRIAL " in record.getMessage()
    ]
    assert ordinals == ["1/6", "2/6", "3/6"]


def test_an_interrupted_trial_takes_no_ordinal(tmp_path: Path) -> None:
    """An interrupted trial reruns, so it leaves the count where it was and the next trial follows."""
    run = _execution(tmp_path)
    numbers = [_finish(run, "success"), _finish(run, "interrupted"), _finish(run, "failure")]

    assert [run.finished_ordinals[number] for number in numbers] == [1, 1, 2]
    assert run.finished_trials == 2


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_run_log_reports_abort_after_its_triggering_trial(tmp_path: Path, n_jobs: int) -> None:
    """The abort stays in its triggering outcome and prints after that trial's TRIAL line."""
    experiment = make_experiment(
        persistent=tmp_path,
        trial_command="false {overrides}",
        n_trials=2,
        n_jobs=n_jobs,
        max_consecutive_failures=1,
        search_space={"x": IntParam(type="int", low=0, high=2)},
    )

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(experiment)

    study = optuna.load_study(
        study_name="t::p", storage=_resolve_storage(experiment.resolved_storage)
    )
    triggering = next(
        trial for trial in study.trials if "abort" in trial.user_attrs[TRIAL_OUTCOME_ATTR]
    )
    abort = triggering.user_attrs[TRIAL_OUTCOME_ATTR]["abort"]
    lines = (_experiment_dir(experiment) / "run.log").read_text().splitlines()
    abort_lines = [index for index, line in enumerate(lines) if "phase=p ABORTED:" in line]
    trial_index = next(
        index
        for index, line in enumerate(lines)
        if f"phase=p TRIAL trial={triggering.number} " in line
    )
    assert len(abort_lines) == 1
    assert trial_index < abort_lines[0]
    assert abort["cause"] in lines[abort_lines[0]]
