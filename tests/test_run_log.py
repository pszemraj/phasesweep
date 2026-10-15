"""``run.log`` carries one TRIAL line per finished trial, the process-free progress stream."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Metric, Phase
from phasesweep.config.search import CategoricalParam, IntParam, Repeats, Sampler
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.paths import _experiment_dir, _phase_dir
from phasesweep.engine.state import FAILURE_REASON_ATTR, TRIAL_OUTCOME_ATTR
from phasesweep.evidence.models import LogRegexExtractor
from tests.conftest import make_experiment, write_trainer


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
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
            max=50,
        ),
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
    assert "outcome=failure value=none" in failed
    assert failed.endswith(" reason=non-zero exit code 3; stderr: second line")
    infeasible = by_state["INFEASIBLE"]
    assert "outcome=failure value=52.5" in infeasible
    assert "reason=metric or evidence constraints were infeasible" in infeasible
    feasible = [line for line in lines if "state=COMPLETE" in line]
    assert len(feasible) == 2
    assert all("outcome=success" in line and "reason=" not in line for line in feasible)
    assert all(re.search(r"duration=\d+\.\ds", line) for line in lines)
    assert all("repeat=" not in line for line in lines)


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
    monkeypatch.setattr("phasesweep.engine.trial._NARROWED_ENV_EXIT_WARNED", set())
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=_stderr_cause_trainer(tmp_path),
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
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
