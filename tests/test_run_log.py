"""``run.log`` carries one TRIAL line per finished trial, the process-free progress stream."""

from __future__ import annotations

import re
from pathlib import Path

from phasesweep import run_experiment
from phasesweep.config import (
    CategoricalParam,
    IntParam,
    LogRegexExtractor,
    Metric,
    Phase,
    Repeats,
    Sampler,
)
from phasesweep.engine.paths import _experiment_dir
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
    assert " reason=" in failed and "second line" not in failed
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
