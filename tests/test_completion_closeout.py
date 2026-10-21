"""Run closeout distinguishes accepted incomplete results and pending MLflow delivery."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

import phasesweep.engine.phase as phase_ops
from phasesweep.config.models import Experiment
from phasesweep.engine.paths import _experiment_dir
from phasesweep.engine.run import TerminalReport, run_experiment
from tests.conftest import make_experiment, write_constant_trainer


def _winner_lines(experiment: Experiment) -> list[str]:
    """Read the phase's terminal winner lines from its durable log."""
    return [
        line
        for line in (_experiment_dir(experiment) / "run.log").read_text().splitlines()
        if "phase=p WINNER " in line
    ]


@pytest.mark.parametrize("guard", ["phase", "run"])
def test_accepted_incomplete_winner_and_replay_log_counts_and_timeout_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, guard: str
) -> None:
    """An intentional partial result stays successful and visibly incomplete on replay."""
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_constant_trainer(tmp_path),
        n_trials=3,
        gpu_policy="none",
        allow_incomplete_on_timeout=True,
        timeout_seconds_per_phase=600.0 if guard == "phase" else None,
    )
    if guard == "run":
        experiment = experiment.model_copy(update={"timeout_seconds_per_run": 600.0})

    elapsed = [0.0]
    monkeypatch.setattr(
        phase_ops,
        "time",
        SimpleNamespace(monotonic=lambda: time.monotonic() + elapsed[0], sleep=time.sleep),
    )
    settled: list[int] = []
    after_trial = phase_ops._after_trial

    def expire_after_first_result(
        run: phase_ops._PhaseExecution,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        after_trial(run, study, trial)
        if trial.state == optuna.trial.TrialState.COMPLETE:
            settled.append(trial.number)
            elapsed[0] = 10_000.0

    monkeypatch.setattr(phase_ops, "_after_trial", expire_after_first_result)
    reports: list[TerminalReport] = []
    winner = run_experiment(experiment, terminal_callback=reports.append)["p"]
    assert winner.completion["incomplete"] is True
    assert winner.completion["finished_trials"] == 1
    assert reports[0].primary_error is None

    replayed = run_experiment(experiment, terminal_callback=reports.append)["p"]
    assert replayed.completion == winner.completion
    assert reports[1].primary_error is None
    assert settled == [0]
    lines = _winner_lines(experiment)
    assert len(lines) == 2
    for line in lines:
        assert "INCOMPLETE" in line
        assert "1/3 finished" in line
        assert f"timeout_scope={guard}" in line


def test_completed_winner_has_no_incomplete_marker(tmp_path: Path) -> None:
    """A fully completed result keeps its ordinary winner closeout."""
    experiment = make_experiment(
        persistent=tmp_path, trainer=write_constant_trainer(tmp_path), n_trials=1, gpu_policy="none"
    )
    winner = run_experiment(experiment)["p"]
    assert winner.completion["incomplete"] is False
    [line] = _winner_lines(experiment)
    assert "INCOMPLETE" not in line
