"""A shutdown after a trainer exits warns once about the trial, in words that are true for what the stopped run did."""

from __future__ import annotations

import logging
import signal
from collections.abc import Callable
from pathlib import Path

import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment
from phasesweep.runtime.shutdown import PhaseSweepShutdown, _shutdown_handler
from tests.conftest import make_experiment, phase_trials, write_constant_trainer

_DEFERRED = "finished training; its result is retained"
_LOST = "finished training, but its result was not extracted and is lost"
_KEPT = "Kept completed trial result"


def _run_until_shutdown(experiment: Experiment, caplog: pytest.LogCaptureFixture) -> None:
    """Run the experiment and require that the shutdown stops it, capturing PhaseSweep's log."""
    with caplog.at_level(logging.INFO, logger="phasesweep"), pytest.raises(PhaseSweepShutdown):
        run_experiment(experiment)


def _shutdown_after_the_first_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliver SIGINT to the handler right after trial 0's success outcome is recorded.

    That is where a signal lands when it arrives between the recording and Optuna's commit
    of the trial.
    """
    import phasesweep.engine.phase as phase_module

    real_record: Callable[..., None] = phase_module._record_outcome
    recorded = False

    def record(run: object, trial: object, outcome: str, **kwargs: object) -> None:
        nonlocal recorded
        real_record(run, trial, outcome, **kwargs)
        if outcome == "success" and not recorded:
            recorded = True
            _shutdown_handler(signal.SIGINT, None)

    monkeypatch.setattr(phase_module, "_record_outcome", record)


def _shutdown_in_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliver SIGINT to the handler as the first trial's extraction begins, before any outcome."""
    import phasesweep.engine.phase as phase_module

    real_extract: Callable[..., object] = phase_module.extract_trial_result
    calls = 0

    def extract(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            _shutdown_handler(signal.SIGINT, None)
        return real_extract(**kwargs)

    monkeypatch.setattr(phase_module, "extract_trial_result", extract)


def _experiment(tmp_path: Path, *, persistent: bool) -> Experiment:
    """Build a two-trial experiment, on a ledger or with ``storage: null``."""
    trainer = write_constant_trainer(tmp_path)
    if persistent:
        return make_experiment(persistent=tmp_path, trainer=trainer, n_trials=2)
    return make_experiment(workdir=tmp_path / "runs", trainer=trainer, n_trials=2)


@pytest.mark.integration
def test_a_trial_the_stopped_run_commits_is_not_also_left_to_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Once the outcome is recorded the exit commits the trial and says so, so no warning promises extraction."""
    experiment = _experiment(tmp_path, persistent=True)
    _shutdown_after_the_first_outcome(monkeypatch)

    _run_until_shutdown(experiment, caplog)

    assert _DEFERRED not in caplog.text
    assert caplog.text.count(_KEPT) == 1
    assert [trial.state.name for trial in phase_trials(experiment)] == ["COMPLETE"]


@pytest.mark.integration
def test_training_the_stopped_run_could_not_settle_keeps_its_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With no outcome recorded the trial stays RUNNING for the next run to extract, and the warning says so once."""
    experiment = _experiment(tmp_path, persistent=True)
    _shutdown_in_extraction(monkeypatch)

    _run_until_shutdown(experiment, caplog)

    assert caplog.text.count(_DEFERRED) == 1
    assert _KEPT not in caplog.text
    assert [trial.state.name for trial in phase_trials(experiment)] == ["RUNNING"]


@pytest.mark.integration
def test_storage_null_still_says_the_stopped_trial_is_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing outlives a `storage: null` run, so its unextracted training is reported as lost."""
    experiment = _experiment(tmp_path, persistent=False)
    _shutdown_in_extraction(monkeypatch)

    _run_until_shutdown(experiment, caplog)

    assert caplog.text.count(_LOST) == 1
    assert _KEPT not in caplog.text
