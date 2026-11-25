"""A rerun warns about overwriting a checkpoint directory only when it holds a finished attempt's model."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric, Phase, Sampler
from phasesweep.config.search import CategoricalParam
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import RERUN_OF_ATTR, TRIAL_DIR_ATTR
from phasesweep.engine.trial_dirs import DirectoryLaunch, overwritable_results
from phasesweep.evidence.models import JsonExtractor, Sha256Gate
from tests.conftest import addressed_phase_study, make_experiment, phase_trials, write_trainer

_METRIC = Metric(extractor=JsonExtractor(type="json", path="r.json", key="loss"))
_MARKER = "overwrit"


def _trainer(tmp_path: Path, *, fail_once: bool = False) -> Path:
    """Write a trainer that saves ``best.ckpt`` as 1, then 2 once the file exists.

    :param Path tmp_path: Scratch directory for the trainer and its failure marker.
    :param bool fail_once: Exit 1 on the first launch overall, before any checkpoint.
    :return Path: The trainer script.
    """
    return write_trainer(
        tmp_path / "trainer.py",
        f"""
        import json, os, sys
        from pathlib import Path
        directory = Path(os.environ["PHASESWEEP_TRIAL_DIR"])
        failed = Path({str(tmp_path / "failed-once")!r})
        if {fail_once!r} and not failed.exists():
            failed.write_text("x")
            sys.exit(1)
        shared = directory / "best.ckpt"
        value = 2 if shared.exists() else 1
        shared.write_text(str(value))
        (directory / "r.json").write_text(json.dumps({{"loss": value}}))
        """,
    )


def _config(tmp_path: Path, trainer: Path, *, cells: int, n_trials: int, gate: int) -> Experiment:
    """Build a grid over ``cells`` points whose gate accepts only ``best.ckpt`` equal to ``gate``.

    :param Path tmp_path: Scratch directory holding the ledger and workdir.
    :param Path trainer: The trainer script.
    :param int cells: Grid points.
    :param int n_trials: Trial target.
    :param int gate: The checkpoint content the gate accepts.
    :return Experiment: The experiment.
    """
    digest = hashlib.sha256(str(gate).encode()).hexdigest()
    phase = Phase(
        name="p",
        gpu_policy="none",
        n_trials=n_trials,
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=list(range(cells)))},
        gates=[Sha256Gate(type="sha256", path="best.ckpt", sha256=digest)],
    )
    return make_experiment(persistent=tmp_path, trainer=trainer, metric=_METRIC, phases=[phase])


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the logged messages that disclose a rerun over a finished attempt's model.

    :param pytest.LogCaptureFixture caplog: Captured log records.
    :return list[str]: The matching messages.
    """
    return [message for message in caplog.messages if _MARKER in message]


def test_reruns_over_finished_models_warn_once_with_the_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Two gate-failed trials retried in their directories print one line naming two reruns."""
    trainer = _trainer(tmp_path)
    run_experiment(_config(tmp_path, trainer, cells=2, n_trials=2, gate=1))
    assert not _warnings(caplog)

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_config(tmp_path, trainer, cells=2, n_trials=2, gate=2))
    caplog.clear()
    run_experiment(_config(tmp_path, trainer, cells=2, n_trials=4, gate=2))

    (line,) = _warnings(caplog)
    assert line.startswith("[p] 2 reruns write where finished attempts' models can be overwritten.")
    assert 'docs/runtime.md ("Interrupted trials rerun")' in line


def test_one_rerun_over_a_finished_model_names_its_directory(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A single gate-failed trial retried in its directory prints one line naming that directory."""
    trainer = _trainer(tmp_path)
    run_experiment(_config(tmp_path, trainer, cells=1, n_trials=1, gate=1))
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_config(tmp_path, trainer, cells=1, n_trials=1, gate=2))
    caplog.clear()
    run_experiment(_config(tmp_path, trainer, cells=1, n_trials=2, gate=2))

    (line,) = _warnings(caplog)
    retried = addressed_phase_study(_config(tmp_path, trainer, cells=1, n_trials=2, gate=2))
    directory = Path(retried.get_trials()[1].user_attrs[TRIAL_DIR_ATTR])
    assert line.startswith(f"[p/trial_1] Reruns in {directory}, where a finished attempt's model")


def test_a_sibling_reading_that_completed_the_attempt_makes_its_model_finished(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A retry by the study that read the attempt as failed warns when a sibling study read it as complete."""
    trainer = _trainer(tmp_path)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_config(tmp_path, trainer, cells=1, n_trials=1, gate=2))
    run_experiment(_config(tmp_path, trainer, cells=1, n_trials=1, gate=1))
    caplog.clear()
    run_experiment(_config(tmp_path, trainer, cells=1, n_trials=2, gate=2))

    (line,) = _warnings(caplog)
    assert line.startswith("[p/trial_1] Reruns in ")


def test_a_retried_failed_trial_has_no_finished_model_to_overwrite(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed grid cell retried in its own directory prints no checkpoint warning."""
    trainer = _trainer(tmp_path, fail_once=True)
    run_experiment(_config(tmp_path, trainer, cells=2, n_trials=2, gate=1))
    caplog.clear()
    run_experiment(_config(tmp_path, trainer, cells=2, n_trials=3, gate=1))

    retried = phase_trials(_config(tmp_path, trainer, cells=2, n_trials=3, gate=1))
    assert any(trial.user_attrs.get(RERUN_OF_ATTR) is not None for trial in retried)
    assert not _warnings(caplog)


@pytest.mark.parametrize("launching", [False, True])
def test_overwritable_results_choose_standing_launches_a_later_launch_follows(
    launching: bool,
) -> None:
    """Reading the ledger spares the last launch; a launch about to start follows them all."""
    launches = [
        DirectoryLaunch("a", "g", 1.0, 1.0, True),
        DirectoryLaunch("b", "g", 2.0, 2.0, True),
        DirectoryLaunch("c", "g", 3.0, 3.0, False),
    ]
    held = overwritable_results(launches, {"a", "b", "c"}, launching=launching)
    assert [item.attempt_id for item in held] == (["a", "b"] if launching else ["a"])
