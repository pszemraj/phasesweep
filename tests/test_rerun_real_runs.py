"""Interrupted trials rerun in real runs: a stopped ``phasesweep run`` resumes to what an uninterrupted run evaluates.

Each test runs a control experiment and an interrupted copy of the same name in
separate workdirs. The interrupted copy runs ``phasesweep run`` in a subprocess
whose trainer holds a chosen trial in flight until the test sends SIGINT to the
run's process group or SIGKILL to the orchestrator alone, which leaves the
trainer for the next run's recovery to kill. Every process group a test spawns,
and every trainer group its trials recorded, is killed on exit when still alive.
"""

from __future__ import annotations

import shlex
import signal

import optuna
import pytest
from click.testing import CliRunner

from phasesweep import run_experiment
from phasesweep.cli import cli
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import CategoricalParam, FloatParam, Repeats, Sampler
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.locking import _experiment_lock
from phasesweep.engine.optuna import _draw_positions, _finished_trial_count, _trial_count_state
from phasesweep.engine.read import read_result
from phasesweep.engine.state import (
    RERUN_OF_ATTR,
    TRIAL_OUTCOME_ATTR,
    _parsed_trial_outcome,
    _trial_interrupted,
)
from phasesweep.engine.study_policy import _load_phase_policy_state
from tests.conftest import FAKE_TRAIN_METRIC, make_experiment, phase_trials
from tests.recovery_helpers import RerunHarness

pytestmark = pytest.mark.integration

_LR = FloatParam(type="float", low=1e-5, high=1e-2, log=True)
_WEIGHT_DECAY = FloatParam(type="float", low=0.0, high=0.3)
_DROPOUT = FloatParam(type="float", low=0.0, high=0.3)


def _finished_draws(trials: list[optuna.trial.FrozenTrial]) -> dict[int, dict[str, object]]:
    """Map each draw position to the params its COMPLETE trial evaluated.

    :param list[optuna.trial.FrozenTrial] trials: Every trial of one study.
    :return dict[int, dict[str, object]]: Params by draw position.
    """
    positions = _draw_positions(trials)
    return {
        positions[t.number]: t.params for t in trials if t.state == optuna.trial.TrialState.COMPLETE
    }


def _assert_matches_control(experiment: Experiment, control: Experiment) -> list[int]:
    """Require the resumed experiment to have evaluated and published what the control did.

    Every interrupted trial reads ``INTERRUPTED`` with outcome ``interrupted``
    and exactly one trial names it as its rerun's source; the finished trials
    meet the target, nothing waits, and the failure streak is zero.

    :param Experiment experiment: The interrupted, then resumed, experiment.
    :param Experiment control: The uninterrupted control.
    :return list[int]: The interrupted trial numbers.
    """
    trials, controls = phase_trials(experiment), phase_trials(control)
    assert _finished_draws(trials) == _finished_draws(controls)
    interrupted = [t.number for t in trials if _trial_interrupted(t)]
    sources = [t.user_attrs[RERUN_OF_ATTR] for t in trials if RERUN_OF_ATTR in t.user_attrs]
    assert sorted(sources) == interrupted
    for trial in trials:
        if trial.number in interrupted:
            parsed = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
            assert parsed is not None and parsed.outcome == "interrupted"
            assert _trial_count_state(trial) == "INTERRUPTED"
    assert "WAITING" not in {t.state.name for t in trials}
    phase = experiment.phases[0]
    assert _finished_trial_count(trials) == phase.n_trials
    storage = _resolve_storage(experiment.resolved_storage)
    study = optuna.load_study(study_name="t::p", storage=storage)
    assert _load_phase_policy_state(study).consecutive_failures == 0
    _, (winner,) = read_result(experiment, phase_names=["p"])
    _, (expected,) = read_result(control, phase_names=["p"])
    assert winner is not None and expected is not None
    repeat_key = None if phase.repeats is None else phase.repeats.key
    assert {k: v for k, v in winner.params.items() if k != repeat_key} == {
        k: v for k, v in expected.params.items() if k != repeat_key
    }
    assert winner.metric == expected.metric
    assert winner.incomplete is False
    return interrupted


def _sources(experiment: Experiment) -> dict[int, int]:
    """Map each rerun to the trial it reran.

    :param Experiment experiment: Experiment whose phase ``p`` is read.
    :return dict[int, int]: Source trial number by rerun trial number.
    """
    return {
        t.number: t.user_attrs[RERUN_OF_ATTR]
        for t in phase_trials(experiment)
        if RERUN_OF_ATTR in t.user_attrs
    }


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_grid_reruns_points_interrupted_by_sigint_and_sigkill(
    harness: RerunHarness, n_jobs: int
) -> None:
    """A grid stopped by SIGINT, then by SIGKILL, covers every point once, as one uninterrupted run does."""
    phase = Phase(
        name="p",
        n_trials=6,
        n_jobs=n_jobs,
        gpu_policy="none",
        allow_no_gpu_isolation=n_jobs > 1,
        sampler=Sampler(type="grid", seed=0),
        search_space={
            "model.n_layers": CategoricalParam(type="categorical", choices=[4, 8]),
            "optimizer.lr": CategoricalParam(type="categorical", choices=[1e-4, 3e-4, 1e-3]),
        },
    )
    control = harness.experiment("control", phase)
    run_experiment(control)
    experiment = harness.experiment("interrupted", phase)
    config = harness.config(experiment)

    harness.interrupt(config, 1, signal.SIGINT)
    # The second run's first trial is the rerun of trial 1, so this interrupts a rerun.
    harness.interrupt(config, len(phase_trials(experiment)), signal.SIGKILL)
    harness.complete(config)

    assert len(_assert_matches_control(experiment, control)) >= 2
    grid_ids = [
        t.system_attrs["grid_id"] for t in phase_trials(experiment) if t.state.name == "COMPLETE"
    ]
    assert sorted(grid_ids) == list(range(6))


def test_seeded_random_repeats_rerun_a_chain_of_interruptions(harness: RerunHarness) -> None:
    """A rerun interrupted again gets its own rerun, and every repeat group completes as in the control."""
    phase = Phase(
        name="p",
        n_trials=6,
        gpu_policy="none",
        sampler=Sampler(type="random", seed=4),
        search_space={"optimizer.lr": _LR},
        repeats=Repeats(key="seed", values=[0, 1]),
    )
    control = harness.experiment("control", phase)
    run_experiment(control)
    experiment = harness.experiment("interrupted", phase)
    config = harness.config(experiment)

    harness.interrupt(config, 2, signal.SIGINT)
    harness.interrupt(config, 3, signal.SIGKILL)
    harness.complete(config)

    assert _assert_matches_control(experiment, control) == [2, 3]
    assert _sources(experiment) == {3: 2, 4: 3}


@pytest.mark.parametrize(
    ("sampler", "space", "n_trials", "first", "second"),
    [
        # Trial 1 is inside the three startup draws; trial 5 is position 4, after them.
        pytest.param(
            Sampler(type="tpe", seed=3, n_startup_trials=3),
            {"optimizer.lr": _LR, "optimizer.weight_decay": _WEIGHT_DECAY},
            8,
            1,
            5,
            id="tpe",
        ),
        # Two floats give popsize 6: trial 3 is mid-generation 0, and trial 8 is
        # position 7, the trial that tells generation 0.
        pytest.param(
            Sampler(type="cmaes", seed=3),
            {"optimizer.weight_decay": _WEIGHT_DECAY, "model.dropout": _DROPOUT},
            10,
            3,
            8,
            id="cmaes",
        ),
    ],
)
def test_model_samplers_resume_real_runs_exactly(
    harness: RerunHarness,
    sampler: Sampler,
    space: dict[str, FloatParam],
    n_trials: int,
    first: int,
    second: int,
) -> None:
    """TPE and CMA-ES stopped by SIGINT, then by SIGKILL, draw and publish what one uninterrupted run does."""
    phase = Phase(
        name="p", n_trials=n_trials, gpu_policy="none", sampler=sampler, search_space=space
    )
    control = harness.experiment("control", phase)
    run_experiment(control)
    experiment = harness.experiment("interrupted", phase)
    config = harness.config(experiment)

    harness.interrupt(config, first, signal.SIGINT)
    harness.interrupt(config, second, signal.SIGKILL)
    harness.complete(config)

    assert _assert_matches_control(experiment, control) == [first, second]


def test_fixed_configuration_killed_mid_trial_reruns_its_one_attempt(harness: RerunHarness) -> None:
    """A fixed configuration whose orchestrator was killed reruns at the same target and publishes."""
    phase = Phase(name="p", n_trials=1, gpu_policy="none", fixed_overrides={"model.n_layers": 8})
    control = harness.experiment("control", phase)
    run_experiment(control)
    experiment = harness.experiment("interrupted", phase)
    config = harness.config(experiment)

    harness.interrupt(config, 0, signal.SIGKILL)
    harness.complete(config)

    assert _assert_matches_control(experiment, control) == [0]
    _, (winner,) = read_result(experiment, phase_names=["p"])
    assert winner is not None and winner.trial_number == 1


def test_phase_deadline_interrupts_a_trial_that_reruns_once_the_phase_continues(
    harness: RerunHarness,
) -> None:
    """A deadline's interrupted trial leaves the phase short, then reruns first when it continues."""
    phase = Phase(
        name="p",
        n_trials=3,
        gpu_policy="none",
        timeout_seconds_per_phase=6.0,
        sampler=Sampler(type="random", seed=5),
        search_space={"optimizer.lr": _LR},
    )
    control = harness.experiment("control", phase)
    run_experiment(control)

    refused = harness.experiment("refused", phase)
    (harness.gate / "hold-1").touch()
    with pytest.raises(TimeoutError, match=r"after 1/3 finished trials \(1 completed\)"):
        run_experiment(refused)
    harness.release()
    longer = phase.model_copy(update={"timeout_seconds_per_phase": 600.0})
    run_experiment(harness.experiment("refused", longer))
    assert _assert_matches_control(harness.experiment("refused", longer), control) == [1]

    partial = phase.model_copy(update={"allow_incomplete_on_timeout": True})
    accepted = harness.experiment("accepted", partial)
    (harness.gate / "hold-1").touch()
    assert run_experiment(accepted)["p"].completion["incomplete"] is True
    harness.release()
    # The same target replays the accepted partial and schedules nothing.
    run_experiment(accepted)
    assert [_trial_count_state(t) for t in phase_trials(accepted)] == [
        "COMPLETE",
        "INTERRUPTED",
    ]
    raised = harness.experiment(
        "accepted", partial.model_copy(update={"n_trials": 4, "timeout_seconds_per_phase": 600.0})
    )
    run_experiment(raised)
    assert _sources(raised) == {2: 1}
    assert _finished_trial_count(phase_trials(raised)) == 4


@pytest.mark.parametrize(
    ("sampler", "space", "first", "target"),
    [
        pytest.param(
            Sampler(type="tpe", seed=3, n_startup_trials=3),
            {"optimizer.lr": _LR, "optimizer.weight_decay": _WEIGHT_DECAY},
            6,
            8,
            id="tpe",
        ),
        pytest.param(
            Sampler(type="cmaes", seed=3),
            {"optimizer.weight_decay": _WEIGHT_DECAY, "model.dropout": _DROPOUT},
            6,
            10,
            id="cmaes",
        ),
    ],
)
def test_model_samplers_top_up_real_runs_exactly(
    harness: RerunHarness,
    sampler: Sampler,
    space: dict[str, FloatParam],
    first: int,
    target: int,
) -> None:
    """Raising a finished TPE or CMA-ES phase's target draws what one run of the larger target draws."""
    phase = Phase(name="p", n_trials=target, gpu_policy="none", sampler=sampler, search_space=space)
    control = harness.experiment("control", phase)
    run_experiment(control)

    run_experiment(harness.experiment("topped", phase.model_copy(update={"n_trials": first})))
    topped = harness.experiment("topped", phase)
    run_experiment(topped)

    assert _assert_matches_control(topped, control) == []


_NO_RUN_ACTIVE = (
    "# No run is active; 1 trial is recorded RUNNING. Rerun the same config to settle it. "
    "A trainer still running is stopped, so let it exit first to keep its result."
)


def test_status_tells_a_killed_run_from_a_live_one(harness: RerunHarness) -> None:
    """After SIGKILL both status views say no run is active; a live lock silences them and a rerun clears them."""
    phase = Phase(
        name="p",
        n_trials=2,
        gpu_policy="none",
        sampler=Sampler(type="grid", seed=0),
        search_space={"model.n_layers": CategoricalParam(type="categorical", choices=[4, 8])},
    )
    experiment = harness.experiment("killed", phase)
    config = harness.config(experiment)

    harness.interrupt(config, 0, signal.SIGKILL)

    assert [t.state.name for t in phase_trials(experiment)] == ["RUNNING"]
    for arguments in (["status"], ["status", "--trials"]):
        shown = CliRunner().invoke(cli, [*arguments, str(config)])
        assert shown.exit_code == 0, shown.output
        # The table form wraps the comment to the terminal width, each line still a comment.
        comment = [line for line in shown.output.splitlines() if line.startswith("# ")]
        assert " ".join(line.removeprefix("# ") for line in comment) == _NO_RUN_ACTIVE.removeprefix(
            "# "
        )
        assert shown.output.splitlines()[0] == comment[0]
        assert shown.output.count("No run is active") == 1
        with _experiment_lock(experiment):
            live = CliRunner().invoke(cli, [*arguments, str(config)])
        assert live.exit_code == 0, live.output
        assert "No run is active" not in live.output

    harness.release()
    harness.complete(config)

    for arguments in (["status"], ["status", "--trials"]):
        recovered = CliRunner().invoke(cli, [*arguments, str(config)])
        assert recovered.exit_code == 0, recovered.output
        assert "No run is active" not in recovered.output


@pytest.mark.parametrize("persistent", [True, False], ids=["persistent", "in-memory"])
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGHUP, signal.SIGTERM])
def test_a_stopped_run_ends_with_the_next_step_once(
    harness: RerunHarness, sig: signal.Signals, persistent: bool
) -> None:
    """After SIGINT, SIGHUP or SIGTERM the line before the log path says to rerun, and what that keeps."""
    phase = Phase(
        name="p",
        n_trials=2,
        gpu_policy="none",
        sampler=Sampler(type="grid", seed=0),
        search_space={"model.n_layers": CategoricalParam(type="categorical", choices=[4, 8])},
    )
    if persistent:
        experiment = harness.experiment("stopped", phase)
    else:
        experiment = make_experiment(
            workdir=harness.root / "stopped" / "runs",
            trial_command=f"python {harness.trainer} {{overrides}}",
            metric=FAKE_TRAIN_METRIC,
            phases=[phase],
        )
    config = harness.config(experiment)

    harness.interrupt(config, 0, sig)

    assert harness.spawned[-1].returncode == 128 + sig
    expected = (
        "phasesweep: stopped. Rerun the same command to resume."
        if persistent
        else "phasesweep: stopped. Rerun the same command to start over: storage: null keeps "
        "no completed trials between runs."
    )
    lines = harness.log.read_text().splitlines()
    assert lines[-3] == expected
    assert lines[-2] == shlex.join(["phasesweep", "run", str(config)])
    assert lines[-1].startswith("log: ") and lines[-1].endswith("run.log")
    assert lines.count(expected) == 1
    assert "Aborted." not in lines
