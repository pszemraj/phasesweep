"""Interrupted trials rerun in real runs: a stopped ``phasesweep run`` resumes to what an uninterrupted run evaluates.

Each test runs a control experiment and an interrupted copy of the same name in
separate workdirs. The interrupted copy runs ``phasesweep run`` in a subprocess
whose trainer holds a chosen trial in flight until the test sends SIGINT to the
run's process group or SIGKILL to the orchestrator alone, which leaves the
trainer for the next run's recovery to kill. Every process group a test spawns,
and every trainer group its trials recorded, is killed on exit when still alive.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import optuna
import pytest
import yaml

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import CategoricalParam, FloatParam, Repeats, Sampler
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.optuna import _draw_positions, _finished_trial_count, _trial_count_state
from phasesweep.engine.read import read_result
from phasesweep.engine.state import (
    RERUN_OF_ATTR,
    TRIAL_OUTCOME_ATTR,
    _parsed_trial_outcome,
    _trial_interrupted,
)
from phasesweep.engine.study_policy import _load_phase_policy_state
from phasesweep.runtime.reaper import PROCESS_IDENTITY_FILE, kill_stale_group
from tests.conftest import (
    FAKE_TRAIN_METRIC,
    _phase_trials,
    copy_fake_train,
    make_experiment,
    write_trainer,
)

pytestmark = pytest.mark.integration

#: How long a run may take to reach a held trial, or to finish, before the test fails.
_WAIT_SECONDS = 120.0

_LR = FloatParam(type="float", low=1e-5, high=1e-2, log=True)
_WEIGHT_DECAY = FloatParam(type="float", low=0.0, high=0.3)
_DROPOUT = FloatParam(type="float", low=0.0, high=0.3)


class _Harness:
    """One test's workdirs, gated trainer and spawned runs."""

    def __init__(self, root: Path) -> None:
        """Write the gated trainer under ``root``.

        :param Path root: The test's scratch directory.
        """
        self.root = root
        self.gate = root / "gate"
        self.gate.mkdir()
        fake_train = copy_fake_train(root)
        self.trainer = write_trainer(
            root / "gated_train.py",
            f"""
            import os, sys, time
            from pathlib import Path
            gate = Path({str(self.gate)!r})
            trial = os.environ["PHASESWEEP_TRIAL_ID"]
            if (gate / f"hold-{{trial}}").exists():
                (gate / f"reached-{{trial}}").touch()
                while (gate / f"hold-{{trial}}").exists():
                    time.sleep(0.05)
            os.execv(sys.executable, [sys.executable, {str(fake_train)!r}, *sys.argv[1:]])
            """,
        )
        self.spawned: list[subprocess.Popen[bytes]] = []
        self.log = root / "runs.log"

    def experiment(self, name: str, phase: Phase) -> Experiment:
        """Build a persistent experiment ``t`` under ``root/name`` running the gated trainer.

        :param str name: Directory holding this copy's ledger and workdir.
        :param Phase phase: The experiment's one phase.
        :return Experiment: The experiment.
        """
        return make_experiment(
            persistent=self.root / name,
            trial_command=f"python {self.trainer} {{overrides}}",
            metric=FAKE_TRAIN_METRIC,
            phases=[phase],
        )

    def config(self, experiment: Experiment) -> Path:
        """Write ``experiment`` as the YAML file a ``phasesweep run`` subprocess loads.

        :param Experiment experiment: The experiment to write.
        :return Path: The config path.
        """
        assert experiment.workdir is not None
        path = Path(experiment.workdir).parent / "experiment.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=False))
        return path

    def _spawn(self, config: Path) -> subprocess.Popen[bytes]:
        """Start ``phasesweep run`` on ``config`` as its own session.

        :param Path config: Config to run.
        :return subprocess.Popen[bytes]: The run.
        """
        with self.log.open("a") as handle:
            proc = subprocess.Popen(
                [sys.executable, "-m", "phasesweep", "run", str(config)],
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.spawned.append(proc)
        return proc

    def interrupt(self, config: Path, trial: int, sig: signal.Signals) -> None:
        """Run ``config`` until ``trial`` is in flight, then stop the run with ``sig``.

        SIGINT goes to the run's process group, as a terminal's Ctrl-C does;
        SIGKILL goes to the orchestrator alone, so its trainer survives it.

        :param Path config: Config to run.
        :param int trial: Trial number the trainer holds in flight.
        :param signal.Signals sig: ``SIGINT`` or ``SIGKILL``.
        """
        (self.gate / f"hold-{trial}").touch()
        proc = self._spawn(config)
        reached = self.gate / f"reached-{trial}"
        deadline = time.monotonic() + _WAIT_SECONDS
        while not reached.exists():
            if proc.poll() is not None:
                raise AssertionError(
                    f"run exited ({proc.returncode}) before trial {trial} started:\n"
                    f"{self.log.read_text()}"
                )
            if time.monotonic() > deadline:
                raise AssertionError(f"trial {trial} never started:\n{self.log.read_text()}")
            time.sleep(0.05)
        if sig == signal.SIGINT:
            os.killpg(proc.pid, signal.SIGINT)
        else:
            os.kill(proc.pid, signal.SIGKILL)
        assert proc.wait(timeout=_WAIT_SECONDS) != 0, self.log.read_text()

    def complete(self, config: Path) -> None:
        """Run ``config`` to completion and require success.

        :param Path config: Config to run.
        """
        proc = self._spawn(config)
        assert proc.wait(timeout=_WAIT_SECONDS) == 0, self.log.read_text()

    def release(self) -> None:
        """Remove every hold, so a trainer still waiting on one finishes."""
        for hold in self.gate.glob("hold-*"):
            hold.unlink()

    def kill_survivors(self) -> None:
        """Kill every spawned run still alive and every trainer group its trials recorded."""
        for proc in self.spawned:
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        for path in self.root.rglob(PROCESS_IDENTITY_FILE):
            identity = json.loads(path.read_text())
            kill_stale_group(
                identity["pid"],
                identity["proc_starttime"],
                pgid=identity["pgid"],
                grace_seconds=1.0,
            )


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[_Harness]:
    """Yield a harness and kill whatever its runs leave alive.

    :param Path tmp_path: The test's scratch directory.
    :return Iterator[_Harness]: The harness.
    """
    built = _Harness(tmp_path)
    try:
        yield built
    finally:
        built.kill_survivors()
        built.release()


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
    trials, controls = _phase_trials(experiment), _phase_trials(control)
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
        for t in _phase_trials(experiment)
        if RERUN_OF_ATTR in t.user_attrs
    }


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_grid_reruns_points_interrupted_by_sigint_and_sigkill(
    harness: _Harness, n_jobs: int
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
    harness.interrupt(config, len(_phase_trials(experiment)), signal.SIGKILL)
    harness.complete(config)

    assert len(_assert_matches_control(experiment, control)) >= 2
    grid_ids = [
        t.system_attrs["grid_id"] for t in _phase_trials(experiment) if t.state.name == "COMPLETE"
    ]
    assert sorted(grid_ids) == list(range(6))


def test_seeded_random_repeats_rerun_a_chain_of_interruptions(harness: _Harness) -> None:
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
    harness: _Harness,
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


def test_fixed_configuration_killed_mid_trial_reruns_its_one_attempt(harness: _Harness) -> None:
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
    harness: _Harness,
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
    with pytest.raises(TimeoutError, match=r"after 1/3 completed evaluations"):
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
    assert [_trial_count_state(t) for t in _phase_trials(accepted)] == [
        "COMPLETE",
        "INTERRUPTED",
    ]
    raised = harness.experiment(
        "accepted", partial.model_copy(update={"n_trials": 4, "timeout_seconds_per_phase": 600.0})
    )
    run_experiment(raised)
    assert _sources(raised) == {2: 1}
    assert _finished_trial_count(_phase_trials(raised)) == 4


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
    harness: _Harness,
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
