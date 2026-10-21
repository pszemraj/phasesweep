"""A resume needs winners only for the phases it inherits."""

from pathlib import Path

import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Metric, Phase
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.evidence.models import LogRegexExtractor
from tests.conftest import make_experiment


def test_from_phase_runs_independent_arm_after_failure(tmp_path: Path) -> None:
    """An aborted arm cannot prevent an independent arm from being resumed."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage="auto",
        trial_command="test {phase} = arm_b && echo x=0.5 {overrides}",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.]+)")
        ),
        phases=[
            Phase(
                name=name,
                n_trials=1,
                max_consecutive_failures=1,
                gpu_policy="none",
                sampler=Sampler(type="random", seed=0),
                search_space={name: IntParam(type="int", low=0, high=1)},
            )
            for name in ("arm_a", "arm_b")
        ],
    )
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)

    winners = run_experiment(experiment, from_phase="arm_b")

    assert set(winners) == {"arm_b"}
    assert winners["arm_b"].metric == 0.5
    assert run_experiment(experiment, from_phase="arm_b")["arm_b"].metric == 0.5
