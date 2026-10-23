"""Independent resumes retain published winners and skip absent unrelated ones."""

from pathlib import Path

import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.errors import ResumeWinnerMissingError
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.errors import OperatorAction
from phasesweep.evidence.models import LogRegexExtractor
from tests.conftest import invoke_cli_boundary, make_experiment
from tests.experiments import phase_payload, write_experiment


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


def test_from_phase_retains_a_published_independent_winner(tmp_path: Path) -> None:
    """An independent resume carries earlier published results without requiring absent ones."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage="auto",
        trial_command="echo x=0.5 {overrides}",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.]+)")
        ),
        phases=[Phase(name=name, n_trials=1, gpu_policy="none") for name in ("arm_a", "arm_b")],
    )
    original = run_experiment(experiment)

    resumed = run_experiment(experiment, from_phase="arm_b")

    assert set(resumed) == {"arm_a", "arm_b"}
    assert resumed["arm_a"].source == original["arm_a"].source
    assert resumed["arm_a"].metric == original["arm_a"].metric


def _chained_experiment(workdir: Path) -> Experiment:
    """Two phases where ``child`` inherits ``parent``, on auto storage under ``workdir``."""
    return make_experiment(
        workdir=workdir,
        storage="auto",
        trial_command="echo x=0.5 {overrides}",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.]+)")
        ),
        phases=[
            Phase(name="parent", n_trials=1, gpu_policy="none"),
            Phase(name="child", n_trials=1, gpu_policy="none", inherits=["parent"]),
        ],
    )


def test_from_phase_into_a_workdir_without_results_refuses_before_creating_it(
    tmp_path: Path,
) -> None:
    """Earlier winners that were never published leave nothing to resume, and nothing is written."""
    workdir = tmp_path / "alt"

    with pytest.raises(ResumeWinnerMissingError) as refused:
        run_experiment(_chained_experiment(workdir), from_phase="child")

    message = str(refused.value)
    assert "Winner file missing for phase 'parent'" in message
    assert "no earlier results" in message
    assert "Run without --from-phase" in message
    assert refused.value.action is OperatorAction.FIX_CONFIG
    assert not workdir.exists()


def test_from_phase_names_the_phase_a_published_generation_lacks(tmp_path: Path) -> None:
    """A publication that holds other phases' winners still routes the missing one to a fix."""
    workdir = tmp_path / "runs"
    unrelated = _chained_experiment(workdir).model_copy(
        update={"phases": [Phase(name="other", n_trials=1, gpu_policy="none")]}
    )
    run_experiment(unrelated)

    with pytest.raises(ResumeWinnerMissingError) as refused:
        run_experiment(_chained_experiment(workdir), from_phase="child")

    message = str(refused.value)
    assert "Winner file missing for phase 'parent'" in message
    assert "holds no winner for it" in message
    assert "Run without --from-phase" in message
    assert refused.value.action is OperatorAction.FIX_CONFIG


def test_cli_from_phase_into_a_workdir_without_results_names_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI prints the refusal in its `phasesweep:` format with a fix line, and creates no directory."""
    config = write_experiment(
        tmp_path / "refine.yaml",
        experiment="t",
        trial_command="echo x=0.5 {overrides}",
        override_format="argparse",
        metric={"extractor": {"type": "log_regex", "pattern": r"x=(?P<value>[0-9.]+)"}},
        storage="auto",
        workdir=str(tmp_path / "runs"),
        phases=[
            phase_payload(name="parent", gpu_policy="none"),
            phase_payload(
                name="child",
                gpu_policy="none",
                inherits=["parent"],
                search_space={"y": {"type": "int", "low": 0, "high": 10}},
            ),
        ],
    )

    status = invoke_cli_boundary(
        ["run", "--set", f"workdir={tmp_path / 'alt'}", "--from-phase", "child", str(config)],
        monkeypatch,
    )

    assert status == 1
    error = capsys.readouterr().err
    assert error.startswith("phasesweep: Winner file missing for phase 'parent'")
    assert "operational error" not in error
    assert "no earlier results" in error
    assert "Run without --from-phase" in error
    assert error.rstrip().endswith(
        "fix: correct the experiment configuration or the environment it runs in."
    )
    assert not (tmp_path / "alt").exists()
