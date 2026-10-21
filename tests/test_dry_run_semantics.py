"""A preview refuses incompatible recorded semantics before promising replay."""

from pathlib import Path

import pytest

from phasesweep import run_experiment
from phasesweep.config.models import ExecutionContext, Metric, Phase
from phasesweep.config.search import CategoricalParam, Sampler
from phasesweep.engine.errors import ResumeWinnerMissingError, StudyFingerprintMismatchError
from phasesweep.evidence.models import LogRegexExtractor
from tests.conftest import make_experiment
from tests.ledger_fixtures import tree_snapshot


@pytest.mark.parametrize("changed", ["search_space", "trainer_config", "cwd"])
@pytest.mark.parametrize("phase_name", ["a", "b"])
def test_dry_run_refuses_changed_recorded_semantics(
    tmp_path: Path, changed: str, phase_name: str
) -> None:
    """First and later studies use the real run's precise fingerprint refusal."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage="auto",
        trial_command="echo x=0.5 {overrides}",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.]+)")
        ),
        execution=ExecutionContext(cwd=str(tmp_path)),
        phases=[
            Phase(
                name=name,
                n_trials=1,
                gpu_policy="none",
                sampler=Sampler(type="random", seed=0),
                search_space={name: CategoricalParam(type="categorical", choices=[1, 2])},
            )
            for name in ("a", "b")
        ],
    )
    run_experiment(experiment)
    updated = experiment.model_dump(mode="json")
    if changed == "search_space":
        updated["phases"][0 if phase_name == "a" else 1]["search_space"][phase_name]["choices"] = [
            1,
            3,
        ]
    elif changed == "trainer_config":
        updated["trainer_config"] = {"base": 2}
    else:
        updated["execution"]["cwd"] = str(tmp_path / "other")
    modified = type(experiment).model_validate(updated)
    before = tree_snapshot(tmp_path / "runs")

    with pytest.raises(StudyFingerprintMismatchError, match="Changed:"):
        run_experiment(modified, dry_run=True)

    assert tree_snapshot(tmp_path / "runs") == before


def test_dry_run_refuses_missing_inherited_winner(tmp_path: Path) -> None:
    """A missing required parent cannot become a plausible preview placeholder."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        phases=[
            Phase(name="parent", n_trials=1, fixed_overrides={"depth": 2}),
            Phase(name="child", n_trials=1, inherits=["parent"]),
        ],
    )

    with pytest.raises(ResumeWinnerMissingError, match="Winner file missing for phase 'parent'"):
        run_experiment(experiment, from_phase="child", dry_run=True)

    assert not (tmp_path / "runs").exists()
