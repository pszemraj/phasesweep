"""Public configuration-boundary regressions for lossless revalidation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from phasesweep import NoFeasibleTrialError, run_experiment
from phasesweep.config import (
    Constraint,
    Experiment,
    LogRegexExtractor,
    Metric,
    Phase,
    RequiredFileGate,
    Sampler,
)


def _metric() -> Metric:
    return Metric(
        extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
    )


def _phase(*, explicitly_empty: bool = False) -> Phase:
    options: dict[str, Any] = {"gates": []} if explicitly_empty else {}
    return Phase(
        name="p",
        n_trials=1,
        sampler=Sampler(type="grid"),
        gpu_policy="none",
        **options,
    )


def _experiment(path: Path, phase: Phase, **options: Any) -> Experiment:
    return Experiment(
        experiment="revalidation",
        workdir=str(path),
        storage="auto",
        provenance={"trainer": "test-shell-v1"},
        trial_command="echo x=1 {overrides}",
        override_format="argparse",
        metric=_metric(),
        phases=[phase],
        **options,
    )


@pytest.mark.parametrize("explicitly_empty", [False, True])
@pytest.mark.integration
def test_revalidation_preserves_attached_gate(
    tmp_path: Path,
    explicitly_empty: bool,
) -> None:
    """A missing-artifact gate applies regardless of initial field presence."""
    phase = _phase(explicitly_empty=explicitly_empty)
    phase.gates.append(RequiredFileGate(type="required_file", path="model.pt"))

    experiment = _experiment(tmp_path / "runs", phase)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)

    assert not list((tmp_path / "runs").rglob("model.pt"))
    assert not list((tmp_path / "runs").rglob("last_successful_generation.yaml"))


@pytest.mark.parametrize("explicitly_empty", [False, True])
@pytest.mark.integration
def test_revalidation_preserves_attached_constraint(
    tmp_path: Path,
    explicitly_empty: bool,
) -> None:
    """An added resource bound remains operative at the execution boundary."""
    options: dict[str, Any] = {"constraints": []} if explicitly_empty else {}
    experiment = _experiment(tmp_path / "runs", _phase(), **options)
    experiment.constraints.append(
        Constraint(
            name="resource",
            max=0.5,
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        )
    )

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)

    assert not list((tmp_path / "runs").rglob("last_successful_generation.yaml"))


@pytest.mark.parametrize("explicitly_empty", [False, True])
@pytest.mark.integration
def test_revalidation_preserves_attached_fixed_override(
    tmp_path: Path,
    explicitly_empty: bool,
) -> None:
    """An added override survives execution, publication, and downstream inheritance."""
    options: dict[str, Any] = {"fixed_overrides": {}} if explicitly_empty else {}
    phase = Phase(
        name="p",
        n_trials=1,
        sampler=Sampler(type="grid"),
        gpu_policy="none",
        **options,
    )
    phase.fixed_overrides["temperature"] = 0.125
    experiment = _experiment(tmp_path / "runs", phase)
    downstream = Phase(
        name="downstream",
        n_trials=1,
        sampler=Sampler(type="grid"),
        gpu_policy="none",
        inherits=["p"],
    )
    experiment = experiment.model_copy(update={"phases": [phase, downstream]})

    winners = run_experiment(experiment)

    assert winners["p"].effective_overrides == {"temperature": 0.125}
    assert winners["downstream"].effective_overrides == {"temperature": 0.125}


@pytest.mark.parametrize("explicitly_empty", [False, True])
@pytest.mark.integration
def test_revalidation_preserves_attached_environment(
    tmp_path: Path,
    explicitly_empty: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An added environment value reaches the trainer instead of being discarded."""
    monkeypatch.delenv("PHASESWEEP_REVALIDATION_VALUE", raising=False)
    options: dict[str, Any] = {"env": {}} if explicitly_empty else {}
    experiment = Experiment(
        experiment="revalidation",
        workdir=str(tmp_path / "runs"),
        storage="auto",
        provenance={"trainer": "test-shell-v1"},
        trial_command='echo x="$PHASESWEEP_REVALIDATION_VALUE" {overrides}',
        override_format="argparse",
        metric=_metric(),
        phases=[_phase()],
        **options,
    )
    experiment.env["PHASESWEEP_REVALIDATION_VALUE"] = "0.125"

    assert run_experiment(experiment)["p"].metric == 0.125


@pytest.mark.parametrize("level", ["experiment", "phase"])
def test_revalidation_rejects_unknown_copied_field(tmp_path: Path, level: str) -> None:
    """A misspelled copied timeout is rejected before artifacts or trials exist."""
    experiment = _experiment(tmp_path / "runs", _phase())
    if level == "phase":
        phase = experiment.phases[0].model_copy(update={"timeout_seconds_per_trail": 5})
        experiment = experiment.model_copy(update={"phases": [phase]})
    else:
        experiment = experiment.model_copy(update={"timeout_seconds_per_rnu": 5})

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        run_experiment(experiment)

    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("mutation", ["phase", "experiment", "phases"])
def test_revalidation_rejects_nested_mutations_before_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import phasesweep.engine.run as run_engine

    experiment = Experiment(
        experiment="mutated",
        workdir=str(tmp_path / "runs"),
        storage=None,
        trial_command="echo x=1 {overrides}",
        override_format="argparse",
        metric=_metric(),
        phases=[_phase()],
    )
    if mutation == "phase":
        experiment.phases[0].search_space["bad"] = {"type": "int", "low": 5, "high": 0}
    elif mutation == "experiment":
        experiment.env["BAD"] = ["not", "a", "string"]
    else:
        experiment.phases.append(_phase())
    monkeypatch.setattr(
        run_engine,
        "_run_experiment_outcome",
        lambda *args, **kwargs: pytest.fail("invalid experiment reached execution"),
    )
    with pytest.raises(ValueError):
        run_experiment(experiment)
    assert not (tmp_path / "runs").exists()
