"""Public configuration-boundary regressions for lossless revalidation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from phasesweep import NoFeasibleTrialError, run_experiment, run_suite
from phasesweep.cli import main
from phasesweep.config import (
    Constraint,
    ExecutionContext,
    Experiment,
    LogRegexExtractor,
    Metric,
    Phase,
    RequiredFileGate,
    Sampler,
    StudySpec,
    Suite,
    SuiteDefaults,
)
from phasesweep.config.io import load_config, load_config_bytes


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
@pytest.mark.parametrize("entrypoint", ["experiment", "suite"])
def test_revalidation_preserves_attached_gate(
    tmp_path: Path,
    explicitly_empty: bool,
    entrypoint: str,
) -> None:
    """A missing-artifact gate applies regardless of initial field presence."""
    phase = _phase(explicitly_empty=explicitly_empty)
    phase.gates.append(RequiredFileGate(type="required_file", path="model.pt"))

    if entrypoint == "experiment":
        experiment = _experiment(tmp_path / "runs", phase)
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(experiment)
    else:
        suite = Suite(
            suite="revalidation",
            defaults=SuiteDefaults(
                workdir=str(tmp_path / "runs"),
                storage="auto",
                provenance={"trainer": "test-shell-v1"},
                trial_command="echo x=1 {overrides}",
                override_format="argparse",
                metric=_metric(),
            ),
            studies=[StudySpec(name="s", phases=[phase])],
        )
        with pytest.raises(NoFeasibleTrialError):
            run_suite(suite)

    assert not list((tmp_path / "runs").rglob("model.pt"))
    assert not list((tmp_path / "runs").rglob("last_successful_generation.yaml"))


@pytest.mark.parametrize("explicitly_empty", [False, True])
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


def test_suite_revalidation_preserves_defaults_and_explicit_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lossless revalidation must preserve omission versus explicit-reset semantics."""
    monkeypatch.setenv("PHASESWEEP_REVALIDATION_INHERIT", "7")
    suite = Suite(
        suite="revalidation_defaults",
        defaults=SuiteDefaults(
            workdir=str(tmp_path / "runs"),
            storage="auto",
            provenance={"trainer": "test-shell-v1"},
            trial_command='echo x="${{PHASESWEEP_REVALIDATION_INHERIT:-0}}" {overrides}',
            override_format="argparse",
            metric=_metric(),
            execution=ExecutionContext(inherit_env="none"),
        ),
        studies=[
            StudySpec(name="inherited", phases=[_phase()]),
            StudySpec(name="reset", phases=[_phase()], execution=None, storage=None),
        ],
    )

    compiled = []
    original_compile = Suite.experiment_for_study

    def compile_once(self, study):
        compiled.append(study.name)
        return original_compile(self, study)

    monkeypatch.setattr(Suite, "experiment_for_study", compile_once)
    results = run_suite(suite)

    assert compiled == ["inherited", "reset"]
    assert results["inherited"]["p"].metric == 0
    assert results["reset"]["p"].metric == 7
    assert suite.experiment_for_study(suite.studies[0]).resolved_storage is not None
    assert suite.experiment_for_study(suite.studies[1]).resolved_storage is None


@pytest.mark.parametrize("field", ["trial_command", "metric"])
@pytest.mark.parametrize("invalid_index", [0, 1])
@pytest.mark.parametrize("entrypoint", ["bytes", "path", "direct", "cli-validate", "cli-run"])
def test_suite_resolves_all_studies_before_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid_index: int,
    entrypoint: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing required resolved field fails before directories or trainer launch."""
    import phasesweep.engine.run as run_engine

    monkeypatch.setattr(
        run_engine,
        "_run_experiment_outcome",
        lambda *args, **kwargs: pytest.fail("invalid suite reached component execution"),
    )
    defaults = {
        "workdir": str(tmp_path / "runs"),
        "storage": None,
        "trial_command": "echo x=1 {overrides}",
        "override_format": "argparse",
        "metric": _metric().model_dump(mode="json"),
    }
    studies = [
        {"name": name, "phases": [{"name": "p", "n_trials": 1, "sampler": {"type": "grid"}}]}
        for name in ("first", "second")
    ]
    # Cover absent defaults and explicit null overriding a valid default.
    if invalid_index == 0:
        del defaults[field]
    else:
        studies[invalid_index][field] = None
    payload = {"suite": "invalid", "defaults": defaults, "studies": studies}
    data = yaml.safe_dump(payload).encode()
    path = tmp_path / "suite.yaml"
    path.write_bytes(data)
    if entrypoint.startswith("cli-"):
        monkeypatch.setattr(sys, "argv", ["phasesweep", entrypoint.removeprefix("cli-"), str(path)])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        output = capsys.readouterr().err
        assert f"must define '{field}'" in output
        assert "Traceback" not in output
    else:
        with pytest.raises(ValueError, match=f"must define '{field}'"):
            if entrypoint == "direct":
                run_suite(Suite.model_validate(payload))
            elif entrypoint == "bytes":
                load_config_bytes(data)
            else:
                load_config(path)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("mutation", ["phase", "defaults", "study"])
def test_suite_revalidates_nested_mutations_before_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import phasesweep.engine.run as run_engine

    suite = Suite(
        suite="mutated",
        defaults=SuiteDefaults(
            workdir=str(tmp_path / "runs"),
            storage=None,
            trial_command="echo x=1 {overrides}",
            override_format="argparse",
            metric=_metric(),
        ),
        studies=[StudySpec(name="s", phases=[_phase()])],
    )
    if mutation == "phase":
        suite.studies[0].phases[0].search_space["bad"] = {"type": "int", "low": 5, "high": 0}
    elif mutation == "defaults":
        suite.defaults.env["BAD"] = ["not", "a", "string"]
    else:
        suite.studies[0].phases.append(_phase())
    monkeypatch.setattr(
        run_engine,
        "_run_experiment_outcome",
        lambda *args, **kwargs: pytest.fail("invalid suite reached component execution"),
    )
    with pytest.raises(ValueError):
        run_suite(suite)
    assert not (tmp_path / "runs").exists()
