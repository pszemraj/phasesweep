"""The stage an MCP run's failure reports follows whether the run claimed its generation."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from phasesweep.config.models import ExecutionContext, Phase
from phasesweep.engine.errors import (
    ArtifactRootConflictError,
    LaunchEnvironmentError,
    StudyFingerprintMismatchError,
    TrackingTargetUnavailableError,
)
from phasesweep.engine.paths import _generation_path
from phasesweep.engine.run import TerminalReport, run_experiment
from phasesweep.mcp import runner as mcp_runner
from phasesweep.mcp.runs import RunStore
from tests.conftest import make_experiment, write_experiment_config
from tests.mcp_helpers import SEEDED_RANDOM, run_claimed_runner
from tests.tracking_helpers import switchable_wandb, wandb_objective_experiment


def test_a_failure_the_engine_never_reported_is_a_preflight_failure() -> None:
    """The engine reports only from the block that can claim its generation, so no report means no claim."""
    error = ArtifactRootConflictError("bound to /a, offered /b")
    reported = TerminalReport(
        generation_id="g",
        primary_error=error,
        cleanup_confirmed=True,
        recovered_attempt_ids=frozenset(),
        uncertain_attempt_ids=frozenset(),
        failure_stage="execution",
    )

    assert mcp_runner._terminal_error(None, error) == (error, "preflight")
    assert mcp_runner._terminal_error(reported, RuntimeError("other")) == (error, "execution")


@pytest.mark.integration
def test_fingerprint_refusal_on_an_existing_tree_reports_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later phase's fingerprint refusal is raised inside the phase, before it launches a trial."""
    monkeypatch.chdir(tmp_path)
    base = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'fingerprint.journal'}",
        n_trials=1,
        trial_command="echo x=0.5 {overrides}",
    )
    experiment = base.model_copy(
        update={
            "phases": [
                *base.phases,
                Phase(name="q", n_trials=1, inherits=["p"], sampler=SEEDED_RANDOM),
            ]
        }
    )
    run_experiment(experiment)
    generation_before = _generation_path(experiment).read_bytes()
    edited = experiment.model_copy(
        update={
            "phases": [
                experiment.phases[0],
                Phase(name="q", n_trials=1, inherits=["p"], sampler={"type": "random", "seed": 1}),
            ]
        }
    )
    config_path = tmp_path / "experiment.yaml"
    write_experiment_config(config_path, edited)
    store = RunStore(tmp_path / "state")

    with pytest.raises(StudyFingerprintMismatchError):
        run_claimed_runner(store, "refused", config_path, experiment_id="t", cwd=tmp_path)

    failure = json.loads(store.status_path("refused").read_text())["failure"]
    assert (failure["code"], failure["stage"]) == ("fingerprint_mismatch", "preflight")
    assert _generation_path(experiment).read_bytes() == generation_before


@pytest.mark.parametrize(
    ("refusal", "remediation"),
    [
        pytest.param(
            "program_missing",
            "Put it on the PATH PhaseSweep is launched with.",
            id="program_missing",
        ),
        pytest.param(
            "trainer_cwd_missing",
            "Create that directory, or set execution.cwd to an existing one.",
            id="trainer_cwd_missing",
        ),
    ],
)
def test_a_refused_launch_environment_reports_its_own_code_and_fix(
    tmp_path: Path, refusal: str, remediation: str
) -> None:
    """A launch whose trials could not start fails at preflight with the raise's own fix and the runner log that holds the error."""
    experiment = (
        make_experiment(
            workdir=tmp_path / "runs", trial_command="phasesweep-no-such-trainer {overrides}"
        )
        if refusal == "program_missing"
        else make_experiment(
            workdir=tmp_path / "runs", execution=ExecutionContext(cwd=str(tmp_path / "gone"))
        )
    )
    config_path = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    store = RunStore(tmp_path / "state")

    with pytest.raises(LaunchEnvironmentError):
        run_claimed_runner(store, "refused", config_path, experiment_id="t", cwd=tmp_path)

    failure = json.loads(store.status_path("refused").read_text())["failure"]
    assert failure == {
        "code": "launch_environment_invalid",
        "stage": "preflight",
        "retryable": False,
        "actor": "operator",
        "remediation": (
            f"Ask the operator: {remediation} The error in this run's log, `logs/refused.log` "
            "under the catalog's `state_dir`, gives the details."
        ),
    }


@pytest.mark.integration
def test_an_unreadable_tracker_stops_the_live_run_and_refuses_the_next_at_preflight(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """The run that met the outage claimed its generation; the next is refused before claiming one."""
    tracker = switchable_wandb(wandb_worker_sdk, tmp_path)
    experiment, _launches = wandb_objective_experiment(tmp_path, n_trials=1)
    config_path = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    store = RunStore(tmp_path / "state")
    tracker.switch("denied")

    for run_id, stage in (("live", "execution"), ("next", "preflight")):
        with pytest.raises(TrackingTargetUnavailableError):
            run_claimed_runner(store, run_id, config_path, experiment_id="t", cwd=tmp_path)
        failure = json.loads(store.status_path(run_id).read_text())["failure"]
        assert (failure["code"], failure["stage"]) == ("tracking_unavailable", stage)
        assert "the finished training is kept" in failure["remediation"]
