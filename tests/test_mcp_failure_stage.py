"""The stage an MCP run's failure reports follows whether the run claimed its generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phasesweep.config.models import Phase
from phasesweep.engine.errors import ArtifactRootConflictError, StudyFingerprintMismatchError
from phasesweep.engine.paths import _generation_path
from phasesweep.engine.run import TerminalReport, run_experiment
from phasesweep.mcp import runner as mcp_runner
from phasesweep.mcp.runs import RunStore
from tests.conftest import make_experiment
from tests.mcp_helpers import SEEDED_RANDOM, run_claimed_runner, write_experiment_config


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
