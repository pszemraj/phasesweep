"""A result that reran an interrupted trial keeps that history in every study that holds it."""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest

import phasesweep.mcp.recovery as mcp_recovery
from phasesweep import run_experiment
from phasesweep.config.io import load_experiment
from phasesweep.config.models import Experiment, Metric
from phasesweep.engine.read import read_result, read_status
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    EARLIER_ATTEMPT_RESULT_ATTR,
    GENERATION_ID_ATTR,
    RERUN_OF_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    InterruptedReruns,
)
from phasesweep.mcp.snapshots import RunResultSnapshot, capture_result_snapshot
from tests.conftest import (
    addressed_phase_study,
    invoke_cli_boundary,
    new_phase_study,
    reaped_pid,
    trial_outcome,
    write_experiment_config,
)
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    make_mcp_app,
    stage_dead_run,
    write_run_status,
    write_srv_catalog,
    write_srv_config,
)
from tests.recovery_helpers import (
    make_stale_trial_dirs,
    stale_trial_rerun_config,
    write_trial_identity,
)

_LINE = "interrupted: 1 of 2 results came from a rerun, the winner among them"


def test_a_flipped_goal_keeps_the_winners_interrupted_rerun_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A goal change that keeps the same winning attempt, adopting it without a launch or its rerun link, still counts it as a rerun of the interrupted trial in the run summary, ``status --trials``, ``show-winners`` and the status and result reads MCP serves."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("COLUMNS", "500")
    experiment, _config = stale_trial_rerun_config(tmp_path)
    run_experiment(experiment)
    first = addressed_phase_study(experiment).trials
    flipped = experiment.model_copy(
        update={"metric": Metric(**{**experiment.metric.model_dump(), "goal": "maximize"})}
    )
    config = write_experiment_config(tmp_path / "flipped.yaml", flipped)
    capsys.readouterr()

    silent = []
    for command in (["run"], ["status", "--trials"], ["show-winners"]):
        assert invoke_cli_boundary([command[0], str(config), *command[1:]], monkeypatch) == 0
        captured = capsys.readouterr()
        if _LINE not in captured.out + captured.err:
            silent.append(command[0])
    assert silent == []
    status, (winner,) = read_result(flipped)
    assert status["phases"][0]["interrupted_reruns"] == 1
    assert winner.reruns == InterruptedReruns(results=2, reruns=1, winner=True)

    adopted = addressed_phase_study(flipped).trials
    assert [t.user_attrs.get(ATTEMPT_ID_ATTR) for t in adopted if t.state.is_finished()] != []
    assert {t.user_attrs.get(ATTEMPT_ID_ATTR) for t in adopted} <= {
        t.user_attrs.get(ATTEMPT_ID_ATTR) for t in first
    }
    assert not any(RERUN_OF_ATTR in t.user_attrs for t in adopted)


def test_recovery_counts_a_settled_pending_rerun_in_the_frozen_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A captured RUNNING rerun of an interrupted trial that recovery settles as COMPLETE, scored from an earlier attempt's result, adds one to each frozen interrupted-rerun counter, as a live status read of the same ledger counts it."""
    run_id = "srv-pending-rerun"
    config = write_srv_config(tmp_path)
    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)
    study = new_phase_study(experiment, "p")
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.FAIL,
            user_attrs={TRIAL_OUTCOME_ATTR: trial_outcome(1, "interrupted")},
        )
    )
    rerun = study.ask()
    attempt_id = "pending-rerun"
    trial_dir, attempt_dir = make_stale_trial_dirs(experiment, "p", rerun.number, attempt_id)
    write_trial_identity(attempt_dir, attempt_id=attempt_id, pid=reaped_pid(), starttime=222)
    for key, value in {
        TRIAL_DIR_ATTR: str(trial_dir),
        GENERATION_ID_ATTR: run_id,
        ATTEMPT_ID_ATTR: attempt_id,
        CLEANUP_CONFIRMED_ATTR: False,
        RERUN_OF_ATTR: 0,
    }.items():
        rerun.set_user_attr(key, value)
    _app, registry, store = make_mcp_app(
        write_srv_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS)
    )
    stage_dead_run(store, run_id, config, registry.get("srv").id, cleanup_uncertain=True)
    frozen = RunResultSnapshot.model_validate(
        capture_result_snapshot(experiment, generation_id=run_id)
    )
    (phase,) = frozen.status.phases
    assert phase.running_attempts is not None
    phase.running_attempts[0].recovers_as = "EVIDENCE_PENDING"
    assert (phase.interrupted_reruns, phase.interrupted_reruns_earlier_attempt) == (0, 0)
    write_run_status(
        store,
        run_id,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
        result_snapshot_state="complete",
        result_snapshot=frozen.model_dump(mode="json"),
    )
    rerun.set_user_attr(TRIAL_OUTCOME_ATTR, trial_outcome(2, "success"))
    rerun.set_user_attr(EARLIER_ATTEMPT_RESULT_ATTR, True)
    study.tell(rerun.number, 1.0)
    study.set_user_attr(CLEANUP_RECOVERED_TRIALS_ATTR, [rerun.number])
    monkeypatch.setattr(mcp_recovery, "kill_stale_group", lambda *_args, **_kwargs: True)

    mcp_recovery.recover_run(registry.state_dir, run_id, confirm=True, emit=lambda _: None)

    status = json.loads(store.status_path(run_id).read_text())
    (recovered,) = status["result_snapshot"]["status"]["phases"]
    (live,) = read_status(experiment)["phases"]
    counters = ("interrupted_reruns", "interrupted_reruns_earlier_attempt")
    assert [recovered[key] for key in counters] == [live[key] for key in counters] == [1, 1]
