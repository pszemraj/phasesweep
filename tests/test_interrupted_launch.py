"""A controller killed while it launches a worker, before any payload crossed, strands no trial."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment
from phasesweep.engine.cleanup import _reap_stale_trials
from phasesweep.engine.paths import _attempts_dir
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    RERUN_OF_ATTR,
    RETURN_CODE_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    _trial_attempt_dir,
)
from phasesweep.errors import ProcessCleanupUncertainError
from phasesweep.runtime.process import (
    ATTEMPT_LIFECYCLE_FILE,
    read_attempt_lifecycle,
    write_attempt_lifecycle,
)
from phasesweep.runtime.reaper import PROCESS_IDENTITY_FILE, PROCESS_IDENTITY_SCHEMA_VERSION
from tests.conftest import (
    addressed_phase_study,
    make_experiment,
    phase_trials,
    reaped_pid,
    write_experiment_config,
    write_trainer,
)
from tests.mcp_helpers import make_mcp_app, stage_dead_run, write_srv_catalog
from tests.recovery_helpers import fabricate_stale_trial, recover_run_cli, write_trial_identity

_GENERATION = "stranded-generation"
_WITHHELD = "no worker received its payload; settles without signalling."

# The runner SIGKILLs itself at one launch boundary. Launch 1 is the trainer's own; launch 2
# is the evidence worker an MLflow capture starts in the same attempt slot after the trainer
# finished. Before the identity write the guardian is killed too: a surviving guardian sees
# end-of-file instead of a payload and records ``exited`` itself, so only a kill that takes
# the guardian with the controller (a reboot, the OOM killer) leaves ``launching`` behind.
_RUNNER = """
import os, signal, sys
from dataclasses import replace
from phasesweep import run_experiment
from phasesweep.config.io import load_experiment
from phasesweep.engine import trial
from phasesweep.evidence.evaluation import _capture_mlflow
from phasesweep.evidence.mlflow import compile_query
from phasesweep.evidence.models import MlflowExtractor
from phasesweep.runtime import process
from phasesweep.tracking import session
from phasesweep.tracking.models import MlflowSettings

config, slot, point, generation_id = sys.argv[1:5]
kill_on = 1 if slot == "trainer" else 2
guardians = []
original_spawn = process._spawn_blocked_supervisor
original_identity = process._write_process_identity

def spawn(*args, **kwargs):
    if point == "before-spawn" and len(guardians) + 1 == kill_on:
        os.kill(os.getpid(), signal.SIGKILL)
    spawned = original_spawn(*args, **kwargs)
    guardians.append(spawned[0])
    return spawned

def write_identity(*args, **kwargs):
    if point == "before-identity" and len(guardians) == kill_on:
        guardians[-1].kill()
        guardians[-1].wait()
        os.kill(os.getpid(), signal.SIGKILL)
    return original_identity(*args, **kwargs)

def extract_with_worker(ctx, cfg, **kwargs):
    settings = MlflowSettings(tracking_uri="http://localhost:5000", experiment_id="1")
    binding = {
        "schema_version": 1,
        "tracking_uri": settings.tracking_uri,
        "experiment_id": "1",
        "run_id": "native-id",
        "run_mode": settings.run_mode,
        "experiment": ctx.experiment,
        "phase": ctx.phase,
        "generation_id": ctx.generation_id,
        "attempt_id": ctx.attempt_id,
        "overrides_sha256": ctx.overrides_sha256,
    }
    tracking_dir = session.attempt_tracking_dir(
        ctx.trial_dir.parent.parent, ctx.generation_id, ctx.attempt_id
    )
    session.write_record(tracking_dir / session.BINDING_FILE, binding)
    session.write_record(
        tracking_dir / session.EXPECTED_BINDING_FILE,
        {k: v for k, v in binding.items() if k != "run_id"},
    )
    remote = experiment.model_copy(update={
        "mlflow": settings,
        "metric": experiment.metric.model_copy(update={
            "extractor": MlflowExtractor(type="mlflow", metric_key="x", select="last"),
        }),
    })
    _capture_mlflow(
        replace(ctx, mlflow_settings=settings.target(), mlflow_query=compile_query(remote, [])),
        deadline=kwargs.get("deadline"),
    )
    return original_extract(ctx, cfg, **kwargs)

process._spawn_blocked_supervisor = spawn
process._write_process_identity = write_identity
experiment = load_experiment(config)
original_extract = trial.run_extractor
if slot == "evidence":
    trial.run_extractor = extract_with_worker
run_experiment(experiment, generation_id=generation_id)
"""


def _strand_launch(tmp_path: Path, *, slot: str, point: str) -> tuple[Experiment, Path, Path]:
    """Run one sweep in a runner that is SIGKILLed at a launch boundary.

    :param Path tmp_path: Directory for the ledger, trainer and its launch log.
    :param str slot: ``trainer`` to kill at the trainer's launch, ``evidence`` at the evidence worker's.
    :param str point: ``before-spawn`` or ``before-identity``.
    :return tuple[Experiment, Path, Path]: The experiment, its config file and the trainer's launch log.
    """
    launches = tmp_path / "launches"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        from pathlib import Path
        with Path({str(launches)!r}).open("a") as handle:
            handle.write("launch\\n")
        print("x=0.25", flush=True)
        """,
    )
    experiment = make_experiment(persistent=tmp_path, trainer=trainer, n_trials=1)
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    with (tmp_path / "runner.log").open("w") as output:
        runner = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(config), slot, point, _GENERATION],
            stdout=output,
            stderr=output,
            timeout=60,
            check=False,
        )
    assert runner.returncode == -9, (tmp_path / "runner.log").read_text()
    (stranded,) = phase_trials(experiment)
    attempt_dir = _trial_attempt_dir(stranded)
    assert attempt_dir is not None
    lifecycle = read_attempt_lifecycle(
        attempt_dir, expected_attempt_id=stranded.user_attrs[ATTEMPT_ID_ATTR]
    )
    assert lifecycle is not None and lifecycle.state == "launching"
    assert not (attempt_dir / PROCESS_IDENTITY_FILE).exists()
    assert stranded.state == optuna.trial.TrialState.RUNNING
    return experiment, config, launches


@pytest.mark.integration
@pytest.mark.parametrize("point", ["before-spawn", "before-identity"])
def test_finished_trainer_keeps_its_result_when_its_evidence_worker_never_got_a_payload(
    tmp_path: Path, point: str
) -> None:
    """A kill while the evidence worker launches settles the finished trainer without retraining."""
    experiment, _config, launches = _strand_launch(tmp_path, slot="evidence", point=point)
    (stranded,) = phase_trials(experiment)
    assert stranded.user_attrs[RETURN_CODE_ATTR] == 0
    attempt = stranded.user_attrs[ATTEMPT_ID_ATTR]
    assert (_attempts_dir(experiment) / f"{attempt}.json").is_file()

    assert run_experiment(experiment)["p"].metric == 0.25

    (settled,) = phase_trials(experiment)
    assert settled.state == optuna.trial.TrialState.COMPLETE
    assert settled.value == 0.25
    assert launches.read_text() == "launch\n"
    assert not (_attempts_dir(experiment) / f"{attempt}.json").exists()
    # Recovery proved no process received the payload; it never claims a group exited.
    settled_attempt_dir = _trial_attempt_dir(settled)
    assert settled_attempt_dir is not None
    lifecycle = read_attempt_lifecycle(settled_attempt_dir, expected_attempt_id=attempt)
    assert lifecycle is not None and lifecycle.state == "launching"


@pytest.mark.integration
@pytest.mark.parametrize("point", ["before-spawn", "before-identity"])
def test_trainer_whose_payload_never_crossed_is_interrupted_and_rerun(
    tmp_path: Path, point: str
) -> None:
    """A kill at the trainer's own launch records an interruption, and the rerun completes."""
    experiment, _config, launches = _strand_launch(tmp_path, slot="trainer", point=point)
    assert not launches.exists()

    assert run_experiment(experiment)["p"].metric == 0.25

    interrupted, rerun = phase_trials(experiment)
    assert interrupted.state == optuna.trial.TrialState.FAIL
    assert interrupted.user_attrs[TRIAL_OUTCOME_ATTR]["outcome"] == "interrupted"
    assert rerun.state == optuna.trial.TrialState.COMPLETE
    assert rerun.user_attrs[RERUN_OF_ATTR] == 0
    assert launches.read_text() == "launch\n"


@pytest.mark.integration
def test_a_new_reading_reruns_an_adopted_trial_whose_payload_never_crossed(
    tmp_path: Path,
) -> None:
    """A changed reading adopts the settled attempt and reruns it once, in its own directory."""
    experiment, _config, launches = _strand_launch(
        tmp_path, slot="trainer", point="before-identity"
    )
    (stranded,) = phase_trials(experiment)
    changed = experiment.model_copy(
        update={"metric": experiment.metric.model_copy(update={"goal": "maximize"})}
    )

    assert run_experiment(changed)["p"].metric == 0.25

    adopted, rerun = addressed_phase_study(changed).trials
    assert adopted.user_attrs[ATTEMPT_ID_ATTR] == stranded.user_attrs[ATTEMPT_ID_ATTR]
    assert adopted.user_attrs[TRIAL_OUTCOME_ATTR]["outcome"] == "interrupted"
    assert rerun.user_attrs[RERUN_OF_ATTR] == 0
    assert rerun.user_attrs[TRIAL_DIR_ATTR] == stranded.user_attrs[TRIAL_DIR_ATTR]
    assert launches.read_text() == "launch\n"


@pytest.mark.integration
def test_recover_run_preview_names_a_withheld_payload_and_changes_nothing(
    tmp_path: Path,
) -> None:
    """``mcp recover-run`` without --confirm reports the withheld payload and writes nothing."""
    experiment, config, launches = _strand_launch(tmp_path, slot="evidence", point="before-spawn")
    (stranded,) = phase_trials(experiment)
    stranded_attempt_dir = _trial_attempt_dir(stranded)
    assert stranded_attempt_dir is not None
    lifecycle_path = stranded_attempt_dir / ATTEMPT_LIFECYCLE_FILE
    before = lifecycle_path.read_bytes()
    _app, registry, store = make_mcp_app(write_srv_catalog(tmp_path, config))
    stage_dead_run(store, _GENERATION, config, "srv", cleanup_uncertain=False)

    preview = recover_run_cli(registry.state_dir, _GENERATION)

    assert preview.exit_code == 0, preview.output
    assert "lifecycle=launching with no process identity: " + _WITHHELD in preview.output
    assert lifecycle_path.read_bytes() == before
    (unchanged,) = phase_trials(experiment)
    assert unchanged.state == optuna.trial.TrialState.RUNNING

    confirmed = recover_run_cli(registry.state_dir, _GENERATION, confirm=True)

    assert confirmed.exit_code == 0, confirmed.output
    assert "kept 1 completed trial result(s)" in confirmed.output
    (settled,) = phase_trials(experiment)
    assert settled.state == optuna.trial.TrialState.COMPLETE
    assert settled.value == 0.25
    assert launches.read_text() == "launch\n"


@pytest.mark.parametrize("identity", ["empty", "malformed", "partial", "foreign"])
def test_launching_with_any_identity_file_keeps_the_strict_read(
    tmp_path: Path, identity: str
) -> None:
    """An identity file that exists at all, however damaged, may follow a delivered payload."""
    experiment = make_experiment(persistent=tmp_path)
    study, _trial_dir, number, attempt_dir = fabricate_stale_trial(
        experiment, "p", attempt_id="launched"
    )
    write_attempt_lifecycle(attempt_dir, attempt_id="launched", state="launching")
    path = attempt_dir / PROCESS_IDENTITY_FILE
    if identity == "foreign":
        write_trial_identity(
            attempt_dir, attempt_id="another-attempt", pid=reaped_pid(), starttime=1
        )
    else:
        path.write_text(
            {
                "empty": "",
                "malformed": "{",
                "partial": json.dumps(
                    {"schema_version": PROCESS_IDENTITY_SCHEMA_VERSION, "attempt_id": "launched"}
                ),
            }[identity]
        )

    with pytest.raises(
        ProcessCleanupUncertainError,
        match="durable process identity is missing, malformed, partial",
    ):
        _reap_stale_trials(study, experiment, "p", confirm=True)

    assert study.get_trials(deepcopy=False)[number].state == optuna.trial.TrialState.RUNNING
