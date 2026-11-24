"""A two-phase W&B-only sweep against a real W&B project.

Opt in with ``pytest -m live tests/test_wandb_live.py`` and name a project the
current W&B credentials may write to::

    PHASESWEEP_WANDB_ENTITY=<entity> PHASESWEEP_WANDB_PROJECT=<project> \\
        pytest -m live tests/test_wandb_live.py

``PHASESWEEP_WANDB_BASE_URL`` selects another W&B server. The sweep test makes four
sequential CPU trainer runs, each a few seconds, and waits about a minute per run for W&B to serve the finished run's summary. The resume test interrupts one trial after its first attempt's rows reach W&B and resumes it in a second run, a few minutes in all.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import ExecutionContext, Experiment, Metric, Phase
from phasesweep.config.search import CategoricalParam, Sampler
from phasesweep.engine.read import read_result
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    RERUN_OF_ATTR,
    TRIAL_DIR_ATTR,
    _trial_attempt_dir,
)
from phasesweep.evidence.baseline import read_evidence_baseline
from phasesweep.evidence.models import WandbExtractor
from tests.conftest import phase_trials, write_experiment_config
from tests.recovery_helpers import runner_once
from tests.wandb_live_trainer import CURVE_STEPS, curve_loss

pytestmark = pytest.mark.live

_TRAINER = Path(__file__).with_name("wandb_live_trainer.py")
_PASSTHROUGH = [
    "WANDB_API_KEY",
    "WANDB_IDENTITY_TOKEN_FILE",
    "WANDB_CREDENTIALS_FILE",
    "WANDB_MODE",
    "WANDB_DISABLED",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
]


def _live_target() -> tuple[str, str, str]:
    """Return the W&B server, entity and project the live tests write to, or skip.

    :return tuple[str, str, str]: Base URL, entity and project.
    """
    pytest.importorskip("wandb")
    entity = os.environ.get("PHASESWEEP_WANDB_ENTITY")
    project = os.environ.get("PHASESWEEP_WANDB_PROJECT")
    if not entity or not project:
        pytest.skip("set PHASESWEEP_WANDB_ENTITY and PHASESWEEP_WANDB_PROJECT to run")
    return os.environ.get("PHASESWEEP_WANDB_BASE_URL", "https://api.wandb.ai"), entity, project


def _grid_phase(name: str, key: str, choices: list[float], **kwargs: object) -> Phase:
    """Build a two-trial grid phase with the live test's time caps.

    :param str name: Phase name.
    :param str key: The one searched override key.
    :param list[float] choices: Its two values.
    :param object kwargs: Extra phase fields, such as ``inherits``.
    :return Phase: The phase.
    """
    return Phase(  # type: ignore[arg-type]
        name=name,
        n_trials=2,
        gpu_policy="none",
        sampler=Sampler(type="grid"),
        timeout_seconds_per_trial=120,
        # Each of the two trials' reads may wait out W&B's summary lag.
        timeout_seconds_per_phase=1800,
        search_space={key: CategoricalParam(type="categorical", choices=choices)},
        **kwargs,
    )


def test_wandb_only_sweep_publishes_and_replays_without_remote_reads(tmp_path: Path) -> None:
    """Every objective comes from its trial's W&B run, and a replay reads none of it."""
    base_url, entity, project = _live_target()
    extractor = WandbExtractor(
        type="wandb",
        base_url=base_url,
        entity=entity,
        project=project,
        metric_key="eval/loss.min",
        evaluation_axis="eval/step",
        timeout_seconds=600,
        poll_seconds=2,
    )
    experiment = Experiment(
        experiment="wandb_live",
        workdir=str(tmp_path),
        storage="auto",
        provenance={"trainer": "tests/wandb_live_trainer.py"},
        override_format="argparse",
        trial_command=(
            f"{shlex.join([sys.executable, str(_TRAINER)])} "
            "--receipt {trial_dir}/parameters.json {overrides}"
        ),
        execution=ExecutionContext(inherit_env="none", passthrough_env=_PASSTHROUGH),
        metric=Metric(name="eval_loss", goal="minimize", extractor=extractor),
        timeout_seconds_per_run=3600,
        phases=[
            _grid_phase("learning_rate", "learning_rate", [0.01, 0.1]),
            _grid_phase("regularization", "weight_decay", [0.0, 0.01], inherits=["learning_rate"]),
        ],
    )
    winners = run_experiment(experiment)

    assert winners["learning_rate"].effective_overrides["learning_rate"] == 0.1
    attempts: dict[str, list[str]] = {}
    for phase in experiment.phases:
        trials = phase_trials(experiment, phase.name)
        assert [trial.state for trial in trials] == [optuna.trial.TrialState.COMPLETE] * 2
        assert all(trial.value is not None and math.isfinite(trial.value) for trial in trials)
        winner = winners[phase.name]
        assert winner.metric == min(trial.value for trial in trials if trial.value is not None)
        attempts[phase.name] = [trial.user_attrs[ATTEMPT_ID_ATTR] for trial in trials]
        for trial in trials:
            provenance = json.loads(trial.user_attrs[OBJECTIVE_PROVENANCE_ATTR])
            capture = provenance["remote_capture"]
            assert capture["run_id"] == trial.user_attrs[ATTEMPT_ID_ATTR]
            assert capture["run_state"] == "finished"
            assert capture["values"]["eval/loss.min"] == trial.value
            # Loss falls monotonically, so the min is the last evaluation.
            assert provenance["source"]["evaluation"] == {
                "progress": {"axis": "eval/step", "value": 20}
            }
            assert (capture["base_url"], capture["entity"], capture["project"]) == (
                extractor.base_url,
                entity,
                project,
            )
            receipt = json.loads(
                (Path(trial.user_attrs[TRIAL_DIR_ATTR]) / "parameters.json").read_text()
            )
            if phase.name == "regularization":
                assert receipt["learning_rate"] == 0.1
                assert receipt["weight_decay"] == trial.params["weight_decay"]
            else:
                assert receipt["learning_rate"] == trial.params["learning_rate"]
    assert len({attempt for ids in attempts.values() for attempt in ids}) == 4
    assert (
        len(read_result(experiment, phase_names=[phase.name for phase in experiment.phases])[1])
        == 2
    )

    # A no-op replay needs no SDK, contacts no W&B server, and launches no trainer.
    with (
        patch(
            "phasesweep.evidence.wandb.require_wandb_sdk",
            side_effect=AssertionError("SDK on replay"),
        ),
        patch(
            "phasesweep.evidence.evaluation.poll_wandb_summary",
            side_effect=AssertionError("remote read on replay"),
        ),
        patch(
            "phasesweep.engine.phase.launch_trial", side_effect=AssertionError("trainer on replay")
        ),
    ):
        replayed = run_experiment(experiment)
    for phase in experiment.phases:
        assert replayed[phase.name].attempt_id == winners[phase.name].attempt_id
        trials = phase_trials(experiment, phase.name)
        assert [trial.user_attrs[ATTEMPT_ID_ATTR] for trial in trials] == attempts[phase.name]


def test_a_resumed_trial_logs_every_attempt_to_its_one_wandb_run(tmp_path: Path) -> None:
    """An interrupted trial's rerun resumes the trial's W&B run, so its history and min summary span both attempts.

    The first attempt logs steps 1 to 6 of a curve whose minimum is at step 3, waits until
    W&B serves those rows, and is killed with its runner; the next run reruns the trial,
    whose trainer resumes from its checkpoint and logs steps 7 to 12 into the same run.
    """
    import wandb

    base_url, entity, project = _live_target()
    marker = tmp_path / "interrupt"
    extractor = WandbExtractor(
        type="wandb",
        base_url=base_url,
        entity=entity,
        project=project,
        metric_key="eval/loss.min",
        evaluation_axis="eval/step",
        # A resumed run served the earlier attempt's summary for minutes after the rerun ended.
        timeout_seconds=600,
        poll_seconds=2,
    )
    experiment = Experiment(
        experiment="wandb_live_resume",
        workdir=str(tmp_path / "runs"),
        storage="auto",
        provenance={"trainer": "tests/wandb_live_trainer.py"},
        override_format="argparse",
        trial_command=(
            f"{shlex.join([sys.executable, str(_TRAINER)])} --receipt {{trial_dir}}/parameters.json "
            f"--interrupt_marker {shlex.quote(str(marker))} {{overrides}}"
        ),
        execution=ExecutionContext(
            inherit_env="none", passthrough_env=_PASSTHROUGH, cwd=str(tmp_path)
        ),
        metric=Metric(name="eval_loss", goal="minimize", extractor=extractor),
        timeout_seconds_per_run=3600,
        phases=[
            Phase(
                name="resume",
                n_trials=1,
                gpu_policy="none",
                sampler=Sampler(type="grid"),
                # The first attempt waits up to 600 s for W&B to serve its rows.
                timeout_seconds_per_trial=1200,
                timeout_seconds_per_phase=2400,
                search_space={"learning_rate": CategoricalParam(type="categorical", choices=[0.1])},
            )
        ],
    )
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    with runner_once(config, marker, timeout=900):
        pass

    winner = run_experiment(experiment)["resume"]
    first, rerun = phase_trials(experiment, "resume")
    opener = first.user_attrs[ATTEMPT_ID_ATTR]
    api = wandb.Api(overrides={"base_url": base_url}, timeout=60)
    run = api.run(f"{entity}/{project}/{opener}")
    history = list(run.scan_history(keys=["_step", "eval/step", "eval/loss"]))
    details = (
        f"W&B run {run.url}: state {run.state!r}, summary {dict(run.summary_metrics)!r}, "
        f"history {history!r}"
    )
    assert rerun.user_attrs[RERUN_OF_ATTR] == first.number, details
    assert winner.trial_number == rerun.number, details
    assert run.state == "finished", details
    with pytest.raises(wandb.errors.CommError, match="Could not find run"):
        api.run(f"{entity}/{project}/{rerun.user_attrs[ATTEMPT_ID_ATTR]}")
    # Every step once, and W&B's own step continuing across the resume.
    assert [row["eval/step"] for row in history] == list(range(1, CURVE_STEPS + 1)), details
    assert [row["_step"] for row in history] == list(range(CURVE_STEPS)), details
    # The rerun's row is what made its read count.
    rerun_dir = _trial_attempt_dir(rerun)
    assert rerun_dir is not None
    launched_at = read_evidence_baseline(rerun_dir).launched_at
    assert launched_at is not None and run.summary_metrics["_timestamp"] > launched_at, details
    expected = min(curve_loss(step) for step in range(1, CURVE_STEPS + 1))
    provenance = json.loads(rerun.user_attrs[OBJECTIVE_PROVENANCE_ATTR])
    capture = provenance["remote_capture"]
    assert capture["run_id"] == opener, details
    assert rerun.value == capture["values"]["eval/loss.min"] == expected, details
    assert provenance["source"]["evaluation"] == {
        "progress": {"axis": "eval/step", "value": CURVE_STEPS}
    }, details
