"""A two-phase W&B-only sweep against a real W&B project.

Opt in with ``pytest -m live tests/test_wandb_live.py`` and name a project the
current W&B credentials may write to::

    PHASESWEEP_WANDB_ENTITY=<entity> PHASESWEEP_WANDB_PROJECT=<project> \\
        pytest -m live tests/test_wandb_live.py

``PHASESWEEP_WANDB_BASE_URL`` selects another W&B server. The test makes four
sequential CPU trainer runs, each a few seconds, and polls W&B for at most two
minutes per run.
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
from phasesweep.config import (
    CategoricalParam,
    ExecutionContext,
    Experiment,
    Metric,
    Phase,
    Sampler,
    WandbExtractor,
)
from phasesweep.engine import read_winners
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.state import ATTEMPT_ID_ATTR, OBJECTIVE_PROVENANCE_ATTR, TRIAL_DIR_ATTR

pytestmark = pytest.mark.live

_TRAINER = Path(__file__).with_name("wandb_live_trainer.py")


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
        timeout_seconds_per_phase=600,
        search_space={key: CategoricalParam(type="categorical", choices=choices)},
        **kwargs,
    )


def test_wandb_only_sweep_publishes_and_replays_without_remote_reads(tmp_path: Path) -> None:
    """Every objective comes from W&B, keyed by attempt, and a replay reads none of it."""
    pytest.importorskip("wandb")
    entity = os.environ.get("PHASESWEEP_WANDB_ENTITY")
    project = os.environ.get("PHASESWEEP_WANDB_PROJECT")
    if not entity or not project:
        pytest.skip("set PHASESWEEP_WANDB_ENTITY and PHASESWEEP_WANDB_PROJECT to run")
    extractor = WandbExtractor(
        type="wandb",
        base_url=os.environ.get("PHASESWEEP_WANDB_BASE_URL", "https://api.wandb.ai"),
        entity=entity,
        project=project,
        metric_key="eval/loss.min",
        evaluation_axis="eval/step",
        timeout_seconds=120,
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
        execution=ExecutionContext(
            inherit_env="none",
            passthrough_env=[
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
            ],
        ),
        metric=Metric(name="eval_loss", goal="minimize", extractor=extractor),
        timeout_seconds_per_run=1200,
        phases=[
            _grid_phase("learning_rate", "learning_rate", [0.01, 0.1]),
            _grid_phase("regularization", "weight_decay", [0.0, 0.01], inherits=["learning_rate"]),
        ],
    )
    winners = run_experiment(experiment)

    storage = _resolve_storage(experiment.resolved_storage)

    assert winners["learning_rate"].effective_overrides["learning_rate"] == 0.1
    attempts: dict[str, list[str]] = {}
    for phase in experiment.phases:
        trials = optuna.load_study(
            study_name=f"{experiment.experiment}::{phase.name}", storage=storage
        ).trials
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
    assert len(read_winners(experiment)) == 2

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
        trials = optuna.load_study(
            study_name=f"{experiment.experiment}::{phase.name}", storage=storage
        ).trials
        assert [trial.user_attrs[ATTEMPT_ID_ATTR] for trial in trials] == attempts[phase.name]
