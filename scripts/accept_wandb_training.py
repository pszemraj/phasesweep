"""Manually accept W&B-only training, winner inheritance, and offline replay."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import optuna

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
from phasesweep.engine.state import ATTEMPT_ID_ATTR, OBJECTIVE_PROVENANCE_ATTR, TRIAL_DIR_ATTR


def main(argv: list[str] | None = None) -> int:
    """Run the explicitly approved four-attempt acceptance command."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--entity", required=True, help="Authorized W&B account or team")
    parser.add_argument("--project", required=True, help="Authorized W&B project")
    parser.add_argument("--base-url", default="https://api.wandb.ai", help="W&B API endpoint")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu", help="Trainer compute device"
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="Host GPU index leased when --device=cuda"
    )
    parser.add_argument(
        "--workdir", type=Path, required=True, help="Fresh acceptance output directory"
    )
    args = parser.parse_args(argv)
    if args.gpu_id < 0:
        parser.error("--gpu-id must be non-negative")
    workdir = args.workdir.expanduser().resolve()
    trainer = Path(__file__).resolve().parents[1] / "examples" / "wandb_linear_train.py"
    experiment_name = (
        "wandb_linear_acceptance" if args.device == "cpu" else "wandb_linear_gpu_acceptance"
    )
    gpu_policy: Literal["none", "single_per_trial"] = (
        "none" if args.device == "cpu" else "single_per_trial"
    )
    gpu_ids = None if args.device == "cpu" else [args.gpu_id]
    extractor = WandbExtractor(
        type="wandb",
        base_url=args.base_url,
        entity=args.entity,
        project=args.project,
        metric_key="eval/loss",
        timeout_seconds=120,
        poll_seconds=2,
    )
    experiment = Experiment(
        experiment=experiment_name,
        workdir=str(workdir),
        storage="auto",
        provenance={
            "trainer": "wandb-linear-20-steps-v1",
            "data": "fixed-linear-64-32-v1",
            "device": args.device,
        },
        override_format="argparse",
        trial_command=(
            f"{shlex.join([sys.executable, str(trainer)])} "
            f"--device {args.device} --receipt {{trial_dir}}/parameters.json "
            "--device-receipt {trial_dir}/device.json {overrides}"
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
            Phase(
                name="learning_rate",
                n_trials=2,
                n_jobs=1,
                gpu_policy=gpu_policy,
                gpu_ids=gpu_ids,
                sampler=Sampler(type="grid"),
                timeout_seconds_per_trial=120,
                timeout_seconds_per_phase=600,
                search_space={
                    "learning_rate": CategoricalParam(type="categorical", choices=[0.01, 0.1])
                },
            ),
            Phase(
                name="regularization",
                inherits=["learning_rate"],
                n_trials=2,
                n_jobs=1,
                gpu_policy=gpu_policy,
                gpu_ids=gpu_ids,
                sampler=Sampler(type="grid"),
                timeout_seconds_per_trial=120,
                timeout_seconds_per_phase=600,
                search_space={
                    "weight_decay": CategoricalParam(type="categorical", choices=[0.0, 0.01])
                },
            ),
        ],
    )
    storage = experiment.resolved_storage
    assert storage is not None
    print(
        f"Launch budget: four sequential {args.device.upper()} attempts at "
        f"{extractor.base_url}/"
        f"{args.entity}/{args.project}; 20 full-batch steps each; "
        "64 training / 32 held-out examples; trainer cap=120s, polling cap=120s, "
        f"phase cap=600s, experiment cap=1200s; gpu_id="
        f"{args.gpu_id if args.device == 'cuda' else 'none'}; cleanup grace is separate.",
        flush=True,
    )
    winners = run_experiment(experiment)
    assert winners["learning_rate"].effective_overrides["learning_rate"] == 0.1
    identities: dict[str, list[str]] = {}
    for phase in experiment.phases:
        study = optuna.load_study(
            study_name=f"{experiment.experiment}::{phase.name}", storage=storage
        )
        trials = study.trials
        assert len(trials) == 2
        assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in trials)
        assert all(trial.value is not None and math.isfinite(trial.value) for trial in trials)
        winner = winners[phase.name]
        assert winner.metric == min(trial.value for trial in trials if trial.value is not None)
        objective_provenance = winner.objective_provenance
        assert objective_provenance is not None
        capture = objective_provenance["remote_capture"]
        assert capture["run_id"] == winner.attempt_id
        assert capture["run_state"] == "finished"
        assert capture["values"]["eval/loss"] == winner.metric
        assert all(
            capture[key] == getattr(extractor, key) for key in ("base_url", "entity", "project")
        )
        identities[phase.name] = [trial.user_attrs[ATTEMPT_ID_ATTR] for trial in trials]
        for trial in trials:
            evidence = json.loads(trial.user_attrs[OBJECTIVE_PROVENANCE_ATTR])
            trial_capture = evidence["remote_capture"]
            assert trial_capture["run_id"] == trial.user_attrs[ATTEMPT_ID_ATTR]
            assert trial_capture["run_state"] == "finished"
            assert trial_capture["values"]["eval/loss"] == trial.value
            assert all(
                trial_capture[key] == getattr(extractor, key)
                for key in ("base_url", "entity", "project")
            )
            receipt = json.loads(
                (Path(trial.user_attrs[TRIAL_DIR_ATTR]) / "parameters.json").read_text(
                    encoding="utf-8"
                )
            )
            assert set(receipt) == {"learning_rate", "weight_decay"}
            device_receipt = json.loads(
                (Path(trial.user_attrs[TRIAL_DIR_ATTR]) / "device.json").read_text(encoding="utf-8")
            )
            assert set(device_receipt) == {
                "requested_device",
                "model_device",
                "train_tensor_device",
                "eval_tensor_device",
                "cuda_visible_devices",
                "cuda_device_name",
            }
            assert device_receipt["requested_device"] == args.device
            expected_device = "cuda:0" if args.device == "cuda" else "cpu"
            assert device_receipt["model_device"] == expected_device
            assert device_receipt["train_tensor_device"] == expected_device
            assert device_receipt["eval_tensor_device"] == expected_device
            if args.device == "cuda":
                assert device_receipt["cuda_visible_devices"] == str(args.gpu_id)
                assert device_receipt["cuda_device_name"]
            else:
                assert device_receipt["cuda_visible_devices"] is None
                assert device_receipt["cuda_device_name"] is None
            if phase.name == "regularization":
                assert (
                    receipt["learning_rate"]
                    == winners["learning_rate"].effective_overrides["learning_rate"]
                )
                assert receipt["weight_decay"] == trial.params["weight_decay"]
            else:
                assert receipt["learning_rate"] == trial.params["learning_rate"]
    assert len({attempt for phase_ids in identities.values() for attempt in phase_ids}) == 4
    assert len(read_winners(experiment)) == 2
    # A no-op replay must not need SDK availability, contact W&B, or launch a trainer.
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
        study = optuna.load_study(
            study_name=f"{experiment.experiment}::{phase.name}", storage=storage
        )
        assert [trial.user_attrs[ATTEMPT_ID_ATTR] for trial in study.trials] == identities[
            phase.name
        ]
    print(
        json.dumps(
            {
                "result": "passed",
                "workdir": str(workdir),
                "device": args.device,
                "gpu_id": args.gpu_id if args.device == "cuda" else None,
                "winners": {
                    name: {
                        "metric": winner.metric,
                        "parameters": winner.effective_overrides,
                        "attempt_id": winner.attempt_id,
                    }
                    for name, winner in winners.items()
                },
                "replay": "same attempts; no trainer or remote reader",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
