"""Protocol layer: contracts, gates, promotion, and suites."""

from __future__ import annotations

from pathlib import Path

import pytest

from phasesweep import load_config, run_config
from phasesweep.config import (
    Contract,
    Experiment,
    IntParam,
    JsonExtractor,
    Metric,
    Phase,
    RequiredFileGate,
    Suite,
)
from phasesweep.orchestrator import run_experiment
from tests.conftest import make_experiment, write_trainer, write_yaml


def test_contract_fixed_overrides_and_gates_apply_to_trial(tmp_path: Path) -> None:
    """Contract overrides are immutable trial inputs and contract gates must pass."""
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        with open(args.out, "w") as f:
            json.dump({"x": 0.5}, f)
        """,
    )
    exp = Experiment(
        experiment="contract_test",
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
        contracts={
            "fixed_eval": Contract(
                fixed_overrides={"eval.seq_len": 1024},
                gates=[RequiredFileGate(type="required_file", path="r.json")],
            )
        },
        phases=[
            Phase(
                name="p",
                contracts=["fixed_eval"],
                n_trials=1,
                search_space={"x": IntParam(type="int", low=0, high=1)},
            )
        ],
    )

    winners = run_experiment(exp)

    assert winners["p"].effective_overrides["eval.seq_len"] == 1024


def test_promotion_can_continue_baseline_on_insufficient_delta(tmp_path: Path) -> None:
    """A phase can run a candidate but expose the baseline if promotion fails."""
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, rest = ap.parse_known_args()
        value = 1.0
        for item in rest:
            if item.startswith("score="):
                value = float(item.split("=", 1)[1])
        with open(args.out, "w") as f:
            json.dump({"x": value}, f)
        """,
    )
    exp = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        phases=[
            Phase(
                name="baseline",
                n_trials=1,
                fixed_overrides={"score": 1.0},
                search_space={},
            ),
            Phase(
                name="candidate",
                n_trials=1,
                fixed_overrides={"score": 0.95},
                search_space={},
                promotion={
                    "min_delta_vs": "baseline",
                    "min_delta": 0.1,
                    "on_fail": "continue_baseline",
                },
            ),
        ],
    )

    winners = run_experiment(exp)

    assert winners["candidate"].metric == winners["baseline"].metric
    assert winners["candidate"].effective_overrides == winners["baseline"].effective_overrides


def test_promotion_can_treat_failed_gates_as_advisory(tmp_path: Path) -> None:
    """``requires_gates: false`` records gate failures without failing the trial."""
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, rest = ap.parse_known_args()
        value = 1.0
        for item in rest:
            if item.startswith("score="):
                value = float(item.split("=", 1)[1])
        with open(args.out, "w") as f:
            json.dump({"x": value}, f)
        """,
    )
    exp = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        phases=[
            Phase(
                name="baseline",
                n_trials=1,
                fixed_overrides={"score": 1.0},
                search_space={},
            ),
            Phase(
                name="candidate",
                n_trials=1,
                fixed_overrides={"score": 0.5},
                search_space={},
                gates=[RequiredFileGate(type="required_file", path="missing.txt")],
                promotion={
                    "min_delta_vs": "baseline",
                    "min_delta": 0.1,
                    "requires_gates": False,
                    "on_fail": "stop",
                },
            ),
        ],
    )

    winners = run_experiment(exp)

    assert winners["candidate"].metric == 0.5
    assert winners["candidate"].gates[0]["passed"] is False


def test_suite_promotion_can_continue_baseline_study(tmp_path: Path) -> None:
    """Suite-level promotion compares final study winners across studies."""
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, rest = ap.parse_known_args()
        value = 1.0
        for item in rest:
            if item.startswith("score="):
                value = float(item.split("=", 1)[1])
        with open(args.out, "w") as f:
            json.dump({"x": value}, f)
        """,
    )
    p = write_yaml(
        tmp_path,
        f"""
        suite: promote_suite
        defaults:
          workdir: {tmp_path}/runs
          trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
          metric:
            name: x
            goal: minimize
            extractor: {{ type: json, path: r.json, key: x }}
        studies:
          - name: baseline
            phases:
              - name: eval
                n_trials: 1
                fixed_overrides: {{ score: 1.0 }}
                search_space: {{}}
          - name: candidate
            depends_on: [baseline]
            promotion:
              min_delta_vs: baseline
              min_delta: 0.1
              on_fail: continue_baseline
            phases:
              - name: eval
                n_trials: 1
                fixed_overrides: {{ score: 0.95 }}
                search_space: {{}}
        """,
    )

    config = load_config(p)
    winners = run_config(config)

    assert winners["candidate"]["eval"].metric == winners["baseline"]["eval"].metric


def test_suite_config_runs_dry_without_artifacts(tmp_path: Path) -> None:
    """Suite configs compile studies to isolated experiments and run through dispatch."""
    p = write_yaml(
        tmp_path,
        f"""
        suite: suite_t
        defaults:
          workdir: {tmp_path}/runs
          trial_command: "echo {{overrides}}"
          metric:
            name: x
            goal: minimize
            extractor: {{ type: json, path: r.json, key: x }}
        studies:
          - name: ablation_a
            phases:
              - name: p
                n_trials: 1
                search_space: {{ x: {{ type: int, low: 0, high: 1 }} }}
        """,
    )

    config = load_config(p)
    assert isinstance(config, Suite)
    winners = run_config(config, dry_run=True)

    assert "ablation_a" in winners
    assert not (tmp_path / "runs").exists()


def test_contract_keys_cannot_be_resampled() -> None:
    """Contracts are fixed-comparison inputs, not phase-local suggestions."""
    with pytest.raises(ValueError, match="contract-locked"):
        Experiment(
            experiment="bad_contract",
            trial_command="echo {overrides}",
            metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
            contracts={"c": Contract(fixed_overrides={"seq_len": 1024})},
            phases=[
                Phase(
                    name="p",
                    contracts=["c"],
                    n_trials=1,
                    search_space={"seq_len": IntParam(type="int", low=512, high=2048)},
                )
            ],
        )
