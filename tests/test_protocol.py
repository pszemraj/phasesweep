"""Protocol layer: contracts, gates, promotion, and suites."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from phasesweep import load_config, run_config
from phasesweep.config import (
    ArtifactSizeGate,
    Contract,
    Experiment,
    IntParam,
    JsonEqualsGate,
    JsonExtractor,
    Metric,
    Phase,
    RequiredFileGate,
    Sha256Gate,
    Suite,
)
from phasesweep.engine import run_experiment
from phasesweep.engine.state import Winner, _trial_dir_for
from phasesweep.evidence.evaluation import evaluate_gates
from tests.conftest import make_experiment, make_trial_context, write_trainer, write_yaml


def _write_score_trainer(tmp_path: Path) -> Path:
    """Write a trainer that records the ``--score`` override as metric ``x``."""
    return write_trainer(
        tmp_path,
        """
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        ap.add_argument("--score", type=float, default=None)
        args, rest = ap.parse_known_args()
        value = 1.0 if args.score is None else args.score
        for item in rest:
            if item.startswith("score="):
                value = float(item.split("=", 1)[1])
        with open(args.out, "w") as f:
            json.dump({"x": value}, f)
        """,
    )


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
    trainer = _write_score_trainer(tmp_path)
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
    stored = yaml.safe_load((tmp_path / "runs" / "t" / "candidate" / "winner.yaml").read_text())
    assert stored["metric"]["objective"] == pytest.approx(winners["baseline"].metric)
    assert stored["effective_overrides"] == winners["baseline"].effective_overrides
    assert stored["promotion"]["action"] == "continue_baseline"

    decision = yaml.safe_load(
        (tmp_path / "runs" / "t" / "candidate" / "promotion.yaml").read_text()
    )
    assert decision["promoted"] is False
    assert decision["candidate_metric"] == pytest.approx(0.95)
    assert decision["baseline_metric"] == pytest.approx(1.0)
    assert decision["action"] == "continue_baseline"
    assert decision["exposed_source"] == "baseline"
    assert decision["candidate_trial_number"] == 0
    assert decision["candidate_generation_id"] == decision["generation_id"]
    assert decision["candidate_attempt_id"] != winners["baseline"].attempt_id
    assert decision["baseline_trial_number"] == winners["baseline"].trial_number
    assert decision["baseline_generation_id"] == winners["baseline"].generation_id
    assert decision["baseline_attempt_id"] == winners["baseline"].attempt_id
    assert _trial_dir_for(
        exp,
        "candidate",
        decision["candidate_trial_number"],
        generation_id=decision["candidate_generation_id"],
        attempt_id=decision["candidate_attempt_id"],
    ).is_dir()
    summary = yaml.safe_load((tmp_path / "runs" / "t" / "summary.yaml").read_text())
    assert summary["promotion_decisions"][0] == decision
    assert summary["phases"][1]["promotion"] == decision


def test_promotion_can_treat_failed_gates_as_advisory(tmp_path: Path) -> None:
    """``requires_gates: false`` records gate failures without failing the trial."""
    trainer = _write_score_trainer(tmp_path)
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


def test_phase_promotion_requires_prior_baseline(tmp_path: Path) -> None:
    p = write_yaml(
        tmp_path,
        f"""
        experiment: bad_promo
        storage: journal:///{tmp_path}/study.journal
        workdir: {tmp_path}/runs
        trial_command: "python train.py {{overrides}}"
        metric:
          name: objective
          goal: minimize
          extractor: {{ type: json, path: metrics.json, key: objective }}
        phases:
          - name: baseline
            n_trials: 1
            fixed_overrides:
              model.depth: 2
          - name: candidate
            n_trials: 1
            fixed_overrides:
              model.width: 128
            promotion:
              min_delta_vs: typo
        """,
    )

    with pytest.raises(ValueError, match="promotion references 'typo'.*prior phase"):
        load_config(p)


def test_suite_promotion_can_continue_baseline_study(tmp_path: Path) -> None:
    """Suite-level promotion compares final study winners across studies."""
    trainer = _write_score_trainer(tmp_path)
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
    assert isinstance(config, Suite)
    summary_path = tmp_path / "runs" / "promote_suite" / "suite_summary.yaml"
    first_summary = yaml.safe_load(summary_path.read_text())
    first_decision = first_summary["promotion_decisions"][0]
    assert first_summary["studies"][1]["promotion"] == first_decision
    first_baseline = winners["baseline"]["eval"]
    assert first_decision["candidate_trial_number"] == 0
    assert first_decision["candidate_attempt_id"] != first_baseline.attempt_id
    assert first_decision["baseline_trial_number"] == first_baseline.trial_number
    assert first_decision["baseline_generation_id"] == first_baseline.generation_id
    assert first_decision["baseline_attempt_id"] == first_baseline.attempt_id
    candidate_experiment = config.experiment_for_study(config.studies[1])
    first_candidate_dir = _trial_dir_for(
        candidate_experiment,
        "eval",
        first_decision["candidate_trial_number"],
        generation_id=first_decision["candidate_generation_id"],
        attempt_id=first_decision["candidate_attempt_id"],
    )
    assert first_candidate_dir.is_dir()

    second_winners = run_config(config)

    second_summary = yaml.safe_load(summary_path.read_text())
    second_decision = second_summary["promotion_decisions"][0]
    assert second_summary["studies"][1]["promotion"] == second_decision
    second_baseline = second_winners["baseline"]["eval"]
    assert second_decision["candidate_generation_id"] != first_decision["candidate_generation_id"]
    assert second_decision["candidate_attempt_id"] != first_decision["candidate_attempt_id"]
    assert second_decision["baseline_generation_id"] == second_baseline.generation_id
    assert second_decision["baseline_attempt_id"] == second_baseline.attempt_id
    assert first_candidate_dir.is_dir()
    assert _trial_dir_for(
        candidate_experiment,
        "eval",
        second_decision["candidate_trial_number"],
        generation_id=second_decision["candidate_generation_id"],
        attempt_id=second_decision["candidate_attempt_id"],
    ).is_dir()


def test_suite_promotion_study_phase_selector_requires_prior_phase(tmp_path: Path) -> None:
    p = write_yaml(
        tmp_path,
        f"""
        suite: bad_suite_promo
        defaults:
          workdir: {tmp_path}/runs
          trial_command: "echo {{overrides}}"
          metric:
            name: x
            goal: minimize
            extractor: {{ type: json, path: r.json, key: x }}
        studies:
          - name: baseline
            phases:
              - name: eval
                n_trials: 1
          - name: candidate
            promotion:
              min_delta_vs: baseline.typo
            phases:
              - name: eval
                n_trials: 1
        """,
    )

    with pytest.raises(ValueError, match="promotion references missing baseline phase"):
        load_config(p)


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


def test_failed_suite_rerun_clears_previous_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_yaml(
        tmp_path,
        f"""
        suite: stale_suite
        defaults:
          workdir: {tmp_path}/runs
          trial_command: "echo {{overrides}}"
          metric:
            name: x
            goal: minimize
            extractor: {{ type: json, path: r.json, key: x }}
        studies:
          - name: one
            phases:
              - name: eval
                n_trials: 1
                search_space: {{}}
        """,
    )
    config = load_config(config_path)
    assert isinstance(config, Suite)
    winner = Winner(
        trial_number=0,
        params={},
        effective_overrides={},
        metric=1.0,
        completion={"incomplete": False},
        phase_fingerprint="fingerprint",
        generation_id="generation-one",
        attempt_id="attempt-one",
    )
    monkeypatch.setattr(
        "phasesweep.engine.run.run_experiment",
        lambda *_args, **_kwargs: {"eval": winner},
    )

    run_config(config)
    summary_path = tmp_path / "runs" / "stale_suite" / "suite_summary.yaml"
    assert summary_path.is_file()

    def fail_rerun(*_args: object, **_kwargs: object) -> dict[str, Winner]:
        raise RuntimeError("later suite invocation failed")

    monkeypatch.setattr("phasesweep.engine.run.run_experiment", fail_rerun)

    with pytest.raises(RuntimeError, match="later suite invocation failed"):
        run_config(config)

    assert not summary_path.exists()


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


def test_artifact_size_gate_supports_file_directory_and_json_estimate(tmp_path: Path) -> None:
    """Artifact byte gates cover materialized artifacts and trainer-reported estimates."""
    (tmp_path / "model.bin").write_bytes(b"abcd")
    artifact_dir = tmp_path / "bundle"
    nested = artifact_dir / "nested"
    nested.mkdir(parents=True)
    (artifact_dir / "a.bin").write_bytes(b"abc")
    (nested / "b.bin").write_bytes(b"defg")
    (tmp_path / "result.json").write_text('{"artifact_estimate_bytes": 7}')
    ctx = make_trial_context(tmp_path, experiment="e")

    results = evaluate_gates(
        ctx,
        [
            ArtifactSizeGate(
                type="artifact_size",
                source="file",
                path="model.bin",
                min_bytes=4,
                max_bytes=4,
            ),
            ArtifactSizeGate(
                type="artifact_size",
                source="directory",
                path="bundle",
                min_bytes=7,
                max_bytes=7,
            ),
            ArtifactSizeGate(
                type="artifact_size",
                source="json",
                path="result.json",
                key="artifact_estimate_bytes",
                max_bytes=7,
            ),
        ],
    )

    assert [result.passed for result in results] == [True, True, True]


def test_sha256_gate_streams_file_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = (b"phasesweep" * 131_072) + b"tail"
    (tmp_path / "model.bin").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    def fail_read_bytes(self: Path) -> bytes:
        raise AssertionError("sha256 gate must stream instead of Path.read_bytes()")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    results = evaluate_gates(
        make_trial_context(tmp_path),
        [Sha256Gate(type="sha256", path="model.bin", sha256=digest)],
    )

    assert results[0].passed is True


def test_file_metadata_gate_io_failures_are_failed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    original_open = Path.open

    def fail_model_open(self: Path, *args: object, **kwargs: object):
        if self == model_path:
            raise OSError("artifact became unreadable")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_model_open)

    result = evaluate_gates(
        make_trial_context(tmp_path),
        [Sha256Gate(type="sha256", path="model.bin", sha256=digest)],
    )[0]

    assert result.passed is False
    assert "could not read model.bin" in result.detail


def test_artifact_size_io_failure_is_failed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"payload")
    original_stat = Path.stat

    def fail_model_stat(self: Path, *args: object, **kwargs: object):
        if self == model_path:
            raise OSError("artifact metadata unavailable")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_model_stat)

    result = evaluate_gates(
        make_trial_context(tmp_path),
        [
            ArtifactSizeGate(
                type="artifact_size",
                source="file",
                path="model.bin",
                max_bytes=1024,
            )
        ],
    )[0]

    assert result.passed is False
    assert "could not inspect model.bin" in result.detail


def test_json_equals_gate_requires_matching_json_type(tmp_path: Path) -> None:
    """Protocol equality is type-strict; numeric tolerance belongs in scalar bounds."""
    (tmp_path / "result.json").write_text('{"flag": true, "count": 1}')
    ctx = make_trial_context(tmp_path)

    results = evaluate_gates(
        ctx,
        [
            JsonEqualsGate(type="json_equals", path="result.json", key="flag", value=True),
            JsonEqualsGate(type="json_equals", path="result.json", key="flag", value=1),
            JsonEqualsGate(type="json_equals", path="result.json", key="count", value=1.0),
        ],
    )

    assert [result.passed for result in results] == [True, False, False]
    assert "bool" in results[1].detail
    assert "float" in results[2].detail


def test_artifact_size_gate_reports_bad_sources(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text('{"artifact_estimate_bytes": "7"}')
    ctx = make_trial_context(tmp_path, experiment="e")

    results = evaluate_gates(
        ctx,
        [
            ArtifactSizeGate(type="artifact_size", source="file", path="missing.bin", max_bytes=1),
            ArtifactSizeGate(
                type="artifact_size",
                source="json",
                path="result.json",
                key="artifact_estimate_bytes",
                max_bytes=10,
            ),
        ],
    )

    assert results[0].passed is False
    assert "not a file" in results[0].detail
    assert results[1].passed is False
    assert "not an integer" in results[1].detail
