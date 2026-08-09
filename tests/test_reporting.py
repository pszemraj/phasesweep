"""Trainer-side objective reporting contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from phasesweep import report_objective
from phasesweep.cli import cli as cli_main
from phasesweep.evidence import run_extractor
from phasesweep.evidence.models import JsonEnvelopeExtractor
from tests.conftest import make_trial_context


@pytest.fixture
def reporting_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install the trial identity and configured destination injected at launch."""
    destination = tmp_path / "reports" / "objective.json"
    monkeypatch.setenv("PHASESWEEP_OBJECTIVE_PATH", str(destination))
    monkeypatch.setenv("PHASESWEEP_GENERATION_ID", "generation-test")
    monkeypatch.setenv("PHASESWEEP_ATTEMPT_ID", "attempt-test")
    monkeypatch.setenv("PHASESWEEP_OVERRIDES_SHA256", "a" * 64)
    return destination


def test_report_objective_writes_extractable_attempt_envelope(
    reporting_environment: Path,
    tmp_path: Path,
) -> None:
    """The helper fills managed identity and preserves extra scalar evidence."""
    reporting_environment.parent.mkdir(parents=True)
    reporting_environment.write_text("previous incomplete result")

    written = report_objective(
        0.125,
        name="eval_loss",
        split="validation",
        policy="best_checkpoint",
        checkpoint="checkpoint-40",
        step=40,
        extra={"param_bytes": 1024},
    )

    assert written == reporting_environment
    payload = json.loads(written.read_text())
    assert payload == {
        "attempt_id": "attempt-test",
        "evaluation": {
            "checkpoint": "checkpoint-40",
            "policy": "best_checkpoint",
            "step": 40,
        },
        "generation_id": "generation-test",
        "objective": {
            "name": "eval_loss",
            "split": "validation",
            "value": 0.125,
        },
        "overrides_sha256": "a" * 64,
        "param_bytes": 1024,
        "schema_version": 1,
        "status": "complete",
    }
    assert not list(reporting_environment.parent.glob(".*.tmp"))

    extractor = JsonEnvelopeExtractor(
        type="json_envelope",
        path="reports/objective.json",
        objective_name="eval_loss",
        split="validation",
        policy="best_checkpoint",
        checkpoint="checkpoint-40",
        expected_step=40,
    )
    context = make_trial_context(tmp_path)
    assert run_extractor(context, extractor) == pytest.approx(0.125)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "0.5"])
def test_report_objective_rejects_non_json_or_nonfinite_values(
    reporting_environment: Path,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="value must be"):
        report_objective(
            value,  # type: ignore[arg-type]
            name="loss",
            split="validation",
            policy="final_checkpoint",
            checkpoint="final.pt",
            step=1,
        )

    assert not reporting_environment.exists()


def test_report_objective_rejects_reserved_extra_fields(reporting_environment: Path) -> None:
    with pytest.raises(ValueError, match="extra cannot replace.*objective"):
        report_objective(
            0.5,
            name="loss",
            split="validation",
            policy="final_checkpoint",
            checkpoint="final.pt",
            step=1,
            extra={"objective": {"value": 999}},
        )

    assert not reporting_environment.exists()


def test_report_objective_requires_injected_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHASESWEEP_OBJECTIVE_PATH", raising=False)

    with pytest.raises(RuntimeError, match="requires PHASESWEEP_OBJECTIVE_PATH"):
        report_objective(
            0.5,
            name="loss",
            split="validation",
            policy="final_checkpoint",
            checkpoint="final.pt",
            step=1,
        )


def test_report_objective_cli_uses_the_same_envelope_writer(
    reporting_environment: Path,
) -> None:
    result = CliRunner().invoke(
        cli_main,
        [
            "report-objective",
            "0.25",
            "--name",
            "accuracy",
            "--split",
            "validation",
            "--policy",
            "final_checkpoint",
            "--checkpoint",
            "final.pt",
            "--step",
            "80",
        ],
    )

    assert result.exit_code == 0, result.output
    assert str(reporting_environment) in result.output
    payload = json.loads(reporting_environment.read_text())
    assert payload["objective"] == {
        "name": "accuracy",
        "split": "validation",
        "value": 0.25,
    }
    assert payload["evaluation"] == {
        "checkpoint": "final.pt",
        "policy": "final_checkpoint",
        "step": 80,
    }
