"""Seed help states which namespace determines seeded draws."""

from __future__ import annotations

from click.testing import CliRunner

from phasesweep.cli import cli


def test_explain_sampler_seed_names_the_study_namespace() -> None:
    result = CliRunner().invoke(cli, ["explain", "phases[].sampler.seed"])

    assert result.exit_code == 0, result.output
    explanation = result.output.lower()
    assert "study name" in explanation
    assert "experiment::phase" in explanation
    assert "changes the draws" in explanation
