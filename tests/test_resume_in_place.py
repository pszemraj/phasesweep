"""A rerun continues in its trial's directory and is scored only from files its own attempt wrote."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from phasesweep.config.models import Metric
from phasesweep.evidence.baseline import read_evidence_baseline, write_evidence_baseline
from phasesweep.evidence.evaluation import ExtractorError, run_extractor
from phasesweep.evidence.models import JsonExtractor
from tests.conftest import make_experiment, make_trial_context

_JSON_METRIC = Metric(extractor=JsonExtractor(type="json", path="r.json", key="loss"))


def test_a_value_file_left_before_launch_fails_as_not_written_by_this_attempt(
    tmp_path: Path,
) -> None:
    """A result file unchanged since before the attempt ran is refused; the attempt's own rewrite of it reads."""
    experiment = make_experiment(persistent=tmp_path, metric=_JSON_METRIC)
    trial_dir, attempt_dir = tmp_path / "trial", tmp_path / "attempt"
    trial_dir.mkdir()
    attempt_dir.mkdir()
    (trial_dir / "r.json").write_text('{"loss": 0.1}', encoding="utf-8")

    baseline = write_evidence_baseline(experiment, experiment.phases[0], trial_dir, attempt_dir)
    assert read_evidence_baseline(attempt_dir) == baseline and set(baseline) == {"r.json"}
    ctx = dataclasses.replace(
        make_trial_context(tmp_path),
        trial_dir=trial_dir,
        attempt_dir=attempt_dir,
        baseline=baseline,
    )
    with pytest.raises(ExtractorError, match="was not written by this attempt"):
        run_extractor(ctx, _JSON_METRIC.extractor)

    (trial_dir / "r.json").write_text('{"loss": 0.25}', encoding="utf-8")
    assert run_extractor(ctx, _JSON_METRIC.extractor) == 0.25
