"""The real decoder example supports seed repeats and persistent inspection as written."""

from pathlib import Path

import pytest

from phasesweep.config.io import load_experiment
from phasesweep.config.search import CategoricalParam
from phasesweep.evidence.models import JsonEnvelopeExtractor
from tests.conftest import REPO

EXAMPLE = REPO / "examples/tiny_decoder_enwik8/repeats.yaml"
GUIDE = REPO / "examples/tiny_decoder_enwik8/README.md"


def test_real_decoder_repeats_example_loads_for_resume_from_another_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shipped recipe includes persistence identity, seed groups, and its trainer cwd."""
    monkeypatch.chdir(tmp_path)
    experiment = load_experiment(EXAMPLE)
    assert experiment.resolved_storage is not None
    assert experiment.provenance
    assert experiment.execution.cwd is not None
    assert Path(experiment.execution.cwd).resolve() == REPO
    assert len(experiment.phases) == 1
    phase = experiment.phases[0]
    assert phase.repeats is not None
    assert phase.repeats.key == "seed"
    assert phase.repeats.reduce == "mean"
    assert len(phase.repeats.values) >= 2
    rates = phase.search_space["learning_rate"]
    assert isinstance(rates, CategoricalParam)
    assert len(rates.choices) >= 3
    assert phase.n_trials == len(rates.choices) * len(phase.repeats.values)
    assert phase.n_jobs == 1
    assert phase.gpu_ids == [0]
    assert phase.gates
    assert "run_trial.py" in experiment.trial_command
    extractor = experiment.metric.extractor
    assert isinstance(extractor, JsonEnvelopeExtractor)
    assert extractor.policy == "final_checkpoint"
    assert extractor.checkpoint == "final.pt"
    assert extractor.expected_step == experiment.trainer_config["num_batches"]


def test_real_decoder_guide_has_a_complete_persistent_repeats_workflow() -> None:
    """The example's entry point makes launch, group inspection, and replay discoverable."""
    guide = GUIDE.read_text(encoding="utf-8")
    assert "## Learning rates over seeds" in guide
    section = guide.split("## Learning rates over seeds", 1)[1].split("\n## ", 1)[0]
    config = "examples/tiny_decoder_enwik8/repeats.yaml"
    assert f"phasesweep validate {config}" in section
    assert f"phasesweep run {config} --dry-run" in section
    assert f"phasesweep status {config} --trials" in section
    assert f"phasesweep show-winners {config}" in section
    assert section.count(f"phasesweep run {config}\n") >= 2
