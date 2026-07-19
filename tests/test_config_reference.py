"""Keep the per-key config reference synchronized with the Pydantic schema."""

from __future__ import annotations

import re
from pathlib import Path

from phasesweep.config import (
    ArtifactSizeGate,
    CategoricalParam,
    Constraint,
    Contract,
    Experiment,
    FloatParam,
    IntParam,
    JsonEqualsGate,
    JsonExtractor,
    JsonScalarBoundGate,
    LogRegexExtractor,
    Metric,
    Phase,
    Promotion,
    RequiredFileGate,
    Sampler,
    Sha256Gate,
    StudySpec,
    Suite,
    SuiteDefaults,
    WandbExtractor,
    WandbSummaryRequiredGate,
)

_REFERENCE = Path(__file__).parents[1] / "docs" / "config_reference.yaml"
_FIELD_DECLARATION = re.compile(r"^# +([a-z][a-z0-9_]*):", re.MULTILINE)
_SECTION_MODELS = {
    "EXPERIMENT ROOT": (Experiment, Metric, Constraint, Contract, Phase, Sampler),
    "SEARCH PARAMETER OBJECTS": (FloatParam, IntParam, CategoricalParam),
    "EXTRACTOR OBJECTS": (JsonExtractor, LogRegexExtractor, WandbExtractor),
    "GATE OBJECTS": (
        RequiredFileGate,
        JsonEqualsGate,
        JsonScalarBoundGate,
        ArtifactSizeGate,
        Sha256Gate,
        WandbSummaryRequiredGate,
    ),
    "PROMOTION OBJECT": (Promotion,),
    "SUITE ROOT": (Suite, SuiteDefaults, StudySpec),
}


def test_config_reference_covers_all_model_fields() -> None:
    """Every documented section exactly covers its corresponding model fields."""
    reference = _REFERENCE.read_text()
    headings = list(_SECTION_MODELS)
    section_starts = [reference.index(f"# {heading}") for heading in headings]
    section_starts.append(len(reference))

    for index, (heading, models) in enumerate(_SECTION_MODELS.items()):
        section = reference[section_starts[index] : section_starts[index + 1]]
        documented = set(_FIELD_DECLARATION.findall(section))
        expected = {field for model in models for field in model.model_fields}

        assert documented == expected, heading
