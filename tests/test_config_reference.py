"""Keep the per-key config reference synchronized with the Pydantic schema."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from phasesweep.config import (
    ArtifactSizeGate,
    CategoricalParam,
    Constraint,
    Contract,
    Experiment,
    FloatParam,
    IntParam,
    JsonEnvelopeExtractor,
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
    "EXTRACTOR OBJECTS": (
        JsonExtractor,
        JsonEnvelopeExtractor,
        LogRegexExtractor,
        WandbExtractor,
    ),
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


def _reference_sections() -> dict[str, str]:
    reference = _REFERENCE.read_text()
    headings = list(_SECTION_MODELS)
    starts = [reference.index(f"# {heading}") for heading in headings]
    starts.append(len(reference))
    return {
        heading: reference[starts[index] : starts[index + 1]]
        for index, heading in enumerate(headings)
    }


def _render_default(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = {}
    return json.dumps(value, separators=(",", ":"))


def test_config_reference_covers_all_model_fields_and_defaults() -> None:
    """Every documented section covers model fields and their effective defaults."""
    for heading, section in _reference_sections().items():
        models = _SECTION_MODELS[heading]
        documented = set(_FIELD_DECLARATION.findall(section))
        expected = {field for model in models for field in model.model_fields}
        assert documented == expected, heading

        declarations = section.splitlines()
        for model in models:
            for name, field in model.model_fields.items():
                matching = [line for line in declarations if re.match(rf"^# +{name}:", line)]
                assert matching, f"{heading}: {model.__name__}.{name}"
                if field.is_required():
                    assert any("(required" in line for line in matching), (
                        f"{heading}: {model.__name__}.{name}"
                    )
                    continue
                default = (
                    field.default_factory() if field.default_factory is not None else field.default
                )
                rendered = _render_default(default)
                assert any(f" = {rendered}" in line for line in matching), (
                    f"{heading}: {model.__name__}.{name} default {rendered}"
                )
