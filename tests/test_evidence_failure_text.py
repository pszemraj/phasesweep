"""The failure text of an unreadable objective or gate file names what happened and what the trainer must write."""

from __future__ import annotations

from pathlib import Path

import pytest

from phasesweep.evidence.evaluation import (
    ExtractorError,
    evaluate_gates,
    run_extractor,
)
from phasesweep.evidence.models import (
    ArtifactSizeGate,
    JsonEnvelopeExtractor,
    JsonEqualsGate,
    JsonExtractor,
    JsonScalarBoundGate,
    LogRegexExtractor,
)
from phasesweep.runtime.json import NonFiniteJsonConstantError, strict_json_loads
from tests.conftest import make_trial_context

_NON_FINITE_TAIL = "which is not a finite number; a diverged run writes it"


def _envelope() -> JsonEnvelopeExtractor:
    return JsonEnvelopeExtractor(
        type="json_envelope",
        objective_name="val_loss",
        split="validation",
        policy="final",
    )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_a_diverged_objective_file_says_it_holds_a_non_finite_number(
    tmp_path: Path, constant: str
) -> None:
    """The reason names the constant the file holds, for the plain and the envelope reader."""
    target = tmp_path / "result.json"
    target.write_text(f'{{"loss": {constant}}}')

    with pytest.raises(ExtractorError) as plain:
        run_extractor(
            make_trial_context(tmp_path), JsonExtractor(type="json", path="result.json", key="loss")
        )
    with pytest.raises(ExtractorError) as envelope:
        run_extractor(make_trial_context(tmp_path), _envelope())

    assert str(plain.value) == f"JSON at {target} holds {constant}, {_NON_FINITE_TAIL}."
    assert str(envelope.value) == (
        f"JSON envelope at {target} holds {constant}, {_NON_FINITE_TAIL}."
    )


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_the_strict_parser_names_the_non_finite_token_it_refuses(token: str) -> None:
    """The parser's refusal of NaN or Infinity carries the token, which the readers report."""
    with pytest.raises(NonFiniteJsonConstantError) as refused:
        strict_json_loads(f"[{token}]")

    assert refused.value.constant == token


@pytest.mark.parametrize(
    "gate",
    [
        JsonEqualsGate(type="json_equals", path="result.json", key="loss", value=1),
        JsonScalarBoundGate(type="json_scalar_bound", path="result.json", key="loss", max=1),
        ArtifactSizeGate(
            type="artifact_size", source="json", path="result.json", key="loss", max_bytes=1
        ),
    ],
    ids=["json_equals", "json_scalar_bound", "artifact_size"],
)
def test_a_gate_reading_a_diverged_file_says_it_holds_a_non_finite_number(
    tmp_path: Path, gate: JsonEqualsGate | JsonScalarBoundGate | ArtifactSizeGate
) -> None:
    """A gate's detail reads the same way as the objective's reason, and other failures stay as they were."""
    (tmp_path / "result.json").write_text('{"loss": NaN}')
    (tmp_path / "duplicate.json").write_text('{"loss": 1, "loss": 2}')

    [diverged] = evaluate_gates(make_trial_context(tmp_path), [gate])
    [duplicate] = evaluate_gates(
        make_trial_context(tmp_path), [gate.model_copy(update={"path": "duplicate.json"})]
    )

    assert diverged.passed is False
    assert (
        diverged.detail == f"result.json:loss unavailable: the file holds NaN, {_NON_FINITE_TAIL}"
    )
    assert duplicate.detail == "duplicate.json:loss unavailable: duplicate JSON key 'loss'"


@pytest.mark.parametrize(
    ("extractor", "file", "step"),
    [
        pytest.param(
            JsonExtractor(type="json", path="result.json", key="loss"),
            "result.json",
            "the trainer must write `result.json` under PHASESWEEP_TRIAL_DIR before it exits, "
            "with the numeric key `loss`",
            id="json",
        ),
        pytest.param(
            _envelope(),
            "result.json",
            "the trainer must call `phasesweep.report_objective(...)` before it exits, from "
            "rank 0 only; it writes `result.json` under PHASESWEEP_TRIAL_DIR",
            id="json-envelope",
        ),
        pytest.param(
            LogRegexExtractor(type="log_regex", file="train.log", pattern=r"loss=(?P<value>\S+)"),
            "train.log",
            "the trainer must write `train.log` under PHASESWEEP_TRIAL_DIR; `stdout.log` and "
            "`stderr.log` are written automatically",
            id="log-regex",
        ),
    ],
)
def test_a_missing_objective_file_says_what_the_trainer_must_write(
    tmp_path: Path,
    extractor: JsonExtractor | JsonEnvelopeExtractor | LogRegexExtractor,
    file: str,
    step: str,
) -> None:
    """A trainer that exits 0 without its objective file is told which file, where and how.

    The path is followed by a space, so a copied path carries no punctuation, and the trainer's
    step comes after it in parentheses.
    """
    with pytest.raises(ExtractorError) as missing:
        run_extractor(make_trial_context(tmp_path), extractor)

    assert str(missing.value).endswith(f" ({step})")
    assert str(missing.value).split(" not found: ")[1].startswith(f"{tmp_path / file} (")


def test_no_log_match_ends_on_the_path_it_searched(tmp_path: Path) -> None:
    """The log path ends the message, so nothing sticks to it when a reader copies it."""
    (tmp_path / "train.log").write_text("nothing here\n")
    extractor = LogRegexExtractor(
        type="log_regex", file="train.log", pattern=r"loss=(?P<value>\S+)"
    )

    with pytest.raises(ExtractorError) as missing:
        run_extractor(make_trial_context(tmp_path), extractor)

    assert str(missing.value).endswith(f" in {tmp_path / 'train.log'}")
