from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from phasesweep.config import JsonEqualsGate, JsonExtractor, LogRegexExtractor
from phasesweep.evidence import ExtractorError, run_extractor
from tests.conftest import make_trial_context


def test_json_basic(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"eval": {"loss": 0.42}}))
    cfg = JsonExtractor(type="json", path="result.json", key="eval.loss")
    assert run_extractor(make_trial_context(tmp_path), cfg) == pytest.approx(0.42)


def test_json_missing_file(tmp_path):
    cfg = JsonExtractor(type="json", path="nope.json", key="x")
    with pytest.raises(ExtractorError, match="not found"):
        run_extractor(make_trial_context(tmp_path), cfg)


def test_json_missing_key(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"a": {"b": 1}}))
    cfg = JsonExtractor(type="json", path="result.json", key="a.c")
    with pytest.raises(ExtractorError, match="not found"):
        run_extractor(make_trial_context(tmp_path), cfg)


def test_json_non_numeric(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"x": "abc"}))
    cfg = JsonExtractor(type="json", path="result.json", key="x")
    with pytest.raises(ExtractorError, match="not numeric"):
        run_extractor(make_trial_context(tmp_path), cfg)


@pytest.mark.parametrize("bad_path", ["/tmp/result.json", "../result.json", ""])
def test_evidence_paths_must_be_trial_relative(bad_path: str) -> None:
    with pytest.raises(ValidationError, match="trial-relative path required"):
        JsonExtractor(type="json", path=bad_path, key="x")


@pytest.mark.parametrize("bad_key", ["", ".x", "x."])
def test_json_keys_must_be_non_empty_dotted_paths(bad_key: str) -> None:
    with pytest.raises(ValidationError, match="JSON key"):
        JsonEqualsGate(type="json_equals", path="result.json", key=bad_key, value=1)


def test_log_regex_last(tmp_path):
    (tmp_path / "stdout.log").write_text(
        "step=1 eval_loss=1.0\nstep=2 eval_loss=0.5\nstep=3 eval_loss=0.25\n"
    )
    cfg = LogRegexExtractor(
        type="log_regex",
        file="stdout.log",
        pattern=r"eval_loss=(?P<value>[0-9.eE+-]+)",
        select="last",
    )
    assert run_extractor(make_trial_context(tmp_path), cfg) == 0.25


def test_log_regex_min(tmp_path):
    (tmp_path / "stdout.log").write_text("eval_loss=1.0\neval_loss=0.5\neval_loss=0.7\n")
    cfg = LogRegexExtractor(
        type="log_regex",
        file="stdout.log",
        pattern=r"eval_loss=(?P<value>[0-9.eE+-]+)",
        select="min",
    )
    assert run_extractor(make_trial_context(tmp_path), cfg) == 0.5


def test_log_regex_no_value_group(tmp_path):
    (tmp_path / "stdout.log").write_text("eval_loss=1.0\n")
    cfg = LogRegexExtractor(
        type="log_regex",
        file="stdout.log",
        pattern=r"eval_loss=([0-9.]+)",  # no named 'value'
        select="last",
    )
    with pytest.raises(ExtractorError, match="named group 'value'"):
        run_extractor(make_trial_context(tmp_path), cfg)


def test_log_regex_no_match(tmp_path):
    (tmp_path / "stdout.log").write_text("nothing here\n")
    cfg = LogRegexExtractor(
        type="log_regex",
        file="stdout.log",
        pattern=r"eval_loss=(?P<value>[0-9.]+)",
        select="last",
    )
    with pytest.raises(ExtractorError, match="No matches"):
        run_extractor(make_trial_context(tmp_path), cfg)
