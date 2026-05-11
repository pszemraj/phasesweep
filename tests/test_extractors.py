from __future__ import annotations

import json

import pytest

from phasesweep.config import JsonExtractor, LogRegexExtractor
from phasesweep.extractors import ExtractorError, TrialContext, run_extractor


def _ctx(tmp_path):
    return TrialContext(
        experiment="t",
        phase="p",
        trial_id=0,
        trial_dir=tmp_path,
        run_name="t-p-0",
        return_code=0,
        duration_seconds=0.0,
    )


def test_json_basic(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"eval": {"loss": 0.42}}))
    cfg = JsonExtractor(type="json", path="result.json", key="eval.loss")
    assert run_extractor(_ctx(tmp_path), cfg) == pytest.approx(0.42)


def test_json_missing_file(tmp_path):
    cfg = JsonExtractor(type="json", path="nope.json", key="x")
    with pytest.raises(ExtractorError, match="not found"):
        run_extractor(_ctx(tmp_path), cfg)


def test_json_missing_key(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"a": {"b": 1}}))
    cfg = JsonExtractor(type="json", path="result.json", key="a.c")
    with pytest.raises(ExtractorError, match="not found"):
        run_extractor(_ctx(tmp_path), cfg)


def test_json_non_numeric(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"x": "abc"}))
    cfg = JsonExtractor(type="json", path="result.json", key="x")
    with pytest.raises(ExtractorError, match="not numeric"):
        run_extractor(_ctx(tmp_path), cfg)


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
    assert run_extractor(_ctx(tmp_path), cfg) == 0.25


def test_log_regex_min(tmp_path):
    (tmp_path / "stdout.log").write_text("eval_loss=1.0\neval_loss=0.5\neval_loss=0.7\n")
    cfg = LogRegexExtractor(
        type="log_regex",
        file="stdout.log",
        pattern=r"eval_loss=(?P<value>[0-9.eE+-]+)",
        select="min",
    )
    assert run_extractor(_ctx(tmp_path), cfg) == 0.5


def test_log_regex_no_value_group(tmp_path):
    (tmp_path / "stdout.log").write_text("eval_loss=1.0\n")
    cfg = LogRegexExtractor(
        type="log_regex",
        file="stdout.log",
        pattern=r"eval_loss=([0-9.]+)",  # no named 'value'
        select="last",
    )
    with pytest.raises(ExtractorError, match="named group 'value'"):
        run_extractor(_ctx(tmp_path), cfg)


def test_log_regex_no_match(tmp_path):
    (tmp_path / "stdout.log").write_text("nothing here\n")
    cfg = LogRegexExtractor(
        type="log_regex",
        file="stdout.log",
        pattern=r"eval_loss=(?P<value>[0-9.]+)",
        select="last",
    )
    with pytest.raises(ExtractorError, match="No matches"):
        run_extractor(_ctx(tmp_path), cfg)
