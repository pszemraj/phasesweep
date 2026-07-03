from __future__ import annotations

import json
import sys
import time
import types

import pytest
from pydantic import ValidationError

from phasesweep.config import JsonEqualsGate, JsonExtractor, LogRegexExtractor, WandbExtractor
from phasesweep.evidence import ExtractorError, run_extractor
from tests.conftest import make_trial_context


class _FakeRun:
    def __init__(self, name: str, state: str, summary: dict):
        self.display_name = name
        self.state = state
        self.summary = summary


class _FakeApi:
    def __init__(self, runs_for_filter):
        self._runs_for_filter = runs_for_filter

    def runs(self, path, filters):
        name = filters["display_name"]
        return self._runs_for_filter(path, name)


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``wandb.apis.public.Api`` backed by a supplied runs callable."""

    def install(runs_for_filter):
        wandb_mod = types.ModuleType("wandb")
        apis_mod = types.ModuleType("wandb.apis")
        public_mod = types.ModuleType("wandb.apis.public")

        class Api:
            def __init__(self):
                self._inner = _FakeApi(runs_for_filter)

            def runs(self, path, filters):
                return self._inner.runs(path, filters)

        public_mod.Api = Api
        wandb_mod.apis = apis_mod  # type: ignore[attr-defined]
        apis_mod.public = public_mod  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "wandb", wandb_mod)
        monkeypatch.setitem(sys.modules, "wandb.apis", apis_mod)
        monkeypatch.setitem(sys.modules, "wandb.apis.public", public_mod)

    return install


def test_json_basic(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"eval": {"loss": 0.42}}))
    cfg = JsonExtractor(type="json", path="result.json", key="eval.loss")
    assert run_extractor(make_trial_context(tmp_path), cfg) == pytest.approx(0.42)


def test_json_extractor_reports_missing_or_nonnumeric_values(tmp_path):
    cases = [
        ("missing_file", None, JsonExtractor(type="json", path="nope.json", key="x"), "not found"),
        (
            "missing_key",
            {"result.json": {"a": {"b": 1}}},
            JsonExtractor(type="json", path="result.json", key="a.c"),
            "not found",
        ),
        (
            "non_numeric",
            {"result.json": {"x": "abc"}},
            JsonExtractor(type="json", path="result.json", key="x"),
            "not numeric",
        ),
    ]

    for case, files, cfg, match in cases:
        case_dir = tmp_path / case
        case_dir.mkdir()
        for filename, data in (files or {}).items():
            (case_dir / filename).write_text(json.dumps(data))
        with pytest.raises(ExtractorError, match=match):
            run_extractor(make_trial_context(case_dir), cfg)


def test_extractor_config_rejects_unsafe_paths_and_keys() -> None:
    bad_paths = ["/tmp/result.json", "../result.json", ""]
    bad_keys = ["", ".x", "x."]

    for bad_path in bad_paths:
        with pytest.raises(ValidationError, match="trial-relative path required"):
            JsonExtractor(type="json", path=bad_path, key="x")

    for bad_key in bad_keys:
        with pytest.raises(ValidationError, match="JSON key"):
            JsonEqualsGate(type="json_equals", path="result.json", key=bad_key, value=1)


def test_log_regex_selects_last_or_min_value(tmp_path):
    cases = [
        (
            "last",
            "step=1 eval_loss=1.0\nstep=2 eval_loss=0.5\nstep=3 eval_loss=0.25\n",
            "last",
            0.25,
        ),
        ("min", "eval_loss=1.0\neval_loss=0.5\neval_loss=0.7\n", "min", 0.5),
    ]

    for case, text, select, expected in cases:
        case_dir = tmp_path / case
        case_dir.mkdir()
        (case_dir / "stdout.log").write_text(text)
        cfg = LogRegexExtractor(
            type="log_regex",
            file="stdout.log",
            pattern=r"eval_loss=(?P<value>[0-9.eE+-]+)",
            select=select,
        )
        assert run_extractor(make_trial_context(case_dir), cfg) == expected


def test_log_regex_reports_invalid_patterns_or_no_matches(tmp_path):
    cases = [
        ("no_value_group", "eval_loss=1.0\n", r"eval_loss=([0-9.]+)", "named group 'value'"),
        ("no_match", "nothing here\n", r"eval_loss=(?P<value>[0-9.]+)", "No matches"),
    ]

    for case, text, pattern, match in cases:
        case_dir = tmp_path / case
        case_dir.mkdir()
        (case_dir / "stdout.log").write_text(text)
        cfg = LogRegexExtractor(
            type="log_regex",
            file="stdout.log",
            pattern=pattern,
            select="last",
        )
        with pytest.raises(ExtractorError, match=match):
            run_extractor(make_trial_context(case_dir), cfg)


def test_wandb_extractor_finds_metric(fake_wandb, tmp_path):
    fake_wandb(
        lambda path, name: [_FakeRun(name=name, state="finished", summary={"eval/loss": 0.123})]
    )
    cfg = WandbExtractor(
        type="wandb",
        entity="me",
        project="proj",
        run_name_template="{experiment}-{phase}-{trial_id}",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )
    ctx = make_trial_context(tmp_path, experiment="exp", phase="ph", trial_id=7)
    assert run_extractor(ctx, cfg) == pytest.approx(0.123)


def test_wandb_extractor_timeout(fake_wandb, tmp_path):
    fake_wandb(lambda path, name: [])
    cfg = WandbExtractor(
        type="wandb",
        entity="me",
        project="proj",
        run_name_template="{experiment}-{phase}-{trial_id}",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=0.1,
    )
    with pytest.raises(ExtractorError, match="not found or metric"):
        ctx = make_trial_context(tmp_path, experiment="exp", phase="ph", trial_id=7)
        run_extractor(ctx, cfg)


def test_wandb_extractor_stuck_api_call_does_not_spawn_repeated_threads(fake_wandb, tmp_path):
    calls = 0

    def stuck_runs(path: str, name: str) -> list[_FakeRun]:
        nonlocal calls
        calls += 1
        time.sleep(0.2)
        return []

    fake_wandb(stuck_runs)
    cfg = WandbExtractor(
        type="wandb",
        entity="me",
        project="proj",
        run_name_template="{experiment}-{phase}-{trial_id}",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=0.05,
    )

    with pytest.raises(ExtractorError, match="not found or metric"):
        ctx = make_trial_context(tmp_path, experiment="exp", phase="ph", trial_id=7)
        run_extractor(ctx, cfg)

    assert calls == 1
