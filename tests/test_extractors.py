from __future__ import annotations

import json
import sys
import types
from dataclasses import replace

import pytest
from pydantic import ValidationError

from phasesweep.config import (
    JsonEnvelopeExtractor,
    JsonEqualsGate,
    JsonExtractor,
    LogRegexExtractor,
    WandbExtractor,
)
from phasesweep.evidence import ExtractorError, run_extractor
from tests.conftest import make_trial_context


class _FakeRun:
    def __init__(self, state: str, summary: dict):
        self.state = state
        self.summary = summary


class _FakeApi:
    def __init__(self, run_for_path):
        self._run_for_path = run_for_path

    def run(self, path):
        return self._run_for_path(path)


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``wandb.apis.public.Api`` backed by a supplied run callable."""

    def install(run_for_path):
        wandb_mod = types.ModuleType("wandb")
        apis_mod = types.ModuleType("wandb.apis")
        public_mod = types.ModuleType("wandb.apis.public")
        timeouts: list[int | None] = []

        class Api:
            def __init__(self, timeout=None):
                timeouts.append(timeout)
                self._inner = _FakeApi(run_for_path)

            def run(self, path):
                return self._inner.run(path)

        public_mod.Api = Api
        wandb_mod.apis = apis_mod  # type: ignore[attr-defined]
        apis_mod.public = public_mod  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "wandb", wandb_mod)
        monkeypatch.setitem(sys.modules, "wandb.apis", apis_mod)
        monkeypatch.setitem(sys.modules, "wandb.apis.public", public_mod)
        return timeouts

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
            "not a JSON number",
        ),
        (
            "numeric_string",
            {"result.json": {"x": "1.25"}},
            JsonExtractor(type="json", path="result.json", key="x"),
            "not a JSON number",
        ),
        (
            "boolean",
            {"result.json": {"x": True}},
            JsonExtractor(type="json", path="result.json", key="x"),
            "not a JSON number",
        ),
    ]

    for case, files, cfg, match in cases:
        case_dir = tmp_path / case
        case_dir.mkdir()
        for filename, data in (files or {}).items():
            (case_dir / filename).write_text(json.dumps(data))
        with pytest.raises(ExtractorError, match=match):
            run_extractor(make_trial_context(case_dir), cfg)


@pytest.mark.parametrize(
    "payload",
    [
        '{"x": 1, "x": 2}',
        '{"x": NaN}',
        '{"x": Infinity}',
    ],
)
def test_json_extractor_rejects_ambiguous_nonstandard_json(tmp_path, payload):
    (tmp_path / "result.json").write_text(payload)
    cfg = JsonExtractor(type="json", path="result.json", key="x")

    with pytest.raises(ExtractorError, match="Invalid JSON"):
        run_extractor(make_trial_context(tmp_path), cfg)


def test_file_extractors_report_invalid_utf8_as_trial_evidence_failure(tmp_path):
    cases = [
        (
            "json",
            "result.json",
            JsonExtractor(type="json", path="result.json", key="x"),
            "not valid UTF-8",
        ),
        (
            "log",
            "stdout.log",
            LogRegexExtractor(
                type="log_regex",
                file="stdout.log",
                pattern=r"value=(?P<value>[0-9.]+)",
            ),
            "not valid UTF-8",
        ),
        (
            "envelope",
            "result.json",
            JsonEnvelopeExtractor(
                type="json_envelope",
                objective_name="val_loss",
                split="validation",
                policy="final_checkpoint",
            ),
            "not valid UTF-8",
        ),
    ]

    for case, filename, cfg, match in cases:
        case_dir = tmp_path / case
        case_dir.mkdir()
        (case_dir / filename).write_bytes(b"\xff\xfe")
        with pytest.raises(ExtractorError, match=match):
            run_extractor(make_trial_context(case_dir), cfg)


def test_json_envelope_binds_objective_to_current_attempt(tmp_path):
    payload = {
        "schema_version": 1,
        "status": "complete",
        "generation_id": "generation-test",
        "attempt_id": "attempt-test",
        "overrides_sha256": "a" * 64,
        "objective": {"name": "val_loss", "split": "validation", "value": 0.25},
        "evaluation": {
            "policy": "final_checkpoint",
            "checkpoint": "final.pt",
            "step": 1000,
        },
    }
    (tmp_path / "result.json").write_text(json.dumps(payload))
    cfg = JsonEnvelopeExtractor(
        type="json_envelope",
        path="result.json",
        objective_name="val_loss",
        split="validation",
        policy="final_checkpoint",
        checkpoint="final.pt",
        expected_step=1000,
    )

    assert run_extractor(make_trial_context(tmp_path), cfg) == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("attempt_id",), "prior-attempt", "does not match attempt"),
        (("generation_id",), "prior-generation", "does not match generation"),
        (("overrides_sha256",), "b" * 64, "resolved overrides"),
        (("status",), "failed", "status='complete'"),
        (("objective", "name"), "train_loss", "objective 'val_loss'"),
        (("evaluation", "policy"), "last_periodic", "policy 'final_checkpoint'"),
        (("evaluation", "step"), 900, "step 1000"),
    ],
)
def test_json_envelope_rejects_mismatched_provenance(tmp_path, path, value, match):
    payload = {
        "schema_version": 1,
        "status": "complete",
        "generation_id": "generation-test",
        "attempt_id": "attempt-test",
        "overrides_sha256": "a" * 64,
        "objective": {"name": "val_loss", "split": "validation", "value": 0.25},
        "evaluation": {
            "policy": "final_checkpoint",
            "checkpoint": "final.pt",
            "step": 1000,
        },
    }
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    (tmp_path / "result.json").write_text(json.dumps(payload))
    cfg = JsonEnvelopeExtractor(
        type="json_envelope",
        path="result.json",
        objective_name="val_loss",
        split="validation",
        policy="final_checkpoint",
        checkpoint="final.pt",
        expected_step=1000,
    )

    with pytest.raises(ExtractorError, match=match):
        run_extractor(make_trial_context(tmp_path), cfg)


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
    paths: list[str] = []

    def finished_run(path: str) -> _FakeRun:
        paths.append(path)
        return _FakeRun(state="finished", summary={"eval/loss": 0.123})

    timeouts = fake_wandb(finished_run)
    cfg = WandbExtractor(
        type="wandb",
        entity="me",
        project="proj",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )
    ctx = make_trial_context(tmp_path, experiment="exp", phase="ph", trial_id=7)
    assert run_extractor(ctx, cfg) == pytest.approx(0.123)
    assert paths == ["me/proj/attempt-test"]
    assert timeouts == [1]


def test_wandb_extractor_timeout(fake_wandb, tmp_path):
    def missing_run(_path: str) -> _FakeRun:
        raise LookupError("not found")

    fake_wandb(missing_run)
    cfg = WandbExtractor(
        type="wandb",
        entity="me",
        project="proj",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=0.03,
    )
    with pytest.raises(ExtractorError, match="not found or metric"):
        ctx = make_trial_context(tmp_path, experiment="exp", phase="ph", trial_id=7)
        run_extractor(ctx, cfg)


def test_wandb_request_timeout_shrinks_with_poll_budget(
    fake_wandb, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"now": 0.0}

    def slow_missing_run(_path: str) -> _FakeRun:
        clock["now"] += 6.0
        raise LookupError("not found")

    timeouts = fake_wandb(slow_missing_run)
    monkeypatch.setattr("phasesweep.evidence.wandb.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        "phasesweep.evidence.wandb.time.sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    cfg = WandbExtractor(
        type="wandb",
        entity="me",
        project="proj",
        metric_key="eval/loss",
        poll_seconds=1.0,
        timeout_seconds=10.0,
    )

    with pytest.raises(ExtractorError, match="not found or metric"):
        run_extractor(make_trial_context(tmp_path), cfg)

    assert timeouts == [10, 3]


@pytest.mark.parametrize("state", ["failed", "crashed", "killed"])
def test_wandb_extractor_rejects_unsuccessful_terminal_run(fake_wandb, tmp_path, state):
    fake_wandb(lambda _path: _FakeRun(state=state, summary={"eval/loss": 99.0}))
    cfg = WandbExtractor(
        type="wandb",
        entity="me",
        project="proj",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )

    with pytest.raises(ExtractorError, match=rf"state '{state}'.*only finished"):
        ctx = make_trial_context(tmp_path)
        run_extractor(ctx, cfg)


def test_wandb_extractor_correlates_by_attempt_not_reused_display_name(fake_wandb, tmp_path):
    runs = {
        "old-attempt": _FakeRun(state="failed", summary={"eval/loss": 99.0}),
        "new-attempt": _FakeRun(state="finished", summary={"eval/loss": 0.1}),
    }
    seen: list[str] = []

    def run_for_path(path: str) -> _FakeRun:
        run_id = path.rsplit("/", 1)[-1]
        seen.append(run_id)
        return runs[run_id]

    fake_wandb(run_for_path)
    cfg = WandbExtractor(
        type="wandb",
        entity="me",
        project="proj",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )
    ctx = replace(
        make_trial_context(tmp_path),
        attempt_id="new-attempt",
        run_name="reused-display-name",
    )

    assert run_extractor(ctx, cfg) == pytest.approx(0.1)
    assert seen == ["new-attempt"]
