from __future__ import annotations

import hashlib
import json
import os
import sys
import textwrap
import time
import types
from dataclasses import replace
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from phasesweep.config import (
    ArtifactSizeGate,
    JsonEnvelopeExtractor,
    JsonEqualsGate,
    JsonExtractor,
    LogRegexExtractor,
    WandbExtractor,
    WandbSummaryRequiredGate,
)
from phasesweep.errors import UnsafeProcessCleanupError
from phasesweep.evidence import ExtractorError, evaluate_gates, run_extractor
from phasesweep.evidence.evaluation import DeadlineExceededError, extractor_config_fingerprint
from phasesweep.evidence.wandb import (
    WandbPollTimeout,
    WandbRunTerminalError,
    WandbSetupError,
    _poll_wandb_summary,
    poll_wandb_summary,
)
from phasesweep.runtime.process import (
    PROCESS_IDENTITY_FILE,
    is_pid_alive,
    read_attempt_lifecycle,
    read_stale_process_identity,
)
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
    """Run polling-loop unit tests inline with a fake W&B API and controllable clock."""

    def poll_inline(*, trial_dir, **kwargs):
        return _poll_wandb_summary(**kwargs)

    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", poll_inline)

    def install(run_for_path=None, *, api_class=None):  # noqa: ANN001, ANN202
        wandb_mod = types.ModuleType("wandb")
        apis_mod = types.ModuleType("wandb.apis")
        public_mod = types.ModuleType("wandb.apis.public")
        timeouts: list[int | None] = []

        if api_class is None:

            class Api:
                def __init__(self, overrides=None, timeout=None):
                    timeouts.append(timeout)
                    self._inner = _FakeApi(run_for_path)

                def run(self, path):
                    return self._inner.run(path)

            api_class = Api

        public_mod.Api = api_class
        wandb_mod.apis = apis_mod  # type: ignore[attr-defined]
        apis_mod.public = public_mod  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "wandb", wandb_mod)
        monkeypatch.setitem(sys.modules, "wandb.apis", apis_mod)
        monkeypatch.setitem(sys.modules, "wandb.apis.public", public_mod)
        return timeouts

    return install


@pytest.fixture
def wandb_worker_sdk(tmp_path, monkeypatch):
    """Put a fake SDK on the real polling worker's import path."""
    root = tmp_path / "sdk"
    apis = root / "wandb" / "apis"
    apis.mkdir(parents=True)
    (root / "wandb" / "__init__.py").write_text("", encoding="utf-8")
    (apis / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setattr(
        "phasesweep.evidence.wandb.TemporaryDirectory",
        lambda **kwargs: TemporaryDirectory(dir=tmp_path, **kwargs),
    )

    def install(source):
        (apis / "public.py").write_text(textwrap.dedent(source), encoding="utf-8")

    return install


def test_wandb_worker_returns_summary(wandb_worker_sdk, tmp_path):
    wandb_worker_sdk(
        """
        class Api:
            def __init__(self, overrides=None, timeout=None):
                assert overrides == {"base_url": "https://example.test"}
                assert 0 < timeout <= 5

            def run(self, path):
                assert path == "entity/project/attempt"
                class Run:
                    state = "finished"
                    summary = {"loss": 0.25, "extra": [1, "two"]}
                return Run()
        """
    )

    assert poll_wandb_summary(
        base_url="https://example.test",
        entity="entity",
        project="project",
        run_id="attempt",
        trial_dir=tmp_path,
        poll_seconds=0.01,
        timeout_seconds=5,
        required_keys=["loss"],
    ) == {"loss": 0.25, "extra": [1, "two"]}
    assert list(tmp_path.glob("phasesweep-wandb-*")) == []
    assert (
        read_stale_process_identity(tmp_path, expected_attempt_id="attempt").attempt_id == "attempt"
    )
    assert read_attempt_lifecycle(tmp_path, expected_attempt_id="attempt").cleanup_confirmed


@pytest.mark.parametrize("stage", ["constructor", "lookup"])
def test_wandb_deadline_stops_sdk_retries_and_reaps_workers(
    wandb_worker_sdk, tmp_path, monkeypatch, stage
):
    marker = tmp_path / "pids.json"
    monkeypatch.setenv("POLL_TEST_PIDS", str(marker))
    monkeypatch.setenv("POLL_TEST_STAGE", stage)
    wandb_worker_sdk(
        """
        import json, os, subprocess, sys, time
        from pathlib import Path

        def sdk_retries():
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            Path(os.environ["POLL_TEST_PIDS"]).write_text(json.dumps([os.getpid(), child.pid]))
            retry_deadline = time.monotonic() + 30
            while time.monotonic() < retry_deadline:
                try:
                    raise ConnectionError("transient transport failure")
                except ConnectionError:
                    time.sleep(0.05)

        class Api:
            def __init__(self, overrides=None, timeout=None):
                if os.environ["POLL_TEST_STAGE"] == "constructor":
                    sdk_retries()

            def run(self, path):
                sdk_retries()
        """
    )
    started = time.monotonic()

    with pytest.raises(WandbPollTimeout):
        poll_wandb_summary(
            base_url="https://example.test",
            entity="e",
            project="p",
            run_id="attempt",
            trial_dir=tmp_path,
            poll_seconds=0.01,
            timeout_seconds=1,
        )

    assert time.monotonic() - started < 3
    pids = json.loads(marker.read_text(encoding="utf-8"))
    assert all(not is_pid_alive(pid) for pid in pids)
    assert list(tmp_path.glob("phasesweep-wandb-*")) == []
    assert (tmp_path / PROCESS_IDENTITY_FILE).is_file()
    assert read_attempt_lifecycle(tmp_path, expected_attempt_id="attempt").cleanup_confirmed


@pytest.mark.parametrize(
    ("source", "error", "match"),
    [
        ("raise ImportError('fake SDK missing')", ImportError, "fake SDK missing"),
        (
            "class Api:\n    def __init__(self, **kwargs):\n        raise ValueError('bad credentials')",
            WandbSetupError,
            "bad credentials",
        ),
        (
            "class Api:\n    def __init__(self, **kwargs): pass\n"
            "    def run(self, path):\n        return type('Run', (), {'state': 'crashed'})()",
            WandbRunTerminalError,
            "crashed",
        ),
    ],
)
def test_wandb_worker_preserves_typed_errors(wandb_worker_sdk, tmp_path, source, error, match):
    wandb_worker_sdk(source)
    with pytest.raises(error, match=match):
        poll_wandb_summary(
            base_url="https://example.test",
            entity="e",
            project="p",
            run_id="attempt",
            trial_dir=tmp_path,
            poll_seconds=0.01,
            timeout_seconds=5,
        )


def test_wandb_launch_failure_cannot_recover_using_previous_process_identity(tmp_path, monkeypatch):
    from phasesweep.runtime import process

    # The completed trainer's identity must not survive a failed worker launch.
    with (tmp_path / "trainer.log").open("w") as output:
        result = process.run_supervised(
            "true",
            env=dict(os.environ),
            stdout=output,
            stderr=output,
            timeout=5,
            trial_dir=tmp_path,
            attempt_id="attempt",
        )
    assert result.cleanup_confirmed
    assert (tmp_path / PROCESS_IDENTITY_FILE).is_file()
    real_abort_launch = process._abort_launch

    def fail_identity_write(path, identity):
        assert path == tmp_path / PROCESS_IDENTITY_FILE
        assert not path.exists()
        raise OSError("injected worker identity write failure")

    def uncertain_abort(proc, pgid):
        # Reap the actual helper, then simulate an unconfirmed cleanup verdict.
        assert real_abort_launch(proc, pgid)
        return False

    monkeypatch.setattr(process, "_write_process_identity", fail_identity_write)
    monkeypatch.setattr(process, "_abort_launch", uncertain_abort)
    with pytest.raises(UnsafeProcessCleanupError, match="Recovery records remain"):
        poll_wandb_summary(
            base_url="https://example.test",
            entity="e",
            project="p",
            run_id="attempt",
            trial_dir=tmp_path,
            poll_seconds=0.01,
            timeout_seconds=5,
        )
    assert not (tmp_path / PROCESS_IDENTITY_FILE).exists()
    lifecycle = read_attempt_lifecycle(tmp_path, expected_attempt_id="attempt")
    assert lifecycle.state == "launching"
    assert lifecycle.cleanup_confirmed is None


@pytest.mark.parametrize("stage", ["constructor", "lookup"])
def test_wandb_poll_rejects_late_sdk_response(fake_wandb, monkeypatch, stage):
    clock = {"now": 0.0}
    lookups = []

    class Api:
        def __init__(self, **kwargs):
            if stage == "constructor":
                clock["now"] = 2.0

        def run(self, path):
            lookups.append(path)
            clock["now"] = 2.0
            return _FakeRun("finished", {"loss": 0.1})

    fake_wandb(api_class=Api)
    monkeypatch.setattr("phasesweep.evidence.wandb.time.monotonic", lambda: clock["now"])
    with pytest.raises(WandbPollTimeout):
        _poll_wandb_summary(
            base_url="https://example.test",
            entity="e",
            project="p",
            run_id="attempt",
            poll_seconds=0.01,
            timeout_seconds=1,
        )
    assert lookups == ([] if stage == "constructor" else ["e/p/attempt"])


def _assert_log_regex_provenance(
    tmp_path,
    text: str,
    cases: list[tuple[str, float, int, int]],
) -> None:
    """Run log-regex selection cases and assert their source provenance."""
    raw = text.encode("utf-8")
    (tmp_path / "stdout.log").write_bytes(raw)

    for select, expected_value, expected_line, expected_count in cases:
        cfg = LogRegexExtractor(
            type="log_regex",
            file="stdout.log",
            pattern=r"eval_loss=(?P<value>[0-9.eE+-]+)",
            select=select,
        )
        provenance: dict = {}
        value = run_extractor(make_trial_context(tmp_path), cfg, provenance=provenance)
        assert value == expected_value, select
        source = provenance["source"]
        assert source["sha256"] == hashlib.sha256(raw).hexdigest(), select
        assert source["size_bytes"] == len(raw), select
        assert source["matched_line"] == expected_line, select
        assert source["match_count"] == expected_count, select


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
    bad_paths = ["/tmp/result.json", "../result.json", "", ".", "result\0.json"]
    bad_keys = ["", ".x", "x."]

    for bad_path in bad_paths:
        with pytest.raises(ValidationError, match="trial-relative path required"):
            JsonExtractor(type="json", path=bad_path, key="x")

    for bad_key in bad_keys:
        with pytest.raises(ValidationError, match="JSON key"):
            JsonEqualsGate(type="json_equals", path="result.json", key=bad_key, value=1)

    with pytest.raises(ValidationError, match="valid only with source=directory"):
        ArtifactSizeGate(type="artifact_size", source="file", path=".", max_bytes=1)

    assert (
        ArtifactSizeGate(type="artifact_size", source="directory", path=".", min_bytes=0).path
        == "."
    )


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


def test_log_regex_reports_no_matches(tmp_path):
    (tmp_path / "stdout.log").write_text("nothing here\n", encoding="utf-8")
    cfg = LogRegexExtractor(type="log_regex", pattern=r"eval_loss=(?P<value>[0-9.]+)")
    with pytest.raises(ExtractorError, match="No matches"):
        run_extractor(make_trial_context(tmp_path), cfg)


def test_provenance_freezes_file_digest_and_extractor_identity(tmp_path):
    """The provenance sink freezes the exact evidence bytes and the extractor's
    config identity at extraction time (review v0.5.17 / finding F)."""
    raw = json.dumps({"eval": {"loss": 0.42}}).encode("utf-8")
    (tmp_path / "result.json").write_bytes(raw)
    cfg = JsonExtractor(type="json", path="result.json", key="eval.loss")

    provenance: dict = {}
    value = run_extractor(make_trial_context(tmp_path), cfg, provenance=provenance)

    assert value == pytest.approx(0.42)
    assert provenance["schema_version"] == 1
    assert provenance["extractor"]["kind"] == "json"
    assert provenance["extractor"]["config_sha256"] == extractor_config_fingerprint(cfg)
    assert provenance["source"] == {
        "kind": "file",
        "path": "result.json",
        "key": "eval.loss",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    assert provenance["recorded_at"]
    json.dumps(provenance)  # must survive the JSON study-attr round trip

    other = JsonExtractor(type="json", path="result.json", key="eval.acc")
    assert extractor_config_fingerprint(other) != extractor_config_fingerprint(cfg)


def test_provenance_absent_when_extraction_fails(tmp_path):
    cfg = JsonExtractor(type="json", path="missing.json", key="x")
    provenance: dict = {}
    with pytest.raises(ExtractorError):
        run_extractor(make_trial_context(tmp_path), cfg, provenance=provenance)
    assert "source" not in provenance


def test_provenance_envelope_freezes_validated_evaluation_metadata(tmp_path):
    payload = {
        "schema_version": 1,
        "status": "complete",
        "generation_id": "generation-test",
        "attempt_id": "attempt-test",
        "overrides_sha256": "a" * 64,
        "objective": {"name": "val_loss", "split": "validation", "value": 0.25},
        "evaluation": {
            "policy": "final_checkpoint",
            "checkpoint": "step_1000.pt",
            "step": 1000,
        },
    }
    raw = json.dumps(payload).encode("utf-8")
    (tmp_path / "result.json").write_bytes(raw)
    cfg = JsonEnvelopeExtractor(
        type="json_envelope",
        path="result.json",
        objective_name="val_loss",
        split="validation",
        policy="final_checkpoint",
    )

    provenance: dict = {}
    run_extractor(make_trial_context(tmp_path), cfg, provenance=provenance)

    source = provenance["source"]
    assert source["sha256"] == hashlib.sha256(raw).hexdigest()
    # The checkpoint and step are the envelope's own validated values even
    # when the config did not pin them.
    assert source["evaluation"] == {
        "objective_name": "val_loss",
        "split": "validation",
        "policy": "final_checkpoint",
        "checkpoint": "step_1000.pt",
        "step": 1000,
    }


def test_provenance_log_regex_records_selected_line_and_whole_file_digest(tmp_path):
    _assert_log_regex_provenance(
        tmp_path,
        "eval_loss=1.0\neval_loss=0.5\neval_loss=0.7\n",
        [
            ("min", 0.5, 2, 3),
            ("last", 0.7, 3, 3),
            # "first" stops matching at line 1 but must still digest the whole file.
            ("first", 1.0, 1, 1),
        ],
    )


def test_log_regex_splits_carriage_return_progress_lines(tmp_path):
    """Lone carriage returns are line boundaries, matching the historical
    text-mode universal-newline reader — tqdm-style progress logs separate
    updates with bare "\\r" (review v0.5.17 / finding F follow-up)."""
    _assert_log_regex_provenance(
        tmp_path,
        "eval_loss=2.5\reval_loss=1.9\reval_loss=2.1\nfinal eval_loss=2.2\r\n",
        [
            # 1.9 sits mid-"\r"-run: only reachable when "\r" splits lines.
            ("min", 1.9, 2, 4),
            ("max", 2.5, 1, 4),
            ("last", 2.2, 4, 4),
            ("first", 2.5, 1, 1),
        ],
    )


def test_provenance_wandb_freezes_summary_subset(fake_wandb, tmp_path):
    fake_wandb(lambda path: _FakeRun(state="finished", summary={"eval/loss": 0.123, "extra": 9}))
    cfg = WandbExtractor(
        type="wandb",
        entity="me",
        project="proj",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )

    provenance: dict = {}
    run_extractor(make_trial_context(tmp_path), cfg, provenance=provenance)

    source = provenance["source"]
    assert source["kind"] == "wandb"
    assert source["base_url"] == "https://api.wandb.ai"
    assert source["entity"] == "me"
    assert source["project"] == "proj"
    assert source["run_id"] == "attempt-test"
    assert source["run_state"] == "finished"
    # Only the subset that justified the metric is frozen.
    assert source["summary"] == {"eval/loss": 0.123}
    assert source["retrieved_at"]


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


def test_wandb_extractor_uses_explicit_normalized_endpoint(fake_wandb, tmp_path) -> None:
    constructed: list[tuple[dict[str, str] | None, int | None]] = []

    class Api:
        def __init__(self, overrides=None, timeout=None):
            constructed.append((overrides, timeout))

        def run(self, _path):
            return _FakeRun(state="finished", summary={"eval/loss": 0.123})

    fake_wandb(api_class=Api)
    cfg = WandbExtractor(
        type="wandb",
        base_url="https://wandb.example.test///",
        entity="me",
        project="proj",
        metric_key="eval/loss",
        timeout_seconds=1.0,
    )

    assert run_extractor(make_trial_context(tmp_path), cfg) == pytest.approx(0.123)
    assert cfg.base_url == "https://wandb.example.test"
    assert constructed == [({"base_url": "https://wandb.example.test"}, 1)]


def test_wandb_extractor_timeout(fake_wandb, tmp_path, monkeypatch: pytest.MonkeyPatch):
    def missing_run(_path: str) -> _FakeRun:
        raise LookupError("not found")

    fake_wandb(missing_run)
    clock = {"now": 0.0}
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
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )
    with pytest.raises(ExtractorError, match="not found or metric"):
        ctx = make_trial_context(tmp_path, experiment="exp", phase="ph", trial_id=7)
        run_extractor(ctx, cfg)


@pytest.mark.parametrize("model", [WandbExtractor, WandbSummaryRequiredGate])
def test_wandb_timeouts_require_whole_second_transport_budget(model):
    """Both wandb extractor models share one pydantic ``ge=1.0`` constraint on
    ``timeout_seconds``; any sub-1.0 value hits the same validation branch, so a
    single boundary-adjacent value (0.999) pins it without repeating instances.
    """
    common = {
        "type": "wandb" if model is WandbExtractor else "wandb_summary_required",
        "entity": "me",
        "project": "proj",
        "timeout_seconds": 0.999,
    }
    if model is WandbExtractor:
        common["metric_key"] = "eval/loss"
    else:
        common["keys"] = ["eval/loss"]
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        model(**common)


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


def test_wandb_extractor_rejects_unsuccessful_terminal_run(fake_wandb, tmp_path):
    """WandbExtractor rejects any terminal state via one
    ``run.state in {"crashed", "failed", "killed"}`` set-membership check; one
    representative value (``"failed"``) pins that single branch.
    """
    state = "failed"
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


def test_wandb_api_constructor_failure_is_typed_extractor_error(fake_wandb, tmp_path):
    """A credential/settings failure in ``Api(...)`` must not escape the error model.

    Construction used to happen outside the polling ``try`` block, so an
    exception there surfaced as an arbitrary exception instead of a typed
    extractor failure (review v0.5.17 / finding D).
    """

    class Api:
        def __init__(self, overrides=None, timeout=None):
            raise RuntimeError("credential loader exploded during Api construction")

    fake_wandb(api_class=Api)

    cfg = WandbExtractor(
        type="wandb",
        entity="team",
        project="proj",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=60,
    )

    with pytest.raises(ExtractorError, match="W&B client setup failed"):
        run_extractor(make_trial_context(tmp_path), cfg)


def test_wandb_transient_api_construction_failure_mid_poll_is_retried(
    fake_wandb, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """A mid-poll ``Api(...)`` failure is transient, not a setup error.

    Construction happens every iteration (budget-aware request timeouts) and
    performs a network round-trip, so once one client has been built a later
    constructor failure must be retried like any other poll error instead of
    aborting the wait with most of the budget unspent.
    """
    constructions = {"count": 0}

    class Api:
        def __init__(self, overrides=None, timeout=None):
            constructions["count"] += 1
            if constructions["count"] == 2:
                raise ConnectionError("transient network blip during Api construction")

        def run(self, path):
            if constructions["count"] < 3:
                return _FakeRun(state="running", summary={})
            return _FakeRun(state="finished", summary={"eval/loss": 0.123})

    fake_wandb(api_class=Api)

    clock = {"now": 0.0}
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
        poll_seconds=0.01,
        timeout_seconds=60,
    )

    assert run_extractor(make_trial_context(tmp_path), cfg) == pytest.approx(0.123)
    assert constructions["count"] == 3


@pytest.mark.parametrize("model", [WandbExtractor, WandbSummaryRequiredGate])
@pytest.mark.parametrize("field", ["poll_seconds", "timeout_seconds"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_wandb_time_budgets_must_be_finite(model, field, value):
    """Non-finite poll/timeout budgets must fail validation, not first use.

    ``ge=1.0``/``gt=0.0`` alone admit ``.inf``, which previously survived
    ``phasesweep validate`` and then crashed the first poll inside the setup
    error boundary (``ceil(inf)``) with a misleading credentials diagnosis.
    """
    common = {
        "type": "wandb" if model is WandbExtractor else "wandb_summary_required",
        "entity": "me",
        "project": "proj",
        field: value,
    }
    if model is WandbExtractor:
        common["metric_key"] = "eval/loss"
    else:
        common["keys"] = ["eval/loss"]
    with pytest.raises(ValidationError, match="finite"):
        model(**common)


def test_wandb_extractor_poll_budget_is_capped_by_phase_deadline(fake_wandb, tmp_path):
    """A 60s W&B poll budget must shrink to the remaining phase/run budget.

    Without the cap, evidence extraction could extend a 1-second phase budget
    by a full extractor timeout (review v0.5.17 / blocker 8).
    """
    import time as _time

    fake_wandb(lambda _path: (_ for _ in ()).throw(ConnectionError("unreachable")))
    cfg = WandbExtractor(
        type="wandb",
        entity="team",
        project="proj",
        metric_key="eval/loss",
        poll_seconds=0.05,
        timeout_seconds=60,
    )

    started = _time.monotonic()
    with pytest.raises(DeadlineExceededError):
        run_extractor(
            make_trial_context(tmp_path),
            cfg,
            deadline=_time.monotonic() + 1.0,
        )
    elapsed = _time.monotonic() - started

    # Enormous two-sided margin: capped polling ends in ~1s; an uncapped
    # budget would need the full 60s.
    assert elapsed < 20.0


def test_wandb_extractor_rejects_expired_deadline_before_polling(fake_wandb, tmp_path):
    """An exhausted phase budget reports the deadline, not a zero-second W&B miss."""
    timeouts = fake_wandb(lambda _path: _FakeRun("finished", {"eval/loss": 0.1}))
    cfg = WandbExtractor(
        type="wandb",
        entity="team",
        project="proj",
        metric_key="eval/loss",
        timeout_seconds=60,
    )

    with pytest.raises(DeadlineExceededError, match="wallclock deadline exceeded"):
        run_extractor(make_trial_context(tmp_path), cfg, deadline=0.0)

    assert timeouts == []


def test_gate_results_mark_only_deadline_caused_failures(fake_wandb, tmp_path):
    """Gate failures carry a causal deadline marker, not an elapsed-clock guess."""
    fake_wandb(lambda _path: (_ for _ in ()).throw(ConnectionError("unreachable")))
    ctx = make_trial_context(tmp_path)
    gate = WandbSummaryRequiredGate(
        type="wandb_summary_required",
        entity="team",
        project="proj",
        keys=["eval/loss"],
        poll_seconds=0.01,
        timeout_seconds=60,
    )

    capped = evaluate_gates(ctx, [gate], deadline=time.monotonic() + 0.05)[0]
    expired = evaluate_gates(
        ctx,
        [JsonEqualsGate(type="json_equals", path="result.json", key="ok", value=True)],
        deadline=0.0,
    )[0]
    ordinary = evaluate_gates(
        ctx,
        [JsonEqualsGate(type="json_equals", path="result.json", key="ok", value=True)],
    )[0]

    assert capped.deadline_exhausted is True
    assert expired.deadline_exhausted is True
    assert ordinary.deadline_exhausted is False


def test_deadline_capped_extractor_error_keeps_underlying_diagnostic(fake_wandb, tmp_path):
    """A deadline-attributed W&B failure must still name the real error.

    The rewrap that blames an exhausted budget used to discard the extractor's
    own message, so an expired API key or a wrong entity/project surfaced to the
    operator as a pure timeout: they would raise ``timeout_seconds`` and rerun
    into the identical failure forever.
    """
    fake_wandb(lambda _path: (_ for _ in ()).throw(ConnectionError("401 unauthorized")))
    cfg = WandbExtractor(
        type="wandb",
        entity="team",
        project="proj",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=60,
    )

    with pytest.raises(DeadlineExceededError) as excinfo:
        run_extractor(
            make_trial_context(tmp_path),
            cfg,
            deadline=time.monotonic() + 0.05,
        )

    message = str(excinfo.value)
    # Both halves must survive: the causal attribution the engine keys off, and
    # the diagnostic that says more budget would not have helped.
    assert "deadline exhausted" in message
    assert "401 unauthorized" in message
    assert "eval/loss" in message
    # trial.py records ``f"metric extractor: {exc}"`` verbatim as failure_reason,
    # so the message above is exactly what lands in the Optuna user attrs.
    cause = excinfo.value.__cause__
    assert isinstance(cause, ExtractorError)
    assert "401 unauthorized" in str(cause)


class _UnregisteredGate:
    """Stand-in for a ``Gate`` union member with no ``_GATE_DISPATCH`` entry."""

    type = "unregistered"

    def __repr__(self) -> str:
        return "_UnregisteredGate()"


@pytest.mark.parametrize("deadline", [None, 0.0, "future"])
def test_unregistered_gate_type_fails_instead_of_raising(tmp_path, deadline):
    """An unhandled gate type degrades to one failing gate, never a KeyError.

    A bare ``_GATE_DISPATCH[type(gate)]`` lookup raises ``KeyError``, which is
    not an ``ExtractorError`` and is not in Optuna's ``catch=`` tuple, so it
    would escape the objective and abort the entire phase and run mid-sweep.
    """
    ctx = make_trial_context(tmp_path)
    resolved = time.monotonic() + 5.0 if deadline == "future" else deadline
    known = JsonEqualsGate(type="json_equals", path="result.json", key="ok", value=True)

    results = evaluate_gates(ctx, [_UnregisteredGate(), known], deadline=resolved)  # type: ignore[list-item]

    assert len(results) == 2
    assert results[0].passed is False
    assert results[0].gate_type == "_UnregisteredGate"
    assert "unknown gate" in results[0].detail
    # Evaluation continues past the unknown gate rather than aborting the trial.
    assert results[1].gate_type == "json_equals"
    assert results[1].passed is False
