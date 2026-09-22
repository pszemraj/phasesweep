from __future__ import annotations

import hashlib
import json
import os
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
from phasesweep.evidence import ExtractorError, evaluate_gates, run_extractor
from phasesweep.evidence.evaluation import extractor_config_fingerprint
from tests.conftest import make_trial_context


def _wandb_poll(**kwargs):
    from phasesweep.evidence.wandb import _poll_wandb_summary

    return _poll_wandb_summary(
        base_url="https://example.test",
        entity="entity",
        project="project",
        run_id="attempt",
        poll_seconds=0.001,
        timeout_seconds=1,
        **kwargs,
    )


def test_json_float_overflow_is_invalid_evidence(tmp_path):
    from phasesweep.evidence.evaluation import json_float

    with pytest.raises(ValueError, match="finite scalar range"):
        json_float(10**400, label="objective")
    (tmp_path / "result.json").write_text(json.dumps({"x": 10**400}))
    with pytest.raises(ExtractorError, match="finite scalar range"):
        run_extractor(
            make_trial_context(tmp_path), JsonExtractor(type="json", path="result.json", key="x")
        )


def test_wandb_real_sdk_decodes_refreshed_finished_summary(tmp_path, monkeypatch):
    public = pytest.importorskip("wandb.apis.public")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    calls = []
    responses = iter(
        [
            {"state": "running", "summaryMetrics": '{"eval/loss": 9}'},
            {
                "state": "finished",
                "summaryMetrics": '{"eval/loss": 0.25, "large": {"unrelated": true}}',
            },
        ]
    )

    class Transport:
        def execute_graphql(self, query, variables, **kwargs):
            calls.append(variables)
            return {"project": {"run": next(responses)}}

    # Keep actual Api.run, Run construction, and summary decoding. Replace only
    # authenticated transport setup; this test never contacts the service.
    def initialize(self, overrides=None, timeout=None):
        assert overrides == {"base_url": "https://example.test"}
        self.settings = {"entity": "entity", "project": "project"}
        self._runs = {}
        self._service_api = Transport()
        self.api_key = None

    monkeypatch.setattr(public.Api, "__init__", initialize)
    capture = _wandb_poll(required_keys=["eval/loss"], presence_keys=["large", "absent"])
    assert capture["values"] == {"eval/loss": 0.25}
    assert capture["present_keys"] == ["large"]
    assert len(calls) == 2
    assert all(
        call == {"entity": "entity", "project": "project", "name": "attempt"} for call in calls
    )


@pytest.mark.parametrize("status", [401, 403, 429, 503, "connection", "timeout"])
def test_wandb_real_sdk_error_causes_are_preserved(monkeypatch, status):
    public = pytest.importorskip("wandb.apis.public")
    from requests import HTTPError, Response
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import Timeout as RequestsTimeout
    from wandb.errors import CommError

    from phasesweep.evidence.wandb import WandbSetupError

    if isinstance(status, int):
        response = Response()
        response.status_code = status
        cause = HTTPError("secret-token", response=response)
    else:
        cause = (RequestsConnectionError if status == "connection" else RequestsTimeout)(
            "secret-token"
        )
    error = CommError("secret-token", exc=cause)
    attempts = []

    def initialize(self, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise error

    monkeypatch.setattr(public.Api, "__init__", initialize)
    monkeypatch.setattr(
        public.Api,
        "run",
        lambda self, path: SimpleNamespace(state="finished", summary_metrics={"eval/loss": 0.25}),
    )
    if status in {401, 403}:
        with pytest.raises(WandbSetupError) as caught:
            _wandb_poll(required_keys=["eval/loss"])
        assert str(status) in caught.value.cause
        assert "secret-token" not in caught.value.cause
        assert len(attempts) == 1
    else:
        assert _wandb_poll(required_keys=["eval/loss"])["values"] == {"eval/loss": 0.25}
        assert len(attempts) == 2


@pytest.mark.parametrize("state", ["failed", "crashed", "killed", "preempted"])
def test_wandb_terminal_runs_cannot_win(monkeypatch, state):
    public = pytest.importorskip("wandb.apis.public")
    from phasesweep.evidence.wandb import WandbRunTerminalError

    monkeypatch.setattr(
        public,
        "Api",
        lambda **kwargs: SimpleNamespace(
            run=lambda path: SimpleNamespace(state=state, summary_metrics={"loss": 0.2})
        ),
    )
    with pytest.raises(WandbRunTerminalError) as caught:
        _wandb_poll(required_keys=["loss"])
    assert caught.value.state == state


@pytest.mark.parametrize("value", [True, "0.2", None, [], {}, float("inf"), 10**400])
def test_wandb_invalid_scalar_is_not_retryable(monkeypatch, value):
    public = pytest.importorskip("wandb.apis.public")
    monkeypatch.setattr(
        public,
        "Api",
        lambda **kwargs: SimpleNamespace(
            run=lambda path: SimpleNamespace(state="finished", summary_metrics={"loss": value})
        ),
    )
    with pytest.raises(ValueError):
        _wandb_poll(required_keys=["loss"])


def test_wandb_consumers_share_one_capture(tmp_path, monkeypatch):
    from phasesweep.evidence.models import WandbQuery

    objective = WandbExtractor(
        type="wandb", entity="entity", project="project", metric_key="eval/loss"
    )
    constraint = objective.model_copy(update={"metric_key": "memory"})
    gate = WandbSummaryRequiredGate(
        type="wandb_summary_required", entity="entity", project="project", keys=["done"]
    )
    query = WandbQuery(objective, ("eval/loss", "memory"), ("done",))
    ctx = replace(make_trial_context(tmp_path), wandb_query=query)
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return {
            "values": {"eval/loss": 0.2, "memory": 4.0},
            "present_keys": ["done"],
            "retrieved_at": "2026-09-20T00:00:00Z",
        }

    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", capture)
    provenance = {}
    assert run_extractor(ctx, objective, provenance=provenance) == 0.2
    assert run_extractor(ctx, constraint) == 4
    assert all(result.passed for result in evaluate_gates(ctx, [gate]))
    assert len(calls) == 1
    assert calls[0]["required_keys"] == ("eval/loss", "memory")
    assert provenance["remote_capture"]["run_id"] == ctx.attempt_id


def test_wandb_gate_only_missing_key_fails_first_finished_capture(tmp_path, monkeypatch):
    from phasesweep.evidence.wandb import _poll_wandb_summary

    public = pytest.importorskip("wandb.apis.public")
    calls = []

    def lookup(path):
        calls.append(path)
        return SimpleNamespace(state="finished", summary_metrics={})

    monkeypatch.setattr(public, "Api", lambda **kwargs: SimpleNamespace(run=lookup))
    monkeypatch.setattr(
        "phasesweep.evidence.evaluation.poll_wandb_summary",
        lambda trial_dir, environment, **kwargs: _poll_wandb_summary(**kwargs),
    )
    result = evaluate_gates(
        make_trial_context(tmp_path),
        [
            WandbSummaryRequiredGate(
                type="wandb_summary_required", entity="e", project="p", keys=["complete"]
            )
        ],
    )[0]
    assert not result.passed
    assert not result.deadline_exhausted
    assert len(calls) == 1


@pytest.mark.parametrize("stage", ["constructor", "lookup"])
def test_wandb_late_sdk_success_is_rejected(monkeypatch, stage):
    public = pytest.importorskip("wandb.apis.public")
    from phasesweep.evidence.wandb import WandbPollTimeout

    clock = [0.0]
    monkeypatch.setattr("phasesweep.evidence.wandb.time.monotonic", lambda: clock[0])

    class Api:
        def __init__(self, **kwargs):
            if stage == "constructor":
                clock[0] = 2.0

        def run(self, path):
            clock[0] = 2.0
            return SimpleNamespace(state="finished", summary_metrics={"loss": 0.2})

    monkeypatch.setattr(public, "Api", Api)
    with pytest.raises(WandbPollTimeout):
        _wandb_poll(required_keys=["loss"])


def test_wandb_worker_transfers_only_requested_evidence(wandb_worker_sdk, tmp_path, monkeypatch):
    from phasesweep.evidence.wandb import poll_wandb_summary
    from phasesweep.runtime.process import read_attempt_lifecycle

    monkeypatch.setenv("WANDB_API_KEY", "excluded-parent-secret")
    wandb_worker_sdk("""
        import os
        class Api:
            def __init__(self, **kwargs):
                assert "WANDB_API_KEY" not in os.environ
                assert os.environ["HTTPS_PROXY"] == "https://transport.test"
                assert "PYTHONHOME" not in os.environ
            def run(self, path):
                assert path == "entity/project/attempt"
                return type("Run", (), {"state": "finished", "summary_metrics":
                    {"loss": 0.25, "done": object(), "unrelated": object()}})()
    """)
    capture = poll_wandb_summary(
        base_url="https://example.test",
        entity="entity",
        project="project",
        run_id="attempt",
        trial_dir=tmp_path,
        poll_seconds=0.01,
        timeout_seconds=5,
        required_keys=["loss"],
        presence_keys=["done", "missing"],
        environment={"HTTPS_PROXY": "https://transport.test", "PYTHONHOME": "/trainer/python"},
    )
    assert capture["values"] == {"loss": 0.25}
    assert capture["present_keys"] == ["done"]
    assert read_attempt_lifecycle(tmp_path, expected_attempt_id="attempt").cleanup_confirmed


def test_wandb_worker_crash_preserves_private_stderr(wandb_worker_sdk, tmp_path):
    from phasesweep.evidence.wandb import poll_wandb_summary

    wandb_worker_sdk("raise RuntimeError('worker-diagnostic-marker')")

    with pytest.raises(RuntimeError, match="Diagnostic preserved") as exc_info:
        poll_wandb_summary(
            base_url="https://example.test",
            entity="e",
            project="p",
            run_id="attempt",
            trial_dir=tmp_path,
            poll_seconds=0.01,
            timeout_seconds=5,
        )

    diagnostic = tmp_path / "wandb-worker.stderr.log"
    assert "worker-diagnostic-marker" in diagnostic.read_text()
    assert diagnostic.stat().st_mode & 0o777 == 0o600
    assert "worker-diagnostic-marker" not in str(exc_info.value)


def test_wandb_launch_failure_cannot_use_trainer_identity(tmp_path, monkeypatch):
    from phasesweep.errors import UnsafeProcessCleanupError
    from phasesweep.evidence.wandb import poll_wandb_summary
    from phasesweep.runtime import process

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
    original_abort = process._abort_launch

    def fail_identity(path, identity):
        assert not path.exists()
        raise OSError("injected identity failure")

    def uncertain_abort(proc, pgid):
        assert original_abort(proc, pgid)
        return False

    monkeypatch.setattr(process, "_write_process_identity", fail_identity)
    monkeypatch.setattr(process, "_abort_launch", uncertain_abort)
    with pytest.raises(UnsafeProcessCleanupError, match="cleanup is uncertain"):
        poll_wandb_summary(
            base_url="https://example.test",
            entity="e",
            project="p",
            run_id="attempt",
            trial_dir=tmp_path,
            poll_seconds=0.01,
            timeout_seconds=5,
        )
    lifecycle = process.read_attempt_lifecycle(tmp_path, expected_attempt_id="attempt")
    assert lifecycle.state == "launching"
    assert lifecycle.cleanup_confirmed is None


@pytest.mark.parametrize("stage", ["constructor", "lookup"])
@pytest.mark.integration
def test_wandb_supervision_bounds_blocked_sdk_and_descendants(wandb_worker_sdk, tmp_path, stage):
    from phasesweep.evidence.wandb import WandbPollTimeout, poll_wandb_summary
    from phasesweep.runtime.process import is_pid_alive
    from tests.conftest import is_pid_zombie

    pid_file = tmp_path / "descendant"
    wandb_worker_sdk(f"""
        import os, subprocess, sys, time
        from pathlib import Path
        def block():
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            Path({str(pid_file)!r}).write_text(str(child.pid))
            time.sleep(60)
        class Api:
            def __init__(self, **kwargs):
                if {stage!r} == "constructor": block()
            def run(self, path):
                block()
    """)
    start = time.monotonic()
    with pytest.raises(WandbPollTimeout):
        poll_wandb_summary(
            base_url="https://example.test",
            entity="e",
            project="p",
            run_id="attempt",
            trial_dir=tmp_path,
            poll_seconds=0.01,
            timeout_seconds=2,
        )
    assert time.monotonic() - start < 12
    pid = int(pid_file.read_text())
    assert not is_pid_alive(pid) or is_pid_zombie(pid)


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


@pytest.mark.parametrize(
    ("select", "line", "expected", "match_count"),
    [
        ("last", "loss=9 loss=1\n", 1.0, 2),
        ("min", "loss=9 loss=1\n", 1.0, 2),
        ("max", "loss=1 loss=9\n", 9.0, 2),
        ("first", "loss=bad loss=1\n", 1.0, 1),
    ],
)
def test_log_regex_considers_every_numeric_match_per_line(
    tmp_path: Path, select: str, line: str, expected: float, match_count: int
) -> None:
    """One log line can contribute multiple numeric candidates."""
    raw = line.encode("utf-8")
    (tmp_path / "stdout.log").write_bytes(raw)
    cfg = LogRegexExtractor(
        type="log_regex",
        pattern=r"loss=(?P<value>[^\s]+)",
        select=select,
    )

    provenance: dict = {}
    assert run_extractor(make_trial_context(tmp_path), cfg, provenance=provenance) == expected
    assert provenance["source"] == {
        "kind": "file",
        "path": "stdout.log",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "matched_line": 1,
        "match_count": match_count,
    }


def test_log_regex_reports_no_matches(tmp_path):
    (tmp_path / "stdout.log").write_text("nothing here\n", encoding="utf-8")
    cfg = LogRegexExtractor(type="log_regex", pattern=r"eval_loss=(?P<value>[0-9.]+)")
    with pytest.raises(ExtractorError, match="No matches"):
        run_extractor(make_trial_context(tmp_path), cfg)


def test_artifact_size_directory_gate_rejects_unreadable_subtree(tmp_path: Path) -> None:
    """An incomplete traversal cannot establish a checkpoint-size bound."""
    if os.geteuid() == 0:
        pytest.skip("Permission-denial reproduction requires an unprivileged user")
    private = tmp_path / "checkpoint" / "weights"
    private.mkdir(parents=True)
    (private / "model.bin").write_bytes(b"x" * 16_384)
    private.chmod(0)
    try:
        result = evaluate_gates(
            make_trial_context(tmp_path),
            [
                ArtifactSizeGate(
                    type="artifact_size",
                    path="checkpoint",
                    source="directory",
                    max_bytes=1024,
                )
            ],
        )[0]
    finally:
        private.chmod(0o700)

    assert result.passed is False
    assert "could not inspect checkpoint" in result.detail


def test_artifact_size_directory_gate_counts_file_symlinks_not_directory_symlinks(
    tmp_path: Path,
) -> None:
    """File-link targets count, while linked directories are not traversed."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.bin").write_bytes(b"meta")
    external = tmp_path / "external"
    external.mkdir()
    (external / "weights.bin").write_bytes(b"x" * 16_384)
    (external / "metadata-link.bin").write_bytes(b"linked!")
    (checkpoint / "linked-weights").symlink_to(external, target_is_directory=True)
    (checkpoint / "linked-metadata.bin").symlink_to(external / "metadata-link.bin")

    result = evaluate_gates(
        make_trial_context(tmp_path),
        [
            ArtifactSizeGate(
                type="artifact_size",
                path="checkpoint",
                source="directory",
                max_bytes=11,
            )
        ],
    )[0]

    assert result.passed is True
    assert "directory size 11 within bounds" in result.detail


def test_artifact_size_directory_gate_reports_root_inspection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root metadata error is failed evidence with its original path detail."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    real_stat = Path.stat

    def fail_root_stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if self == checkpoint and not follow_symlinks:
            raise OSError("root metadata unavailable")
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fail_root_stat)

    result = evaluate_gates(
        make_trial_context(tmp_path),
        [
            ArtifactSizeGate(
                type="artifact_size",
                path="checkpoint",
                source="directory",
                max_bytes=1024,
            )
        ],
    )[0]

    assert result.passed is False
    assert str(checkpoint) in result.detail
    assert "root metadata unavailable" in result.detail


def test_artifact_size_directory_gate_reports_descendant_stat_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child metadata error cannot be silently excluded from a directory size."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    child = checkpoint / "model.bin"
    child.write_bytes(b"payload")
    real_stat = Path.stat

    def fail_child_stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if self == child and follow_symlinks:
            raise OSError("descendant metadata unavailable")
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fail_child_stat)

    result = evaluate_gates(
        make_trial_context(tmp_path),
        [
            ArtifactSizeGate(
                type="artifact_size",
                path="checkpoint",
                source="directory",
                max_bytes=1024,
            )
        ],
    )[0]

    assert result.passed is False
    assert str(child) in result.detail
    assert "descendant metadata unavailable" in result.detail


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
    # Unchanged JSON provenance keeps its old identity; changed log-regex
    # interpretation must be distinguishable even under identical YAML.
    legacy_json = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert extractor_config_fingerprint(cfg) == hashlib.sha256(legacy_json.encode()).hexdigest()
    log = LogRegexExtractor(type="log_regex", pattern=r"loss=(?P<value>[0-9.]+)")
    legacy_log = json.dumps(log.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert extractor_config_fingerprint(log) != hashlib.sha256(legacy_log.encode()).hexdigest()


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


def test_log_regex_streams_carriage_return_progress_lines(tmp_path):
    """CR-only progress output must not be accumulated as one binary line."""
    raw = b"eval_loss=1.0\r" * 10_000
    (tmp_path / "stdout.log").write_bytes(raw)
    cfg = LogRegexExtractor(
        type="log_regex",
        file="stdout.log",
        pattern=r"eval_loss=(?P<value>[0-9.eE+-]+)",
        select="last",
    )

    provenance: dict = {}
    tracemalloc.start()
    try:
        value = run_extractor(make_trial_context(tmp_path), cfg, provenance=provenance)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert value == 1.0
    assert provenance["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert provenance["source"]["size_bytes"] == len(raw)
    assert provenance["source"]["matched_line"] == 10_000
    assert provenance["source"]["match_count"] == 10_000
    assert peak_bytes < len(raw) * 8


def test_log_regex_first_hashes_invalid_utf8_suffix_without_decoding_it(tmp_path):
    """First-match selection still hashes later bytes without validating them."""
    raw = b"eval_loss=1.0\n\xff"
    (tmp_path / "stdout.log").write_bytes(raw)
    cfg = LogRegexExtractor(
        type="log_regex",
        file="stdout.log",
        pattern=r"eval_loss=(?P<value>[0-9.eE+-]+)",
        select="first",
    )

    provenance: dict = {}
    assert run_extractor(make_trial_context(tmp_path), cfg, provenance=provenance) == 1.0
    assert provenance["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert provenance["source"]["size_bytes"] == len(raw)


def test_gate_results_mark_expired_local_gate(tmp_path):
    """An expired deadline marks an unevaluated local gate as exhausted."""
    gate = JsonEqualsGate(type="json_equals", path="result.json", key="ok", value=True)

    expired = evaluate_gates(make_trial_context(tmp_path), [gate], deadline=0.0)[0]
    ordinary = evaluate_gates(make_trial_context(tmp_path), [gate])[0]

    assert expired.deadline_exhausted is True
    assert ordinary.deadline_exhausted is False


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
