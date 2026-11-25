from __future__ import annotations

import hashlib
import json
import math
import os
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from phasesweep.evidence.evaluation import (
    EVIDENCE_PROVENANCE_SCHEMA_VERSION,
    DeadlineExceededError,
    EvidenceUnreadableError,
    ExtractorError,
    evaluate_gates,
    extractor_config_fingerprint,
    run_extractor,
)
from phasesweep.evidence.models import (
    ArtifactSizeGate,
    JsonEnvelopeExtractor,
    JsonEqualsGate,
    JsonExtractor,
    LogRegexExtractor,
    MlflowExtractor,
    MlflowRequiredGate,
    RequiredFileGate,
    Sha256Gate,
    WandbExtractor,
    WandbSummaryRequiredGate,
)
from tests.conftest import make_trial_context
from tests.tracking_helpers import ManualClock, attach_wandb_transport


def _wandb_poll(**kwargs):
    from phasesweep.evidence.wandb import _poll_wandb_summary

    return _poll_wandb_summary(
        **{
            "base_url": "https://example.test",
            "entity": "entity",
            "project": "project",
            "run_id": "attempt",
            "poll_seconds": 0.001,
            "timeout_seconds": 1,
            **kwargs,
        }
    )


def test_reread_artifact_gates_require_state_from_before_the_next_launch(tmp_path):
    """A shared file, directory or digest from a later retry cannot pass an earlier gate."""
    ctx = make_trial_context(tmp_path)
    earlier = ctx.trial_dir / "earlier.txt"
    earlier.write_bytes(b"checkpoint")
    time.sleep(0.01)
    with_later_contents = ctx.trial_dir / "checkpoints"
    with_later_contents.mkdir()
    file = with_later_contents / "new.pt"
    file.write_bytes(b"checkpoint")
    cutoff = (earlier.stat().st_ctime_ns + file.stat().st_ctime_ns) / 2e9
    gates = [
        RequiredFileGate(type="required_file", path="earlier.txt"),
        RequiredFileGate(type="required_file", path="checkpoints/new.pt"),
        Sha256Gate(
            type="sha256",
            path="checkpoints/new.pt",
            sha256=hashlib.sha256(b"checkpoint").hexdigest(),
        ),
        ArtifactSizeGate(
            type="artifact_size", source="file", path="checkpoints/new.pt", min_bytes=1
        ),
        ArtifactSizeGate(type="artifact_size", source="directory", path="checkpoints", min_bytes=1),
    ]
    assert all(gate.passed for gate in evaluate_gates(ctx, gates))
    old, *new = evaluate_gates(replace(ctx, evidence_before=cutoff), gates)
    assert old.passed
    assert all(not gate.passed and "another attempt reused" in gate.detail for gate in new), new


def test_json_float_overflow_is_invalid_evidence(tmp_path):
    from phasesweep.evidence.evaluation import json_float

    with pytest.raises(ValueError, match="finite scalar range"):
        json_float(10**400, label="objective")
    (tmp_path / "result.json").write_text(json.dumps({"x": 10**400}))
    with pytest.raises(ExtractorError, match="finite scalar range"):
        run_extractor(
            make_trial_context(tmp_path), JsonExtractor(type="json", path="result.json", key="x")
        )


@pytest.mark.parametrize("deadline", [None, 0.0])
def test_unregistered_extractor_fails_before_deadline_or_provenance_changes(tmp_path, deadline):
    class UnregisteredExtractor(JsonExtractor):
        pass

    provenance = {"retained": True}
    with pytest.raises(ExtractorError, match="No extractor registered for UnregisteredExtractor"):
        run_extractor(
            make_trial_context(tmp_path),
            UnregisteredExtractor(type="json", path="result.json", key="x"),
            deadline=deadline,
            provenance=provenance,
        )
    assert provenance == {"retained": True}


@pytest.mark.parametrize("backend", ["wandb", "mlflow"])
@pytest.mark.parametrize("consumer", ["extractor", "gate"])
@pytest.mark.parametrize("failure", [ExtractorError, DeadlineExceededError])
def test_remote_consumers_forward_deadline_and_preserve_failure_kind(
    tmp_path, monkeypatch, backend, consumer, failure
):
    expected_deadline = time.monotonic() + 60.0

    def capture(*args, deadline):
        assert deadline == expected_deadline
        raise failure("capture stopped")

    monkeypatch.setattr(f"phasesweep.evidence.evaluation._capture_{backend}", capture)
    if backend == "wandb":
        extractor = WandbExtractor(type="wandb", entity="e", project="p", metric_key="loss")
        gate = WandbSummaryRequiredGate(
            type="wandb_summary_required", entity="e", project="p", keys=["done"]
        )
    else:
        extractor = MlflowExtractor(type="mlflow", metric_key="loss", select="last")
        gate = MlflowRequiredGate(type="mlflow_required", metrics=["loss"])
    ctx = make_trial_context(tmp_path)
    if consumer == "extractor":
        with pytest.raises(failure, match="capture stopped") as stopped:
            run_extractor(ctx, extractor, deadline=expected_deadline)
        assert stopped.type is failure
    else:
        result = evaluate_gates(ctx, [gate], deadline=expected_deadline)[0]
        assert result.passed is False
        assert result.detail == "capture stopped"
        assert result.deadline_exhausted is (failure is DeadlineExceededError)


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
        attach_wandb_transport(self, Transport())

    monkeypatch.setattr(public.Api, "__init__", initialize)
    capture = _wandb_poll(required_keys=["eval/loss"], presence_keys=["large", "absent"])
    assert capture["values"] == {"eval/loss": 0.25}
    assert capture["present_keys"] == ["large"]
    assert len(calls) == 2
    assert all(
        call == {"entity": "entity", "project": "project", "name": "attempt"} for call in calls
    )


@pytest.mark.parametrize("status", [401, 403, 429, 503, "connection", "timeout"])
def test_wandb_real_sdk_error_causes_are_preserved(tmp_path, monkeypatch, status):
    """A denial ends polling as unreadable evidence; a transient fault is retried."""
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
        # Rejected credentials say nothing about the trial, so its evidence is unreadable.
        attempts.clear()
        monkeypatch.setattr(
            "phasesweep.evidence.evaluation.poll_wandb_summary",
            lambda attempt_dir, attempt_id, environment, **kwargs: _wandb_poll(**kwargs),
        )
        extractor = WandbExtractor(type="wandb", entity="e", project="p", metric_key="eval/loss")
        with pytest.raises(EvidenceUnreadableError, match=str(status)):
            run_extractor(make_trial_context(tmp_path), extractor)
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


@pytest.mark.parametrize("value", [True, "0.2", None, float("inf"), 10**400])
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


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("WandbPollTimeout", ("attempt", 1.0, False)),
        ("WandbRunTerminalError", ("attempt", "crashed")),
        ("WandbSetupError", ("attempt", "denied")),
        ("WandbMissingKeysError", ("attempt", ["loss"], ["eval/loss"], 1.0)),
        ("WandbRunSupersededError", ("attempt",)),
    ],
)
def test_wandb_errors_unwind_through_context_managers(name, arguments):
    import contextlib

    from phasesweep.evidence import wandb as wandb_evidence

    @contextlib.contextmanager
    def span():
        yield

    error = getattr(wandb_evidence, name)(*arguments)
    with pytest.raises(type(error)) as caught, span():
        raise error
    assert caught.value is error


class _HistoryRun:
    """A finished run whose history, and the index describing it, a test controls.

    The index looks like the live service's: it lists every logged key and every
    summary key, aggregates such as ``eval/loss.min`` included although no row carries
    them, each with a count of 2 whatever the row count. ``indexed_steps`` are the
    index's last steps on successive loads before it reaches the last row, ``None`` for
    a null index and -1 for one without keys, as W&B serves before indexing; ``visible``
    slices the rows successive sampled reads return, as W&B has served scanned rows
    before sampled ones.
    """

    def __init__(self, summary, rows, indexed_steps=(), visible=()):
        self.state = "finished"
        self.summary_metrics = summary
        self.rows = rows
        self.indexed_steps = iter(indexed_steps)
        self.visible = iter(visible)
        self.history_keys = None
        self.reads = []

    def load(self, force=False):
        assert force is True
        listed = {key for row in self.rows for key in row} | set(self.summary_metrics)
        indexed = next(self.indexed_steps, max((row["_step"] for row in self.rows), default=-1))
        self.history_keys = (
            None
            if indexed is None
            else {
                "lastStep": indexed,
                "keys": {
                    key: {"typeCounts": [{"type": "number", "count": 2}]}
                    for key in listed | {"_runtime", "_timestamp"}
                    if indexed >= 0
                },
            }
        )

    def _logged(self, keys):
        # Like the server, only rows that logged every requested key.
        return [row for row in self.rows if all(key in row for key in keys)]

    def history(self, keys, samples=500, pandas=True):
        assert pandas is False
        self.reads.append(("history", keys))
        rows = self._logged(keys)[next(self.visible, slice(None))]
        # The server returns a series of at least ``samples`` rows as a sample.
        return rows if len(rows) < samples else rows[::2][:samples]

    def scan_history(self, keys, page_size=1000, use_cache=True):
        assert use_cache is False
        self.reads.append(("scan", keys))
        self.page_size = page_size
        return self._logged(keys)


def _history_position(
    monkeypatch, run, metric_key="eval/loss", axis="_step", poll_seconds=0.001, timeout_seconds=1
):
    from phasesweep.evidence.wandb import _locate_wandb_position

    public = pytest.importorskip("wandb.apis.public")
    monkeypatch.setattr(public, "Api", lambda **_: SimpleNamespace(run=lambda path: run))
    return _locate_wandb_position(
        base_url="https://example.test",
        entity="entity",
        project="project",
        run_id="attempt",
        poll_seconds=poll_seconds,
        metric_key=metric_key,
        axis=axis,
        value=run.summary_metrics[metric_key],
        deadline=time.monotonic() + timeout_seconds,
    )


_EVAL_ROWS = [
    {"_step": 0, "train/loss": 2.0},
    {"_step": 1, "eval/loss": 0.4, "eval/step": 100},
    {"_step": 2, "train/loss": 1.0},
    {"_step": 3, "eval/loss": 0.2, "eval/step": 200},
    {"_step": 4, "eval/loss": 0.3, "eval/step": 300},
    {"_step": 5, "eval/loss": 0.2, "eval/step": 400},
]


@pytest.mark.parametrize(
    ("summary", "metric_key", "axis", "expected"),
    [
        ({"eval/loss": 0.2, "_step": 5}, "eval/loss", "_step", 5),
        ({"eval/loss": 0.2, "_step": 5}, "eval/loss", "eval/step", 400),
        # The server returns define_metric aggregates as flat series.aggregation
        # keys. An aggregate covers its whole series, so it is positioned where
        # the series ends, not where its best value fell (eval/step 100 here).
        ({"eval/loss.max": 0.4, "_step": 5}, "eval/loss.max", "eval/step", 400),
        ({"eval/loss.min": 0.2, "_step": 5}, "eval/loss.min", "eval/step", 400),
        ({"eval/loss.mean": 0.275, "_step": 5}, "eval/loss.mean", "eval/step", 400),
        ({"eval/loss.last": 0.2, "_step": 5}, "eval/loss.last", "eval/step", 400),
        ({"eval/loss.first": 0.4, "_step": 5}, "eval/loss.first", "eval/step", 400),
        ({"eval/loss": 0.1, "_step": 5}, "eval/loss", "eval/step", None),
        ({"eval/loss": 0.2}, "eval/loss", "eval/step", None),
    ],
    ids=[
        "last-on-_step",
        "last-on-eval-step",
        "max",
        "min",
        "mean",
        "last",
        "first",
        "assigned",
        "no-history",
    ],
)
def test_wandb_locates_where_the_objective_summary_was_logged(
    monkeypatch, summary, metric_key, axis, expected
):
    run = _HistoryRun(summary, _EVAL_ROWS)
    assert _history_position(monkeypatch, run, metric_key, axis) == expected


def test_wandb_aggregate_key_listed_in_the_index_is_read_from_its_series(monkeypatch):
    """An aggregate key the index lists but no row carries is read from its head series, after one scan confirms it is empty."""
    run = _HistoryRun({"eval/loss.min": 0.2, "_step": 5}, _EVAL_ROWS)
    run.load(force=True)
    assert "eval/loss.min" in run.history_keys["keys"]
    assert _history_position(monkeypatch, run, "eval/loss.min", "eval/step") == 400
    assert run.reads == [
        ("history", ["eval/loss.min", "_step"]),
        ("scan", ["eval/loss.min", "_step"]),
        ("history", ["eval/loss", "eval/step"]),
    ]


@pytest.mark.parametrize(
    ("evaluations", "expected"),
    [([5, 10, 15, 20], 20), (list(range(1, 13)), 12)],
    ids=["sweep-trainer", "resumed-trainer"],
)
def test_wandb_live_trainer_histories_are_positioned_at_their_last_evaluation(
    monkeypatch, evaluations, expected
):
    """The live trainers' runs, one eval row per ``_step`` and an ``eval/loss.min`` summary, are positioned at their last ``eval/step``."""
    rows = [
        {"_step": step, "eval/loss": 1.0 / evaluation, "eval/step": evaluation}
        for step, evaluation in enumerate(evaluations)
    ]
    summary = {"eval/loss.min": min(row["eval/loss"] for row in rows), "_step": len(rows) - 1}
    run = _HistoryRun(summary, rows)
    assert _history_position(monkeypatch, run, "eval/loss.min", "eval/step") == expected


def test_wandb_axis_never_logged_settles_without_a_read(monkeypatch):
    """An axis the index does not list leaves no position to find, so nothing is read or waited for."""
    run = _HistoryRun({"eval/loss": 0.2, "_step": 5}, _EVAL_ROWS)
    start = time.monotonic()
    assert _history_position(monkeypatch, run, axis="tokens", timeout_seconds=60) is None
    assert time.monotonic() - start < 5
    assert run.reads == []


def test_wandb_empty_sampled_read_is_scanned_once(monkeypatch):
    """Rows the sampled call does not serve yet are read by one scan instead of a wait."""
    run = _HistoryRun({"eval/loss": 0.2, "_step": 5}, _EVAL_ROWS, visible=[slice(0, 0)])
    assert _history_position(monkeypatch, run, axis="eval/step") == 400
    assert [kind for kind, _ in run.reads] == ["history", "scan"]


@pytest.mark.parametrize("direction", [-math.inf, math.inf])
def test_wandb_final_history_neighbor_supplies_only_advisory_position(monkeypatch, direction):
    """The two API representations may differ by one binary64 value."""
    value = 0.0965803563594818
    logged = math.nextafter(value, direction)
    rows = [
        {"_step": 0, "eval/loss": 3.9606740474700928, "eval/step": 0},
        {"_step": 25, "eval/loss": logged, "eval/step": 500},
        {"_step": 27, "eval/loss": logged, "eval/step": 500},
    ]
    summary = {"eval/loss": value, "_step": 28}
    run = _HistoryRun(summary, rows + [{"_step": 28, "train/loss": 0.04}])
    assert _history_position(monkeypatch, run, axis="eval/step") == 500
    assert run.summary_metrics == {"eval/loss": value, "_step": 28}
    assert run.rows[1]["eval/loss"] == logged


@pytest.mark.parametrize("exact_axis", [100, None])
def test_wandb_exact_history_match_precedes_later_neighbor(monkeypatch, exact_axis):
    """An exact observation stays authoritative even without a usable axis."""
    value = 0.2
    rows = [
        {"_step": 0, "eval/loss": value, "eval/step": exact_axis},
        {"_step": 1, "eval/loss": math.nextafter(value, math.inf), "eval/step": 500},
    ]
    run = _HistoryRun({"eval/loss": value, "_step": 1}, rows)
    assert _history_position(monkeypatch, run, axis="eval/step") == exact_axis


@pytest.mark.parametrize(
    "values",
    [
        [math.nextafter(0.2, -math.inf), math.nextafter(0.2, math.inf)],
        [math.nextafter(math.nextafter(0.2, math.inf), math.inf)],
        [math.nextafter(0.2, math.inf), 0.3],
        [math.nextafter(0.2, math.inf), "not-a-number"],
        [math.nextafter(0.2, math.inf), True],
        [math.nextafter(0.2, math.inf), math.nan],
        [math.nextafter(0.2, math.inf), math.inf],
    ],
    ids=["ambiguous", "two-ulps", "older-only", "malformed", "boolean", "nan", "infinite"],
)
def test_wandb_history_neighbor_cannot_guess_a_position(monkeypatch, values):
    rows = [
        {"_step": step, "eval/loss": logged, "eval/step": (step + 1) * 100}
        for step, logged in enumerate(values)
    ]
    run = _HistoryRun({"eval/loss": 0.2, "_step": len(rows) - 1}, rows)
    assert _history_position(monkeypatch, run, axis="eval/step") is None


def test_wandb_series_at_the_sample_cap_is_scanned_in_pages_never_sampled(monkeypatch):
    monkeypatch.setattr("phasesweep.evidence.wandb._HISTORY_SAMPLES", 2)
    run = _HistoryRun({"eval/loss.min": 0.2, "_step": 5}, _EVAL_ROWS)
    assert _history_position(monkeypatch, run, "eval/loss.min", "eval/step") == 400
    # Six rows exceed the cap of two, so the aggregate key and then its head are scanned.
    assert [kind for kind, _ in run.reads] == ["scan", "scan"]
    # A page of as many steps as the cap holds at most that many rows.
    assert run.page_size == 2


def test_wandb_slow_summary_polling_still_retries_history(monkeypatch):
    """A summary poll interval as long as the budget must not leave one history read."""
    monkeypatch.setattr("phasesweep.evidence.wandb._HISTORY_RETRY_SECONDS", 0.001)
    run = _HistoryRun({"eval/loss": 0.2, "_step": 5}, _EVAL_ROWS, indexed_steps=[3])
    position = _history_position(
        monkeypatch, run, "eval/loss", "eval/step", poll_seconds=60, timeout_seconds=60
    )
    assert position == 400
    assert len(run.reads) == 1


@pytest.mark.parametrize(
    ("metric_key", "value"),
    [("best/loss", 0.1), ("eval/loss.selected", 0.123456), ("eval/loss.selected", 0.4)],
    ids=["absent-series", "custom-suffix-unlogged-value", "custom-suffix-matching-value"],
)
def test_wandb_summary_key_absent_from_history_settles_with_no_position(
    monkeypatch, metric_key, value
):
    """A summary key no row carries settles after one sampled read and one scan, without waiting."""
    run = _HistoryRun({metric_key: value, "_step": 5}, _EVAL_ROWS)
    assert _history_position(monkeypatch, run, metric_key, "eval/step", timeout_seconds=60) is None
    assert run.reads == [
        ("history", [metric_key, "eval/step"]),
        ("scan", [metric_key, "eval/step"]),
    ]


@pytest.mark.parametrize("suffix", ["ema", "min"])
def test_wandb_dotted_history_series_is_read_before_its_head(monkeypatch, suffix):
    """A series really logged under a dotted name, even an aggregate-looking one, is positioned by its own rows."""
    key = f"eval/loss.{suffix}"
    rows = [
        {"_step": 0, "eval/loss": 0.2, key: 0.3, "eval/step": 10},
        {"_step": 1, "eval/loss": 0.3, key: 0.2, "eval/step": 20},
        {"_step": 2, "eval/loss": 0.4, key: 0.25, "eval/step": 30},
    ]
    run = _HistoryRun({key: 0.2, "_step": 2}, rows)
    assert _history_position(monkeypatch, run, key, "eval/step") == 20
    reads = [("history", [key, "eval/step"])]
    if suffix == "min":
        reads.insert(0, ("history", [key, "_step"]))
    assert run.reads == reads


def test_wandb_dotted_history_series_without_its_axis_has_no_position(monkeypatch):
    """A literal metric without the axis cannot acquire its head's unrelated position."""
    key = "eval/loss.min"
    rows = [
        {"_step": 0, key: 0.2},
        {"_step": 1, "eval/loss": 0.9, "eval/step": 20},
    ]
    run = _HistoryRun({key: 0.2, "_step": 1}, rows)
    assert _history_position(monkeypatch, run, key, "eval/step") is None


@pytest.mark.parametrize("indexed_steps", [[3, 4], [None, -1, -1]], ids=["behind", "unindexed"])
def test_wandb_waits_for_history_to_reach_the_summary_step(monkeypatch, indexed_steps):
    """An index behind the summary, or not yet served at all, is waited for before any read."""
    run = _HistoryRun({"eval/loss": 0.2, "_step": 5}, _EVAL_ROWS, indexed_steps=indexed_steps)
    assert _history_position(monkeypatch, run, axis="eval/step") == 400
    assert len(run.reads) == 1


def test_wandb_history_that_never_settles_stops_at_the_deadline(monkeypatch):
    run = _HistoryRun({"eval/loss": 0.2, "_step": 5}, _EVAL_ROWS, indexed_steps=[3] * 10**6)
    start = time.monotonic()
    assert _history_position(monkeypatch, run, timeout_seconds=0.2) is None
    assert 0.1 < time.monotonic() - start < 30
    assert run.reads == []


def test_wandb_transient_history_failure_retries_until_the_deadline(monkeypatch):
    """A history read that fails in transport after a long wait is read again within the deadline."""
    clock = _fake_wandb_clock(monkeypatch)

    class FlakyRun(_HistoryRun):
        failures = 1

        def history(self, keys, samples=500, pandas=True):
            if self.failures:
                self.failures -= 1
                clock.now += 40.0  # a long read, then a dropped connection
                raise ConnectionError("connection reset by peer")
            return super().history(keys, samples, pandas)

    run = FlakyRun({"eval/loss": 0.2, "_step": 5}, _EVAL_ROWS)
    assert _history_position(monkeypatch, run, axis="eval/step", timeout_seconds=120) == 400


@pytest.mark.parametrize(("timeout", "expected"), [(120, 400), (70, None)])
def test_wandb_history_read_is_bounded_by_the_deadline(monkeypatch, timeout, expected):
    """A slow history read is used when it ends before the deadline."""
    clock = ManualClock()
    monkeypatch.setattr("phasesweep.evidence.wandb.time.monotonic", clock.monotonic)
    monkeypatch.setattr("phasesweep.evidence.wandb._HISTORY_SAMPLES", 1)

    class SlowScanRun(_HistoryRun):
        def scan_history(self, keys, page_size=1000, use_cache=True):
            # Four evaluation rows, 20 s apart: the scan ends 80 s in.
            for row in super().scan_history(keys, page_size, use_cache):
                clock.now += 20.0
                yield row

    run = SlowScanRun({"eval/loss": 0.2, "_step": 5}, _EVAL_ROWS)
    position = _history_position(monkeypatch, run, axis="eval/step", timeout_seconds=timeout)
    assert position == expected


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
    # Constraints and gates may trigger the shared capture first.
    assert run_extractor(ctx, constraint) == 4
    provenance = {}
    assert run_extractor(ctx, objective, provenance=provenance) == 0.2
    assert all(result.passed for result in evaluate_gates(ctx, [gate]))
    assert len(calls) == 1
    assert calls[0]["required_keys"] == ("eval/loss", "memory")
    assert provenance["remote_capture"]["run_id"] == ctx.attempt_id
    # The position is read only once the trial is accepted.
    assert provenance["source"]["evaluation"] == {"progress": {"axis": "_step", "value": None}}


def test_wandb_gates_share_failed_capture(tmp_path, monkeypatch):
    from phasesweep.evidence.models import WandbQuery
    from phasesweep.evidence.wandb import WandbRunTerminalError

    gate = WandbSummaryRequiredGate(
        type="wandb_summary_required", entity="entity", project="project", keys=["done"]
    )
    other_gate = gate.model_copy(update={"keys": ["complete"]})
    ctx = replace(
        make_trial_context(tmp_path),
        wandb_query=WandbQuery(gate, (), ("done", "complete")),
    )
    calls = 0

    def capture(**kwargs):
        nonlocal calls
        calls += 1
        raise WandbRunTerminalError(ctx.attempt_id, "failed")

    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", capture)
    results = evaluate_gates(ctx, [gate, other_gate])
    assert calls == 1
    assert len(results) == 2
    assert all(not result.passed and not result.deadline_exhausted for result in results)
    assert results[0].detail == results[1].detail
    assert "ended in 'failed'" in results[0].detail


def _wandb_failures() -> dict[str, tuple[Exception, bool]]:
    """Return each W&B reader failure and whether it leaves the evidence unreadable."""
    from phasesweep.evidence import wandb as wandb_evidence

    attempt = "attempt-test"
    return {
        "transport-timeout": (wandb_evidence.WandbPollTimeout(attempt, 1.0, reached=False), True),
        "reached-timeout": (wandb_evidence.WandbPollTimeout(attempt, 1.0, reached=True), False),
        "setup": (wandb_evidence.WandbSetupError(attempt, "HTTPError (HTTP 401)"), True),
        "sdk": (ImportError("wandb"), True),
        "worker": (RuntimeError("worker exited"), True),
        "missing-run": (wandb_evidence.WandbRunMissingError(attempt), False),
        "terminal": (wandb_evidence.WandbRunTerminalError(attempt, "crashed"), False),
        "missing-keys": (wandb_evidence.WandbMissingKeysError(attempt, ["loss"], [], 1.0), False),
        "invalid": (ValueError("not a finite number"), False),
    }


@pytest.mark.parametrize("failure", list(_wandb_failures()))
def test_wandb_capture_is_unreadable_only_when_wandb_gave_no_answer(tmp_path, monkeypatch, failure):
    """The objective and every gate see one cached failure, unreadable only without an answer."""
    from phasesweep.evidence.models import WandbQuery

    error, unreadable = _wandb_failures()[failure]
    calls = 0

    def capture(**kwargs):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", capture)
    extractor = WandbExtractor(type="wandb", entity="e", project="p", metric_key="loss")
    gate = WandbSummaryRequiredGate(
        type="wandb_summary_required", entity="e", project="p", keys=["loss"]
    )
    ctx = replace(
        make_trial_context(tmp_path), wandb_query=WandbQuery(extractor, ("loss",), ("loss",))
    )
    with pytest.raises(ExtractorError) as refused:
        run_extractor(ctx, extractor)
    assert isinstance(refused.value, EvidenceUnreadableError) is unreadable
    assert not isinstance(refused.value, DeadlineExceededError)
    (result,) = evaluate_gates(ctx, [gate])
    assert (result.passed, result.deadline_exhausted) == (False, False)
    assert result.evidence_unreadable is unreadable
    assert calls == 1


# The capture budget includes the supervised worker's startup, which takes 0.5 to 0.9 s
# from spawn to its first poll, alone or under ``pytest -n 12`` on this file. A test that
# needs a poll before its deadline, or polling to continue past startup, gets several
# times that so a loaded host cannot leave it without one.
_WORKER_POLL_BUDGET_SECONDS = 6


@pytest.mark.integration
@pytest.mark.parametrize(
    ("answer", "limit"),
    [("connection", "source"), ("running", "source"), ("running", "phase")],
)
def test_wandb_worker_timeout_says_whether_the_run_answered(
    wandb_worker_sdk, tmp_path, answer, limit
):
    """The source's timeout after transport failures is unreadable and after an unfinished run fails it.

    A phase deadline nearer than the source's timeout ends either poll as the phase interruption,
    although the worker stops a reserve before it and answers with the clock still short of it.
    """
    from phasesweep.evidence.wandb import WandbPollTimeout

    wandb_worker_sdk(f"""
        class Api:
            def __init__(self, **kwargs): pass
            def run(self, path):
                if {answer!r} == "connection":
                    raise ConnectionError("tracker unreachable")
                return type("Run", (), {{"state": "running", "summary_metrics": {{}}}})()
    """)
    phase = limit == "phase"
    source = WandbExtractor(
        type="wandb",
        entity="e",
        project="p",
        metric_key="loss",
        poll_seconds=0.01,
        timeout_seconds=10 * _WORKER_POLL_BUDGET_SECONDS if phase else _WORKER_POLL_BUDGET_SECONDS,
    )
    deadline = time.monotonic() + _WORKER_POLL_BUDGET_SECONDS if phase else None
    with pytest.raises(ExtractorError) as refused:
        run_extractor(make_trial_context(tmp_path), source, deadline=deadline)
    assert isinstance(refused.value, DeadlineExceededError) is phase
    assert isinstance(refused.value, EvidenceUnreadableError) is (
        answer == "connection" and not phase
    )
    timeout = refused.value.__cause__
    assert isinstance(timeout, WandbPollTimeout)
    assert timeout.reached is (answer == "running")


@pytest.mark.integration
def test_wandb_worker_polls_until_shortly_before_its_deadline(wandb_worker_sdk, tmp_path):
    """A run that finishes 3.5 seconds inside a ten-second budget is captured, not timed out.

    The run finishes 6.5 seconds after the extraction starts, timed from the parent rather than
    from the worker's first lookup, so the worker's start-up time cannot push it past the
    polling deadline; a five-second reserve stops polling before it.
    """
    started = time.monotonic()
    wandb_worker_sdk(f"""
        import time
        class Api:
            def __init__(self, **kwargs): pass
            def run(self, path):
                finished = time.monotonic() - {started!r} >= 6.5
                return type("Run", (), {{
                    "state": "finished" if finished else "running",
                    "summary_metrics": {{"loss": 0.25}},
                }})()
    """)
    source = WandbExtractor(
        type="wandb",
        entity="e",
        project="p",
        metric_key="loss",
        poll_seconds=0.2,
        timeout_seconds=10,
    )
    assert run_extractor(make_trial_context(tmp_path), source) == 0.25


@pytest.mark.parametrize(
    ("server", "unreadable"),
    [("refused", True), ("denied", True), ("failed-run", False), ("deleted-run", False)],
)
def test_mlflow_capture_is_unreadable_only_when_mlflow_gave_no_answer(
    tmp_path, monkeypatch, server, unreadable
):
    """A refused connection or denied access leaves the evidence unreadable; a FAILED or deleted run fails it."""
    import errno

    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.protos import databricks_pb2

    from phasesweep.evidence.evaluation import _capture_mlflow
    from tests.tracking_helpers import bind_mlflow_attempt, mlflow_run, run_mlflow_worker_in_process

    binding, ctx = bind_mlflow_attempt(tmp_path)

    def get_run(_run_id):
        if server == "refused":
            raise MlflowException("private") from ConnectionRefusedError(
                errno.ECONNREFUSED, "private"
            )
        if server == "denied":
            raise MlflowException("private", error_code=databricks_pb2.PERMISSION_DENIED)
        if server == "deleted-run":
            raise MlflowException("private", error_code=databricks_pb2.RESOURCE_DOES_NOT_EXIST)
        return mlflow_run(binding, status="FAILED")

    monkeypatch.setattr(
        mlflow, "MlflowClient", lambda tracking_uri: SimpleNamespace(get_run=get_run)
    )

    run_mlflow_worker_in_process(monkeypatch)
    with pytest.raises(ExtractorError) as refused:
        _capture_mlflow(ctx, deadline=None)
    assert isinstance(refused.value, EvidenceUnreadableError) is unreadable
    assert "private" not in str(refused.value)


@pytest.mark.parametrize("definite_gate", [False, True], ids=["unreadable-only", "with-definite"])
def test_a_definite_gate_failure_outranks_an_unreadable_one(tmp_path, monkeypatch, definite_gate):
    """A result is unreadable only when every failed gate is; one definite failure fails the trial."""
    from phasesweep.config.models import Metric
    from phasesweep.engine.trial import ExecutedTrial, extract_trial_result
    from phasesweep.evidence.models import RequiredFileGate
    from phasesweep.evidence.wandb import WandbSetupError
    from phasesweep.runtime.process import ProcessResult
    from tests.conftest import make_experiment

    def denied(**kwargs):
        raise WandbSetupError("attempt-test", "HTTPError (HTTP 401)")

    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", denied)
    (tmp_path / "r.json").write_text(json.dumps({"x": 0.5}))
    experiment = make_experiment(
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x"))
    )
    gates = [
        WandbSummaryRequiredGate(
            type="wandb_summary_required", entity="e", project="p", keys=["done"]
        )
    ]
    if definite_gate:
        gates.append(RequiredFileGate(type="required_file", path="checkpoint.pt"))
    result = extract_trial_result(
        experiment=experiment,
        executed=ExecutedTrial(make_trial_context(tmp_path), ProcessResult(0, False, -1, 1.0)),
        gates=gates,
    )
    assert result.failure_reason is not None
    assert result.gate_results is not None
    assert [gate.passed for gate in result.gate_results] == [False] * len(gates)
    assert result.evidence_unreadable is not definite_gate
    assert not result.deadline_exhausted


@pytest.mark.parametrize("failure_kind", ["deadline", "cleanup"])
def test_wandb_consumers_retry_uncached_failures(tmp_path, monkeypatch, failure_kind):
    from phasesweep.errors import UnsafeProcessCleanupError

    extractor = WandbExtractor(type="wandb", entity="entity", project="project", metric_key="loss")
    ctx = make_trial_context(tmp_path)
    failure = (
        DeadlineExceededError("capture deadline expired")
        if failure_kind == "deadline"
        else UnsafeProcessCleanupError("worker cleanup is uncertain")
    )
    calls = 0

    def capture(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        return {"values": {"loss": 0.25}, "present_keys": ["loss"]}

    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", capture)
    with pytest.raises(type(failure), match=str(failure)):
        run_extractor(ctx, extractor)
    assert run_extractor(ctx, extractor) == 0.25
    assert calls == 2


@pytest.mark.parametrize("stopped_at", [1.0, 0.25], ids=["at-deadline", "in-reserve"])
@pytest.mark.parametrize("answer", ["unanswered", "reached", "missing-run"])
def test_wandb_consumers_retry_capture_cut_short_by_deadline(
    tmp_path, monkeypatch, answer, stopped_at
):
    """A capture the execution deadline cut short is the phase interruption and can be retried.

    The worker stops polling a reserve before the deadline, so the poll ends with the clock
    still short of it whatever its last answer was.
    """
    from phasesweep.evidence.wandb import WandbPollTimeout, WandbRunMissingError

    gate = WandbSummaryRequiredGate(
        type="wandb_summary_required", entity="entity", project="project", keys=["done"]
    )
    ctx = make_trial_context(tmp_path)
    clock = ManualClock()
    calls = 0

    def capture(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            clock.now = stopped_at
            if answer == "missing-run":
                raise WandbRunMissingError(ctx.attempt_id)
            raise WandbPollTimeout(
                ctx.attempt_id, gate.timeout_seconds, reached=answer == "reached"
            )
        return {"values": {}, "present_keys": ["done"]}

    monkeypatch.setattr("phasesweep.evidence.evaluation.time.monotonic", clock.monotonic)
    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", capture)
    first = evaluate_gates(ctx, [gate], deadline=1.0)[0]
    assert not first.passed
    assert first.deadline_exhausted
    assert not first.evidence_unreadable
    assert evaluate_gates(ctx, [gate], deadline=2.0)[0].passed
    assert calls == 2


@pytest.mark.parametrize("stopped_at", [118.0, 120.0], ids=["in-reserve", "at-timeout"])
@pytest.mark.parametrize("answer", ["unanswered", "reached", "missing-run"])
def test_wandb_capture_stopped_by_the_source_timeout_is_an_ordinary_failure(
    tmp_path, monkeypatch, answer, stopped_at
):
    """A poll that ends under the source's own timeout, before the execution deadline, is no phase stop."""
    from phasesweep.evidence.wandb import WandbPollTimeout, WandbRunMissingError

    gate = WandbSummaryRequiredGate(
        type="wandb_summary_required", entity="entity", project="project", keys=["done"]
    )
    ctx = make_trial_context(tmp_path)
    clock = ManualClock()

    def capture(**kwargs):
        clock.now = stopped_at
        if answer == "missing-run":
            raise WandbRunMissingError(ctx.attempt_id)
        raise WandbPollTimeout(ctx.attempt_id, gate.timeout_seconds, reached=answer == "reached")

    monkeypatch.setattr("phasesweep.evidence.evaluation.time.monotonic", clock.monotonic)
    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", capture)
    (result,) = evaluate_gates(ctx, [gate], deadline=gate.timeout_seconds + 1.0)
    assert not result.passed
    assert not result.deadline_exhausted
    assert result.evidence_unreadable is (answer == "unanswered")


def test_wandb_gate_only_missing_key_fails_first_finished_capture(tmp_path, monkeypatch):
    from phasesweep.evidence.wandb import _poll_wandb_summary

    public = pytest.importorskip("wandb.apis.public")
    calls = []

    def lookup(path):
        calls.append(path)
        return SimpleNamespace(state="finished", summary_metrics={"other_key": 1})

    monkeypatch.setattr(public, "Api", lambda **kwargs: SimpleNamespace(run=lookup))
    monkeypatch.setattr(
        "phasesweep.evidence.evaluation.poll_wandb_summary",
        lambda attempt_dir, attempt_id, environment, **kwargs: _poll_wandb_summary(**kwargs),
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
def test_wandb_late_lookup_answer_is_kept_but_no_lookup_starts_late(monkeypatch, stage):
    """A lookup answered after the polling deadline is used; none starts once it has passed."""
    public = pytest.importorskip("wandb.apis.public")
    from phasesweep.evidence.wandb import WandbPollTimeout

    clock = ManualClock()
    monkeypatch.setattr("phasesweep.evidence.wandb.time.monotonic", clock.monotonic)
    lookups = []

    class Api:
        def __init__(self, **kwargs):
            if stage == "constructor":
                clock.now = 2.0

        def run(self, path):
            lookups.append(path)
            clock.now = 2.0
            return SimpleNamespace(state="finished", summary_metrics={"loss": 0.2})

    monkeypatch.setattr(public, "Api", Api)
    if stage == "constructor":
        with pytest.raises(WandbPollTimeout):
            _wandb_poll(required_keys=["loss"])
        assert lookups == []
    else:
        assert _wandb_poll(required_keys=["loss"])["values"] == {"loss": 0.2}


def test_wandb_worker_transfers_only_requested_evidence(wandb_worker_sdk, tmp_path, monkeypatch):
    from phasesweep.evidence.wandb import locate_wandb_position, poll_wandb_summary
    from phasesweep.runtime.process import read_attempt_lifecycle

    monkeypatch.setenv("WANDB_API_KEY", "excluded-parent-secret")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    wandb_worker_sdk("""
        import os
        class Run:
            state = "finished"
            summary_metrics = {"loss": 0.25, "_step": 7, "done": object(), "unrelated": object()}
            history_keys = {"lastStep": 7, "keys": {"loss": {"typeCounts": [{"count": 2}]},
                                                    "_step": {"typeCounts": [{"count": 2}]}}}
            def load(self, force=False):
                pass
            def history(self, keys, samples=500, pandas=True):
                return [{"loss": 0.25, "_step": 7}]
        class Api:
            def __init__(self, **kwargs):
                assert "WANDB_API_KEY" not in os.environ
                assert os.environ["HTTPS_PROXY"] == "https://transport.test"
                assert "PYTHONHOME" not in os.environ
                assert os.environ["DO_NOT_TRACK"] == "1"
                assert kwargs["overrides"] == {"base_url": "https://example.test"}
            def run(self, path):
                assert path == "entity/project/attempt"
                return Run()
    """)
    target = {
        "base_url": "https://example.test",
        "entity": "entity",
        "project": "project",
        "run_id": "attempt",
        "attempt_id": "attempt",
        "attempt_dir": tmp_path,
        "poll_seconds": 0.01,
        "timeout_seconds": 5,
        "environment": {"HTTPS_PROXY": "https://transport.test", "PYTHONHOME": "/trainer/python"},
    }
    capture = poll_wandb_summary(
        **target, required_keys=["loss"], presence_keys=["done", "missing"]
    )
    assert capture == {
        "values": {"loss": 0.25},
        "present_keys": ["done"],
        "retrieved_at": capture["retrieved_at"],
    }
    assert read_attempt_lifecycle(tmp_path, expected_attempt_id="attempt").cleanup_confirmed
    # The advisory position is a separate request in the same attempt slot.
    assert locate_wandb_position(**target, metric_key="loss", axis="_step", value=0.25) == 7
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
            attempt_id="attempt",
            attempt_dir=tmp_path,
            poll_seconds=0.01,
            timeout_seconds=5,
        )

    diagnostic = tmp_path / "wandb-worker.stderr.log"
    assert "worker-diagnostic-marker" in diagnostic.read_text()
    assert diagnostic.stat().st_mode & 0o777 == 0o600
    assert "worker-diagnostic-marker" not in str(exc_info.value)


def test_wandb_missing_summary_key_reports_available_names_after_polling(
    wandb_worker_sdk, tmp_path
):
    """A finished summary that never gains the key fails at the deadline, naming the missing and available keys."""
    calls = tmp_path / "lookup-count"
    wandb_worker_sdk(f"""
        from pathlib import Path
        class Run:
            state = "finished"
            summary_metrics = {{"eval/loss": 0.25, "train/runtime": 123456.789}}
        class Api:
            def __init__(self, **kwargs): pass
            def run(self, path):
                target = Path({str(calls)!r})
                target.write_text(str(int(target.read_text()) + 1) if target.exists() else "1")
                return Run()
    """)
    extractor = WandbExtractor(
        type="wandb",
        entity="e",
        project="p",
        metric_key="eval_loss",
        poll_seconds=0.02,
        timeout_seconds=_WORKER_POLL_BUDGET_SECONDS,
    )

    with pytest.raises(ExtractorError) as caught:
        run_extractor(make_trial_context(tmp_path), extractor)

    message = str(caught.value)
    assert "eval_loss" in message
    assert "available" in message.lower()
    assert "eval/loss" in message
    assert "train/runtime" in message
    assert "123456.789" not in message
    assert int(calls.read_text()) > 1


def test_wandb_finished_summary_can_gain_a_required_key(wandb_worker_sdk, tmp_path):
    wandb_worker_sdk("""
        calls = 0
        class Run:
            state = "finished"
        class Api:
            def __init__(self, **kwargs): pass
            def run(self, path):
                global calls
                calls += 1
                run = Run()
                run.summary_metrics = {"loss": 0.25} if calls >= 3 else {"train/runtime": 1}
                return run
    """)
    extractor = WandbExtractor(
        type="wandb",
        entity="e",
        project="p",
        metric_key="loss",
        poll_seconds=0.01,
        timeout_seconds=_WORKER_POLL_BUDGET_SECONDS,
    )

    assert run_extractor(make_trial_context(tmp_path), extractor) == 0.25


def _fake_wandb_clock(monkeypatch: pytest.MonkeyPatch) -> ManualClock:
    """Advance fake monotonic time by the poller's sleeps.

    :param pytest.MonkeyPatch monkeypatch: Fixture that replaces the poller's clock functions.
    :return ManualClock: The clock shared by the poller and test assertions.
    """
    clock = ManualClock()
    monkeypatch.setattr("phasesweep.evidence.wandb.time.monotonic", clock.monotonic)
    monkeypatch.setattr("phasesweep.evidence.wandb.time.sleep", clock.sleep)
    return clock


@pytest.mark.parametrize("request_kind", ["summary", "position"])
def test_wandb_launch_failure_cannot_use_trainer_identity(tmp_path, monkeypatch, request_kind):
    """Uncertain worker cleanup is never downgraded, even for the advisory position read."""
    from phasesweep.errors import UnsafeProcessCleanupError
    from phasesweep.evidence.wandb import locate_wandb_position, poll_wandb_summary
    from phasesweep.runtime import process

    with (tmp_path / "trainer.log").open("w") as output:
        result = process.run_supervised(
            "true",
            env=dict(os.environ),
            stdout=output,
            stderr=output,
            timeout=5,
            attempt_dir=tmp_path,
            attempt_id="attempt",
        )
    assert result.cleanup_confirmed
    original_abort = process._abort_launch
    aborted_groups: list[int] = []

    def fail_identity(path, identity):
        assert not path.exists()
        raise OSError("injected identity failure")

    def uncertain_abort(proc, pgid):
        assert original_abort(proc, pgid)
        assert isinstance(pgid, int)
        aborted_groups.append(pgid)
        return False

    monkeypatch.setattr(process, "_write_process_identity", fail_identity)
    monkeypatch.setattr(process, "_abort_launch", uncertain_abort)
    target = {
        "base_url": "https://example.test",
        "entity": "e",
        "project": "p",
        "run_id": "attempt",
        "attempt_id": "attempt",
        "attempt_dir": tmp_path,
        "poll_seconds": 0.01,
        "timeout_seconds": 5,
    }
    with pytest.raises(UnsafeProcessCleanupError, match="cleanup is uncertain"):
        if request_kind == "summary":
            poll_wandb_summary(**target)
        else:
            locate_wandb_position(**target, metric_key="loss", axis="_step", value=0.25)
    lifecycle = process.read_attempt_lifecycle(tmp_path, expected_attempt_id="attempt")
    assert lifecycle is not None and lifecycle.state == "exited"
    assert lifecycle.cleanup_confirmed is True
    assert lifecycle.duration_seconds is not None and lifecycle.duration_seconds >= 0
    assert lifecycle.ended_by == "not_started"
    assert lifecycle.return_code is None
    assert aborted_groups
    assert all(not process._process_group_alive(pgid) for pgid in aborted_groups)


@pytest.mark.parametrize("stage", ["constructor", "lookup"])
@pytest.mark.integration
def test_wandb_supervision_bounds_blocked_sdk_and_descendants(wandb_worker_sdk, tmp_path, stage):
    from phasesweep.evidence.wandb import WandbPollTimeout, poll_wandb_summary
    from phasesweep.runtime.reaper import is_pid_alive
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
            attempt_id="attempt",
            attempt_dir=tmp_path,
            poll_seconds=0.01,
            timeout_seconds=2,
        )
    assert time.monotonic() - start < 12
    pid = int(pid_file.read_text())
    assert not is_pid_alive(pid) or is_pid_zombie(pid)


@pytest.mark.integration
def test_wandb_position_worker_stopped_at_its_deadline_records_no_position(
    wandb_worker_sdk, tmp_path
):
    from phasesweep.evidence.wandb import locate_wandb_position
    from phasesweep.runtime.process import read_attempt_lifecycle

    wandb_worker_sdk("""
        import time
        class Run:
            state = "finished"
            summary_metrics = {"loss": 0.25, "_step": 7}
            history_keys = {"lastStep": 7, "keys": {"loss": {"typeCounts": [{"count": 2}]},
                                                    "_step": {"typeCounts": [{"count": 2}]}}}
            def load(self, force=False):
                pass
            def history(self, keys, samples=500, pandas=True):
                time.sleep(60)
                return []
        class Api:
            def __init__(self, **kwargs):
                pass
            def run(self, path):
                return Run()
    """)
    start = time.monotonic()
    position = locate_wandb_position(
        base_url="https://example.test",
        entity="e",
        project="p",
        run_id="attempt",
        attempt_id="attempt",
        attempt_dir=tmp_path,
        poll_seconds=0.01,
        timeout_seconds=2,
        metric_key="loss",
        axis="_step",
        value=0.25,
    )
    assert time.monotonic() - start < 12
    assert position is None
    assert read_attempt_lifecycle(tmp_path, expected_attempt_id="attempt").cleanup_confirmed


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


@pytest.mark.parametrize(
    "filename,key,payload,match",
    [
        ("nope.json", "x", None, "not found"),
        ("result.json", "a.c", {"a": {"b": 1}}, "not found"),
        ("result.json", "x", {"x": "abc"}, "not a JSON number"),
        ("result.json", "x", {"x": "1.25"}, "not a JSON number"),
        ("result.json", "x", {"x": True}, "not a JSON number"),
    ],
    ids=["missing_file", "missing_key", "non_numeric", "numeric_string", "boolean"],
)
def test_json_extractor_reports_missing_or_nonnumeric_values(
    tmp_path, filename, key, payload, match
):
    if payload is not None:
        (tmp_path / filename).write_text(json.dumps(payload))
    cfg = JsonExtractor(type="json", path=filename, key=key)
    with pytest.raises(ExtractorError, match=match):
        run_extractor(make_trial_context(tmp_path), cfg)


def test_json_extractor_rejects_duplicate_keys(tmp_path):
    (tmp_path / "result.json").write_text('{"x": 1, "x": 2}')
    cfg = JsonExtractor(type="json", path="result.json", key="x")

    with pytest.raises(ExtractorError, match="Invalid JSON"):
        run_extractor(make_trial_context(tmp_path), cfg)


@pytest.mark.parametrize(
    "filename,cfg",
    [
        ("result.json", JsonExtractor(type="json", path="result.json", key="x")),
        (
            "stdout.log",
            LogRegexExtractor(
                type="log_regex",
                file="stdout.log",
                pattern=r"value=(?P<value>[0-9.]+)",
            ),
        ),
        (
            "result.json",
            JsonEnvelopeExtractor(
                type="json_envelope",
                objective_name="val_loss",
                split="validation",
                policy="final_checkpoint",
            ),
        ),
    ],
    ids=["json", "log", "envelope"],
)
def test_file_extractors_report_invalid_utf8_as_trial_evidence_failure(tmp_path, filename, cfg):
    (tmp_path / filename).write_bytes(b"\xff\xfe")
    with pytest.raises(ExtractorError, match="not valid UTF-8"):
        run_extractor(make_trial_context(tmp_path), cfg)


@pytest.mark.parametrize("kind", ["json", "json_envelope"])
def test_json_extractors_preserve_read_failure_diagnostics(tmp_path, monkeypatch, kind):
    target = tmp_path / "result.json"
    target.write_text("{}")
    failure = PermissionError("read denied")
    original = Path.read_bytes

    def unreadable(path):
        if path == target:
            raise failure
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    cfg = (
        JsonExtractor(type="json", path="result.json", key="x")
        if kind == "json"
        else JsonEnvelopeExtractor(
            type="json_envelope", objective_name="loss", split="validation", policy="final"
        )
    )
    with pytest.raises(ExtractorError) as caught:
        run_extractor(make_trial_context(tmp_path), cfg)
    noun = "JSON" if kind == "json" else "JSON envelope"
    assert str(caught.value) == f"Could not read {noun} at {target}: read denied"
    assert caught.value.__cause__ is failure


@pytest.fixture
def envelope_evidence():
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
    cfg = JsonEnvelopeExtractor(
        type="json_envelope",
        path="result.json",
        objective_name="val_loss",
        split="validation",
        policy="final_checkpoint",
    )
    return payload, cfg


def test_json_envelope_binds_objective_to_current_attempt(tmp_path, envelope_evidence):
    payload, cfg = envelope_evidence
    (tmp_path / "result.json").write_text(json.dumps(payload))
    cfg = cfg.model_copy(update={"checkpoint": "final.pt", "expected_step": 1000})

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
def test_json_envelope_rejects_mismatched_provenance(
    tmp_path, envelope_evidence, path, value, match
):
    payload, cfg = envelope_evidence
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    (tmp_path / "result.json").write_text(json.dumps(payload))
    cfg = cfg.model_copy(update={"checkpoint": "final.pt", "expected_step": 1000})

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


@pytest.mark.parametrize(
    ("select", "line", "expected", "match_count"),
    [
        ("last", "loss=9 loss=1\n", 1.0, 2),
        ("min", "loss=0.5, loss=bad loss=9 loss=1\n", 1.0, 2),
        ("max", "loss=1 loss=9\n", 9.0, 2),
        ("first", "loss=0.5, loss=1\n", 1.0, 1),
    ],
)
def test_log_regex_considers_every_numeric_match_per_line(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    select: str,
    line: str,
    expected: float,
    match_count: int,
) -> None:
    """One log line can contribute multiple numeric candidates."""
    raw = line.encode("utf-8")
    (tmp_path / "stdout.log").write_bytes(raw)
    cfg = LogRegexExtractor(type="log_regex", pattern=r"loss=(?P<value>[^\s]+)", select=select)

    provenance: dict = {}
    assert run_extractor(make_trial_context(tmp_path), cfg, provenance=provenance) == expected
    assert len(caplog.messages) == int(select in {"first", "min"})
    if caplog.messages:
        assert "capture '0.5,' on line 1" in caplog.messages[0]
    assert provenance["source"] == {
        "kind": "file",
        "path": "stdout.log",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "matched_line": 1,
        "match_count": match_count,
    }


@pytest.mark.parametrize(
    ("select", "text", "bad_line"),
    [
        ("min", "loss=nan\nloss=0.5\nloss=0.3\n", 1),
        ("min", "loss=0.5\nloss=0.3\nloss=nan\n", 3),
        ("min", "loss=0.5\nloss=inf\nloss=0.3\n", 2),
        ("max", "loss=nan\nloss=0.5\nloss=0.3\n", 1),
        ("max", "loss=0.5\nloss=0.3\nloss=nan\n", 3),
        ("max", "loss=0.5\nloss=-inf\n", 2),
    ],
)
def test_log_regex_min_max_fails_on_any_non_finite_match(
    tmp_path: Path, select: str, text: str, bad_line: int
) -> None:
    """A min or max over matches that include nan or inf fails naming the offending line."""
    (tmp_path / "stdout.log").write_text(text, encoding="utf-8")
    cfg = LogRegexExtractor(type="log_regex", pattern=r"loss=(?P<value>[^\s]+)", select=select)
    provenance: dict = {}
    with pytest.raises(
        ExtractorError,
        match=rf"non-finite match on line {bad_line}\b.*cannot score this trial",
    ):
        run_extractor(make_trial_context(tmp_path), cfg, provenance=provenance)
    assert "source" not in provenance


@pytest.mark.parametrize(
    ("line", "message"),
    [("nothing here", "No matches"), ("eval_loss=0.5,", r"was numeric\..*'0\.5,' on line 1")],
)
def test_log_regex_reports_no_matches(tmp_path: Path, line: str, message: str) -> None:
    (tmp_path / "stdout.log").write_text(line, encoding="utf-8")
    cfg = LogRegexExtractor(type="log_regex", pattern=r"eval_loss=(?P<value>\S+)")
    with pytest.raises(ExtractorError, match=message):
        run_extractor(make_trial_context(tmp_path), cfg)


def test_provenance_freezes_file_digest_and_extractor_identity(tmp_path):
    """The provenance sink freezes the exact evidence bytes and the extractor's
    config identity at extraction time."""
    raw = json.dumps({"eval": {"loss": 0.42}}).encode("utf-8")
    (tmp_path / "result.json").write_bytes(raw)
    cfg = JsonExtractor(type="json", path="result.json", key="eval.loss")

    provenance: dict = {}
    value = run_extractor(make_trial_context(tmp_path), cfg, provenance=provenance)

    assert value == pytest.approx(0.42)
    assert provenance["schema_version"] == EVIDENCE_PROVENANCE_SCHEMA_VERSION
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


def test_provenance_envelope_freezes_validated_evaluation_metadata(tmp_path, envelope_evidence):
    payload, cfg = envelope_evidence
    payload["evaluation"]["checkpoint"] = "step_1000.pt"
    raw = json.dumps(payload).encode("utf-8")
    (tmp_path / "result.json").write_bytes(raw)

    provenance: dict = {}
    run_extractor(make_trial_context(tmp_path), cfg, provenance=provenance)

    source = provenance["source"]
    assert source["sha256"] == hashlib.sha256(raw).hexdigest()
    # The checkpoint and step are the envelope's own validated values even
    # when the config did not pin them; the default axis is that step.
    assert source["evaluation"] == {
        "objective_name": "val_loss",
        "split": "validation",
        "policy": "final_checkpoint",
        "checkpoint": "step_1000.pt",
        "step": 1000,
        "progress": {"axis": "step", "value": 1000},
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
    """Lone carriage returns are line boundaries, matching a text-mode
    universal-newline reader — tqdm-style progress logs separate updates with
    bare "\\r"."""
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


def test_wandb_real_sdk_missing_run_polls_until_the_visibility_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK's wrapped missing-run error remains visible after all allowed polls."""
    public = pytest.importorskip("wandb.apis.public")
    from phasesweep.evidence.wandb import WandbRunMissingError

    clock = _fake_wandb_clock(monkeypatch)
    lookups: list[object] = []

    class Transport:
        def execute_graphql(
            self, query: object, variables: object, **kwargs: object
        ) -> dict[str, None]:
            lookups.append(variables)
            return {"project": None}

    def initialize(self, overrides: object = None, timeout: object = None) -> None:
        self.settings = {"entity": "entity", "project": "project"}
        attach_wandb_transport(self, Transport())

    monkeypatch.setattr(public.Api, "__init__", initialize)
    with pytest.raises(WandbRunMissingError) as refused:
        _wandb_poll(required_keys=["loss"], poll_seconds=5, timeout_seconds=15)
    assert refused.value.run_id == "attempt"
    assert clock.now == 15
    assert len(lookups) == 3


def test_wandb_missing_run_is_recognized_from_the_sdk_wrapper_alone(monkeypatch) -> None:
    """The service's ``CommError: Could not find run`` answers the lookup without its cause."""
    public = pytest.importorskip("wandb.apis.public")
    from wandb.errors import CommError

    from phasesweep.evidence.wandb import WandbRunMissingError

    _fake_wandb_clock(monkeypatch)

    def lookup(path: str) -> None:
        raise CommError(f"Could not find run <Run {path} (None)>")

    monkeypatch.setattr(public, "Api", lambda **_: SimpleNamespace(run=lookup))
    with pytest.raises(WandbRunMissingError):
        _wandb_poll(required_keys=["loss"], poll_seconds=5, timeout_seconds=15)


@pytest.mark.parametrize(
    "last_observation", ["missing", "running", "network", "summary", "permission"]
)
def test_wandb_missing_run_diagnostic_follows_the_last_observation(
    monkeypatch: pytest.MonkeyPatch, last_observation: str
) -> None:
    """A later successful lookup, transport fault, or denial supersedes an earlier absence."""
    public = pytest.importorskip("wandb.apis.public")
    from requests import HTTPError, Response
    from wandb.errors import CommError

    from phasesweep.evidence.wandb import (
        WandbPollTimeout,
        WandbRunMissingError,
        WandbSetupError,
    )

    _fake_wandb_clock(monkeypatch)
    calls = 0

    class Api:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, path: str) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            if calls == 1 or last_observation == "missing":
                raise CommError("private SDK message", exc=ValueError(f"Could not find run {path}"))
            if last_observation == "network":
                raise CommError("private SDK message", exc=ConnectionError("private token"))
            if last_observation == "permission":
                response = Response()
                response.status_code = 403
                raise CommError(
                    "private SDK message", exc=HTTPError("private token", response=response)
                )
            return SimpleNamespace(
                state="running" if last_observation == "running" else "finished", summary_metrics={}
            )

    monkeypatch.setattr(public, "Api", Api)
    expected = (
        WandbRunMissingError
        if last_observation == "missing"
        else WandbSetupError
        if last_observation == "permission"
        else WandbPollTimeout
    )
    with pytest.raises(expected) as refused:
        _wandb_poll(required_keys=["loss"], poll_seconds=5, timeout_seconds=15)
    assert type(refused.value) is expected
    assert calls >= 2
    assert "private" not in str(refused.value)
    if isinstance(refused.value, WandbPollTimeout):
        # A finished run with an empty summary gives no usable answer either.
        assert refused.value.reached is (last_observation == "running")
        detail = str(refused.value.last_error)
        assert "private" not in detail
        assert "Could not find run" not in detail


@pytest.mark.integration
def test_wandb_missing_run_worker_names_target_and_init_overrides(
    wandb_worker_sdk,
    tmp_path: Path,
) -> None:
    """A missing-run deadline crosses the supervised worker with concrete init advice."""
    wandb_worker_sdk("""
        class Api:
            def __init__(self, **kwargs): pass
            def run(self, path):
                raise ValueError(f"Could not find run {path}")
    """)
    source = WandbExtractor(
        type="wandb",
        entity="entity",
        project="project",
        metric_key="loss",
        poll_seconds=0.01,
        timeout_seconds=_WORKER_POLL_BUDGET_SECONDS,
    )
    context = make_trial_context(tmp_path)
    with pytest.raises(ExtractorError) as refused:
        run_extractor(context, source)
    message = str(refused.value)
    assert f"No W&B run {context.attempt_id!r} was found in entity/project" in message
    assert "The trainer should call wandb.init() without its own project, entity or id" in message
    assert "Last observation: ValueError" not in message
    assert len(message.splitlines()) == 1


@pytest.mark.integration
def test_wandb_worker_reports_a_missing_run_before_the_supervisor_stops_it(
    wandb_worker_sdk,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A lookup still in flight at the worker's polling deadline leaves time to report the missing run."""
    from phasesweep.evidence import wandb as wandb_evidence

    # The client's timeout is the whole seconds left to the polling deadline, so a lookup that
    # begins in the last second is the one a slow link leaves in flight when the deadline passes.
    wandb_worker_sdk("""
        import time
        class Api:
            def __init__(self, overrides=None, timeout=None):
                self.slow = timeout <= 1
            def run(self, path):
                if self.slow:
                    time.sleep(1.5)
                raise ValueError(f"Could not find run {path}")
    """)
    launched = []
    run_worker = wandb_evidence.run_worker

    def recording_worker(*args, **kwargs):
        outcome = run_worker(*args, **kwargs)
        launched.append(outcome)
        return outcome

    monkeypatch.setattr(wandb_evidence, "run_worker", recording_worker)
    with (
        caplog.at_level("WARNING", logger="phasesweep.runtime.process"),
        pytest.raises(wandb_evidence.WandbRunMissingError),
    ):
        wandb_evidence.poll_wandb_summary(
            base_url="https://example.test",
            entity="e",
            project="p",
            run_id="attempt",
            attempt_id="attempt",
            attempt_dir=tmp_path,
            poll_seconds=0.01,
            timeout_seconds=_WORKER_POLL_BUDGET_SECONDS,
        )
    ((result, response, _),) = launched
    assert response == {"status": "run_missing"}
    assert not result.timed_out
    assert "terminating group" not in caplog.text
