"""W&B serves a finished run's summary late; the reader waits for it until the deadline and never reads a half-ingested one.

The sequences follow what the public API served on a live project: a run reported ``finished``
with its whole history at once and its summary a minute later, and a resumed run that kept
the earlier attempt's summary for minutes after the rerun finished.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from phasesweep import run_experiment
from phasesweep.engine.errors import TrackingTargetUnavailableError
from phasesweep.engine.state import ATTEMPT_ID_ATTR
from phasesweep.evidence.evaluation import DeadlineExceededError, ExtractorError, run_extractor
from phasesweep.evidence.models import WandbExtractor
from tests.conftest import make_trial_context, phase_trials
from tests.tracking_helpers import ManualClock, switchable_wandb, wandb_objective_experiment

_LAUNCHED_AT = 1_000.0
_KEY = "eval/loss.min"

Answer = tuple[str, dict[str, object], int | None]
"""A run's state, its summary, and its history index's last step in the same answer.

-1 is an index with no keys and ``None`` a null index, as W&B serves before indexing.
"""


def _history_index(last_step: int | None) -> dict[str, object] | None:
    """Build the ``historyKeys`` index W&B serves for a last step.

    :param int | None last_step: The index's last step, -1 unindexed, ``None`` null.
    :return dict[str, object] | None: The index.
    """
    if last_step is None:
        return None
    keys = {} if last_step < 0 else {"eval/loss": {"typeCounts": [{"count": last_step + 1}]}}
    return {"lastStep": last_step, "keys": keys}


def _poll(
    *, logged_after: float | None = None, required: tuple[str, ...] = (_KEY,), **kwargs: object
) -> dict[str, object]:
    """Run the in-process summary poll every 5 s with a 240 s budget.

    :param float | None logged_after: A rerun's launch time, defaults to None.
    :param tuple[str, ...] required: Required numeric keys, defaults to the objective.
    :param object kwargs: Further poll arguments.
    :return dict[str, object]: The capture.
    """
    from phasesweep.evidence.wandb import _poll_wandb_summary

    return _poll_wandb_summary(
        base_url="https://example.test",
        entity="e",
        project="p",
        run_id="opener",
        poll_seconds=5,
        timeout_seconds=240,
        required_keys=required,
        logged_after=logged_after,
        **kwargs,
    )


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    answers: Callable[[int], Answer],
    *,
    rows: tuple[Mapping[str, object], ...] = (),
    on_scan: Callable[[list[str]], None] | None = None,
) -> tuple[ManualClock, list[str]]:
    """Answer each zero-based run lookup with ``answers(read)`` on a manual clock.

    Each lookup must come from a client built for it, as the real client caches the run.

    :param pytest.MonkeyPatch monkeypatch: Fixture that replaces the SDK client and clock.
    :param Callable[[int], Answer] answers: State, summary and last history step per read.
    :param tuple[Mapping[str, object], ...] rows: Rows available in unsampled history.
    :param Callable[[list[str]], None] | None on_scan: Callback for each history read.
    :return tuple[ManualClock, list[str]]: The clock and the paths looked up.
    """
    public = pytest.importorskip("wandb.apis.public")
    clock = ManualClock()
    monkeypatch.setattr("phasesweep.evidence.wandb.time.monotonic", clock.monotonic)
    monkeypatch.setattr("phasesweep.evidence.wandb.time.sleep", clock.sleep)
    paths: list[str] = []

    def scan_history(
        *, keys: list[str], page_size: int, use_cache: bool
    ) -> list[Mapping[str, object]]:
        assert page_size > 0 and not use_cache
        if on_scan is not None:
            on_scan(keys)
        return [row for row in rows if all(key in row for key in keys)]

    class Api:
        def __init__(self, **kwargs: object) -> None:
            self.used = False

        def run(self, path: str) -> SimpleNamespace:
            assert not self.used, "a client served a second lookup"
            self.used = True
            paths.append(path)
            state, summary, last_step = answers(len(paths) - 1)
            return SimpleNamespace(
                state=state,
                summary_metrics=summary,
                history_keys=_history_index(last_step),
                scan_history=scan_history,
            )

    monkeypatch.setattr(public, "Api", Api)
    return clock, paths


def test_a_finished_runs_summary_served_a_minute_late_is_read_once_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished run whose summary is empty, then behind its history, is read only once the summary reaches the history's last step."""

    def answer(read: int) -> Answer:
        if read < 6:
            return "finished", {}, 3
        if read < 12:
            return "finished", {"_step": 2, _KEY: 0.3}, 3
        return "finished", {"_step": 3, _KEY: 0.1, "_timestamp": 50.0}, 3

    clock, paths = _serve(monkeypatch, answer)
    assert _poll()["values"] == {_KEY: 0.1}
    assert len(paths) == 13
    assert clock.now == 60


def test_reread_only_accepts_a_shared_summary_logged_during_the_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished run can still supply an earlier attempt's summary if it predates the cutoff.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the SDK client and clock.
    """
    from phasesweep.evidence.wandb import WandbRunStaleError

    _serve(monkeypatch, lambda _: ("finished", {"_step": 3, _KEY: 0.1, "_timestamp": 50.0}, 3))
    assert _poll(logged_before=51.0)["values"] == {_KEY: 0.1}
    with pytest.raises(WandbRunStaleError) as failure:
        _poll(logged_before=50.0)
    assert failure.value.superseded


@pytest.mark.parametrize("unindexed", [-1, None], ids=["no-keys", "null"])
def test_a_summary_served_before_the_history_index_is_not_read(
    monkeypatch: pytest.MonkeyPatch, unindexed: int | None
) -> None:
    """A summary that arrives while the history index is still unserved waits for the index, as on the live service."""
    served = {"_step": 3, _KEY: 0.1, "_timestamp": 50.0}

    def answer(read: int) -> Answer:
        # Polls at 0 and 5 s: running, then finished, both before the summary arrives.
        if read == 0:
            return "running", {}, unindexed
        if read < 12:
            return "finished", {}, unindexed
        if read == 12:
            return "finished", served, unindexed
        return "finished", served, 3

    clock, paths = _serve(monkeypatch, answer)
    assert _poll()["values"] == {_KEY: 0.1}
    assert len(paths) == 14
    assert clock.now == 65


@pytest.mark.parametrize("initial_step", [None, 3], ids=["both-unpublished", "history-first"])
def test_a_gate_only_read_waits_for_the_summary_to_reach_the_history(
    monkeypatch: pytest.MonkeyPatch,
    initial_step: int | None,
) -> None:
    """A gate-only read waits for a summary even when the history index is unpublished too."""
    clock, _ = _serve(
        monkeypatch,
        lambda read: (
            ("finished", {}, initial_step)
            if read < 2
            else ("finished", {"_step": 3, "complete": True}, 3)
        ),
    )
    assert _poll(required=(), presence_keys=["complete"])["present_keys"] == ["complete"]
    assert clock.now == 10


def test_a_finished_summary_behind_its_history_stays_unreadable_on_a_rerun(monkeypatch) -> None:
    """A partially ingested summary at the deadline cannot establish which attempt logged."""
    from phasesweep.evidence.wandb import WandbPollTimeout

    _serve(monkeypatch, lambda _: ("finished", {"_step": 1, "_timestamp": 1.0, _KEY: 0.1}, 2))
    with pytest.raises(WandbPollTimeout) as caught:
        _poll(logged_after=_LAUNCHED_AT)
    assert not caught.value.reached


def test_a_finished_summary_missing_a_key_fails_only_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete finished summary without the required key is polled to the deadline, then fails naming the keys and the wait."""
    from phasesweep.evidence.wandb import WandbMissingKeysError

    clock, _ = _serve(monkeypatch, lambda read: ("finished", {"_step": 3, "eval/loss": 0.1}, 3))
    with pytest.raises(WandbMissingKeysError) as caught:
        _poll()
    assert (caught.value.missing, caught.value.available) == ([_KEY], ["_step", "eval/loss"])
    assert caught.value.waited_seconds == clock.now == 240


@pytest.mark.integration
def test_a_late_summary_scores_the_original_trial_on_resume(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """A finished run whose summary is empty past the deadline costs one trainer launch."""
    tracker = switchable_wandb(wandb_worker_sdk, tmp_path)
    tracker.switch("empty")
    experiment, launches = wandb_objective_experiment(tmp_path, n_trials=1, timeout_seconds=6)
    with pytest.raises(TrackingTargetUnavailableError, match="summary"):
        run_experiment(experiment)
    (pending,) = phase_trials(experiment)
    assert pending.state.name == "RUNNING"
    assert tracker.lookups("empty") > 0

    tracker.switch("up")
    assert run_experiment(experiment)["p"].metric == 1.0
    (recovered,) = phase_trials(experiment)
    assert recovered.state.name == "COMPLETE"
    assert recovered.user_attrs[ATTEMPT_ID_ATTR] == pending.user_attrs[ATTEMPT_ID_ATTR]
    assert launches.read_text().splitlines() == ["launch"]


@pytest.mark.parametrize("recovers", [False, True], ids=["outage", "recovers"])
def test_an_outage_after_an_incomplete_summary_proves_nothing_missing(
    monkeypatch: pytest.MonkeyPatch, recovers: bool
) -> None:
    """A summary read without the key, then no answer until the deadline, is unreadable; an answer before it is read."""
    from phasesweep.evidence.wandb import WandbPollTimeout

    def answer(read: int) -> Answer:
        if read == 0:
            return "finished", {"_step": 3}, 3
        if recovers and read == 40:
            return "finished", {"_step": 3, _KEY: 0.1}, 3
        raise ConnectionError("tracker offline")

    clock, _ = _serve(monkeypatch, answer)
    if recovers:
        assert _poll()["values"] == {_KEY: 0.1}
        assert clock.now == 200
    else:
        with pytest.raises(WandbPollTimeout) as caught:
            _poll()
        assert not caught.value.reached
        assert "ConnectionError" in str(caught.value.last_error)
        assert clock.now == 240


def test_a_resumed_runs_earlier_summary_is_waited_out_for_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed run that keeps the earlier attempt's finished summary and history for three minutes, then the new ones, is scored from the new summary."""
    left = {"_step": 5, _KEY: 0.4, "_timestamp": _LAUNCHED_AT - 60}
    logged = {"_step": 11, _KEY: 0.1, "_timestamp": _LAUNCHED_AT + 30}

    def answer(read: int) -> Answer:
        if read < 36:
            return "finished", left, 5
        if read < 40:
            return "finished", left, 11
        return "finished", logged, 11

    clock, paths = _serve(monkeypatch, answer)
    assert _poll(logged_after=_LAUNCHED_AT)["values"] == {_KEY: 0.1}
    assert paths == ["e/p/opener"] * 41
    assert clock.now == 200


def test_a_reruns_read_waits_out_the_state_its_earlier_attempt_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run still crashed, then finished, as the earlier attempt left it is neither failed nor scored; the rerun's own finished summary is."""
    left = {"_step": 5, _KEY: 0.5, "_timestamp": _LAUNCHED_AT - 60}
    logged = {"_step": 11, _KEY: 0.25, "_timestamp": _LAUNCHED_AT + 5}
    answers: list[Answer] = [("crashed", left, 5), ("finished", left, 5), ("finished", logged, 11)]
    clock, _ = _serve(monkeypatch, lambda read: answers[min(read, 2)])
    assert _poll(logged_after=_LAUNCHED_AT)["values"] == {_KEY: 0.25}
    assert clock.now == 10


def test_a_reruns_run_that_never_moves_is_refused_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run whose summary never passes the rerun's launch fails as not written by this attempt only at the deadline."""
    from phasesweep.evidence.wandb import WandbRunStaleError

    left = {"_step": 5, _KEY: 0.5, "_timestamp": _LAUNCHED_AT - 60}
    clock, _ = _serve(monkeypatch, lambda read: ("finished", left, 5))
    with pytest.raises(WandbRunStaleError) as caught:
        _poll(logged_after=_LAUNCHED_AT)
    assert caught.value.waited_seconds == clock.now == 240


def test_a_reruns_own_crash_fails_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that crashed after the rerun logged to it is the rerun's own failure."""
    from phasesweep.evidence.wandb import WandbRunTerminalError

    logged = {"_step": 11, _KEY: 0.25, "_timestamp": _LAUNCHED_AT + 5}
    _serve(monkeypatch, lambda read: ("crashed", logged, 11))
    with pytest.raises(WandbRunTerminalError):
        _poll(logged_after=_LAUNCHED_AT)


@pytest.mark.parametrize(
    ("axis", "mode", "expected"),
    [
        ("eval/step", "native", (0.02, 0.9)),
        ("_step", "native", (0.01, 0.99)),
        ("eval/step", "literal", (0.01, 0.99)),
        ("eval/step", "summary-only", (0.11, 0.88)),
        ("eval/step", "missing-axis", (0.01, 0.99)),
        ("eval/step", "history-outage", (0.02, 0.9)),
    ],
)
def test_a_rerun_replaces_the_observed_training_suffix_for_native_extrema(
    monkeypatch: pytest.MonkeyPatch, axis: str, mode: str, expected: tuple[float, float]
) -> None:
    """Native min and max exclude overwritten points while preserving earlier valid history."""
    series = [
        (975.0, 10, 0.4),
        (985.0, 20, 0.2),
        (990.0, 30, 0.01),
        (995.0, 30, 0.99),
        (1005.0, 30, 0.3),
        (1010.0, 40, 0.9),
        (1015.0, 40, 0.02),
        (1025.0, 50, 0.4),
        (1030.0, 60, 0.35),
    ]
    rows: list[dict[str, object]] = [
        {"_step": index, "_timestamp": stamp, "eval/step": step, "eval/loss": loss}
        for index, (stamp, step, loss) in enumerate(series)
    ]
    if mode == "literal":
        rows.append({"_step": len(rows), "_timestamp": 1031.0, _KEY: 0.01, "eval/loss.max": 0.99})
    if mode == "missing-axis":
        rows[0].pop("eval/step")
    native = (0.11, 0.88) if mode == "summary-only" else (0.01, 0.99)
    last_step = len(rows) - 1
    summary = {
        "_step": last_step,
        "_timestamp": 1031.0,
        _KEY: native[0],
        "eval/loss.max": native[1],
        "eval/loss": 0.35,
    }
    readings: list[list[str]] = []

    def on_scan(keys: list[str]) -> None:
        readings.append(keys)
        if mode == "history-outage" and len(readings) == 1:
            raise ConnectionError("history temporarily unavailable")

    clock, _ = _serve(
        monkeypatch, lambda _: ("finished", summary, last_step), rows=tuple(rows), on_scan=on_scan
    )
    keys = (_KEY, "eval/loss.max")
    for index, selected in enumerate(keys):
        values = {**dict(zip(keys, native, strict=True)), "eval/loss": 0.35}
        values[selected] = expected[index]
        assert (
            _poll(
                logged_after=1020.0,
                earlier_launches=(1000.0,),
                resume_metric=(selected, axis),
                required=(*keys, "eval/loss"),
            )["values"]
            == values
        )
    if axis == "_step":
        assert not readings
    if mode == "history-outage":
        assert clock.now == 5


def test_a_rerun_that_discards_every_objective_observation_fails_the_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rerun that re-logs progress alone cannot win with a discarded metric."""
    rows = (
        {"_step": 0, "_timestamp": 990.0, "eval/step": 10, "eval/loss": 0.1},
        {"_step": 1, "_timestamp": 1010.0, "eval/step": 5},
    )
    summary = {"_step": 1, "_timestamp": 1010.0, _KEY: 0.1}
    _serve(monkeypatch, lambda _: ("finished", summary, 1), rows=rows)
    with pytest.raises(ValueError, match="no finite observation in the retained history"):
        _poll(logged_after=_LAUNCHED_AT, resume_metric=(_KEY, "eval/step"))


def test_the_real_client_reads_the_history_index_with_the_summary(tmp_path, monkeypatch) -> None:
    """The SDK's own run lookup carries the history index, one fresh client per poll, so an empty finished summary is not read."""
    public = pytest.importorskip("wandb.apis.public")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    queries: list[str] = []
    clients: list[object] = []
    responses = iter(
        [
            {"state": "finished", "summaryMetrics": "{}", "historyKeys": _history_index(3)},
            {
                "state": "finished",
                "summaryMetrics": json.dumps({"_step": 3, _KEY: 0.1}),
                "historyKeys": _history_index(3),
            },
        ]
    )

    class Transport:
        def execute_graphql(self, query: object, variables: object, **kwargs: object) -> object:
            queries.append(str(query))
            return {"project": {"run": next(responses)}}

    # Keep the real Api.run, Run construction and decoding; replace only transport setup.
    def initialize(self, overrides=None, timeout=None):
        clients.append(self)
        self.settings = {"entity": "e", "project": "p"}
        self._runs = {}
        self._service_api = Transport()
        self.api_key = None

    monkeypatch.setattr(public.Api, "__init__", initialize)
    from phasesweep.evidence.wandb import _poll_wandb_summary

    capture = _poll_wandb_summary(
        base_url="https://example.test",
        entity="e",
        project="p",
        run_id="opener",
        poll_seconds=0.001,
        timeout_seconds=5,
        required_keys=[_KEY],
    )
    assert capture["values"] == {_KEY: 0.1}
    assert len(queries) == len(clients) == 2
    assert all("historyKeys" in query for query in queries)


def test_the_missing_key_report_crosses_the_worker_with_its_wait(tmp_path, monkeypatch) -> None:
    """The worker's missing-keys reply carries how long it polled, and the trial's failure says so."""
    from phasesweep.evidence import wandb as wandb_evidence

    _serve(monkeypatch, lambda read: ("finished", {"_step": 3, "eval/loss": 0.1}, 3))

    def in_process_worker(request, _worker_path, _backend, **_):
        directory = tmp_path / "worker"
        directory.mkdir(mode=0o700)
        request_path, response_path = directory / "request.json", directory / "response.json"
        request_path.write_text(json.dumps(dict(request)))
        assert wandb_evidence.main([str(request_path), str(response_path)]) == 0
        return (
            SimpleNamespace(timed_out=False, return_code=0),
            json.loads(response_path.read_text()),
            None,
        )

    monkeypatch.setattr(wandb_evidence, "run_worker", in_process_worker)
    extractor = WandbExtractor(
        type="wandb", entity="e", project="p", metric_key=_KEY, poll_seconds=5, timeout_seconds=240
    )
    ctx = make_trial_context(tmp_path)
    with pytest.raises(ExtractorError) as caught:
        run_extractor(ctx, extractor)
    assert type(caught.value) is ExtractorError
    # The worker keeps its reporting reserve, so it polled 238 of the 240 seconds.
    assert str(caught.value) == (
        f"W&B run https://api.wandb.ai/e/p/{ctx.wandb_run_id} finished without required "
        f"summary key(s) ['{_KEY}'] after 238 s; available summary keys: ['_step', 'eval/loss']."
    )


@pytest.mark.parametrize("verdict", ["missing_keys", "stale"])
@pytest.mark.parametrize("capped", [False, True], ids=["source-timeout", "phase-deadline"])
def test_a_deadline_verdict_cut_short_by_the_phase_is_an_interruption(
    tmp_path, monkeypatch, verdict: str, capped: bool
) -> None:
    """A missing-key or not-written verdict reached at a phase deadline shorter than the source's timeout interrupts the trial instead of failing it."""
    import time

    from phasesweep.evidence import wandb as wandb_evidence

    def poll(**kwargs: object) -> dict[str, object]:
        if verdict == "stale":
            raise wandb_evidence.WandbRunStaleError("opener", 9.0)
        raise wandb_evidence.WandbMissingKeysError("opener", [_KEY], [], 9.0)

    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", poll)
    extractor = WandbExtractor(type="wandb", entity="e", project="p", metric_key=_KEY)
    deadline = time.monotonic() + 10 if capped else None
    with pytest.raises(ExtractorError) as caught:
        run_extractor(make_trial_context(tmp_path), extractor, deadline=deadline)
    assert type(caught.value) is (DeadlineExceededError if capped else ExtractorError)
    assert re.search(r"after 9 s", str(caught.value))
