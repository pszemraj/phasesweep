"""W&B serves a finished run's summary late; the reader waits for it until the deadline and never reads a half-ingested one.

The sequences follow what the public API served on a live project: a run reported ``finished``
with its whole history at once and its summary a minute later, and a resumed run that kept
the earlier attempt's summary for minutes after the rerun finished.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from phasesweep.evidence.evaluation import DeadlineExceededError, ExtractorError, run_extractor
from phasesweep.evidence.models import WandbExtractor
from tests.conftest import make_trial_context
from tests.tracking_helpers import ManualClock

_LAUNCHED_AT = 1_000.0
_KEY = "eval/loss.min"

Answer = tuple[str, dict[str, object], int]
"""A run's state, its summary, and the last history step in the same answer (-1 for none)."""


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
    monkeypatch: pytest.MonkeyPatch, answers: Callable[[int], Answer]
) -> tuple[ManualClock, list[str]]:
    """Answer each zero-based run lookup with ``answers(read)`` on a manual clock.

    Each lookup must come from a client built for it, as the real client caches the run.

    :param pytest.MonkeyPatch monkeypatch: Fixture that replaces the SDK client and clock.
    :param Callable[[int], Answer] answers: State, summary and last history step per read.
    :return tuple[ManualClock, list[str]]: The clock and the paths looked up.
    """
    public = pytest.importorskip("wandb.apis.public")
    clock = ManualClock()
    monkeypatch.setattr("phasesweep.evidence.wandb.time.monotonic", clock.monotonic)
    monkeypatch.setattr("phasesweep.evidence.wandb.time.sleep", clock.sleep)
    paths: list[str] = []

    class Api:
        def __init__(self, **kwargs: object) -> None:
            self.used = False

        def run(self, path: str) -> SimpleNamespace:
            assert not self.used, "a client served a second lookup"
            self.used = True
            paths.append(path)
            state, summary, last_step = answers(len(paths) - 1)
            return SimpleNamespace(
                state=state, summary_metrics=summary, history_keys={"lastStep": last_step}
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


def test_a_gate_only_read_waits_for_the_summary_to_reach_the_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no numeric key to wait for, an empty finished summary is still not read while history exists."""
    clock, _ = _serve(
        monkeypatch,
        lambda read: ("finished", {} if read < 2 else {"_step": 3, "complete": True}, 3),
    )
    assert _poll(required=(), presence_keys=["complete"])["present_keys"] == ["complete"]
    assert clock.now == 10


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


def test_the_real_client_reads_the_history_index_with_the_summary(tmp_path, monkeypatch) -> None:
    """The SDK's own run lookup carries the history index, one fresh client per poll, so an empty finished summary is not read."""
    public = pytest.importorskip("wandb.apis.public")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    queries: list[str] = []
    clients: list[object] = []
    responses = iter(
        [
            {"state": "finished", "summaryMetrics": "{}", "historyKeys": {"lastStep": 3}},
            {
                "state": "finished",
                "summaryMetrics": json.dumps({"_step": 3, _KEY: 0.1}),
                "historyKeys": {"lastStep": 3},
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
