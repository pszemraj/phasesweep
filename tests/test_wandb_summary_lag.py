"""W&B serves a finished run's summary late; the reader waits for it until the deadline and never reads a half-ingested one.

The sequences follow what the public API served on a live project: a run reported ``finished``
with its whole history at once and its summary a minute later, and a resumed run that kept
the earlier attempt's summary for minutes after the rerun finished.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric
from phasesweep.engine.errors import TrackingTargetUnavailableError
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import ATTEMPT_ID_ATTR
from phasesweep.evidence.baseline import EvidenceBaseline
from phasesweep.evidence.evaluation import (
    DeadlineExceededError,
    EvidenceUnreadableError,
    ExtractorError,
    run_extractor,
)
from phasesweep.evidence.mlflow import merge_histories, select_observation
from phasesweep.evidence.models import MlflowExtractor, WandbExtractor, WandbSummaryRequiredGate
from tests.conftest import make_experiment, make_trial_context, phase_trials, write_trainer
from tests.tracking_helpers import (
    ManualClock,
    attach_wandb_transport,
    switchable_wandb,
    wandb_objective_experiment,
)

_LAUNCHED_AT = 1_000.0
_KEY = "eval/loss.min"


def test_sibling_retry_replaces_tracker_history_before_recovery_and_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An attempt uses every earlier checkpoint writer's history across reading studies."""
    from phasesweep.evidence import evaluation, wandb

    launches, history = tmp_path / "launches", tmp_path / "history"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import json, os, time
        from pathlib import Path
        launches, history = Path({str(launches)!r}), Path({str(history)!r})
        attempt = len(launches.read_text().splitlines()) if launches.exists() else 0
        with launches.open("a") as log:
            log.write(os.environ["WANDB_RUN_ID"] + "\\n")
        rows = json.loads(history.read_text()) if history.exists() else []
        points = [[(100, 0.20), (300, 0.01)], [(200, 0.40)], [(400, 0.90)]][attempt]
        for progress, loss in points:
            rows.append({{"_step": len(rows), "_timestamp": time.time(), "eval/step": progress,
                          "eval/loss": loss}})
        history.write_text(json.dumps(rows))
        raise SystemExit(1 if attempt < 2 else 0)
        """,
    )

    source = {"entity": "e", "project": "p", "base_url": "https://example.test"}
    timing = {"poll_seconds": 0.01, "timeout_seconds": 2}

    def config(key: str, target: int, *, gated: bool = False) -> Experiment:
        # A gate changes what the kept attempts' evidence reads as, so a gated study re-reads.
        gate = WandbSummaryRequiredGate(
            type="wandb_summary_required", keys=[_KEY], **source, **timing
        )
        return make_experiment(
            persistent=tmp_path,
            trainer=trainer,
            metric=Metric(
                extractor=WandbExtractor(
                    type="wandb", metric_key=key, evaluation_axis="eval/step", **source, **timing
                ),
            ),
            n_trials=target,
            search_space={},
            gpu_policy="none",
            gates=[gate] if gated else [],
        )

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(config(_KEY, 1))
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(config("eval/loss.max", 2))

    earlier_counts: list[int] = []

    def capture(**options: object) -> dict[str, object]:
        epochs = options["earlier_launches"]
        assert isinstance(epochs, tuple)
        earlier_counts.append(len(epochs))
        if len(earlier_counts) == 1:
            run_id = options["run_id"]
            assert isinstance(run_id, str)
            raise wandb.WandbPollTimeout(run_id, 2, reached=False)
        rows = tuple(json.loads(history.read_text()))
        last = len(rows) - 1
        summary = {"_step": last, "_timestamp": rows[-1]["_timestamp"], _KEY: 0.01}
        after = options["logged_after"]
        assert isinstance(after, float)
        with monkeypatch.context() as sdk:
            _serve(sdk, lambda _: ("finished", summary, last), rows=rows)
            return _poll(
                logged_after=after,
                logged_before=options.get("logged_before"),
                presence_keys=options.get("presence_keys", ()),
                earlier_launches=epochs,
                resume_axis="eval/step",
            )

    monkeypatch.setattr(evaluation, "poll_wandb_summary", capture)
    monkeypatch.setattr(evaluation, "locate_wandb_position", lambda **_: 100)
    # The sibling's failed retry spends the second trial, so the third launches.
    returned = config(_KEY, 3)
    with pytest.raises(TrackingTargetUnavailableError):
        run_experiment(returned)
    assert run_experiment(returned)["p"].metric == pytest.approx(0.20)
    assert run_experiment(config(_KEY, 3, gated=True))["p"].metric == pytest.approx(0.20)
    assert earlier_counts == [1, 1, 1]
    assert len(launches.read_text().splitlines()) == 3
    assert len(set(launches.read_text().splitlines())) == 1


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
    created_at: str | None = None,
) -> tuple[ManualClock, list[str]]:
    """Answer each zero-based run lookup with ``answers(read)`` on a manual clock.

    Each lookup must come from a client built for it, as the real client caches the run.

    :param pytest.MonkeyPatch monkeypatch: Fixture that replaces the SDK client and clock.
    :param Callable[[int], Answer] answers: State, summary and last history step per read.
    :param tuple[Mapping[str, object], ...] rows: Rows available in unsampled history.
    :param Callable[[list[str]], None] | None on_scan: Callback for each history read.
    :param str | None created_at: The run's ISO ``created_at``, defaults to None for a run that serves none.
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
                **({} if created_at is None else {"created_at": created_at}),
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
    from phasesweep.evidence.wandb import WandbRunSupersededError

    _serve(monkeypatch, lambda _: ("finished", {"_step": 3, _KEY: 0.1, "_timestamp": 50.0}, 3))
    assert _poll(logged_before=51.0)["values"] == {_KEY: 0.1}
    with pytest.raises(WandbRunSupersededError):
        _poll(logged_before=50.0)


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


def test_a_reruns_run_that_never_moves_is_unanswered_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run whose summary never passes the rerun's launch gives no usable answer, only at the deadline."""
    from phasesweep.evidence.wandb import WandbPollTimeout

    left = {"_step": 5, _KEY: 0.5, "_timestamp": _LAUNCHED_AT - 60}
    clock, _ = _serve(monkeypatch, lambda read: ("finished", left, 5))
    with pytest.raises(WandbPollTimeout) as caught:
        _poll(logged_after=_LAUNCHED_AT)
    assert not caught.value.reached
    assert "earlier attempt's summary" in str(caught.value.last_error)
    assert clock.now == 240


@pytest.mark.parametrize("created_at", ["1970-01-01T00:16:41Z", "1970-01-01T00:16:41"])
def test_a_summary_only_objective_on_a_run_created_after_the_launch_is_read(
    monkeypatch: pytest.MonkeyPatch, created_at: str
) -> None:
    """A finished run created after the rerun launched, whose summary was set directly and has no ``_timestamp``, is read at once.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the SDK client and clock.
    :param str created_at: The run's creation time as W&B serves it, with and without an offset.
    """
    clock, paths = _serve(
        monkeypatch, lambda _: ("finished", {_KEY: 0.25}, None), created_at=created_at
    )
    assert _poll(logged_after=_LAUNCHED_AT)["values"] == {_KEY: 0.25}
    assert len(paths) == 1
    assert clock.now == 0


def test_a_summary_only_objective_on_a_run_created_before_the_launch_still_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished summary without a ``_timestamp`` on a run that predates the rerun may be the earlier attempt's, so the read waits and stays unanswered."""
    from phasesweep.evidence.wandb import WandbPollTimeout

    clock, _ = _serve(
        monkeypatch,
        lambda _: ("finished", {_KEY: 0.25}, None),
        created_at="1970-01-01T00:16:39Z",
    )
    with pytest.raises(WandbPollTimeout) as caught:
        _poll(logged_after=_LAUNCHED_AT)
    assert not caught.value.reached
    assert "earlier attempt's summary" in str(caught.value.last_error)
    assert clock.now == 240


def test_a_resumed_runs_earlier_summary_past_the_deadline_leaves_the_trial_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun whose run still serves the earlier attempt's summary at the deadline stays pending with what was seen, and a later read scores the rerun's summary."""
    served = {
        "left": ({"_step": 5, _KEY: 0.5, "_timestamp": _LAUNCHED_AT - 60}, 5),
        "logged": ({"_step": 11, _KEY: 0.1, "_timestamp": _LAUNCHED_AT + 30}, 11),
    }
    phase = ["left"]

    def answer(_read: int) -> Answer:
        summary, last_step = served[phase[0]]
        return "finished", summary, last_step

    _serve(monkeypatch, answer)
    monkeypatch.setattr("phasesweep.evidence.wandb.run_worker", _in_process_worker(tmp_path))
    extractor = WandbExtractor(
        type="wandb", entity="e", project="p", metric_key=_KEY, poll_seconds=5, timeout_seconds=60
    )

    def read() -> float:
        baseline = EvidenceBaseline(False, {}, _LAUNCHED_AT, extractor.destination)
        return run_extractor(replace(make_trial_context(tmp_path), baseline=baseline), extractor)

    with pytest.raises(EvidenceUnreadableError) as caught:
        read()
    assert type(caught.value) is EvidenceUnreadableError
    assert str(caught.value) == (
        "W&B gave no answer for https://api.wandb.ai/e/p/attempt-test before the evidence "
        f"deadline; required summary keys: ({_KEY!r},). Last observation: The run still shows "
        "only an earlier attempt's summary: it records no row logged since the attempt launched."
    )
    phase[0] = "logged"
    assert read() == 0.1


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
        ("eval/step", "rounded-summary", (0.02, 0.9)),
        ("_step", "native", (0.01, 0.99)),
        ("eval/step", "literal", (0.01, 0.99)),
        ("eval/step", "summary-only", (0.11, 0.88)),
        ("eval/step", "missing-axis", (0.02, 0.9)),
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
    if mode == "rounded-summary":
        native = (math.nextafter(native[0], 0), math.nextafter(native[1], 1))
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
    assert _poll(
        logged_after=1020.0,
        earlier_launches=(1000.0,),
        resume_axis=axis,
        required=(*keys, "eval/loss"),
    )["values"] == {**dict(zip(keys, expected, strict=True)), "eval/loss": 0.35}
    if axis == "_step":
        assert not readings
    if mode == "history-outage":
        assert clock.now == 5


def test_a_rerun_replaces_only_the_history_of_the_metric_it_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpoint loss survives an unrelated row while accuracy is reduced on its own steps."""
    rows = (
        {"_step": 0, "_timestamp": 990.0, "eval/step": 200, "eval/loss": 0.1, "eval/acc": 0.6},
        {"_step": 1, "_timestamp": 995.0, "eval/step": 250, "eval/acc": 0.99},
        {"_step": 2, "_timestamp": 996.0, "eval/step": 260, "eval/acc": 0.2},
        {"_step": 3, "_timestamp": 1010.0, "eval/step": 200, "train_runtime": 1.0},
        {"_step": 4, "_timestamp": 1015.0, "eval/step": 220, "eval/acc": 0.7},
        {"_step": 5, "_timestamp": 1020.0, "eval/step": 300, "eval/loss": 0.35},
    )
    summary = {
        "_step": 5,
        "_timestamp": 1020.0,
        _KEY: 0.1,
        "eval/acc.min": 0.2,
        "eval/acc.max": 0.99,
    }
    readings: list[list[str]] = []

    def on_scan(keys: list[str]) -> None:
        readings.append(keys)
        clock.sleep(0.7)

    clock, _ = _serve(monkeypatch, lambda _: ("finished", summary, 5), rows=rows, on_scan=on_scan)
    assert _poll(
        logged_after=_LAUNCHED_AT,
        resume_axis="eval/step",
        required=(_KEY, "eval/acc.min", "eval/acc.max"),
        deadline=5.0,
    )["values"] == {_KEY: 0.1, "eval/acc.min": 0.6, "eval/acc.max": 0.7}
    assert len(readings) == 7


def test_a_rerun_that_only_logs_teardown_keeps_the_final_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ending a retry at the final checkpoint keeps the objective already measured there."""
    rows = (
        {"_step": 0, "_timestamp": 990.0, "eval/step": 10, "eval/loss": 0.1},
        {"_step": 1, "_timestamp": 1010.0, "eval/step": 10, "train_runtime": 1.0},
    )
    summary = {"_step": 1, "_timestamp": 1010.0, _KEY: 0.1}
    _serve(monkeypatch, lambda _: ("finished", summary, 1), rows=rows)
    assert _poll(logged_after=_LAUNCHED_AT, resume_axis="eval/step")["values"] == {_KEY: 0.1}


def test_a_rerun_that_replaces_every_finite_observation_fails_the_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonfinite remeasurement cannot win with the finite observation it replaced."""
    rows = (
        {"_step": 0, "_timestamp": 990.0, "eval/step": 10, "eval/loss": 0.1},
        {"_step": 1, "_timestamp": 1010.0, "eval/step": 5, "eval/loss": float("nan")},
    )
    summary = {"_step": 1, "_timestamp": 1010.0, _KEY: 0.1}
    _serve(monkeypatch, lambda _: ("finished", summary, 1), rows=rows)
    with pytest.raises(ValueError, match="no finite observation in the retained history"):
        _poll(logged_after=_LAUNCHED_AT, resume_axis="eval/step")


def test_a_measurement_without_the_axis_sits_at_its_attempts_last_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An abandoned measurement logged without the axis is replaced, and the rerun's own is kept."""
    rows = (
        {"_step": 0, "_timestamp": 990.0, "eval/step": 100, "eval/loss": 0.5},
        {"_step": 1, "_timestamp": 991.0, "eval/step": 250, "train/loss": 0.3},
        {"_step": 2, "_timestamp": 992.0, "eval/loss": 0.05},
        {"_step": 3, "_timestamp": 1010.0, "eval/step": 200, "eval/loss": 0.4},
        {"_step": 4, "_timestamp": 1011.0, "eval/loss": 0.45},
    )
    summary = {"_step": 4, "_timestamp": 1011.0, _KEY: 0.05}
    _serve(monkeypatch, lambda _: ("finished", summary, 4), rows=rows)
    assert _poll(logged_after=_LAUNCHED_AT, resume_axis="eval/step")["values"] == {_KEY: 0.4}


def test_an_attempt_that_never_logs_the_axis_fails_the_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without any progress in an attempt, its measurements cannot be retained or replaced."""
    rows = (
        {"_step": 0, "_timestamp": 990.0, "eval/loss": 0.05},
        {"_step": 1, "_timestamp": 1010.0, "eval/step": 5, "eval/loss": 0.4},
    )
    summary = {"_step": 1, "_timestamp": 1010.0, _KEY: 0.05}
    _serve(monkeypatch, lambda _: ("finished", summary, 1), rows=rows)
    with pytest.raises(ValueError, match="never logged 'eval/step'"):
        _poll(logged_after=_LAUNCHED_AT, resume_axis="eval/step")


def test_wandb_and_mlflow_retain_the_same_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both trackers keep an earlier attempt's values only below a later attempt's lowest position."""
    attempts = (((100, 0.5), (200, 0.1)), ((300, 0.3), (150, 0.9)))
    mlflow_history = merge_histories(
        [
            [{"step": step, "timestamp": 10 * n + i, "value": v} for i, (step, v) in enumerate(a)]
            for n, a in enumerate(attempts)
        ]
    )
    expected = select_observation(
        mlflow_history, MlflowExtractor(type="mlflow", metric_key="eval/loss", select="min")
    )["value"]
    rows = tuple(
        {"_step": 2 * n + i, "_timestamp": 990.0 + 20 * n + i, "eval/step": step, "eval/loss": v}
        for n, a in enumerate(attempts)
        for i, (step, v) in enumerate(a)
    )
    summary = {"_step": 3, "_timestamp": 1011.0, _KEY: 0.1}
    _serve(monkeypatch, lambda _: ("finished", summary, 3), rows=rows)
    assert _poll(logged_after=_LAUNCHED_AT, resume_axis="eval/step")["values"] == {_KEY: expected}
    assert expected == 0.3


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
        attach_wandb_transport(self, Transport())

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


def _in_process_worker(tmp_path: Path) -> Callable[..., object]:
    """Build a stand-in for the supervised worker that runs the reader in this process.

    :param Path tmp_path: Directory for each request and response file.
    :return Callable[..., object]: A ``run_worker`` replacement.
    """
    from phasesweep.evidence import wandb as wandb_evidence

    calls = iter(range(10**6))

    def run(request, _worker_path, _backend, **_):
        directory = tmp_path / f"worker-{next(calls)}"
        directory.mkdir(mode=0o700)
        request_path, response_path = directory / "request.json", directory / "response.json"
        request_path.write_text(json.dumps(dict(request)))
        assert wandb_evidence.main([str(request_path), str(response_path)]) == 0
        return (
            SimpleNamespace(timed_out=False, return_code=0),
            json.loads(response_path.read_text()),
            None,
        )

    return run


def test_the_missing_key_report_crosses_the_worker_with_its_wait(tmp_path, monkeypatch) -> None:
    """The worker's missing-keys reply carries how long it polled, and the trial's failure says so."""
    _serve(monkeypatch, lambda read: ("finished", {"_step": 3, "eval/loss": 0.1}, 3))
    monkeypatch.setattr("phasesweep.evidence.wandb.run_worker", _in_process_worker(tmp_path))
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


@pytest.mark.parametrize("capped", [False, True], ids=["source-timeout", "phase-deadline"])
def test_a_deadline_verdict_cut_short_by_the_phase_is_an_interruption(
    tmp_path, monkeypatch, capped: bool
) -> None:
    """A missing-key verdict reached at a phase deadline shorter than the source's timeout interrupts the trial instead of failing it."""
    import time

    from phasesweep.evidence import wandb as wandb_evidence

    def poll(**kwargs: object) -> dict[str, object]:
        raise wandb_evidence.WandbMissingKeysError("opener", [_KEY], [], 9.0)

    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", poll)
    extractor = WandbExtractor(type="wandb", entity="e", project="p", metric_key=_KEY)
    deadline = time.monotonic() + 10 if capped else None
    with pytest.raises(ExtractorError) as caught:
        run_extractor(make_trial_context(tmp_path), extractor, deadline=deadline)
    assert type(caught.value) is (DeadlineExceededError if capped else ExtractorError)
    assert re.search(r"after 9 s", str(caught.value))


def test_a_missing_run_names_the_project_the_attempt_logged_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The missing-run message names the destination recorded at launch, not the configured project a reading study has since changed to."""
    from phasesweep.evidence import wandb as wandb_evidence

    def poll(**kwargs: object) -> dict[str, object]:
        assert kwargs["project"] == "recorded"
        raise wandb_evidence.WandbRunMissingError("opener")

    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", poll)
    extractor = WandbExtractor(type="wandb", entity="e", project="configured", metric_key=_KEY)
    recorded = extractor.destination._replace(project="recorded")
    context = replace(
        make_trial_context(tmp_path), baseline=EvidenceBaseline(False, {}, None, recorded)
    )
    with pytest.raises(ExtractorError) as caught:
        run_extractor(context, extractor)
    assert str(caught.value).startswith(
        "No W&B run 'attempt-test' was found in e/recorded at the evidence deadline."
    )
    assert "configured" not in str(caught.value)
