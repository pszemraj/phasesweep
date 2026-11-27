"""W&B serves a finished run's summary late; the reader waits for it until the deadline and never reads a half-ingested one.

The sequences follow what the public API served on a live project: a run reported ``finished``
with its whole history at once and its summary a minute later.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
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
    ExtractorError,
    NothingToScoreError,
    run_extractor,
)
from phasesweep.evidence.mlflow import merge_histories, select_observation
from phasesweep.evidence.models import MlflowExtractor, WandbExtractor, WandbSummaryRequiredGate
from tests.conftest import make_experiment, make_trial_context, phase_trials, write_trainer
from tests.tracking_helpers import (
    ManualClock,
    attach_wandb_transport,
    serve_wandb_runs,
    switchable_wandb,
    wandb_objective_experiment,
    wandb_run,
)

_KEY = "eval/loss.min"


def test_sibling_retry_replaces_tracker_history_before_recovery_and_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun's read names every earlier checkpoint writer's run across reading studies, and its native min merges their histories, in the live read, in recovery and in a re-read."""
    from phasesweep.evidence import evaluation, wandb

    launches, history = tmp_path / "launches", tmp_path / "history"
    history.mkdir()
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import json, os
        from pathlib import Path
        launches, history = Path({str(launches)!r}), Path({str(history)!r})
        attempt = len(launches.read_text().splitlines()) if launches.exists() else 0
        with launches.open("a") as log:
            log.write(os.environ["WANDB_RUN_ID"] + "\\n")
        points = [[(100, 0.20), (300, 0.01)], [(200, 0.40)], [(400, 0.90)]][attempt]
        rows = [{{"_step": step, "eval/step": progress, "eval/loss": loss}}
                for step, (progress, loss) in enumerate(points)]
        (history / (os.environ["WANDB_RUN_ID"] + ".json")).write_text(json.dumps(rows))
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
        earlier, run_id = options["earlier"], options["run_id"]
        assert isinstance(earlier, list) and isinstance(run_id, str)
        earlier_counts.append(len(earlier))
        if len(earlier_counts) == 1:
            raise wandb.WandbPollTimeout(run_id, 2, reached=False)
        runs = {}
        for path in history.glob("*.json"):
            rows = tuple(json.loads(path.read_text()))
            losses = [row["eval/loss"] for row in rows]
            summary = {"_step": len(rows) - 1, _KEY: min(losses), "eval/loss.max": max(losses)}
            runs[path.stem] = wandb_run("finished", summary, rows)
        with monkeypatch.context() as sdk:
            serve_wandb_runs(sdk, runs)
            return wandb._poll_wandb_summary(
                base_url="https://example.test",
                entity="e",
                project="p",
                run_id=run_id,
                poll_seconds=5,
                timeout_seconds=240,
                required_keys=options["required_keys"],
                presence_keys=options.get("presence_keys", ()),
                objective_key=_KEY,
                earlier=earlier,
                evaluation_axis="eval/step",
            )

    monkeypatch.setattr(evaluation, "poll_wandb_summary", capture)
    monkeypatch.setattr(evaluation, "locate_wandb_position", lambda **_: 100)
    # The sibling's failed retry spends the second trial, so the third launches.
    returned = config(_KEY, 3)
    with pytest.raises(TrackingTargetUnavailableError):
        run_experiment(returned)
    assert run_experiment(returned)["p"].metric == pytest.approx(0.20)
    assert run_experiment(config(_KEY, 3, gated=True))["p"].metric == pytest.approx(0.20)
    assert earlier_counts == [2, 2, 2]
    assert len(set(launches.read_text().splitlines())) == 3


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


def _poll(*, required: tuple[str, ...] = (_KEY,), **kwargs: object) -> dict[str, object]:
    """Run the in-process summary poll of a first attempt every 5 s with a 240 s budget.

    :param tuple[str, ...] required: Required numeric keys, defaults to the objective.
    :param object kwargs: Further poll arguments.
    :return dict[str, object]: The capture.
    """
    from phasesweep.evidence.wandb import _poll_wandb_summary

    return _poll_wandb_summary(
        base_url="https://example.test",
        entity="e",
        project="p",
        run_id="rerun",
        poll_seconds=5,
        timeout_seconds=240,
        required_keys=required,
        **kwargs,
    )


def _poll_rerun(*earlier: str, **kwargs: object) -> dict[str, object]:
    """Poll the rerun's run ``e/p/rerun`` with its trial's earlier runs, oldest first.

    :param str earlier: The earlier runs' IDs.
    :param object kwargs: Further poll arguments.
    :return dict[str, object]: The capture.
    """
    addresses = [
        {"base_url": "https://example.test", "entity": "e", "project": "p", "run_id": run_id}
        for run_id in earlier
    ]
    return _poll(objective_key=_KEY, earlier=addresses, **kwargs)


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
                state=state,
                summary_metrics=summary,
                history_keys=_history_index(last_step),
                history=lambda **_: [],
                scan_history=lambda **_: [],
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


@pytest.mark.parametrize("state", ["killed", "running"], ids=["closed", "open"])
def test_a_reruns_verdict_stands_when_no_lookup_completes_near_the_deadline(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    """The latest answered poll decides a rerun's verdict at the deadline even when it came long before it, because the next client took past the deadline to build: the earlier run's summary is captured, or the attempt left nothing to score.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the SDK client and clock.
    :param str state: The earlier run's state, closed or not.
    """
    from phasesweep.evidence.wandb import WandbNothingToScoreError

    public = pytest.importorskip("wandb.apis.public")
    opener = wandb_run(state, {"_step": 0, _KEY: 0.5}, ({"_step": 0, "eval/loss": 0.5},))
    clock, paths = serve_wandb_runs(monkeypatch, {"opener": opener})
    served = public.Api
    built: list[object] = []

    def slow_second_client(**kwargs: object) -> object:
        client = served(**kwargs)
        built.append(client)
        if len(built) == 2:
            clock.now += 1_000
        return client

    monkeypatch.setattr(public, "Api", slow_second_client)
    if state == "killed":
        capture = _poll_rerun("opener")
        assert capture["values"] == {_KEY: 0.5} and capture["earlier_attempt"] == "opener"
    else:
        with pytest.raises(WandbNothingToScoreError):
            _poll_rerun("opener")
    assert paths == ["e/p/rerun", "e/p/opener"]


def test_a_waiting_verdict_reads_the_earlier_histories_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rerun that logged nothing is scored from its trial's newest earlier run merged with the one before it, whose histories are read on the first poll only while the read waits for the deadline."""
    scanned_at: list[float] = []

    def record(_keys: list[str]) -> None:
        scanned_at.append(clock.now)

    runs = {
        "opener": wandb_run(
            "crashed",
            {"_step": 1, _KEY: 0.05},
            (
                {"_step": 0, "eval/step": 10, "eval/loss": 0.3},
                {"_step": 1, "eval/step": 20, "eval/loss": 0.05},
            ),
            record,
        ),
        "second": wandb_run(
            "killed",
            {"_step": 0, _KEY: 0.2},
            ({"_step": 0, "eval/step": 15, "eval/loss": 0.2},),
            record,
        ),
    }
    clock, paths = serve_wandb_runs(monkeypatch, runs)
    capture = _poll_rerun("opener", "second", evaluation_axis="eval/step")
    assert capture["values"] == {_KEY: 0.2} and capture["earlier_attempt"] == "second"
    assert len(paths) == 48 * 3
    assert scanned_at and set(scanned_at) == {0.0}


def test_a_reruns_own_crash_fails_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rerun's own run that crashed is its own failure, whatever its trial's earlier runs hold."""
    from phasesweep.evidence.wandb import WandbRunTerminalError

    runs = {
        "opener": wandb_run("finished", {"_step": 0, _KEY: 0.5}, ({"_step": 0},)),
        "rerun": wandb_run("crashed", {"_step": 0, _KEY: 0.25}, ({"_step": 0},)),
    }
    serve_wandb_runs(monkeypatch, runs)
    with pytest.raises(WandbRunTerminalError):
        _poll_rerun("opener")


def _extrema_runs(mode: str, on_read: Callable[[list[str]], None]) -> dict[str, object]:
    """Build an opener, a second attempt and a rerun from their checkpoints, logging ``eval/loss``.

    The second attempt resumed at eval step 30 and the rerun at 50, so merged by the axis the
    trial keeps the opener's 0.4 and 0.2, the second's 0.3, 0.9 and 0.02, and the rerun's 0.4
    and 0.35, while the runs' summaries also cover the opener's abandoned 0.01 and 0.99.

    :param str mode: ``native``, ``rounded-summary``, ``literal``, ``summary-only`` or
        ``missing-axis``.
    :param Callable[[list[str]], None] on_read: Called with the keys of each history read.
    :return dict[str, object]: The runs by ID.
    """
    series = {
        "opener": [(10, 0.4), (20, 0.2), (30, 0.01), (30, 0.99)],
        "second": [(30, 0.3), (40, 0.9), (40, 0.02)],
        "rerun": [(50, 0.4), (60, 0.35)],
    }
    runs: dict[str, object] = {}
    for run_id, points in series.items():
        rows: list[dict[str, object]] = [
            {"_step": step, "eval/step": progress, "eval/loss": loss}
            for step, (progress, loss) in enumerate(points)
        ]
        losses = [loss for _, loss in points]
        native = (min(losses), max(losses))
        if run_id == "rerun":
            if mode == "literal":
                rows.append({"_step": len(rows), _KEY: 0.01, "eval/loss.max": 0.99})
                native = (0.01, 0.99)
            elif mode == "summary-only":
                native = (0.11, 0.88)
            elif mode == "rounded-summary":
                native = (math.nextafter(native[0], 0), math.nextafter(native[1], 1))
        if run_id == "opener" and mode == "missing-axis":
            rows[0].pop("eval/step")
        summary = {"_step": len(rows) - 1, _KEY: native[0], "eval/loss.max": native[1]}
        state = "crashed" if run_id != "rerun" else "finished"
        runs[run_id] = wandb_run(state, summary, tuple(rows), on_read)
    return runs


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
def test_native_extrema_exclude_the_history_a_rerun_replaced(
    monkeypatch: pytest.MonkeyPatch, axis: str, mode: str, expected: tuple[float, float]
) -> None:
    """With the axis, native min and max reduce over the trial's runs merged by it, keeping earlier valid history and excluding what a later attempt replaced; without it they reduce over the runs' summaries. A literal dotted series and a summary value the rerun's rows do not reproduce keep the summary, a row without the axis sits at its run's progress, and a history outage is read again.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the SDK client and clock.
    :param str axis: The evaluation axis.
    :param str mode: How the runs log.
    :param tuple[float, float] expected: The min and max.
    """
    readings: list[list[str]] = []

    def on_scan(keys: list[str]) -> None:
        readings.append(keys)
        if mode == "history-outage" and len(readings) == 1:
            raise ConnectionError("history temporarily unavailable")

    runs = _extrema_runs("native" if mode == "history-outage" else mode, on_scan)
    clock, _ = serve_wandb_runs(monkeypatch, runs)
    keys = (_KEY, "eval/loss.max")
    assert _poll_rerun("opener", "second", required=keys, evaluation_axis=axis)["values"] == dict(
        zip(keys, expected, strict=True)
    )
    if axis == "_step":
        assert {tuple(keys) for keys in readings} == {(_KEY, "_step"), ("eval/loss.max", "_step")}
    if mode == "history-outage":
        assert clock.now == 5


def test_a_rerun_replaces_only_the_history_of_the_metric_it_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpoint loss survives the rerun's unrelated row while accuracy is merged on its own positions."""
    opener = wandb_run(
        "crashed",
        {"_step": 2, _KEY: 0.1, "eval/acc.min": 0.2, "eval/acc.max": 0.99},
        (
            {"_step": 0, "eval/step": 200, "eval/loss": 0.1, "eval/acc": 0.6},
            {"_step": 1, "eval/step": 250, "eval/acc": 0.99},
            {"_step": 2, "eval/step": 260, "eval/acc": 0.2},
        ),
    )
    rerun = wandb_run(
        "finished",
        {"_step": 2, _KEY: 0.35, "eval/acc.min": 0.7, "eval/acc.max": 0.7},
        (
            {"_step": 0, "eval/step": 200, "train_runtime": 1.0},
            {"_step": 1, "eval/step": 220, "eval/acc": 0.7},
            {"_step": 2, "eval/step": 300, "eval/loss": 0.35},
        ),
    )
    serve_wandb_runs(monkeypatch, {"opener": opener, "rerun": rerun})
    assert _poll_rerun(
        "opener", required=(_KEY, "eval/acc.min", "eval/acc.max"), evaluation_axis="eval/step"
    )["values"] == {_KEY: 0.1, "eval/acc.min": 0.6, "eval/acc.max": 0.7}


def test_a_measurement_without_the_axis_sits_at_its_runs_last_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An abandoned measurement logged without the axis is replaced, and the rerun's own is kept."""
    opener = wandb_run(
        "crashed",
        {"_step": 2, _KEY: 0.05},
        (
            {"_step": 0, "eval/step": 100, "eval/loss": 0.5},
            {"_step": 1, "eval/step": 250, "train/loss": 0.3},
            {"_step": 2, "eval/loss": 0.05},
        ),
    )
    rerun = wandb_run(
        "finished",
        {"_step": 1, _KEY: 0.4},
        ({"_step": 0, "eval/step": 200, "eval/loss": 0.4}, {"_step": 1, "eval/loss": 0.45}),
    )
    serve_wandb_runs(monkeypatch, {"opener": opener, "rerun": rerun})
    assert _poll_rerun("opener", evaluation_axis="eval/step")["values"] == {_KEY: 0.4}


def test_a_run_that_never_logs_the_axis_fails_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without any progress in a run, its measurements cannot be retained or replaced."""
    opener = wandb_run("crashed", {"_step": 0, _KEY: 0.05}, ({"_step": 0, "eval/loss": 0.05},))
    rerun = wandb_run(
        "finished", {"_step": 0, _KEY: 0.4}, ({"_step": 0, "eval/step": 5, "eval/loss": 0.4},)
    )
    serve_wandb_runs(monkeypatch, {"opener": opener, "rerun": rerun})
    with pytest.raises(ValueError, match="never logged 'eval/step'"):
        _poll_rerun("opener", evaluation_axis="eval/step")


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
    runs = {
        run_id: wandb_run(
            "finished",
            {"_step": 1, _KEY: min(v for _, v in points)},
            tuple(
                {"_step": i, "eval/step": step, "eval/loss": v}
                for i, (step, v) in enumerate(points)
            ),
        )
        for run_id, points in zip(("opener", "rerun"), attempts, strict=True)
    }
    serve_wandb_runs(monkeypatch, runs)
    assert _poll_rerun("opener", evaluation_axis="eval/step")["values"] == {_KEY: expected}
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
        f"W&B run https://api.wandb.ai/e/p/{ctx.attempt_id} finished without required "
        f"summary key(s) ['{_KEY}'] after 238 s; available summary keys: ['_step', 'eval/loss']."
    )


@pytest.mark.parametrize(
    "verdict", ["missing-keys", "nothing-to-score", "earlier-run", "borrowed-key"]
)
@pytest.mark.parametrize("capped", [False, True], ids=["source-timeout", "phase-deadline"])
def test_a_deadline_verdict_cut_short_by_the_phase_is_an_interruption(
    tmp_path, monkeypatch, capped: bool, verdict: str
) -> None:
    """A verdict reached at the poll's deadline (missing keys, a rerun that left nothing to score, a rerun scored from an earlier run, or a key taken from an earlier run) interrupts the trial instead when a phase deadline shorter than the source's timeout cut the poll short."""
    import time

    from phasesweep.evidence import wandb as wandb_evidence

    def poll(**kwargs: object) -> dict[str, object]:
        if verdict == "missing-keys":
            raise wandb_evidence.WandbMissingKeysError("rerun", [_KEY], [], 9.0)
        if verdict == "nothing-to-score":
            raise wandb_evidence.WandbNothingToScoreError("rerun", "was not found")
        source = {"earlier_attempt": "opener"} if verdict == "earlier-run" else {}
        return {"values": {_KEY: 0.5}, "present_keys": [], "deadline_verdict": True, **source}

    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", poll)
    extractor = WandbExtractor(type="wandb", entity="e", project="p", metric_key=_KEY)
    deadline = time.monotonic() + 10 if capped else None
    context = make_trial_context(tmp_path)
    if verdict in {"earlier-run", "borrowed-key"} and not capped:
        assert run_extractor(context, extractor, deadline=deadline) == 0.5
        earlier = ["wandb:opener"] if verdict == "earlier-run" else []
        assert context.earlier_files == earlier
        assert not {"earlier_attempt", "deadline_verdict"} & set(context.wandb_capture)
        return
    with pytest.raises(ExtractorError) as caught:
        run_extractor(context, extractor, deadline=deadline)
    unscored = NothingToScoreError if verdict == "nothing-to-score" else ExtractorError
    assert type(caught.value) is (DeadlineExceededError if capped else unscored)
    if verdict == "missing-keys":
        assert re.search(r"after 9 s", str(caught.value))


def test_a_missing_run_names_the_project_the_attempt_logged_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The missing-run message names the destination recorded at launch, not the configured project a reading study has since changed to."""
    from phasesweep.evidence import wandb as wandb_evidence

    def poll(**kwargs: object) -> dict[str, object]:
        assert kwargs["project"] == "recorded"
        raise wandb_evidence.WandbRunMissingError("attempt-test")

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
