"""Every attempt of a trial logs to a W&B run of its own, and a rerun's result is assembled from its trial's runs."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import phasesweep.runtime.reaper as reaper
from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric
from phasesweep.engine.evidence import _verify_winner_evidence
from phasesweep.engine.paths import _attempts_dir
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    EARLIER_ATTEMPT_RESULT_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    RERUN_OF_ATTR,
    _trial_attempt_dir,
)
from phasesweep.evidence.baseline import EVIDENCE_BASELINE_FILE, EVIDENCE_BASELINE_SCHEMA_VERSION
from phasesweep.evidence.evaluation import locate_objective_position, run_extractor
from phasesweep.evidence.models import WandbExtractor, WandbSummaryRequiredGate
from tests.conftest import (
    make_experiment,
    make_trial_context,
    phase_trials,
    write_experiment_config,
    write_trainer,
)
from tests.recovery_helpers import runner_once
from tests.tracking_helpers import serve_wandb_runs as _serve
from tests.tracking_helpers import wandb_run as _run

_BASE_URL = "https://example.test"


def _poll(*run_ids: str, required: tuple[str, ...] = ("x",), **kwargs: object) -> dict[str, object]:
    """Poll the rerun's run ``e/p/rerun`` every 5 s for 240 s, its trial's earlier runs named in order.

    :param str run_ids: Earlier runs, oldest first.
    :param tuple[str, ...] required: Required numeric keys, defaults to the objective ``x``.
    :param object kwargs: Further poll arguments; the objective key defaults to ``x``.
    :return dict[str, object]: The capture.
    """
    from phasesweep.evidence.wandb import _poll_wandb_summary

    earlier = [
        {"base_url": _BASE_URL, "entity": "e", "project": "p", "run_id": run_id}
        for run_id in run_ids
    ]
    return _poll_wandb_summary(
        base_url=_BASE_URL,
        entity="e",
        project="p",
        run_id="rerun",
        poll_seconds=5,
        timeout_seconds=240,
        required_keys=required,
        objective_key=kwargs.pop("objective_key", "x"),
        earlier=earlier,
        **kwargs,
    )


def _rows(*values: float, axis: tuple[int, ...] = ()) -> tuple[dict[str, int | float], ...]:
    """Build history rows logging ``eval/loss``, at ``eval/step`` positions when given.

    :param float values: The losses, one row each.
    :param tuple[int, ...] axis: Their ``eval/step`` positions, defaults to none.
    :return tuple[dict[str, int | float], ...]: The rows.
    """
    rows: list[dict[str, int | float]] = [
        {"_step": step, "eval/loss": v} for step, v in enumerate(values)
    ]
    for row, position in zip(rows, axis, strict=False):
        row["eval/step"] = position
    return tuple(rows)


def test_a_first_attempt_reads_only_its_own_run_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no earlier runs, a finished summary holding a native min is read at once from the attempt's own run, without reading history or any other run."""
    reads: list[list[str]] = []
    rows = _rows(0.3, 0.1, axis=(10, 20))
    own = _run("finished", {"_step": 1, "eval/loss.min": 0.1}, rows, reads.append)
    clock, paths = _serve(monkeypatch, {"rerun": own})
    capture = _poll(
        required=("eval/loss.min",), objective_key="eval/loss.min", evaluation_axis="eval/step"
    )
    assert capture["values"] == {"eval/loss.min": 0.1} and "deadline_verdict" not in capture
    assert paths == ["e/p/rerun"] and reads == [] and clock.now == 0


def test_a_rerun_that_logged_the_objective_takes_a_key_its_summary_lacks_from_an_earlier_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rerun's finished run holding the objective scores it over an earlier run's value; a required key its summary still lacks at the deadline comes from the newest earlier run holding it, even a crashed one, and one no run holds is a missing key."""
    from phasesweep.evidence.wandb import WandbMissingKeysError

    runs = {
        "opener": _run("finished", {"_step": 0, "x": 0.9, "params": 5.0}, _rows(0.9)),
        "second": _run("crashed", {"_step": 1, "x": 0.5, "params": 7.0}, _rows(0.6, 0.5)),
        "rerun": _run("finished", {"_step": 1, "x": 0.25}, _rows(0.3, 0.25)),
    }
    clock, _ = _serve(monkeypatch, runs)
    capture = _poll("opener", "second", required=("params", "x"))
    assert capture["values"] == {"params": 7.0, "x": 0.25}
    assert capture["deadline_verdict"] is True and "earlier_attempt" not in capture
    assert clock.now == 240
    with pytest.raises(WandbMissingKeysError) as caught:
        _poll("opener", "second", required=("memory", "x"))
    assert caught.value.missing == ["memory"]


@pytest.mark.parametrize(
    ("axis", "expected"),
    [("eval/step", (0.05, 0.4)), ("_step", (0.01, 0.4))],
    ids=["merged-history", "summary-values"],
)
def test_native_extrema_reduce_over_the_trial_runs(
    monkeypatch: pytest.MonkeyPatch, axis: str, expected: tuple[float, float]
) -> None:
    """A rerun from the checkpoint at eval step 20 replaces the earlier run's history from step 25 on, so with the axis its min and max come from the merged series without the abandoned 0.01 at step 30; without the axis they reduce over the runs' summary values, which keep it.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the SDK client and clock.
    :param str axis: The evaluation axis.
    :param tuple[float, float] expected: The min and max.
    """
    keys = ("eval/loss.max", "eval/loss.min")
    runs = {
        "opener": _run(
            "killed",
            {"_step": 2, "eval/loss.min": 0.01, "eval/loss.max": 0.4},
            _rows(0.4, 0.05, 0.01, axis=(10, 20, 30)),
        ),
        "rerun": _run(
            "finished",
            {"_step": 2, "eval/loss.min": 0.1, "eval/loss.max": 0.2},
            _rows(0.2, 0.1, 0.15, axis=(25, 30, 40)),
        ),
    }
    clock, paths = _serve(monkeypatch, runs)
    capture = _poll("opener", required=keys, objective_key="eval/loss.min", evaluation_axis=axis)
    assert capture["values"] == {"eval/loss.max": expected[1], "eval/loss.min": expected[0]}
    # Settled on the first poll: one lookup of each run, and no wait.
    assert "deadline_verdict" not in capture and clock.now == 0
    assert paths == ["e/p/rerun", "e/p/opener"]


@pytest.mark.parametrize(("axis", "expected"), [("eval/step", (0.05, 0.8)), ("_step", (0.01, 1.0))])
def test_borrowed_extrema_reduce_across_attempts(
    monkeypatch: pytest.MonkeyPatch, axis: str, expected: tuple[float, float]
) -> None:
    """A borrowed constraint uses all retained rows on a trainer axis, or all summaries on _step."""
    low, high = "eval/loss.min", "eval/loss.max"
    runs = {
        "opener": _run(
            "killed",
            {"_step": 3, low: 0.01, high: 1.0},
            _rows(0.4, 0.05, 0.01, 1.0, axis=(10, 20, 30, 35)),
        ),
        "second": _run("killed", {"_step": 1, low: 0.2, high: 0.3}, _rows(0.2, 0.3, axis=(25, 40))),
        "rerun": _run("finished", {"_step": 0, "x": 0.7}, _rows(0.8, axis=(35,))),
    }
    _serve(monkeypatch, runs)
    capture = _poll("opener", "second", required=("x", low, high), evaluation_axis=axis)
    assert capture["values"] == {"x": 0.7, low: expected[0], high: expected[1]}
    assert capture["deadline_verdict"] is True


def test_a_borrowed_literal_dotted_key_keeps_its_summary_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dotted series logged by the key's source is a literal even when the rerun omits it."""
    key = "cost.max"
    _serve(
        monkeypatch,
        {
            "opener": _run("killed", {key: 100}),
            "second": _run("killed", {"_step": 0, key: 30}, ({"_step": 0, key: 30},)),
            "rerun": _run("finished", {"x": 1.0}),
        },
    )
    assert _poll("opener", "second", required=("x", key))["values"] == {"x": 1.0, key: 30}


def test_summary_extrema_with_only_one_measured_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """An initial attempt with no extrema leaves the rerun's minimum and maximum intact."""
    low, high = "eval/loss.min", "eval/loss.max"
    _serve(
        monkeypatch,
        {
            "opener": _run("killed", {"params": 3}),
            "rerun": _run("finished", {"_step": 1, low: 0.2, high: 0.8}, _rows(0.2, 0.8)),
        },
    )
    result = _poll("opener", required=(low, high), objective_key=low)
    assert result["values"] == {low: 0.2, high: 0.8}
    assert "deadline_verdict" not in result


def test_borrowed_extremum_with_no_retained_finite_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rollback replacing the whole series with NaN cannot score from the abandoned prefix."""
    key = "eval/loss.min"
    _serve(
        monkeypatch,
        {
            "opener": _run("killed", {"_step": 0, key: 0.1}, _rows(0.1, axis=(10,))),
            "rerun": _run("finished", {"_step": 0, "x": 0.2}, _rows(float("nan"), axis=(10,))),
        },
    )
    with pytest.raises(ValueError, match="no finite observation in the retained history"):
        _poll("opener", required=("x", key), evaluation_axis="eval/step")


def _late_runs(
    opener_summary: dict[str, object], on_read: Callable[[list[str]], None] | None = None
) -> tuple[dict[str, SimpleNamespace], list[dict[str, int | float]]]:
    """Build a rerun from the checkpoint at eval step 20 and its opener, whose history starts unserved.

    The opener's history index already reaches its last row, and its rows are served once
    they are appended to the returned list.

    :param dict[str, object] opener_summary: The summary W&B serves for the opener.
    :param Callable[[list[str]], None] | None on_read: Called before each opener history read.
    :return tuple[dict[str, SimpleNamespace], list[dict[str, int | float]]]: The runs by ID,
        and the opener's served rows.
    """
    served: list[dict[str, int | float]] = []
    opener = _run("killed", opener_summary, served, on_read)
    opener.history_keys = {"lastStep": 2, "keys": {"_step": {}}}
    rerun = _run(
        "finished", {"_step": 2, "eval/loss.min": 0.1}, _rows(0.2, 0.1, 0.15, axis=(25, 30, 40))
    )
    return {"opener": opener, "rerun": rerun}, served


def test_an_earlier_history_served_late_is_waited_for_and_merged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While W&B serves no row of an earlier run whose summary holds the native min, the rerun's own summary value is not returned; once a later poll sees the rows, the merged minimum is returned at once."""

    def serve_late(_keys: list[str]) -> None:
        if clock.now >= 10 and not served:
            served.extend(_rows(0.4, 0.05, 0.01, axis=(10, 20, 30)))

    runs, served = _late_runs({"_step": 2, "eval/loss.min": 0.01}, serve_late)
    clock, _ = _serve(monkeypatch, runs)
    capture = _poll(
        "opener",
        required=("eval/loss.min",),
        objective_key="eval/loss.min",
        evaluation_axis="eval/step",
    )
    assert capture["values"] == {"eval/loss.min": 0.05} and "deadline_verdict" not in capture
    assert clock.now == 10


@pytest.mark.parametrize("empty_rerun", [False, True])
def test_an_unsettled_closed_earlier_run_stays_pending_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch, empty_rerun: bool
) -> None:
    """A closed run's stale summary cannot score either a measured or an empty rerun."""
    from phasesweep.evidence.wandb import WandbPollTimeout

    runs, served = _late_runs({"_step": 0, "eval/loss.min": 0.4})
    served.extend(_rows(0.4, 0.05, 0.01, axis=(10, 20, 30)))
    if empty_rerun:
        runs["rerun"] = _run("finished", {})
    clock, _ = _serve(monkeypatch, runs)
    with pytest.raises(WandbPollTimeout) as caught:
        _poll("opener", required=("eval/loss.min",), objective_key="eval/loss.min")
    assert "history index" in str(caught.value.last_error)
    assert not caught.value.reached and clock.now == 240


@pytest.mark.parametrize("state", ["killed", "crashed"])
def test_a_positive_row_count_before_the_first_indexed_step(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    """A closed opener's counted row can change the winner, while a crashed opener may never serve it."""
    from phasesweep.evidence.wandb import WandbPollTimeout

    key = "eval/loss.min"
    opener = _run(state, {"_wandb": {"runtime": 5}})
    opener.historyLineCount = 1
    runs = {
        "opener": opener,
        "rerun": _run("finished", {"_step": 0, key: 0.9}, _rows(0.9, axis=(8,))),
    }
    _serve(monkeypatch, runs)
    if state == "killed":
        with pytest.raises(WandbPollTimeout) as caught:
            _poll("opener", required=(key,), objective_key=key, evaluation_axis="eval/step")
        assert not caught.value.reached
        runs["opener"] = _run(state, {"_step": 0, key: 0.1}, _rows(0.1, axis=(1,)))
    capture = _poll("opener", required=(key,), objective_key=key, evaluation_axis="eval/step")
    assert capture["values"] == {key: 0.1 if state == "killed" else 0.9}


def test_a_closed_earlier_runs_unserved_aggregate_remains_pending_until_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with matching summary and index steps, delayed keyed rows can change the winner."""
    from phasesweep.evidence.wandb import WandbPollTimeout

    runs, served = _late_runs({"_step": 2, "eval/loss.min": 0.01})
    _serve(monkeypatch, runs)
    key = "eval/loss.min"
    with pytest.raises(WandbPollTimeout) as caught:
        _poll("opener", required=(key,), objective_key=key, evaluation_axis="eval/step")
    assert key in str(caught.value.last_error)
    assert not caught.value.reached
    served.extend(_rows(0.4, 0.05, 0.01, axis=(10, 20, 30)))
    recovered = _poll("opener", required=(key,), objective_key=key, evaluation_axis="eval/step")
    assert recovered["values"] == {key: 0.05}


def test_matching_summary_and_index_must_account_for_the_counted_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closed opener may still serve just its first row. Its final W&B steps can be sparse."""
    from phasesweep.evidence.wandb import WandbPollTimeout

    key = "eval/loss.min"
    first, second = _rows(0.8, 0.1, axis=(1, 2))
    second["_step"] = 10
    opener = _run("killed", {"_step": 0, key: 0.8}, (first,))
    opener.historyLineCount = 2
    runs = {
        "opener": opener,
        "rerun": _run("finished", {"_step": 0, key: 0.9}, _rows(0.9, axis=(8,))),
    }
    _serve(monkeypatch, runs)
    with pytest.raises(WandbPollTimeout) as caught:
        _poll("opener", required=(key,), objective_key=key, evaluation_axis="eval/step")
    assert not caught.value.reached

    runs["opener"] = _run("killed", {"_step": 10, key: 0.1}, (first, second))
    capture = _poll("opener", required=(key,), objective_key=key, evaluation_axis="eval/step")
    assert capture["values"] == {key: 0.1}


@pytest.mark.parametrize("summary_only", ["rerun", "second"])
def test_a_summary_only_run_without_the_objective_does_not_inherit_it(
    monkeypatch: pytest.MonkeyPatch, summary_only: str
) -> None:
    """A user summary key counts as logging even when W&B reports zero history rows."""
    from phasesweep.evidence.wandb import WandbNothingToScoreError

    runs = {"opener": _run("killed", {"x": 0.5}), "rerun": _run("finished", {"_wandb": {}})}
    runs[summary_only] = _run("finished", {"_wandb": {}, "runtime": 0})
    _serve(monkeypatch, runs)
    with pytest.raises(WandbNothingToScoreError, match="'x'"):
        _poll("opener", "second")


@pytest.mark.parametrize("own", ["missing", "empty"])
@pytest.mark.parametrize("state", ["killed", "running"])
def test_a_rerun_that_logged_nothing_is_scored_from_its_newest_closed_earlier_run(
    monkeypatch: pytest.MonkeyPatch, own: str, state: str
) -> None:
    """A rerun whose run is missing at the deadline, or finished with zero rows and no user summary keys, is scored from the newest earlier run W&B holds when that run's session closed; one still ``running`` leaves nothing to score, saying which run and why.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the SDK client and clock.
    :param str own: Whether the rerun's run is missing or empty.
    :param str state: The newest earlier run's state.
    """
    from phasesweep.evidence.wandb import WandbNothingToScoreError

    runs = {
        "opener": _run("finished", {"_step": 0, "x": 0.9}, _rows(0.9)),
        "second": _run(state, {"_step": 1, "x": 0.5}, _rows(0.6, 0.5)),
    }
    if own == "empty":
        runs["rerun"] = _run("finished", {})
    clock, _ = _serve(monkeypatch, runs)
    subject = (
        "was not found"
        if own == "missing"
        else "finished without logging any rows or user summary keys"
    )
    if state == "killed":
        capture = _poll("opener", "second", "vanished")
        assert capture["values"] == {"x": 0.5}
        assert capture["earlier_attempt"] == "second" and capture["deadline_verdict"] is True
    else:
        with pytest.raises(WandbNothingToScoreError) as caught:
            _poll("opener", "second", "vanished")
        assert caught.value.cause == (
            f"{subject}, and its trial's newest earlier run e/p/second is 'running', so it may "
            "lack its session's last row"
        )
    assert clock.now == 240


@pytest.mark.parametrize("state", ["killed", "running"])
def test_an_earlier_run_that_closed_without_logging_is_skipped_for_the_one_before_it(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    """A second attempt that found the training finished and closed with zero rows and no user summary keys is passed over as a missing run is, so a third that logged nothing is scored from the opener; a row-less second run still ``running`` may have trained unseen and leaves nothing to score.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the SDK client and clock.
    :param str state: The row-less second run's state.
    """
    from phasesweep.evidence.wandb import WandbNothingToScoreError

    runs = {
        "opener": _run("killed", {"_step": 1, "x": 0.5}, _rows(0.6, 0.5)),
        "second": _run(state, {"_wandb": {"runtime": 3}}),
        "rerun": _run("finished", {"_wandb": {"runtime": 2}}),
    }
    clock, _ = _serve(monkeypatch, runs)
    if state == "killed":
        capture = _poll("opener", "second")
        assert capture["values"] == {"x": 0.5}
        assert capture["earlier_attempt"] == "opener" and capture["deadline_verdict"] is True
    else:
        with pytest.raises(WandbNothingToScoreError) as caught:
            _poll("opener", "second")
        assert caught.value.cause == (
            "finished without logging any rows or user summary keys, and its trial's newest earlier run e/p/second is "
            "'running', so it may lack its session's last row"
        )
    assert clock.now == 240


def test_an_earlier_run_that_never_closed_is_read_as_wandb_holds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An earlier run whose session was killed outright stays ``crashed`` with its first-row summary and no history; a native min over it with an axis is not waited for, and is returned at the first poll from the rows W&B holds."""
    crashed = _run("crashed", {"_step": 0, "eval/loss.min": 0.4})
    crashed.history_keys = {"lastStep": -1, "keys": {}}
    rerun = _run(
        "finished", {"_step": 2, "eval/loss.min": 0.1}, _rows(0.2, 0.1, 0.15, axis=(25, 30, 40))
    )
    clock, _ = _serve(monkeypatch, {"opener": crashed, "rerun": rerun})
    capture = _poll(
        "opener",
        required=("eval/loss.min",),
        objective_key="eval/loss.min",
        evaluation_axis="eval/step",
    )
    assert capture["values"] == {"eval/loss.min": 0.1} and "deadline_verdict" not in capture
    assert clock.now == 0


@pytest.mark.parametrize("rerun", [True, False], ids=["rerun", "first-attempt"])
def test_a_run_that_logged_rows_without_the_objective_is_not_scored_by_an_earlier_run(
    monkeypatch: pytest.MonkeyPatch, rerun: bool
) -> None:
    """A rerun's finished run that logged rows but not the objective at the deadline leaves nothing to score though an earlier run holds it; a first attempt's is a missing key.

    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the SDK client and clock.
    :param bool rerun: Whether the run has an earlier run in its trial.
    """
    from phasesweep.evidence.wandb import WandbMissingKeysError, WandbNothingToScoreError

    runs = {
        "opener": _run("killed", {"_step": 0, "x": 0.5}, _rows(0.5)),
        "rerun": _run("finished", {"_step": 1, "eval/loss": 0.3}, _rows(0.4, 0.3)),
    }
    _serve(monkeypatch, runs)
    if not rerun:
        with pytest.raises(WandbMissingKeysError):
            _poll()
        return
    with pytest.raises(WandbNothingToScoreError) as caught:
        _poll("opener")
    assert caught.value.cause == (
        "finished with logged data but without 'x', so its trial's earlier runs do not score it"
    )


def test_an_earlier_run_that_scored_the_objective_is_named_and_read_for_its_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader is given each earlier attempt's run at the destination that attempt recorded; when one scores the capture, the objective's source names it, the capture keeps the attempt's own run as its identity, and the position is read in that earlier run.

    :param Path tmp_path: Scratch directory holding the attempts' records.
    :param pytest.MonkeyPatch monkeypatch: Fixture replacing the reader.
    """
    earlier_dir = tmp_path / "attempts" / "earlier-attempt"
    earlier_dir.mkdir(parents=True)
    recorded = {"base_url": "https://api.wandb.ai", "entity": "e", "project": "old"}
    (earlier_dir / EVIDENCE_BASELINE_FILE).write_text(
        json.dumps(
            {
                "schema_version": EVIDENCE_BASELINE_SCHEMA_VERSION,
                "reused": False,
                "files": {},
                "launched_at": None,
                "wandb_destination": recorded,
            }
        )
    )
    polled: list[object] = []
    located: list[tuple[str, str]] = []

    def poll(**kwargs: object) -> dict[str, object]:
        polled.append(kwargs["earlier"])
        return {
            "values": {"x": 0.5},
            "present_keys": [],
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "deadline_verdict": True,
            "earlier_attempt": "earlier-attempt",
        }

    def locate(*, project: str, run_id: str, **_: object) -> int:
        located.append((project, run_id))
        return 7

    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", poll)
    monkeypatch.setattr("phasesweep.evidence.evaluation.locate_wandb_position", locate)
    extractor = WandbExtractor(type="wandb", entity="e", project="new", metric_key="x")
    ctx = replace(
        make_trial_context(tmp_path),
        attempt_dir=tmp_path / "attempts" / "attempt-test",
        earlier_attempts=(("generation-old", "earlier-attempt"),),
    )
    provenance: dict[str, object] = {}
    assert run_extractor(ctx, extractor, provenance=provenance) == 0.5
    assert polled == [[{**recorded, "run_id": "earlier-attempt"}]]
    source, capture = provenance["source"], provenance["remote_capture"]
    assert isinstance(source, dict) and isinstance(capture, dict)
    assert source["run_id"] == "earlier-attempt" and capture["run_id"] == "attempt-test"
    assert ctx.earlier_files == ["wandb:earlier-attempt"]
    assert not {"deadline_verdict", "earlier_attempt"} & set(capture)
    locate_objective_position(ctx, extractor, provenance)
    assert located == [("old", "earlier-attempt")]
    assert source["evaluation"] == {"progress": {"axis": "_step", "value": 7}}


def test_env_wandb_resume_is_refused_whatever_its_value() -> None:
    """A configured WANDB_RESUME conflicts with the new run every attempt starts, so config load refuses it by name."""
    extractor = WandbExtractor(type="wandb", entity="e", project="p", metric_key="x")
    with pytest.raises(
        ValueError,
        match=(
            r"env\.WANDB_RESUME conflicts with the resume policy PhaseSweep sets: every attempt "
            r"starts a new W&B run; remove it\."
        ),
    ):
        make_experiment(metric=Metric(extractor=extractor), env={"WANDB_RESUME": "never"})


def _fake_wandb(install_sdk: Callable[[str], None], root: Path) -> SimpleNamespace:
    """Serve the real reader the runs trainers log by writing them to ``runs``.

    ``runs`` maps a run ID to its ``state`` and ``summary``; a run not in it is missing. Every
    lookup appends the path it asked for to ``lookups``, and while ``hang`` exists a lookup
    touches ``hung`` and blocks.

    :param Callable[[str], None] install_sdk: The ``wandb_worker_sdk`` fixture's installer.
    :param Path root: Directory holding the run, lookup and hang files.
    :return SimpleNamespace: ``runs``, ``lookups``, ``hang`` and ``hung`` paths.
    """
    files = SimpleNamespace(
        runs=root / "runs.json", lookups=root / "lookups", hang=root / "hang", hung=root / "hung"
    )
    install_sdk(f"""
        import json, time
        from pathlib import Path
        from types import SimpleNamespace

        class Api:
            def __init__(self, **kwargs):
                pass

            def run(self, path):
                with Path({str(files.lookups)!r}).open("a") as out:
                    out.write(path + "\\n")
                if Path({str(files.hang)!r}).exists():
                    Path({str(files.hung)!r}).touch()
                    time.sleep(60)
                runs = Path({str(files.runs)!r})
                served = json.loads(runs.read_text()) if runs.exists() else {{}}
                run = served.get(path.rsplit("/", 1)[-1])
                if run is None:
                    raise ValueError("Could not find run " + path)
                return SimpleNamespace(
                    state=run["state"],
                    summary_metrics=run["summary"],
                    history=lambda **_: [],
                    scan_history=lambda **_: [],
                )
    """)
    return files


def _log_run(runs: Path) -> str:
    """Return a trainer line defining ``log_run(state, x)``, which logs the attempt's run to ``runs``.

    :param Path runs: The fake W&B server's run file.
    :return str: One line of Python that needs ``json``, ``os`` and ``Path``.
    """
    served = (
        f"(json.loads(Path({str(runs)!r}).read_text()) if Path({str(runs)!r}).exists() else {{}})"
    )
    entry = '{"state": state, "summary": {"x": x}}'
    return (
        f"log_run = lambda state, x: Path({str(runs)!r}).write_text(json.dumps("
        f'{{**{served}, os.environ["WANDB_RUN_ID"]: {entry}}}))'
    )


def _interrupted_wandb_trial(
    tmp_path: Path, tracker: SimpleNamespace, *, rerun_logs: bool
) -> Experiment:
    """Leave a persistent W&B trial RUNNING after a runner SIGKILL, its trainer still running.

    Every launch appends its ``WANDB_RUN_ID``, ``WANDB_RESUME`` and attempt ID to ``envs``. The
    first logs ``x=0.5`` to a run its session closed as ``killed``, touches ``first`` and sleeps
    until the next run's recovery stops it; a later attempt finds the training finished and
    exits 0, logging ``x=1.0`` to a finished run of its own only when ``rerun_logs``.

    :param Path tmp_path: Scratch directory.
    :param SimpleNamespace tracker: The fake W&B server's files.
    :param bool rerun_logs: Whether a rerun logs a run of its own.
    :return Experiment: The one-trial experiment.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import json, os, time
        from pathlib import Path
        {_log_run(tracker.runs)}
        names =["WANDB_RUN_ID", "WANDB_RESUME", "PHASESWEEP_ATTEMPT_ID"]
        with Path({str(tmp_path / "envs")!r}).open("a") as envs:
            envs.write(json.dumps([os.environ.get(name) for name in names]) + "\\n")
        first = Path({str(tmp_path / "first")!r})
        if not first.exists():
            log_run("killed", 0.5)
            first.touch()
            time.sleep(60)
        elif {rerun_logs!r}:
            log_run("finished", 1.0)
        """,
    )
    extractor = WandbExtractor(
        type="wandb",
        base_url=_BASE_URL,
        entity="e",
        project="p",
        metric_key="x",
        poll_seconds=0.05,
        timeout_seconds=2 if not rerun_logs else 30,
    )
    experiment = make_experiment(
        persistent=tmp_path, trainer=trainer, metric=Metric(extractor=extractor), n_trials=1
    )
    with runner_once(
        write_experiment_config(tmp_path / "experiment.yaml", experiment), tmp_path / "first"
    ):
        pass
    return experiment


def _launch_envs(tmp_path: Path) -> list[list[str | None]]:
    """Read what each launch recorded.

    :param Path tmp_path: Scratch directory holding ``envs``.
    :return list[list[str | None]]: ``[WANDB_RUN_ID, WANDB_RESUME, attempt ID]`` per launch.
    """
    return [json.loads(line) for line in (tmp_path / "envs").read_text().splitlines()]


@pytest.mark.integration
def test_every_attempt_logs_its_own_run_and_a_rerun_that_logs_is_scored_from_its_own(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """The opener and its rerun in the same directory each launch with their own attempt ID as WANDB_RUN_ID and WANDB_RESUME=never; the rerun is scored from its own run, which the capture names, and verifies as a winner."""
    tracker = _fake_wandb(wandb_worker_sdk, tmp_path)
    experiment = _interrupted_wandb_trial(tmp_path, tracker, rerun_logs=True)

    winner = run_experiment(experiment)["p"]
    first, rerun = phase_trials(experiment)
    opener, rerun_id = first.user_attrs[ATTEMPT_ID_ATTR], rerun.user_attrs[ATTEMPT_ID_ATTR]
    assert rerun.user_attrs[RERUN_OF_ATTR] == first.number
    assert winner.trial_number == rerun.number and winner.metric == 1.0
    assert _launch_envs(tmp_path) == [[opener, "never", opener], [rerun_id, "never", rerun_id]]
    assert EARLIER_ATTEMPT_RESULT_ATTR not in rerun.user_attrs
    provenance = json.loads(rerun.user_attrs[OBJECTIVE_PROVENANCE_ATTR])
    assert provenance["remote_capture"]["run_id"] == provenance["source"]["run_id"] == rerun_id
    _verify_winner_evidence(experiment, experiment.phases[0], winner)


@pytest.mark.integration
def test_a_rerun_that_logs_nothing_is_scored_from_its_earlier_run_and_verifies(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """A rerun whose trainer found the training finished and never opened a run is scored at the deadline from the opener's closed run without a second launch: the result is disclosed as an earlier attempt's, its source names the opener's run, and it verifies as a winner."""
    tracker = _fake_wandb(wandb_worker_sdk, tmp_path)
    experiment = _interrupted_wandb_trial(tmp_path, tracker, rerun_logs=False)

    winner = run_experiment(experiment)["p"]
    first, rerun = phase_trials(experiment)
    opener, rerun_id = first.user_attrs[ATTEMPT_ID_ATTR], rerun.user_attrs[ATTEMPT_ID_ATTR]
    assert winner.trial_number == rerun.number and winner.metric == 0.5
    assert rerun.user_attrs[EARLIER_ATTEMPT_RESULT_ATTR] is True
    provenance = json.loads(rerun.user_attrs[OBJECTIVE_PROVENANCE_ATTR])
    assert provenance["source"]["run_id"] == opener
    assert provenance["remote_capture"]["run_id"] == rerun_id
    assert len(_launch_envs(tmp_path)) == 2
    assert {f"e/p/{opener}", f"e/p/{rerun_id}"} <= set(tracker.lookups.read_text().split())
    _verify_winner_evidence(experiment, experiment.phases[0], winner)


@pytest.mark.integration
def test_recovery_after_a_runner_kill_reads_the_reruns_own_run(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """A rerun whose reader was killed with its runner is read again by the next run, from the rerun's own run, without a second launch."""
    tracker = _fake_wandb(wandb_worker_sdk, tmp_path)
    experiment = _interrupted_wandb_trial(tmp_path, tracker, rerun_logs=True)
    tracker.hang.touch()
    reader = None
    try:
        with runner_once(tmp_path / "experiment.yaml", tracker.hung):
            _first, rerun = phase_trials(experiment)
            attempt_dir = _trial_attempt_dir(rerun)
            assert attempt_dir is not None
            reader = reaper.read_stale_process_identity(
                attempt_dir, expected_attempt_id=rerun.user_attrs[ATTEMPT_ID_ATTR]
            )
        tracker.hang.unlink()
        tracker.lookups.write_text("")

        winner = run_experiment(experiment)["p"]
        _first, rerun = phase_trials(experiment)
        rerun_id = rerun.user_attrs[ATTEMPT_ID_ATTR]
        assert winner.trial_number == rerun.number and winner.metric == 1.0
        assert len(_launch_envs(tmp_path)) == 2
        assert set(tracker.lookups.read_text().splitlines()) == {f"e/p/{rerun_id}"}
    finally:
        if reader is not None:
            reaper.cleanup_stale_trial_process(reader, grace_seconds=0.2)


@pytest.mark.integration
def test_recovery_stops_the_reader_of_a_killed_reread(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """The next run stops the registered reader and resumes the fill without training."""
    tracker = _fake_wandb(wandb_worker_sdk, tmp_path)
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import json, os
        from pathlib import Path
        {_log_run(tracker.runs)}
        with Path({str(tmp_path / "launches")!r}).open("a") as out:
            print("launch", file=out)
        log_run("finished", 1.0)
        """,
    )
    extractor = WandbExtractor(
        type="wandb",
        base_url=_BASE_URL,
        entity="e",
        project="p",
        metric_key="x",
        poll_seconds=0.05,
        timeout_seconds=30,
    )
    original = make_experiment(
        persistent=tmp_path, trainer=trainer, metric=Metric(extractor=extractor), n_trials=1
    )
    run_experiment(original)
    (trial,) = phase_trials(original)
    source_slot = _trial_attempt_dir(trial)
    assert source_slot is not None
    source_identity = reaper.read_stale_process_identity(
        source_slot, expected_attempt_id=trial.user_attrs[ATTEMPT_ID_ATTR]
    )
    # A new gate changes what the kept trial's evidence reads as, so the next run re-reads it.
    gate = WandbSummaryRequiredGate(
        type="wandb_summary_required",
        base_url=_BASE_URL,
        entity="e",
        project="p",
        keys=["x"],
        poll_seconds=0.05,
        timeout_seconds=30,
    )
    gated = original.phases[0].model_copy(update={"gates": [gate]})
    changed = original.model_copy(update={"phases": [gated]})
    config = write_experiment_config(tmp_path / "changed.yaml", changed)
    tracker.hang.touch()
    reader = None
    try:
        with runner_once(config, tracker.hung):
            registry = list(_attempts_dir(original).glob("*.json"))
            if registry:
                (entry_path,) = registry
                entry = json.loads(entry_path.read_text())
                worker_path = Path(entry["attempt_dir"])
                worker_id = entry["attempt_id"]
            else:
                worker_path = source_slot
                worker_id = trial.user_attrs[ATTEMPT_ID_ATTR]
            reader = reaper.read_stale_process_identity(worker_path, expected_attempt_id=worker_id)
        tracker.hang.unlink()
        assert run_experiment(changed)["p"].metric == 1
        assert reaper.stale_trial_process_decision(reader).action == "gone"
        assert (
            reaper.read_stale_process_identity(
                source_slot, expected_attempt_id=trial.user_attrs[ATTEMPT_ID_ATTR]
            )
            == source_identity
        )
        assert (tmp_path / "launches").read_text().splitlines() == ["launch"]
    finally:
        if reader is not None and reaper.stale_trial_process_decision(reader).action != "gone":
            reaper.cleanup_stale_trial_process(reader, grace_seconds=0.2)
